import torch
from torch import nn
import numpy as np
from .get_psi import get_psi
from .get_mhat import spherical_wavelet_family
from scipy.special import sph_harm
import math
import time

def associated_legendre_polynomial(l, m, x):
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt((1 - x) * (1 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll

def SH_renormalization(l, m):
    return math.sqrt((2.0 * l + 1.0) * math.factorial(l - m) / \
        (4 * math.pi * math.factorial(l + m)))

def SH(m, l, phi, theta):
    if m == 0:
        return SH_renormalization(l, m) * associated_legendre_polynomial(l, m, torch.cos(theta))
    elif m > 0:
        return math.sqrt(2.0) * SH_renormalization(l, m) * \
            torch.cos(m * phi) * associated_legendre_polynomial(l, m, torch.cos(theta))
    else:
        return math.sqrt(2.0) * SH_renormalization(l, -m) * \
            torch.sin(-m * phi) * associated_legendre_polynomial(l, -m, torch.cos(theta))
    
import numpy as np

def fibonacci_sphere(num_points):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)

def cartesian_to_euler(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # inclination angle (from z-axis down)
    phi = np.arctan2(y, x)    # azimuthal angle (in x-y plane from x-axis)
    psi = phi          # yaw (rotation around z-axis)
    theta = theta      # pitch (rotation around y-axis)
    phi = 0            # roll (rotation around x-axis, set to 0 as it's not defined for a point on a sphere)

    return np.degrees(psi), np.degrees(theta), np.degrees(phi)

def generate_sphere_grid(num_points):
    points = fibonacci_sphere(num_points)
    euler_angles = [cartesian_to_euler(x, y, z) for x, y, z in points]
    return euler_angles

class Wavelets(nn.Module):
    def __init__(self, hparams):
        """
        legendre_polys: determines the number of legendre polynomials.
                        more polynomials lead more fine-grained resolutions
        calculation of spherical harmonics:
            analytic uses pre-computed equations. This is exact, but works only up to degree 50,
            closed-form uses one equation but is computationally slower (especially for high degrees)
        """
        super(Wavelets, self).__init__()

        self.mode, self.max_scale, self.max_rotations, self.embedding_dim = None, None, None, None
        wavelet_params = hparams['embedding']
        try:
            self.time = hparams['data']['time_idx']
            self.max_scale, self.max_rotations = wavelet_params['max_scale'], wavelet_params['max_rotations']
            self.embedding_dim = self.max_scale * self.max_rotations**2 * int(np.ceil(self.max_rotations/2))
            self.embedding_dim = self.max_scale * (self.max_rotations**3)
            self.embedding_dim = self.max_scale * self.max_rotations
            if self.time:
                self.embedding_dim += 1
            self.k_val = wavelet_params['k_val']
            self.scale_factor = wavelet_params['scale_factor']
            self.scale_shift = wavelet_params['scale_shift']
            self.dilation_step = wavelet_params['dilation_step']
        except:
            print("Wavelet parameters not properly initialized.")

        self.rotation_vals = generate_sphere_grid(self.max_rotations)

    def get_needlet(self, j, k, theta, phi):
        return torch.from_numpy(get_psi(2, j, k, theta.numpy(), phi.numpy()))
    
    # TODO:
    # Try this format again
    def _spherical_wavelet(self, j, l, m, phi, theta):
        Y_lm = SH(m, l, phi, theta)
        #Y_lm = 1.
        psi = 2**((j)) * Y_lm * torch.sin(2**(j) * (np.pi - theta))
        return psi.real, Y_lm.real

    # BELOW IS USING NEEDLETS
    # def forward(self, lonlat):
    #     #import pdb
    #     #pdb.set_trace()
    #     lon, lat = lonlat[:, 0].unsqueeze(1), lonlat[:, 1].unsqueeze(1)

    #     # convert degree to rad
    #     phi = torch.deg2rad(90 - lat)
    #     theta = torch.deg2rad(lon)

    #     Y = []
    #     for j in range(self.num_scales):
    #         N_j = 2 ** (2 * j)  # Number of needlets at scale j
    #         for k in range(1, N_j + 1):
    #             y = self.get_needlet(j, k, theta, phi)
    #             # if isinstance(y, float):
    #             #    y = y * torch.ones_like(lat)
    #             Y.append(y)
    #     res = torch.stack(Y, dim=-1).to(torch.float64).squeeze()
    #     #print(res.shape)
    #     return res

    def forward(self, lonlat):
        
        lon, lat = lonlat[:, 0].unsqueeze(1), lonlat[:, 1].unsqueeze(1)
        Y = []
        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(90 + lat)

        scales = [2**(-(j + self.scale_shift) / self.dilation_step) for j in range(0, self.max_scale)]
        for j in scales:
            for alpha, beta, gamma in self.rotation_vals:
                Y.append(spherical_wavelet_family(theta, phi, j, (alpha, beta, gamma), 
                                                self.k_val, scale_factor=self.scale_factor) / (j**3))
        Y = torch.squeeze(torch.stack(Y, dim=-1)).to(lonlat.dtype)
        return Y