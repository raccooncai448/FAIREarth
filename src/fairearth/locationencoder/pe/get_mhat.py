import numpy as np
import torch
from .spherical_harmonics_closed_form import associated_legendre_polynomial
import math
from scipy.special import sph_harm
from scipy.special import lpmv
from .spherical_harmonics_closed_form import SH as SH_closed_form


def rotate_spherical_coords(theta, phi, alpha, beta, gamma):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    R_z_alpha = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])
    
    R_y_beta = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    
    R_z_gamma = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    
    R = R_z_alpha @ R_y_beta @ R_z_gamma
    
    rotated_point = R @ np.array([x, y, z])
    
    x_rot, y_rot, z_rot = rotated_point
    theta_new = np.arccos(z_rot / np.sqrt(x_rot**2 + y_rot**2 + z_rot**2))
    phi_new = np.arctan2(y_rot, x_rot)
    
    return theta_new, phi_new

def mex_hat_wavelet(x, y):
    return (1 / np.pi) * (1 - 0.5*(x**2 + y**2)) * np.exp(-(x**2 + y**2) / 2)

def r_func(theta):
    return 2 * np.tan(theta / 2)

import numpy as np

def spherical_wavelet_proj(theta, phi, k, scale_factor=1, wavelet_type='gabor'):
    '''
    TODO:
    Implement/try the three wavelets explored in:
    https://arxiv.org/pdf/astro-ph/0506308 (pg 4)
    '''
    if wavelet_type == 'gabor':
        # Original Gabor directional wavelet
        scalar_factor = np.exp(-0.5 * np.tan(theta / 2)**2 / (scale_factor ** 2)) / (1 + np.cos(theta))
        return np.cos(k * np.tan(theta / 2) * np.cos(-phi) / scale_factor) * scalar_factor
    
    elif wavelet_type == 'mexican_hat':
        # Spherical Mexican Hat Wavelet (SMHW)
        # Convert spherical coordinates to r for planar approximation
        r = 2 * np.tan(theta / 2)  # Stereographic projection
        return 0.5 * (2 - r**2) * np.exp(-r**2 / 2)
    
    elif wavelet_type == 'butterfly':
        # Symmetric Butterfly Wavelet (SBW)
        # Convert spherical coordinates to x, y for planar approximation
        x = np.tan(theta / 2) * np.cos(phi)
        y = np.tan(theta / 2) * np.sin(phi)
        return x * np.exp(-(x**2 + y**2) / 2)
    
    else:
        raise ValueError(f"Unknown wavelet type: {wavelet_type}. " 
                        "Available options: 'gabor', 'mexican_hat', 'butterfly'")

def spherical_wavelet_family(theta, phi, dil_a, rot_rho, k, scale_factor=1, wavelet_type='butterfly'):
    alpha, beta, gamma = rot_rho
    # FIX FOR GPU TRAINING
    if not torch.cuda.is_available():
        theta, phi = theta[:, 0].numpy(), phi[:, 0].numpy()
    else:
        theta, phi = theta[:, 0].cpu().numpy(), phi[:, 0].cpu().numpy()
    theta, phi = rotate_spherical_coords(theta, phi, alpha, beta, gamma)

    dil_constant = (2 * dil_a) / ((dil_a**2 - 1) * np.cos(theta) + (dil_a**2) + 1)
    inv_theta, inv_phi = 2 * np.arctan2(np.tan(theta / 2), dil_a), phi
    
    wavelet = torch.tensor(dil_constant * spherical_wavelet_proj(inv_theta, inv_phi, 
                                                                 k=k, scale_factor=scale_factor, wavelet_type=wavelet_type))
    # if torch.cuda.is_available():
    #   wavelet = wavelet.cuda()
    return wavelet

def generating_function(m, l, L, dil_a):
    M = 1
    return np.exp((-0.5) * (np.square(l*dil_a - L) + np.square(m - M)))

def spherical_harmonic(l, m, phi, theta):
    l = int(l)
    m = int(m)
    norm = np.sqrt((2*l + 1) / (4 * np.pi) * np.math.factorial(l - abs(m)) / np.math.factorial(l + abs(m)))
    P_l_m = associated_legendre_polynomial(torch.tensor(l), torch.tensor(m), torch.tensor(np.cos(theta)))
    #P_l_m = lpmv(abs(m), l, np.cos(theta))
    
    Y_l_m = norm * P_l_m * np.exp(1j * m * phi)
    if m < 0:
        Y_l_m = (-1)**m * np.conj(Y_l_m)
    
    return Y_l_m

def reconstruct_harmonic(theta, phi, dil_a):
    sm = np.zeros_like(theta, dtype=complex)
    L = 10
    l_max = np.ceil((5+L) / dil_a)
    for l in range(int(l_max)+1):
        for m in range(-l, l+1):
            #legendre_poly = spherical_harmonic(l, m, phi, theta)
            #legendre_poly = associated_legendre_polynomial(l, m, torch.tensor(np.cos(theta))).numpy()
            legendre_poly = lpmv(m, l, np.cos(theta))
            sph_harm_scalar = (-1 ** m) * np.sqrt((2*l + 1)*np.math.factorial(l-m) 
                                                  / ((4*np.pi)*np.math.factorial(l+m)))
            sph_harmonic = np.multiply(legendre_poly, np.cos(m*phi)) * sph_harm_scalar
            scalar_factor = np.sqrt((2*l + 1) / (8*np.pi*np.pi))
            sm += sph_harmonic * scalar_factor * generating_function(m, l, L, dil_a)
    return torch.tensor(sm.real)
    

'''
BELOW IS WITHOUT PROJECTION
'''
# def spherical_wavelet_family(theta, phi, dil_a, rot_rho, k, scale_factor=1):
#     #import pdb
#     #pdb.set_trace()
#     alpha, beta, gamma = rot_rho
#     if True:
#         theta, phi = theta[:, 0].numpy(), phi[:, 0].numpy()
#     else:
#         theta, phi = theta[:, 0].cpu().numpy(), phi[:, 0].cpu().numpy()
#     theta, phi = rotate_spherical_coords(theta, phi, alpha, beta, gamma)

#     return reconstruct_harmonic(theta, phi, dil_a)
    # dil_constant = (2 * dil_a) / ((dil_a**2 - 1) * np.cos(theta) + (dil_a**2) + 1)
    # inv_theta, inv_phi = 2 * np.arctan2(np.tan(theta / 2), dil_a), phi
    
    # wavelet = torch.tensor(dil_constant * spherical_wavelet_proj(inv_theta, inv_phi, 
    #                                                              k=k, scale_factor=scale_factor))
    # # if torch.cuda.is_available():
    # #   wavelet = wavelet.cuda()
    # return wavelet
