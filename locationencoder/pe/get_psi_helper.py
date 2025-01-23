import numpy as np
from scipy.special import legendre
from scipy import integrate

def sph2cart(azimuth, elevation, r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    
    return x, y, z

def get_bl_vector(B, j_max, l_max):
    j_max, l_max = int(j_max), int(l_max)
    bl_vector = np.zeros((j_max + 1, l_max))
    for j in range(j_max + 1):
        for l in range(1, l_max + 1):
            bl_vector[j, l-1] = fun_b(l * 1.0 / B**j, B * 1.0) * (2.0 * l + 1.0) / (4 * np.pi)
    return bl_vector

def fun_b(x, B):
    get_res = get_f3(x/B, B) - get_f3(x, B*1.0)
    if get_res < 0.0:
        return 0.0
    else:
        return np.sqrt(get_res)

def get_f3(x, B):
    if x < 0:
        raise ValueError('x is not in the domain of f3!')
    elif x*1.0 <= 1./B:
        return 1.
    elif x <= 1.:
        return get_f2(1. - 2.*B/(B-1.) * (x - 1./B))
    else:
        return 0

def get_f2(u):
    def f1(x):
        return np.exp(-1 / (1 - x**2))
    if u == -1.0:
        return 0.0
    if u == 1.0:
        return 0.5
    numerator = integrate.quad(f1, -1, u)[0]

    # denominator is erf
    erf = 0.4440
    denominator = erf
    return numerator / denominator

def get_Nside(B, j):
    Nside = 1
    lb = int(B**(j + 1))
    while 2 * Nside < lb:
        Nside *= 2
    return Nside

def p_polynomial_value(m, n, x):
    if n < 0:
        return np.array([])
    
    v = np.zeros((m, n + 1))
    v[:, 0] = 1.0
    
    if n < 1:
        return v
    
    v[:, 1] = x
    
    for i in range(2, n + 1):
        v[:, i] = ((2 * i - 1) * x * v[:, i-1] - (i - 1) * v[:, i-2]) / i
    
    return v

def spneedlet_eval_fast(B, j, bl_vector, P, dist, sqrt_lambda):
    l_min = int(np.ceil(B**(j - 1)))
    l_max = int(np.floor(B**(j + 1)))
    n = len(dist)
    psi = np.zeros(n)
    
    for i in range(n):
        psi[i] = np.sum(bl_vector[j, l_min-1:l_max] * P[i, l_min:l_max+1])
    
    psi *= sqrt_lambda
    return psi