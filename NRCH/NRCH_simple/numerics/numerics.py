import numpy as np
import matplotlib.pyplot as plt

from numpy import pi, random
from matplotlib import animation
from numba import njit, objmode

fast = False
if fast: jit = njit
else: jit = lambda x : x

N = 200
M = 100_000

L = 10.
dx = L / N
dt = 0.1*dx**4
frames = 1000
skip = M//frames


i = np.arange(N)
D2 = np.zeros((N, N))
D2[i, i] = - 2 / dx**2
D2[i, (i+1)%N] = 1 / dx**2
D2[(i+1)%N, i] = 1 / dx**2
eps = np.array([[0, 1], [-1, 0]])

@jit
def f(phi, param):
    r, phibar, a, b = param
    m = ( r  - r * (phi[: ,0]**2 + phi[:, 1]**2) )[:, None] * phi
    m -= D2@phi
    m += a *phi @ eps
    m += random.normal(0, b, size=(N, 2))
    m = D2 @ m
    return m

@jit
def run_euler(param):
    r, p, a, b = param
    phi = np.zeros((N, 2))
    phit = np.empty((M//skip, N, 2))

    phi[:, 0] = p
    phit[0] = phi

    n2 = frames//10

    for i in range(1, frames):
        if ((i+1)//n2) - i//n2 == 1: 
            with objmode(): print("|", end='', flush=True)
        for j in range(0, skip):
            phi += f(phi, param) * dt
        phit[i] = phi
    print('')
    return phit



a = 6.
p = -.7
b = 0.1
u = 10.


from time import time
t = time()
run_euler(param=(-u, p, a, b))
print(time() - t)



# ps = [-.9, -.8, -0.73, -.65, -1/np.sqrt(2)]
# aa = [.55, .55, .55, .55, .5]
# param = [[-1, ps[i], aa[i]] for i in range(len(ps))]

# from multiprocessing import Pool

# with Pool(6) as pool:
#     pool.starmap(make_anim, param)

