# Members: Dakota, Shane, Robert, and Ashton

import numpy as np
import matplotlib.pyplot as plt
from SherrattAshton_Lab10 import make_initialcond
from SherrattAshton_Lab10 import make_tridiagonal
from SherrattAshton_Lab10 import spectral_radius
from SherrattAshton_Lab10 import make_grid

# Part 1 - Ashton, Robbie

def advection1d(method, nspace, ntime, tau_rel, params):
    '''
    Description: function that solves the 1D advection equation using
    matrices.
    -----------------------------------------------------------------
    method : string
        "FTCS" or "Lax", scheme for solving advection equation
    nspace : int
        number of gridspaces
    ntime : int
        number of divisions of the time interval
    tau_rel : int, float
        ratio of timestep (tau) and max timestep (tau_crit)
    params : tuple (L, c)
        L : int, float
            length of computation region
        c : int, float
            wave speed
    '''
    # unpack params tuple
    L, c = params

    # create grid of computation region
    grid = make_grid(nspace, L)

    I = np.identity(nspace)

    # matrix for FTCS and Lax method
    B = make_tridiagonal(nspace, -1, 0, 1)
    B[0, nspace - 1] = -1 
    B[nspace - 1, 0] = 1
    # print(B)

    # matrix for Lax method
    C = np.abs(B)
    # print(C)

    # solve according to method given
    # FTCS method
    if method == 'FTCS':
        A = I - (tau_rel / 2) * B
        
    # Lax method
    if method == 'Lax':
        A = 0.5 * C - (tau_rel / 2) * B

    eig_value = spectral_radius(A)
    if eig_value > 1:
        print("WARNING: Solution is expected to be unstable")
    else:
        print("Solution is expected to be stable")

    # initial condition function values from Lab 10
    sigma0 = 0.2
    k0 = 35

    # set initial and boundary conditions
    a = np.zeros((nspace, ntime))
    a[:,0] = make_initialcond(sigma0, k0, grid)

    # Main loop 
    for istep in range(1, ntime):  
        a[:, istep]  = A.dot(a[:,istep-1])

    return a

print(advection1d("FTCS", 300, 500, 1, (5, 1)))
print(advection1d("Lax", 300, 500, 1, (5, 1)))
