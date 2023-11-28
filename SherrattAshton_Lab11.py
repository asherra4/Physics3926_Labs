# Members: Dakota, Shane, Robert, and Ashton

import numpy as np
import matplotlib.pyplot as plt

# Functions from Lab 10

def make_tridiagonal(N, b, d ,a):
    '''
    Description: function which produced a matrix with N rows and N columns
    with the value d on the diagonal, b on the lower off-diagonal and a on 
    the upper off-diagonal.
    -----------------------------------------------------------------------
    N : int
        size of square matrix
    b : float
        value on lower off-diagonal entries
    d : float
        value on diagonal entries
    a : float
        value on upper off-diagonal entries
    '''
    # create diagonal matrix with d on the diagonal
    diag = d * np.eye(N, M=N, k=0)
    # create matrix with b along lower off-diagonal
    L_diag = b * np.eye(N, M=N, k=-1)
    # create matric with a along upper off-diagonal
    U_diag = a * np.eye(N, M=N, k=1)

    # Sum previous 3 matrices to create the tridiagonal matrix
    tridiag = diag + L_diag + U_diag

    return tridiag

def make_grid(Nspace, L):
    '''
    Description: function that creates an array that extends from -L/2 to 
    L/2 in Nspace steps.
    -----------------------------------------------------------------------
    Nspace : int
        number of grid spaces
    L : float
        length of spacial dimension
    '''
    grid = np.linspace( -L/2 , L/2, Nspace )
    return grid

def make_initialcond(sigma0, k0, grid):
    '''
    Description: function that creates the values for a wavepacket at
    time = 0, for the positions given in the grid array
    -----------------------------------------------------------------------
    sigma0 : float
        standard deviation of guassian function
    k0 : float
        wavenumber
    '''
    # compute the exponent
    exponent = ( - grid ** 2 ) / ( 2 * sigma0 ** 2 ) 
    # compute the exponential and cosine factors seperately
    factor1 = np.exp( exponent )
    factor2 = np.cos( k0 * grid)
    result = factor1 * factor2
    return result

def spectral_radius(A):
    '''
    Description: function that takes a 2D array (Matrix) and returns the
    eigenvalue with a maximum absolute value.
    -----------------------------------------------------------------------
    A : ndarray of ints or floats
        Matrix for which spectral radius is to be computed
    '''
    # Compute the eigenvalues
    eigvals = np.linalg.eig(A)[0]
    # Find max
    eig_max = np.max(np.abs(eigvals))
    return eig_max

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
    x_grid = make_grid(nspace, L)
    t_grid = make_grid(ntime, L)

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
    if method.lower() == 'ftcs':
        A = I - (tau_rel / 2) * B
        
    # Lax method
    if method.lower() == 'lax':
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
    a[:,0] = make_initialcond(sigma0, k0, x_grid)

    # Main loop 
    for istep in range(1, ntime):  
        a[:, istep]  = A.dot(a[:,istep-1])

    return a, x_grid, t_grid

# Part 1 - Shane, Dakota

a, x, t = advection1d('lax', nspace=300, ntime=501, tau_rel=1, params=[5,1])

plotskip = 50
fig, ax = plt.subplots()
# Add vertical offset so plots don't overlap
yoffset = a[:,0].max() - a[:,0].min()
# Loop through t values in reverse order and plot each result
for i in np.arange(len(t)-1,-1,-plotskip): 
    ax.plot(x, a[:,i]+yoffset*i/plotskip,label = 't ={:.3f}'.format(t[i]))
ax.legend(bbox_to_anchor=(1, 1))
ax.set_xlabel('X position')
ax.set_ylabel('a(x,t) [offset]')
plt.show()
