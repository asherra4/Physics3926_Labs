import numpy as np
import matplotlib.pyplot as plt

# Part 1 : tridiagonal function - Ashton

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

# Test make_tridiagonal function
print(make_tridiagonal(5, 3, 1, 5))

# Part 2 : initial condition function - Ashton

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

# test make_initialcond function for Nspace = 300, L = 5, sigma0 = 0.2, k0 = 35
test_grid = make_grid(300, 5)
test_initial_cond = make_initialcond(0.2, 35, test_grid)

plt.plot(test_grid, test_initial_cond, label='a(x,0)')
plt.xlabel('Position')
plt.ylabel('Initial Conditions')
plt.title('Initial Conditions at Given Positions')
plt.legend()
plt.show()

# Part 3 : spectral radius function - Shane

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
