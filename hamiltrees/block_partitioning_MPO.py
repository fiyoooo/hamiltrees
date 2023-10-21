"""
Partitioning of a molecular Hamiltonian in second quantization representation
into two regions A and B as MPO
"""

import numpy as np
import pytenet as ptn
import qib


# matrices for fermionic operators
CREATE = np.array([[0., 0.], [1.,  0.]])
ANNIHIL = np.array([[0., 1.], [0.,  0.]])
I = np.identity(2)
    

# MPO for simplified Hamiltonian

def A(j: int, isites, L: int, D: int, tkin):
    """
    Constructs j-th tensor for MPO of Hamiltonian: sum_{i in isites, j in jsites} t_ij * a_i^dagger * a_j.
    
    Args:
        j: 
        isites: range of i
        D: virtual bond dimensions
        tkin: coefficients
    """
    # check if left- or rightmost tensor created
    left = j == 0
    right = j == L-1

    if left: # if left side
        A = np.zeros((2,2,1,D), dtype = 'complex_') # row vector
    elif right: # if right side
        A = np.zeros((2,2,D,1), dtype = 'complex_') # column vector
    else:
        A = np.zeros((2,2,D,D), dtype = 'complex_') # first physical then virtual dimensions
    
    # diagonal of identity matrices
    for i in range(min(A.shape[2], A.shape[3])):
        A[:,:,i,i] = I
    
    for i in isites:
        if not left:
            A[:,:,i+1,0] = tkin[i,j] * ANNIHIL
        if not right:
            A[:,:,-1,i+1] = CREATE

    return A

def construct_simplified_MPO(regionA, regionB, L, tkin):
    """
    Constructs MPO for Hamiltonian: sum_{i in A, j in B} a_i^t t_ij a_j.
    """
    qd = np.zeros(2) # physical quantum numbers at each site (same for all sites)
    D = L+2
    qD = [[0]] + (L-1)*[D*[0]] + [[0]] # virtual bond quantum numbers (list of quantum number lists)
    
    MPO = ptn.MPO(qd, qD)
    for j in regionB:
        MPO.A[j] = A(j, regionA, L, D, tkin)
    
    return MPO


# first draft and placeholders for the whole MPO

def construct_part_of_MPO(sites, L, tkin, vint):
    """
    Construct the hamiltonian on the given sites as MPO.
    """
    # TODO

    qd = np.zeros(2) # physical quantum numbers at each site (same for all sites)
    # identity MPO as placeholder
    mpo = ptn.MPO.identity(qd, L)

    return mpo

def construct_interacting_MPO(regionA, regionB, L, tkin, vint):
    """
    Construct the interacting hamiltonian H_{AB} in Equ. (10) as MPO.
    """
    # TODO

    qd = np.zeros(2) # physical quantum numbers at each site (same for all sites)
    # identity MPO as placeholder
    mpo = ptn.MPO.identity(qd, L)

    return mpo

def construct_partitioned_hamiltonian_as_MPO(H: qib.operator.MolecularHamiltonian, x: int):
    """
    Construct the partitioned molecular Hamiltonian H as an MPO.
    """
    L = H.field.lattice.nsites

    regionA = range(0, x)
    regionB = range(x, L)

    # construct H_A Hamiltonian
    MPO_A = construct_part_of_MPO(regionA, L, H.tkin, H.vint)
    
    # construct H_B Hamiltonian
    MPO_B = construct_part_of_MPO(regionB, L, H.tkin, H.vint)
    
    # construct H_{AB} Hamiltonian
    MPO_AB = construct_interacting_MPO(regionA, regionB, L, H.tkin, H.vint)
    
    result = MPO_A + MPO_AB + MPO_B # addition of MPOs

    return result
