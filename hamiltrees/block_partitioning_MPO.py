"""
    Hamiltonian construction acc. to Eq. (9) as MPO.
"""

import numpy as np
import pytenet as ptn
import qib


# matrices for fermionic operators
CREATE = np.array([[0., 0.], [1.,  0.]])
ANNIHIL = np.array([[0., 1.], [0.,  0.]])
I = np.identity(2)
    

# MPO for simplified Hamiltonian

def A(j: int, jsites, isites, D: int, tkin):
    """
    Constructs j-th tensor for MPO of Hamiltonian: sum_{i in isites, j in jsites} t_ij * a_i^dagger * a_j.
    
    Args:
        j: 
        jsites: range of j
        isites: range of i
        D: virtual bond dimensions
        tkin: coefficients
    """
    # check if left- or rightmost tensor created
    left = (j-1) not in jsites
    right = (j+1) not in jsites

    if left: # if left side
        A = np.zeros((2,2,1,D)) # row vector
    elif right: # if right side
        A = np.zeros((2,2,D,1)) # column vector
    else:
        A = np.zeros((2,2,D,D)) # first physical then virtual dimensions
    
    A[:,:,0,0] = I
    A[:,:,-1,-1] = I
    
    for i in isites:
        if not left:
            A[:,:,i+1,0] = tkin[i,j] * ANNIHIL
            if not right:
                A[:,:,i+1,i+1] = I
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
        MPO.A[j] = A(j, regionB, regionA, D, tkin)
    
    return MPO


# first draft and placeholders for the whole MPO

def construct_part_of_MPO(sites, L, tkin, vint):
    """
    Construct the hamiltonian on the given sites as MPO.
    """
    # TODO
    return None

def construct_interacting_MPO(regionA, regionB, field, tkin, vint):
    """
    Construct the interacting hamiltonian H_{AB} in Equ. (10) as MPO.
    """
    # TODO
    return None

def construct_partitioned_hamiltonian_as_MPO(H: qib.operator.MolecularHamiltonian, LA: int):
    """
    Construct the partitioned molecular Hamiltonian H as an MPO.
    """
    L = H.field.lattice.nsites

    regionA = range(0, LA)
    regionB = range(LA, L)

    # construct H_A Hamiltonian
    MPO_A = construct_part_of_MPO(regionA, L, H.tkin, H.vint)
    
    # construct H_B Hamiltonian
    MPO_B = construct_part_of_MPO(regionB, L, H.tkin, H.vint)
    
    # construct H_{AB} Hamiltonian
    MPO_AB = construct_interacting_MPO(regionA, regionB, L, H.tkin, H.vint)
    
    result = MPO_A + MPO_AB + MPO_B # addition of MPOs

    return result