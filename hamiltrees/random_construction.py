"""
    Construction of random Hamiltonians and tensors.
"""

import hamiltrees as ham

import numpy as np
import qib
import fermitensor as ftn


def construct_random_coefficients(L, rng):
    # Hamiltonian coefficients
    tkin = 0.5 * qib.util.crandn((L, L), rng)
    vint = 0.5 * qib.util.crandn((L, L, L, L), rng)
    # make hermitian
    tkin = 0.5 * (tkin + tkin.conj().T)
    vint = 0.5 * (vint + vint.conj().transpose((2, 3, 0, 1)))
    # add varchange symmetry
    vint = 0.5 * (vint + vint.transpose(1, 0, 3, 2))
    
    return (tkin, vint)

def construct_random_molecular_hamiltonian(L, rng):
    """
    Construct a random molecular Hamiltonian in second quantization formulation,
    using physicists' convention for the interaction term (note ordering of k and \ell) (dimensions (2^k,2^k)):

    .. math::

        H = (c +) \sum_{i,j} h_{i,j} a^{\dagger}_i a_j 
            + \\frac{1}{2} \sum_{i,j,k,\ell} v_{i,j,k,\ell} 
            a^{\dagger}_i a^{\dagger}_j a_{\ell} a_k
    """
    # underlying lattice
    latt = qib.lattice.FullyConnectedLattice((L,))
    field = qib.field.Field(qib.field.ParticleType.FERMION, latt)

    # random coefficients
    tkin, vint = construct_random_coefficients(L, rng)

    H = qib.operator.MolecularHamiltonian(field, 0., tkin, vint, qib.operator.MolecularHamiltonianSymmetry.HERMITIAN | qib.operator.MolecularHamiltonianSymmetry.VARCHANGE)
    return H

def construct_random_partitioned_hamiltonian(L, LA, rng):
    """
    Construct a random partitioned molecular Hamiltonian acc. to the paper.
    Returns a FieldOperator.
    """
    # underlying lattice
    latt = qib.lattice.FullyConnectedLattice((L,))
    field = qib.field.Field(qib.field.ParticleType.FERMION, latt)

    # random coefficients
    tkin, vint = construct_random_coefficients(L, rng)

    # sizes of regions A and B
    regionA = range(0, LA)
    regionB = range(LA, L)

    # H_A Hamiltonian
    HA = ham.block_partitioning.construct_part_of_hamiltonian(regionA, field, tkin, vint)
    
    # H_B Hamiltonian
    HB = ham.block_partitioning.construct_part_of_hamiltonian(regionB, field, tkin, vint)
    
    # H_{AB} Hamiltonian
    HAB = ham.block_partitioning.construct_interacting_hamiltonian(regionA, regionB, field, tkin, vint)

    return HA + HAB + HB
    
    
# copied test_as_vector from fermionic_tensor_networks.code.tests.test_fMPS
def construct_random_fMPS(L):
    rng = np.random.default_rng()

    # physical and virtual bond dimensions
    d = 2 # because fermionic
    #if parity == "even":
    De = [1] + ((L-1)*[5]) + [1]
    # [ 1,  7,  5,  1]
    Do = [1] + ((L-1)*[4]) + [1]
    # else:
    #     De = [13,  7,  5,  2]
    #     Do = [ 2, 11,  4, 13]

    # number of fermionic modes (or lattice sites)
    L = len(De) - 1
    A = [ftn.fMPSTensor(0.5*ftn.crandn((d//2, De[i], De[i+1]), rng),
                        0.5*ftn.crandn((d//2, De[i], Do[i+1]), rng),
                        0.5*ftn.crandn((d//2, Do[i], De[i+1]), rng),
                        0.5*ftn.crandn((d//2, Do[i], Do[i+1]), rng)) for i in range(L)]
    
    print(d**L)
    return ftn.fMPS(A)

def construct_random_MPS(L):
    """
    Construct a random MPS as list of L matrices.
    The i-th MPS tensor psi[i] is expected to have dimensions (n[i], D[i], D[i+1])
    """
    # logical dimensions
    n = [np.random.randint(1, 10) for i in range(L)] # TODO what

    # virtual bond dimensions (rather arbitrarily chosen) 
    D = [1, ] + [np.random.randint(1, 10) for i in range(L-1)] + [1, ]
    
    # random MPS matrices (the scaling factor keeps the norm of the full tensor in a reasonable range)
    np.random.seed(42)
    psi = [0.4 * qib.util.crandn((2, D[i], D[i+1])) for i in range(L)]
    
    return psi

def mps_to_full_tensor(Alist):
    """
    Construct the full tensor corresponding to the MPS tensors `Alist`.

    The i-th MPS tensor Alist[i] is expected to have dimensions (n[i], D[i], D[i+1]),
    with `n` the list of logical dimensions and `D` the list of virtual bond dimensions.
    """
    # consistency check: dummy singleton dimension
    assert Alist[0].ndim == 3 and Alist[0].shape[1] == 1
    # formally remove dummy singleton dimension
    T = np.reshape(Alist[0], (Alist[0].shape[0], Alist[0].shape[2]))
    # contract virtual bonds
    for i in range(1, len(Alist)):
        T = np.tensordot(T, Alist[i], axes=(-1, 1))
    # consistency check: trailing dummy singleton dimension
    assert T.shape[-1] == 1
    # formally remove trailing singleton dimension
    T = np.reshape(T, T.shape[:-1])
    return T