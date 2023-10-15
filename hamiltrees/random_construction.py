"""
    Construction of random Hamiltonians and tensors.
"""

import hamiltrees as ham

import numpy as np
import pytenet as ptn
import qib
import fermitensor as ftn


def construct_random_coefficients(L, rng):
    """
    Construct random coefficients within range `rng` for a molecular Hamiltonian with L sites.
    """
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
    Construct a random molecular Hamiltonian in second quantization formulation with L sites,
    using physicists' convention for the interaction term (note ordering of k and \ell) (dimensions (2^k,2^k)):
    Returns a MolecularHamiltonian from qib library.

    .. math::

        H = \sum_{i,j} h_{i,j} a^{\dagger}_i a_j 
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

def construct_random_partitioned_hamiltonian_FO(L, LA, rng):
    """
    Construct a random partitioned molecular Hamiltonian acc. to the paper.
    Returns a FieldOperator from qib library.

    Args:
        L: number of sites
        LA: site where the Hamiltonian will be partitioned
        rng: range of random coefficients
    """
    H = construct_random_molecular_hamiltonian(L, rng)
    return ham.block_partitioning.construct_partitioned_hamiltonian_FO(H, LA)
    
def construct_random_partitioned_hamiltonian_MPO(L, LA, rng):
    """
    Construct a random partitioned molecular Hamiltonian acc. to the paper as MPO.
    Returns an MPO from pytenet library.

    Args:
        L: number of sites
        LA: site where the Hamiltonian will be partitioned
        rng: range of random coefficients
    """
    H = construct_random_molecular_hamiltonian(L, rng)
    return ham.block_partitioning_MPO.construct_partitioned_hamiltonian_as_MPO(H, LA)

    
def construct_random_fMPS(L):
    """
    Construct a random fermionic MPS with L tensors.
    Returns an fMPS from fermitensor library.
    """
    rng = np.random.default_rng()

    # physical and virtual bond dimensions
    d = 2 # because fermionic
    De = [1] + ((L-1)*[5]) + [1]
    Do = [1] + ((L-1)*[4]) + [1]
    
    A = [ftn.fMPSTensor(0.5*ftn.crandn((d//2, De[i], De[i+1]), rng),
                        0.5*ftn.crandn((d//2, De[i], Do[i+1]), rng),
                        0.5*ftn.crandn((d//2, Do[i], De[i+1]), rng),
                        0.5*ftn.crandn((d//2, Do[i], Do[i+1]), rng)) for i in range(L)]
    
    return ftn.fMPS(A)

def construct_random_MPS(L):
    """
    Construct a random MPS with L tensors.
    The i-th MPS tensor mps.A[i] is expected to have dimensions (2, D[i], D[i+1])
    Returns an MPS from pytenet library
    """
    rng = np.random.default_rng()

    # physical and virtual bond dimensions
    d = 2 # because fermionic
    D = [1] + [np.random.randint(1, 10) for i in range(L-1)] + [1]
    
    mps = ptn.MPS(rng.integers(-2, 3, size=d), 
                   [rng.integers(-2, 3, size=Di) for Di in D], 
                   fill='random', rng=rng)

    return mps
