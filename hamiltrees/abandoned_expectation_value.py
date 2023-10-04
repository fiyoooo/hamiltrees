"""
    Calculation of expectation value with partitioned Hamiltonian.
"""

from typing import Sequence
import numpy as np
import qib


def apply_part_of_hamiltonian(H: qib.operator.MolecularHamiltonian, sites, psi: Sequence[any]):
    """
    Apply the hamiltonian on the given sites to an MPS.
    """
    print("\nStart applying part of Hamiltonian at sites ", sites)
    
    result = [np.zeros_like(M) for M in psi]

    # kinetic hopping term
    for i in sites:
        for j in sites:
            result += apply_create_op(i, 
                      apply_annihil_op(j, psi, H.tkin[i, j])) # TODO wie addieren
    
    # TODO schöner loopen
    # interaction term
    for i in sites:
        for j in sites:
            for k in sites:
                for l in sites:
                    print("Round: ", i,j,k,l)
                    result += apply_create_op(i, 
                              apply_create_op(j, 
                              apply_annihil_op(k, 
                              apply_annihil_op(l, psi, 0.5 * H.vint[i, j, l, k])))) # TODO wie addieren
    
    return result

def apply_complementary_p(i: int, j: int, sites, field: qib.field.Field, vint, psi: Sequence[any]):
    """
    Construct the complementary operator `P_{ij}^B` in Eq. (11).
    """
    L = field.lattice.nsites
    vint = np.asarray(vint)
    assert vint.shape == (L, L, L, L)
    
    result = [np.zeros_like(M) for M in psi]
    for k in sites:
        for l in sites:
            sint += apply_annihil_op(k, 
                    apply_annihil_op(l, psi, 0.5*vint[i, j, l, k]))

    return result

def apply_complementary_q(i: int, j: int, sites, field: qib.field.Field, vint, psi: Sequence[any]):
    """
    Construct the complementary operator `Q_{ij}^B` in Eq. (12).
    """
    L = field.lattice.nsites
    vint = np.asarray(vint)
    assert vint.shape == (L, L, L, L)
    
    result = [np.zeros_like(M) for M in psi]
    for k in sites:
        for l in sites:
            sint += apply_create_op(k, 
                    apply_annihil_op(l, psi, 0.5*((vint[i, k, j, l] + vint[k, i, l, j])/2 - vint[i, k, l, j])))

    return result

def apply_complementary_s(i: int, sites, field: qib.field.Field, tkin, vint, psi: Sequence[any]):
    """
    Apply the complementary operator `S_i^B` in Eq. (13) to an MPS.
    """
    L = field.lattice.nsites
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)
    assert tkin.shape == (L, L)
    assert vint.shape == (L, L, L, L)

    skin = [np.zeros_like(M) for M in psi]
    for j in sites:
        skin += apply_annihil_op(j, psi, 0.5*tkin[i, j])

    sint = [np.zeros_like(M) for M in psi]
    for j in sites:
        for k in sites:
            for l in sites:
                sint += apply_create_op(j, 
                        apply_annihil_op(k, 
                        apply_annihil_op(l, psi, 0.5*(vint[i, j, l, k] - vint[j, i, l, k]))))
    
    return skin #+ sint


def apply_create_op(i: int, psi: Sequence[any], coeff=1):
    """
    Apply fermionic creation operator to an MPS at site `i`.
    """
    op = np.array([[0,0],[1,0]])
    psi[i] = np.transpose(np.tensordot(psi[i], coeff*op, axes=(0, 0)), (2, 0, 1)) # TODO axes correct?
    
    return psi

def apply_annihil_op(i: int, psi: Sequence[any], coeff=1):
    """
    Apply fermionic annihilation operator to an MPS at site `i`.
    """
    op = np.array([[0,1],[0,0]])
    psi[i] = np.transpose(np.tensordot(psi[i], coeff*op, axes=(0, 0)), (2, 0, 1))
    
    return psi


def apply_interacting_hamiltonian(regionA, regionB, field: qib.field.Field, tkin, vint, psi: Sequence[any]):
    """
    Contruct the interacting hamiltonian H_{AB} in Equ. (10).
    """
    # TODO how to sum???
    S_A_psi = [np.zeros_like(M) for M in psi]
    for i in regionA: 
        S_A_psi += apply_create_op(i, 
                  apply_complementary_s(i, regionB, field, tkin, vint, psi)) 
   
    """
    S_A_psi = sum(apply_create_op(i, 
                  apply_complementary_s(i, regionB, field, tkin, vint, psi)) 
                  for i in regionA)
    S_B_psi = sum(apply_create_op(j, 
                  apply_complementary_s(j, regionA, field, tkin, vint, psi)) 
                  for j in regionB)
    P_psi = sum(apply_create_op(i,
                   apply_create_op(j,
                   apply_complementary_p(i, j, regionB, field, vint, psi))) 
                   for i in regionA for j in regionA)
    Q_psi = sum(apply_create_op(i, 
                   apply_annihil_op(j, 
                   apply_complementary_q(i, j, regionB, field, vint, psi))) 
                   for i in regionA for j in regionA)
    """

    HABhalf = S_A_psi #+ S_B_psi + P_psi + Q_psi
    
    return HABhalf + HABhalf.adjoint() # TODO how???

def apply_hamiltonian(H: qib.operator.MolecularHamiltonian, LA: int, psi: Sequence[any]):
    """
    Apply the hamiltonian H to the MPS psi as list of tensors.
    """
    L = H.field.lattice.nsites

    regionA = range(0, LA)
    regionB = range(LA, L)

    # apply H_A Hamiltonian
    H_A_psi = apply_part_of_hamiltonian(H, regionA, psi)
    
    # apply H_B Hamiltonian
    H_B_psi = apply_part_of_hamiltonian(H, regionB, psi)
    
    # apply H_{AB} Hamiltonian
    H_AB_psi = apply_interacting_hamiltonian(regionA, regionB, H.field, H.tkin, H.vint, psi)
    
    result = H_A_psi + H_AB_psi + H_B_psi # TODO how to add them, elementwise?

    return result

def mps_vdot(Alist, Blist):
    """
    Compute the inner product of two tensors in MPS format, with the convention that
    the complex conjugate of the tensor represented by the first argument is used.

    The i-th MPS tensor Alist[i] is expected to have dimensions (n[i], Da[i], Da[i+1]),
    and similarly Blist[i] must have dimensions                 (n[i], Db[i], Db[i+1]),
    with `n` the list of logical dimensions and `Da`, `Db` the lists of virtual bond dimensions.
    """
    d = len(Alist)
    assert d == len(Blist)
    R = np.tensordot(Blist[-1], Alist[-1].conj(), axes=(0, 0))
    # consistency check of degree and dummy singleton dimensions
    assert R.ndim == 4 and R.shape[1] == 1 and R.shape[3] == 1
    # formally remove dummy singleton dimensions
    R = np.reshape(R, (R.shape[0], R.shape[2]))
    for i in reversed(range(d - 1)):
        # contract with current A tensor
        T = np.tensordot(Alist[i].conj(), R, axes=(2, 1))
        # contract with current B tensor and update R
        R = np.tensordot(Blist[i], T, axes=((0, 2), (0, 2)))
    assert R.ndim == 2 and R.shape[0] == 1 and R.shape[1] == 1
    return R[0, 0]

def expectation_value(H: qib.operator.MolecularHamiltonian, LA: int, psi: Sequence[any]):
    """
    Calculate the expectation value <psi|H|psi> of a Hamiltonian and an MPS
    by using a partition at LA analogously to the paper.
    """
    H_psi = apply_hamiltonian(H, LA, psi)
    exp_val = mps_vdot(psi, H_psi)
    
    return exp_val
