"""
    Test for Hamiltonian construction acc. to eq. (9)
"""

import qib
from qib.operator import FieldOperator, FieldOperatorTerm, IFOType, IFODesc
import fermitensor as ftn # after installing fermitensor (installed with kernel 3.10.11)
import matrix_reference as ref
import hamiltonian_eq_9 as ham

import numpy as np
from scipy.sparse.linalg import norm

def construct_random_coefficients(L):
    rng = np.random.default_rng()
    # Hamiltonian coefficients
    tkin = 0.5 * qib.util.crandn((L, L), rng)
    vint = 0.5 * qib.util.crandn((L, L, L, L), rng)
    # make hermitian
    tkin = 0.5 * (tkin + tkin.conj().T)
    vint = 0.5 * (vint + vint.conj().transpose((2, 3, 0, 1)))
    
    return (tkin, vint)

def is_hermitian(H: FieldOperator, matrix=False):
    if all(term.is_hermitian() for term in H.terms):
        return True
    
    if not matrix:
        H_adj = ham.adjoint(H)  # my adjoint
        return np.allclose(norm(H.as_matrix()-H_adj.as_matrix()), 0)

    matrix = H.as_matrix()
    return np.allclose(norm(matrix-matrix.H), 0)   # adjoint with scipy

def csr_allclose(a, b, rtol=1e-6, atol=1e-8):
    c = np.abs(a - b)# - rtol * np.abs(b)
    help = np.round(c,6)
    help.eliminate_zeros()
    # print(help)
    print(help.max())
    return c.max() <= atol

if __name__ == '__main__':
    L = 8
    L_1 = L//2
    A = [0, L_1]
    B = [L_1, L]

    # latt = qib.lattice.FullyConnectedLattice((L,))
    # field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
    # tkin, vint = construct_random_coefficients(L)

    # create MolecularHamiltonian Object
    H_ref = ref.construct_random_molecular_hamiltonian(L)
    field = H_ref.field
    tkin = H_ref.tkin
    vint = H_ref.vint

    # Create easy parts on left and right H_A and H_B.
    H_A = ham.get_part_of_H_as_FO(field, tkin, vint, A)
    H_B = ham.get_part_of_H_as_FO(field, tkin, vint, B)
    H_AB = ham.get_H_AB_as_FO(field, tkin, vint, A, B)

    H = H_A + H_B + H_AB

    assert norm(H.as_matrix()-H_ref.as_matrix()) == 0