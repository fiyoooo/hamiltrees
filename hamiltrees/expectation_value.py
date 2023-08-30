"""
    Calculation of expectation value with partitioned Hamiltonian.
"""

import numpy as np
import qib


def apply_part_of_hamiltonian(H: qib.operator.MolecularHamiltonian, sites, psi):
    """
    Construct the hamiltonian on the given sites.
    """
    # complementary indices
    compl = [i for i in range(H.field.lattice.nsites) if i not in sites]
    
    # kinetic hopping term
    tkinCopy = np.copy(H.tkin)
    tkinCopy[compl, :] = 0
    tkinCopy[:, compl] = 0
    
    # interaction term
    vintCopy = np.copy(H.vint)
    vintCopy[compl, :, :, :] = 0
    vintCopy[:, compl, :, :] = 0
    vintCopy[:, :, compl, :] = 0
    vintCopy[:, :, :, compl] = 0
    
    # TODO apply to psi
    qib.operator.MolecularHamiltonian(H.field, 0., tkinCopy, vintCopy, qib.operator.MolecularHamiltonianSymmetry.HERMITIAN)

    return psi

def construct_complementary_p(i: int, j: int, sites, field: qib.field.Field, vint):
    """
    Construct the complementary operator `P_{ij}^B` in Eq. (11).
    """
    L = field.lattice.nsites
    vint = np.asarray(vint)
    assert vint.shape == (L, L, L, L)
    # indicator function
    eb = np.zeros(L)
    for k in sites:
        eb[k] = 1
    return qib.operator.FieldOperator([
        qib.operator.FieldOperatorTerm(
            [qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL),
             qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL)],
            [[0.5*vint[i, j, l, k]*eb[l]*eb[k] for l in range(L)] for k in range(L)])])

def construct_complementary_q(i: int, j: int, sites, field: qib.field.Field, vint):
    """
    Construct the complementary operator `Q_{ij}^B` in Eq. (12).
    """
    L = field.lattice.nsites
    vint = np.asarray(vint)
    assert vint.shape == (L, L, L, L)
    # indicator function
    eb = np.zeros(L)
    for k in sites:
        eb[k] = 1
    return qib.operator.FieldOperator([
        qib.operator.FieldOperatorTerm(
            [qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_CREATE),
             qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL)],
            [[0.5*((vint[i, k, j, l] + vint[k, i, l, j])/2 - vint[i, k, l, j])*eb[l]*eb[k] for l in range(L)] for k in range(L)])])

def construct_complementary_s(i: int, sites, field: qib.field.Field, tkin, vint):
    """
    Construct the complementary operator `S_i^B` in Eq. (13).
    """
    L = field.lattice.nsites
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)
    assert tkin.shape == (L, L)
    assert vint.shape == (L, L, L, L)
    # indicator function
    eb = np.zeros(L)
    for k in sites:
        eb[k] = 1
    skin = qib.operator.FieldOperatorTerm([qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL)], [0.5*tkin[i, j]*eb[j] for j in range(L)])
    sint = qib.operator.FieldOperatorTerm([qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_CREATE),
                                           qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL),
                                           qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL)],
                                          [[[0.5*(vint[i, j, l, k] - vint[j, i, l, k])*eb[j]*eb[l]*eb[k] for l in range(L)] for k in range(L)] for j in range(L)])
    return qib.operator.FieldOperator([skin, sint])


def create_op(i: int, field: qib.field.Field):
    """
    Fermionic creation operator at site `i` as field operator.
    """
    # unit vector
    e1 = np.zeros(field.lattice.nsites)
    e1[i] = 1
    return qib.operator.FieldOperator(
        [qib.operator.FieldOperatorTerm(
            [qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_CREATE)], e1)])

def annihil_op(i: int, field: qib.field.Field):
    """
    Fermionic annihilation operator at site `i` as field operator.
    """
    # unit vector
    e1 = np.zeros(field.lattice.nsites)
    e1[i] = 1
    return qib.operator.FieldOperator(
        [qib.operator.FieldOperatorTerm(
            [qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL)], e1)])


def construct_interacting_hamiltonian(regionA, regionB, field, tkin, vint):
    """
    Contruct the interacting hamiltonian H_{AB} in Equ. (10).
    """
    S_i = sum(create_op(i, field) @ construct_complementary_s(i, regionB, field, tkin, vint) for i in regionA)
    S_j = sum(create_op(j, field) @ construct_complementary_s(j, regionA, field, tkin, vint) for j in regionB)
    # TODO check equivalence to paper and necessity
    Q_ii = sum(create_op(i, field) @ annihil_op(i, field) @ construct_complementary_q(i, i, regionB, field, vint) for i in regionA)
    P_ij = sum(create_op(i, field) @  create_op(j, field) @ construct_complementary_p(i, j, regionB, field, vint) for i in regionA for j in regionA)
    Q_ij = sum(create_op(i, field) @ annihil_op(j, field) @ construct_complementary_q(i, j, regionB, field, vint) for i in regionA for j in regionA)

    HABhalf = (S_i + S_j + P_ij + Q_ij)
    
    return HABhalf + HABhalf.adjoint() # TODO make return Molecularhamiltonian?

def apply_operator(op, psi, i: int):
    None

def apply_hamiltonian(H: qib.operator.MolecularHamiltonian, LA, psi):
    """
    Apply the hamiltonian H to the MPS psi as list of tensors.
    """
    # TODO

    L = H.field.lattice.nsites

    # iterate through all sites
    for term in H.terms:
        for i in range(len(term.opdesc)):
            None
        print(len(term.opdesc))
        print(term.opdesc)
        print("\n coeffs starting")
        print(term.coeffs.shape)
        print(term.coeffs.ndim)
    # apply the respective operator
    # apply_operator(op, psi, i)

    regionA = range(0, LA)
    regionB = range(LA, L)

    # H_A Hamiltonian
    HA = apply_part_of_hamiltonian(H, regionA, psi)
    
    # H_B Hamiltonian
    HB = apply_part_of_hamiltonian(H, regionB, psi)
    
    # H_{AB} Hamiltonian
    HAB = apply_interacting_hamiltonian(regionA, regionB, H.field, H.tkin, H.vint)
    
    HA + HAB + HB

    return psi
