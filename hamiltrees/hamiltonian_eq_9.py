"""
    Hamiltonian construction acc. to eq. (9)
"""

import qib
from qib.operator import FieldOperator, FieldOperatorTerm, IFOType, IFODesc
import fermitensor as ftn # after installing fermitensor (installed with kernel 3.10.11)

import numpy as np
from scipy import sparse
from typing import Sequence

# setzt alle coeff ausserhalb von range auf 0 um Dimension nicht zu veraendern
def get_range(coeff, range):
    ret = np.zeros_like(coeff)
    if coeff.ndim == 2:
        ret[range[0]:range[1], range[0]:range[1]] = coeff[range[0]:range[1], range[0]:range[1]]
    elif coeff.ndim ==4:
        ret[range[0]:range[1], range[0]:range[1], range[0]:range[1], range[0]:range[1]] = coeff[range[0]:range[1], range[0]:range[1], range[0]:range[1], range[0]:range[1]]
    else:
        raise NotImplementedError
    return ret

def get_part_of_H_as_field_operator(field, tkin, vint, L):
    # kinetic hopping term
    T = FieldOperatorTerm([IFODesc(field, IFOType.FERMI_CREATE),
                            IFODesc(field, IFOType.FERMI_ANNIHIL)],
                            get_range(tkin, L))
    # interaction term
    V = FieldOperatorTerm([IFODesc(field, IFOType.FERMI_CREATE),
                            IFODesc(field, IFOType.FERMI_CREATE),
                            IFODesc(field, IFOType.FERMI_ANNIHIL),
                            IFODesc(field, IFOType.FERMI_ANNIHIL)],
                            0.5 * get_range(vint, L).transpose((0, 1, 3, 2)))
    return FieldOperator([T, V])

def adjoint(P: FieldOperator):
    # get adjoint of Field Operator
    ret = []

    for term in P.terms:
        opdesc = term.opdesc[::-1] # reverse order
        for desc in opdesc: # replace create with annihil and vice versa
            desc.otype = IFOType.adjoint(desc.otype)
        coeffs = term.coeffs.conjugate().transpose() # reverse order
        
        ret.append(FieldOperatorTerm(opdesc, coeffs)) # create adjoint term

    return FieldOperator(ret)

if __name__ == '__main__':
    L = 8
    L_1 = L//2
    A = [0, L_1]
    B = [L_1, L]

    latt = qib.lattice.FullyConnectedLattice((L,))
    field = qib.field.Field(qib.field.ParticleType.FERMION, latt)

    # create MolecularHamiltonian Object
    H = construct_random_molecular_hamiltonian(L)

    # Create easy parts on left and right H_A and H_B.
    H_A = get_part_of_H_as_field_operator(H.field, H.tkin, H.vint, A)
    H_B = get_part_of_H_as_field_operator(H.field, H.tkin, H.vint, B)