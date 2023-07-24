"""
    Hamiltonian construction acc. to eq. (9)
"""

from qib.operator import FieldOperator, FieldOperatorTerm, IFOType, IFODesc
import numpy as np

def get_part_of_H_as_FO(field, tkin, vint, subsystem):
    coeffs = np.copy(tkin)
    # complementary indices
    compl = [i for i in range(field.lattice.nsites) if i not in subsystem]
    coeffs[compl, :] = 0
    coeffs[:, compl] = 0
    # kinetic hopping term
    T = FieldOperatorTerm([IFODesc(field, IFOType.FERMI_CREATE),
                            IFODesc(field, IFOType.FERMI_ANNIHIL)],
                            coeffs)
    
    coeffs = np.copy(vint)
    # complementary indices
    compl = [i for i in range(field.lattice.nsites) if i not in subsystem]
    coeffs[compl, :, :, :] = 0
    coeffs[:, compl, :, :] = 0
    coeffs[:, :, compl, :] = 0
    coeffs[:, :, :, compl] = 0
    # interaction term
    V = FieldOperatorTerm([IFODesc(field, IFOType.FERMI_CREATE),
                            IFODesc(field, IFOType.FERMI_CREATE),
                            IFODesc(field, IFOType.FERMI_ANNIHIL),
                            IFODesc(field, IFOType.FERMI_ANNIHIL)],
                            0.5 * coeffs.transpose((0, 1, 3, 2)))
    return FieldOperator([T, V])

def get_P_as_FO(field, vint, subsystem, i, j):
    coeffs = np.copy(vint[i, j, :, :])
    # complementary indices
    compl = [i for i in range(field.lattice.nsites) if i not in subsystem]
    coeffs[compl, :] = 0
    coeffs[:, compl] = 0
    V = FieldOperatorTerm([IFODesc(field, IFOType.FERMI_ANNIHIL),
                            IFODesc(field, IFOType.FERMI_ANNIHIL)],
                            coeffs.transpose((1, 0)))
    return FieldOperator([V])
    
def get_Q_as_FO(field, vint, subsystem, i, j):
    coeffs = vint[i, :, j, :] - vint[i, :, :, j]
    # complementary indices
    compl = [i for i in range(field.lattice.nsites) if i not in subsystem]
    coeffs[compl, :] = 0
    coeffs[:, compl] = 0
    V = FieldOperatorTerm([IFODesc(field, IFOType.FERMI_CREATE),
                            IFODesc(field, IFOType.FERMI_ANNIHIL)],
                            coeffs)
    return FieldOperator([V])

def get_S_as_FO(field, tkin, vint, subsystem, i):
    coeffs = np.copy(tkin[i, :])
    # complementary indices
    compl = [i for i in range(field.lattice.nsites) if i not in subsystem]
    coeffs[compl] = 0
    # kinetic hopping term
    T = FieldOperatorTerm([IFODesc(field, IFOType.FERMI_ANNIHIL)],
                            coeffs)
    
    coeffs = np.copy(vint[i, :, :, :])
    # complementary indices
    compl = [i for i in range(field.lattice.nsites) if i not in subsystem]
    coeffs[compl, :, :] = 0
    coeffs[:, compl, :] = 0
    coeffs[:, :, compl] = 0
    # interaction term
    V = FieldOperatorTerm([IFODesc(field, IFOType.FERMI_CREATE),
                            IFODesc(field, IFOType.FERMI_ANNIHIL),
                            IFODesc(field, IFOType.FERMI_ANNIHIL)],
                            coeffs.transpose((0, 2, 1)))
    return FieldOperator([T, V])

def single_term_FO(field, otype, i):
    # single term Field Operator acting on only site i
    coeffs = np.zeros(field.lattice.nsites)
    coeffs[i] = 1
    return FieldOperator([FieldOperatorTerm([IFODesc(field, otype)], coeffs)])

def adjoint(P: FieldOperator):
    # get adjoint of Field Operator
    ret = []

    for term in P.terms:
        opdesc = []
        for desc in term.opdesc[::-1]: # reverse order
            opdesc += [IFODesc(desc.field, IFOType.adjoint(desc.otype))] # replace create with annihil and vice versa
        coeffs = np.copy(term.coeffs).conjugate().T # reverse order
        
        ret.append(FieldOperatorTerm(opdesc, coeffs)) # create adjoint term

    return FieldOperator(ret)

def get_H_AB_as_FO(field, tkin, vint, A, B):
    # acc. to equ. (10) in paper
    
    S_i = sum((single_term_FO(field, IFOType.FERMI_CREATE, i) 
               @ get_S_as_FO(field, tkin, vint, B, i) 
               for i in A), 
               FieldOperator([]))
    Q_ii = sum((single_term_FO(field, IFOType.FERMI_CREATE, i) 
                @ single_term_FO(field, IFOType.FERMI_ANNIHIL, i) 
                @ get_Q_as_FO(field, vint, B, i, i) 
                for i in A), 
                FieldOperator([]))
    S_j = sum((single_term_FO(field, IFOType.FERMI_CREATE, j) 
               @ get_S_as_FO(field, tkin, vint, A, j) 
               for j in B), 
               FieldOperator([]))
    
    # i > j in A
    P_ij = FieldOperator([])
    Q_ij = FieldOperator([])
    for i in A:
        for j in range(i):
            P_ij += single_term_FO(field, IFOType.FERMI_ANNIHIL, i) @ single_term_FO(field, IFOType.FERMI_ANNIHIL, j) @ adjoint(get_P_as_FO(field, vint, B, i, j))
            Q_ij += single_term_FO(field, IFOType.FERMI_CREATE, i) @ single_term_FO(field, IFOType.FERMI_ANNIHIL, j) @ get_Q_as_FO(field, vint, B, i, j)

    
    # P_ij = sum(((sum(single_term_FO(field, IFOType.FERMI_ANNIHIL, i) 
    #              @ single_term_FO(field, IFOType.FERMI_ANNIHIL, j) 
    #              @ adjoint(get_P_as_FO(field, vint, B, i, j)) 
    #              for j in range(i)), FieldOperator([])) for i in A), 
    #              FieldOperator([]))
    
    # Q_ij = sum(((single_term_FO(field, IFOType.FERMI_CREATE, i) 
    #              @ single_term_FO(field, IFOType.FERMI_ANNIHIL, j) 
    #              @ get_Q_as_FO(field, vint, B, i, j) 
    #              for j in range(i)) for i in A), 
    #              FieldOperator([]))

    firstLine = S_i + S_j + Q_ii
    secLine = P_ij + Q_ij
    return firstLine + secLine + adjoint(firstLine) + adjoint(secLine)
