"""
    Referenzwert mit Matrizen
"""

import qib
import fermitensor as ftn # after installing fermitensor (installed with kernel 3.10.11)

import numpy as np

# copied test_molecular_hamiltonian_construction from qib.tests.test_molecular_hamiltonian
def construct_random_molecular_hamiltonian(L):
    """
    Construct a random molecular Hamiltonian in second quantization formulation,
    using physicists' convention for the interaction term (note ordering of k and \ell) (dimensions (2^k,2^k)):

    .. math::

        H = (c +) \sum_{i,j} h_{i,j} a^{\dagger}_i a_j 
            + \\frac{1}{2} \sum_{i,j,k,\ell} v_{i,j,k,\ell} 
            a^{\dagger}_i a^{\dagger}_j a_{\ell} a_k
    """
    rng = np.random.default_rng()
    # underlying lattice
    latt = qib.lattice.FullyConnectedLattice((L,))
    field = qib.field.Field(qib.field.ParticleType.FERMION, latt)

    # Hamiltonian coefficients
    c = 0 # qib.util.crandn(rng=rng)
    tkin = 0.5 * qib.util.crandn((L, L), rng)
    vint = 0.5 * qib.util.crandn((L, L, L, L), rng)
    # make hermitian
    tkin = 0.5 * (tkin + tkin.conj().T)
    vint = 0.5 * (vint + vint.conj().transpose((2, 3, 0, 1)))
    
    H = qib.operator.MolecularHamiltonian(field, c, tkin, vint, qib.operator.MolecularHamiltonianSymmetry(0))
    return H

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

"""
    Berechne Erwartungswert <psi|H|psi> mit Matrizen als Referenzwert für spätere Tests.
"""
if __name__ == '__main__':
    L = 8 # ist k in paper
    
    # constructing H in (8)
    H = construct_random_molecular_hamiltonian(L).as_matrix()
    print(H-H.conj().T)

    # constructing psi
    # Randomly create an fMPS tensors with the `fMPSTensor` class and `crandn()`. 
    # Use it as fMPS with `as_vector()` to get the statevector.
    parity = ["even", "odd"][0] # change for odd parity

    psi = construct_random_fMPS(L)
    psi_even = psi.as_vector("even")
    psi_odd = psi.as_vector("odd")
    psi = np.concatenate((psi_even, psi_odd))
    psi /= np.linalg.norm(psi)

    print(psi.shape)

    # calculating <psi|H|psi>
    print(H.shape)
    print(psi.shape)
    exp_val = np.vdot(psi, H @ psi)
    print(exp_val) # sollte reell sein