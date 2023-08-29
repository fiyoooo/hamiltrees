"""
Expectation value with partitioned Hamiltonian.

    Berechne Erwartungswert <psi|H|psi> mit Matrizen als Referenzwert für spätere Tests.

References:
  - Naoki Nakatani and Garnet Kin-Lic Chan:
    Efficient tree tensor network states (TTNS) for quantum chemistry:
    Generalizations of the density matrix renormalization group algorithm
    J. Chem. Phys. 138, 134113 (2013)
    http://dx.doi.org/10.1063/1.4798639
"""

import hamiltrees as ham

import unittest
import numpy as np
import scipy.sparse.linalg as spla


class TestExpectationValue(unittest.TestCase):

    def test_with_partitioned(self):

        L = 5
        LA = 2
        rng = np.random.default_rng(142)
        
        # random Hamiltonian
        H = ham.random_construction.construct_random_partitioned_hamiltonian(L, LA, rng).as_matrix()
        self.assertAlmostEqual(spla.norm(H - H.conj().T), 0)

        # constructing psi
        psi = ham.random_construction.construct_random_MPS(L)
        # TODO psi as vector
         
        self.assertAlmostEqual(H.shape, psi.shape) # TODO
                
        # calculating <psi|H|psi>
        exp_val = np.vdot(psi, H @ psi)
        self.assertAlmostEqual(exp_val, exp_val.real) # expectation value should be real # TODO
        
    def test_with_fMPS(self):
        
        L = 8
    
        # random Hamiltonian
        H = ham.random_construction.construct_random_molecular_hamiltonian(L).as_matrix()
        self.assertAlmostEqual(spla.norm(H - H.conj().T), 0)

        # constructing psi
        psi = ham.random_construction.construct_random_fMPS(L)
        psi_even = psi.as_vector("even")
        psi_odd = psi.as_vector("odd")
        psi = np.concatenate((psi_even, psi_odd))
        psi /= np.linalg.norm(psi)
        
        self.assertAlmostEqual(H.shape, psi.shape) # TODO
                
        # calculating <psi|H|psi>
        exp_val = np.vdot(psi, H @ psi)
        self.assertAlmostEqual(exp_val, exp_val.real) # expectation value should be real # TODO
        
    def test_with_MPS(self):
        
        L = 8
    
        # random Hamiltonian
        H = ham.random_construction.construct_random_molecular_hamiltonian(L).as_matrix()
        self.assertAlmostEqual(spla.norm(H - H.conj().T), 0)

        # constructing psi
        psi = ham.random_construction.construct_random_MPS(L)
        # TODO psi as vector
        
        self.assertAlmostEqual(H.shape, psi.shape) # TODO
                
        # calculating <psi|H|psi>
        exp_val = np.vdot(psi, H @ psi)
        self.assertAlmostEqual(exp_val, exp_val.real) # expectation value should be real # TODO
        
if __name__ == "__main__":
    unittest.main()
