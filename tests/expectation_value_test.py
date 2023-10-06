"""
Expectation value <psi|H|psi> with partitioned Hamiltonian.

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
        H = ham.random_construction.construct_random_partitioned_hamiltonian_FO(L, LA, rng).as_matrix()
        self.assertAlmostEqual(spla.norm(H - H.conj().T), 0)

        # constructing psi
        mps = ham.random_construction.construct_random_MPS(L).as_vector()
         
        # check if right shapes to multiply
        self.assertAlmostEqual(H.shape[0], mps.shape[0])
        self.assertAlmostEqual(H.shape[1], mps.shape[0])
               
        # calculating <mps|H|mps>
        exp_val = np.vdot(mps, H @ mps)
        self.assertAlmostEqual(exp_val, np.real(exp_val)) # expectation value should be real

    def test_with_fMPS(self):
        
        L = 8
        rng = np.random.default_rng(142)
        
        # random Hamiltonian
        H = ham.random_construction.construct_random_molecular_hamiltonian(L, rng).as_matrix()
        self.assertAlmostEqual(spla.norm(H - H.conj().T), 0)

        # random fMPS
        fmps = ham.random_construction.construct_random_fMPS(L)
        fmps_even = fmps.as_vector("even")
        fmps_odd = fmps.as_vector("odd")
        fmps = np.concatenate((fmps_even, fmps_odd))
        fmps /= np.linalg.norm(fmps)
        
        # check if right shapes to multiply
        self.assertAlmostEqual(H.shape[0], fmps.shape[0])
        self.assertAlmostEqual(H.shape[1], fmps.shape[0])
                
        # calculating <fmps|H|fmps>
        exp_val = np.vdot(fmps, H @ fmps)
        self.assertAlmostEqual(exp_val, np.real(exp_val)) # expectation value should be real

if __name__ == "__main__":
    unittest.main()
