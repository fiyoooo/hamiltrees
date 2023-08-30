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

    @unittest.skip("not done yet")
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
        self.assertAlmostEqual(exp_val, np.real(exp_val)) # expectation value should be real

    @unittest.skip("fMPS not working yet")
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
        self.assertAlmostEqual(exp_val, np.real(exp_val)) # expectation value should be real

    def test_with_MPS(self):
        
        L = 5
        rng = np.random.default_rng(142)
        
        # random Hamiltonian
        H = ham.random_construction.construct_random_molecular_hamiltonian(L, rng)
        H_matrix = H.as_matrix()
        self.assertAlmostEqual(spla.norm(H_matrix - H_matrix.conj().T), 0)

        # constructing psi
        psi = ham.random_construction.construct_random_MPS(L)
        # psi = ham.random_construction.mps_to_full_tensor(psi)

        # self.assertAlmostEqual(H.shape, len(psi)) # TODO
        
        H_psi = ham.expectation_value.apply_hamiltonian(H, psi)

        # calculating <psi|H|psi>
        exp_val = ham.random_construction.mps_vdot(psi, H_psi)
        self.assertAlmostEqual(exp_val, np.real(exp_val)) # expectation value should be real

    @unittest.skip("not done yet")
    def test_full_tensor(self): # TODO
        # construct S and T as full tensors
        # (only for testing - in practice one usually works with the MPS matrices directly!)
        S = ham.random_construction.mps_to_full_tensor(Alist)
        T = ham.random_construction.mps_to_full_tensor(Blist)

        # should all agree
        print("n:", n)
        print("S.shape:", S.shape)
        print("T.shape:", T.shape)

        # dimension consistency checks
        assert np.array_equal(np.array(S.shape), np.array(n))
        assert np.array_equal(np.array(T.shape), np.array(n))
        
if __name__ == "__main__":
    unittest.main()
