"""
Expectation value <psi|H|psi> with partitioned Hamiltonian as MPO.

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

import pytenet as ptn

class TestExpectationValue(unittest.TestCase):

    @unittest.skip("not implemented")
    def test_with_partitioned_MPO(self):

        L = 5
        LA = 2
        rng = np.random.default_rng(142)
        
        # random Hamiltonian
        H = ham.random_construction.construct_random_partitioned_hamiltonian_MPO(L, LA, rng)
        
        # check if H hermitian
        H_matrix = H.as_matrix()
        self.assertAlmostEqual(np.linalg.norm(H_matrix - H_matrix.conj().T), 0)

        # constructing mps
        mps = ham.random_construction.construct_random_MPS(L)
        
        # check if right shapes to contract
        self.assertAlmostEqual(H.nsites, mps.nsites)
               
        # calculating <mps|H|mps>
        exp_val = ptn.operator_average(mps, H)
        self.assertAlmostEqual(exp_val, np.real(exp_val)) # expectation value should be real

    def test_with_simplified_MPO(self):

        L = 5
        LA = 2
        regionA = range(0, LA)
        regionB = range(LA, L)
        rng = np.random.default_rng(142)
        
        # random Hamiltonian
        H_full = ham.random_construction.construct_random_molecular_hamiltonian(L, rng)
        H = ham.block_partitioning_MPO.construct_simplified_MPO(regionA, regionB, L, H_full.tkin)
        
        # check if H hermitian
        H_matrix = H.as_matrix()
        self.assertAlmostEqual(np.linalg.norm(H_matrix - H_matrix.conj().T), 0)

        # constructing mps
        mps = ham.random_construction.construct_random_MPS(L)
        
        # check if right shapes to contract
        self.assertAlmostEqual(H.nsites, mps.nsites)
               
        # calculating <mps|H|mps>
        exp_val = ptn.operator_average(mps, H)
        self.assertAlmostEqual(exp_val, np.real(exp_val)) # expectation value should be real
    
if __name__ == "__main__":
    unittest.main()
