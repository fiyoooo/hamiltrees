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
        H = ham.random_construction.construct_random_partitioned_hamiltonian_MPO(L, LA, rng).as_matrix()
        self.assertAlmostEqual(np.linalg.norm(H - H.conj().T), 0)

        # constructing mps
        mps = ham.random_construction.construct_random_MPS(L).as_vector()
        
        # check if right shapes to multiply
        self.assertAlmostEqual(H.shape[0], mps.shape[0])
        self.assertAlmostEqual(H.shape[1], mps.shape[0])
                
        # calculating <mps|H|mps>
        exp_val = np.vdot(mps, H @ mps) # TODO use ptn.vdot() for vdot of MPS class
        self.assertAlmostEqual(exp_val, np.real(exp_val)) # expectation value should be real

    def test_with_simplified_MPO(self):

        L = 5
        LA = 2
        regionA = range(0, LA)
        regionB = range(LA, L)
        rng = np.random.default_rng(142)
        
        # random Hamiltonian
        Hfull = ham.random_construction.construct_random_molecular_hamiltonian(L, rng)
        H = ham.block_partitioning_MPO.construct_simplified_MPO(regionA, regionB, L, Hfull.tkin).as_matrix()
        self.assertAlmostEqual(np.linalg.norm(H - H.conj().T), 0)

        # constructing mps
        mps = ham.random_construction.construct_random_MPS(L).as_vector()
        
        # check if right shapes to multiply
        self.assertAlmostEqual(H.shape[0], mps.shape[0])
        self.assertAlmostEqual(H.shape[1], mps.shape[0])
                
        # calculating <mps|H|mps>
        exp_val = np.vdot(mps, H @ mps) # TODO use methods for MPO and MPS, ptn.vdot() 
        self.assertAlmostEqual(exp_val, np.real(exp_val)) # expectation value should be real
    
if __name__ == "__main__":
    unittest.main()