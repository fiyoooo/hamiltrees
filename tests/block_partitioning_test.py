"""
Partitioning of a molecular Hamiltonian in second quantization representation
into two regions A and B.

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

import qib


class TestBlockPartitioning(unittest.TestCase):

    def test(self):

        L = 5
        rng = np.random.default_rng(142)
        
        # random Hamiltonian
        H = ham.matrix_reference.construct_random_molecular_hamiltonian(L, rng)
        field = H.field
        tkin = H.tkin
        vint = H.vint
        H = H.as_matrix()
        self.assertAlmostEqual(spla.norm(H - H.conj().T), 0)

        # sizes of regions A and B
        LA = 2
        regionA = range(0, LA)
        regionB = range(LA, L)

        # H_A Hamiltonian
        HA = ham.block_partitioning.construct_part_of_hamiltonian(regionA, field, tkin, vint).as_matrix()
        self.assertAlmostEqual(spla.norm(HA - HA.conj().T), 0)

        # H_B Hamiltonian
        HB = ham.block_partitioning.construct_part_of_hamiltonian(regionB, field, tkin, vint).as_matrix()
        self.assertAlmostEqual(spla.norm(HB - HB.conj().T), 0)

        # H_{AB} Hamiltonian
        HAB = ham.block_partitioning.construct_interacting_hamiltonian(regionA, regionB, field, tkin, vint).as_matrix()
        # check Hermitian property
        self.assertAlmostEqual(spla.norm(HAB - HAB.conj().T), 0)

        # compare with reference Hamiltonian
        self.assertAlmostEqual(spla.norm(HA + HAB + HB - H), 0)

if __name__ == "__main__":
    unittest.main()
