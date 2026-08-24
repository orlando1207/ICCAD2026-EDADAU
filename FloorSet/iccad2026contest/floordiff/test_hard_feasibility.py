"""Regression tests for beta-distribution hard feasibility and MIB handling."""

import unittest

import numpy as np
import torch

from iccad2026_evaluate import (check_area_tolerance,
                                check_dimension_hard_constraints,
                                check_overlap)

from . import legalizer as lg


def _case(areas, cons, target=None):
    n = len(areas)
    if target is None:
        target = np.full((n, 4), -1.0)
    return {
        'area': torch.tensor(areas, dtype=torch.float64),
        'cons': torch.tensor(cons, dtype=torch.long),
        'target': torch.tensor(target, dtype=torch.float64),
        'gt': None,
        'b2b': torch.zeros((0, 3), dtype=torch.float64),
        'p2b': torch.zeros((0, 3), dtype=torch.float64),
        'pins': torch.zeros((0, 2), dtype=torch.float64),
        'metrics': None,
    }


def _official_feasible(sol, case):
    positions = [tuple(map(float, row)) for row in sol]
    cons = case['cons']
    frozen = {i for i in range(len(sol))
              if cons[i, 0] != 0 or cons[i, 1] != 0}
    return not (check_overlap(positions)
                or check_area_tolerance(positions, case['area'],
                                        skip_indices=frozen)
                or check_dimension_hard_constraints(
                    positions, case['target'].tolist(), cons, len(sol)))


class HardFeasibilityTest(unittest.TestCase):
    def assertMatchesOfficial(self, sol, case):
        self.assertEqual(lg.hard_feasibility(sol, case)['feasible'],
                         _official_feasible(sol, case))

    def test_validator_matches_official_hard_checks(self):
        case = _case(
            [4.0, 9.0, 4.0],
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[-1, -1, 2, 2], [3, 0, 3, 3], [-1, -1, -1, -1]],
        )
        valid = np.array([[0, 0, 2, 2], [3, 0, 3, 3], [6, 0, 2, 2.01]])
        fixed_bad = valid.copy()
        fixed_bad[0, 2] += 2e-4
        pre_bad = valid.copy()
        pre_bad[1, 0] += 2e-4
        area_bad = valid.copy()
        area_bad[2, 3] = 2.03
        overlap_bad = valid.copy()
        overlap_bad[2, 0] = 5.999
        for sol in (valid, fixed_bad, pre_bad, area_bad, overlap_bad):
            self.assertMatchesOfficial(sol, case)

    def test_incompatible_fixed_mib_is_not_tied(self):
        case = _case(
            [4.0, 9.0],
            [[1, 0, 1, 0, 0], [1, 0, 1, 0, 0]],
            [[-1, -1, 2, 2], [-1, -1, 3, 3]],
        )
        pred = np.array([[0, 0, 2, 2], [3, 0, 3, 3]], dtype=np.float64)
        before = pred.copy()
        stats = lg._tie_compatible_mib_dims(
            pred, case, np.ones(2, dtype=bool), np.zeros(2, dtype=bool))
        np.testing.assert_allclose(pred, before)
        self.assertEqual(stats['incompatible_frozen'], 1)
        self.assertTrue(lg.hard_feasibility(pred, case)['feasible'])

    def test_incompatible_soft_areas_are_not_tied(self):
        case = _case([4.0, 9.0],
                     [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
        pred = np.array([[0, 0, 2, 1.982], [3, 0, 3, 2.973]])
        before = pred.copy()
        stats = lg._tie_compatible_mib_dims(
            pred, case, np.zeros(2, dtype=bool), np.zeros(2, dtype=bool))
        np.testing.assert_allclose(pred, before)
        self.assertEqual(stats['incompatible_area'], 1)
        self.assertTrue(lg.hard_feasibility(pred, case)['feasible'])
        self.assertGreater(lg._violations_official(
            pred, case['cons'].numpy())[2], 0)

    def test_compatible_mixed_mib_retains_hard_feasibility(self):
        case = _case(
            [4.0, 4.0],
            [[1, 0, 1, 0, 0], [0, 0, 1, 0, 0]],
            [[-1, -1, 2, 2], [-1, -1, -1, -1]],
        )
        pred = np.array([[0, 0, 2, 2], [3, 0, 2.2, 1.8]])
        stats = lg._tie_compatible_mib_dims(
            pred, case, np.array([True, False]), np.zeros(2, dtype=bool))
        np.testing.assert_allclose(pred[:, 2:4], [[2, 2], [2, 2]])
        self.assertEqual(stats['tied'], 1)
        self.assertTrue(lg.hard_feasibility(pred, case)['feasible'])
        self.assertEqual(lg._violations_official(
            pred, case['cons'].numpy())[2], 0)

    def test_selection_always_prefers_feasible_candidate(self):
        infeasible = {
            'proxy_cost': 1.0,
            'hard': {'feasible': False, 'total_violations': 1,
                     'overlap_violations': 0, 'area_violations': 1,
                     'dimension_violations': 0, 'numeric_violations': 0},
        }
        feasible = {
            'proxy_cost': 1.2,
            'hard': {'feasible': True, 'total_violations': 0,
                     'overlap_violations': 0, 'area_violations': 0,
                     'dimension_violations': 0, 'numeric_violations': 0},
        }
        self.assertLess(lg._selection_key(feasible, 1),
                        lg._selection_key(infeasible, 0))


if __name__ == '__main__':
    unittest.main()
