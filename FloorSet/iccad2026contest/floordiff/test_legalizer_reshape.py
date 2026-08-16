"""Focused tests for shape-aware residual detailed placement.

Run from ``iccad2026contest/`` with:
    python -m unittest floordiff.test_legalizer_reshape
"""

import time
import unittest

import numpy as np
import torch

from . import legalizer as lg


def _case(cons):
    return {
        'cons': torch.tensor(cons, dtype=torch.long),
        'b2b': torch.zeros((0, 3), dtype=torch.float64),
        'p2b': torch.zeros((0, 3), dtype=torch.float64),
        'pins': torch.zeros((0, 2), dtype=torch.float64),
    }


class ShapeDetailRepairTest(unittest.TestCase):
    def cfg(self, **overrides):
        return {
            **lg.DEFAULT_CFG,
            'reshape_close_rel': 0.20,
            'reshape_budget_s': 1.0,
            'reshape_trials': 20,
            **overrides,
        }

    def repair(self, sol, case, pre, shrinkable, **overrides):
        area = (sol[:, 0].max() + sol[:, 2].max() - sol[:, 0].min()) \
            * (sol[:, 1].max() + sol[:, 3].max() - sol[:, 1].min())
        S = float(np.sqrt((sol[:, 2] * sol[:, 3]).sum()))
        return lg.shape_detail_repair(
            sol.copy(), case, pre, shrinkable, self.cfg(**overrides), S,
            hpwl_base=1.0, area_base=max(float(area), 1.0), n_soft=1)

    def test_group_gap_closes_by_equal_area_reshape(self):
        sol = np.array([
            [0.0, 0.0, 2.0, 2.0],
            [2.2, 0.0, 2.0, 2.0],
        ])
        case = _case([[0, 1, 0, 1, 0], [0, 0, 0, 1, 0]])
        pre = np.array([True, False])
        shrinkable = np.array([False, True])
        old_area = sol[:, 2] * sol[:, 3]

        out, stats = self.repair(sol, case, pre, shrinkable)

        self.assertEqual(stats['group_moves'], 1)
        self.assertEqual(lg._violations_official(
            out, case['cons'].numpy())[1], 0)
        self.assertGreater(out[1, 2], sol[1, 2])
        self.assertLess(out[1, 3], sol[1, 3])
        np.testing.assert_allclose(out[:, 2] * out[:, 3], old_area)
        np.testing.assert_allclose(out[0], sol[0])
        self.assertLessEqual(lg.max_penetration(out), lg._EPS_OVL)

    def test_high_boundary_contracts_through_shape(self):
        sol = np.array([
            [3.0, 0.0, 1.0, 1.0],
            [0.0, 2.0, 4.2, 1.0],
        ])
        case = _case([[0, 1, 0, 0, 2], [0, 0, 0, 0, 0]])
        pre = np.array([True, False])
        shrinkable = np.array([False, True])
        old_area = sol[:, 2] * sol[:, 3]

        out, stats = self.repair(sol, case, pre, shrinkable)

        self.assertEqual(stats['boundary_moves'], 1)
        self.assertEqual(lg._violations_official(
            out, case['cons'].numpy())[0], 0)
        self.assertAlmostEqual(float((out[:, 0] + out[:, 2]).max()), 4.0)
        self.assertLess(out[1, 2], sol[1, 2])
        np.testing.assert_allclose(out[:, 2] * out[:, 3], old_area)
        np.testing.assert_allclose(out[0], sol[0])
        self.assertLessEqual(lg.max_penetration(out), lg._EPS_OVL)

    def test_fixed_frontier_is_not_reshaped(self):
        sol = np.array([
            [3.0, 0.0, 1.0, 1.0],
            [0.0, 2.0, 4.2, 1.0],
        ])
        case = _case([[0, 1, 0, 0, 2], [1, 0, 0, 0, 0]])
        pre = np.array([True, False])
        out, stats = self.repair(
            sol, case, pre, np.array([False, False]))
        np.testing.assert_allclose(out, sol)
        self.assertEqual(stats['moves'], 0)

    def test_zero_budget_is_constant_time_noop(self):
        sol = np.array([[0.0, 0.0, 2.0, 2.0],
                        [2.2, 0.0, 2.0, 2.0]])
        case = _case([[0, 1, 0, 1, 0], [0, 0, 0, 1, 0]])
        start = time.perf_counter()
        out, stats = self.repair(
            sol, case, np.array([True, False]),
            np.array([False, True]), reshape_budget_s=0.0)
        elapsed = time.perf_counter() - start
        np.testing.assert_allclose(out, sol)
        self.assertEqual(stats['trials'], 0)
        self.assertLess(elapsed, 0.05)


if __name__ == '__main__':
    unittest.main()
