"""Focused tests for bounded cluster ripple repair.

Run from ``iccad2026contest/`` with:
    python -m unittest floordiff.test_legalizer_ripple
"""

import time
import unittest

import numpy as np
import torch

from . import legalizer as lg


class ClusterRippleRepairTest(unittest.TestCase):
    def setUp(self):
        # Blocks 0 and 1 belong to one cluster.  Their x faces already align,
        # but they are separated vertically by 0.2.  Block 2 fills that gap,
        # so translating block 1 alone would overlap it; successful repair has
        # to ripple block 2 downward.
        self.sol = np.array([
            [0.0, 0.0, 2.0, 3.0],
            [2.0, 3.2, 2.0, 2.0],
            [2.0, 3.0, 2.0, 0.2],
        ])
        cons = torch.zeros((3, 5), dtype=torch.long)
        cons[0, 3] = 1
        cons[1, 3] = 1
        self.case = {
            'cons': cons,
            'b2b': torch.zeros((0, 3)),
            'p2b': torch.zeros((0, 3)),
            'pins': torch.zeros((0, 2)),
        }
        self.S = float(np.sqrt((self.sol[:, 2] * self.sol[:, 3]).sum()))
        self.H, self.V = lg.build_graph(
            self.sol,
            self.sol[:, 0] + self.sol[:, 2] / 2,
            self.sol[:, 1] + self.sol[:, 3] / 2,
        )
        self.cfg = {
            **lg.DEFAULT_CFG,
            'ripple_close_rel': 0.10,
            'ripple_drag_rel': 0.10,
            'ripple_budget_s': 1.0,
        }

    def repair(self, pre_mask=None, **overrides):
        cfg = {**self.cfg, **overrides}
        if pre_mask is None:
            pre_mask = np.zeros(3, dtype=bool)
        return lg.cluster_ripple_repair(
            self.sol.copy(), self.case, self.H, self.V, pre_mask, cfg,
            self.S, hpwl_base=1.0, area_base=20.8, n_soft=1)

    def test_blocker_is_rippled_and_group_connects(self):
        before_bbox = [
            (self.sol[:, a] + self.sol[:, 2 + a]).max()
            - self.sol[:, a].min() for a in (0, 1)
        ]
        out, stats = self.repair(np.array([True, False, False]))

        self.assertEqual(stats['group_before'], 1)
        self.assertEqual(stats['group_after'], 0)
        self.assertEqual(stats['moves'], 1)
        self.assertLess(out[2, 1], self.sol[2, 1])
        self.assertLessEqual(lg.max_penetration(out), lg._EPS_OVL)
        self.assertEqual(lg._violations_official(
            out, self.case['cons'].numpy())[1], 0)
        after_bbox = [
            (out[:, a] + out[:, 2 + a]).max() - out[:, a].min()
            for a in (0, 1)
        ]
        np.testing.assert_array_less(
            np.asarray(after_bbox), np.asarray(before_bbox) + 1e-7)

    def test_preplaced_blocker_rejects_ripple(self):
        pre_mask = np.array([True, False, True])
        out, stats = self.repair(pre_mask)
        np.testing.assert_allclose(out, self.sol)
        self.assertEqual(stats['moves'], 0)
        self.assertEqual(stats['group_after'], 1)

    def test_moved_block_cap_rejects_chain(self):
        out, stats = self.repair(
            np.array([True, False, False]), ripple_max_blocks=1)
        np.testing.assert_allclose(out, self.sol)
        self.assertEqual(stats['moves'], 0)

    def test_zero_budget_is_constant_time_noop(self):
        start = time.perf_counter()
        out, stats = self.repair(ripple_budget_s=0.0)
        elapsed = time.perf_counter() - start
        np.testing.assert_allclose(out, self.sol)
        self.assertEqual(stats['trials'], 0)
        self.assertLess(elapsed, 0.05)


if __name__ == '__main__':
    unittest.main()
