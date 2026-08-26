"""Tests for the DP-1 area-preserving reshape fallback (`_reshape_extend`).

Ported from the `final` branch; the properties asserted here are what make it
safe to run inside `cluster_perp_align`:

  * area is preserved exactly (so H2 cannot be violated),
  * the far edge does not move (which is the whole point: it reaches the case
    where graph slack is 0 because that edge is pinned),
  * the perpendicular side never sheds more than `min_frac` allows, and the
    added aspect guard never blocks a move that improves the aspect,
  * the freed perpendicular width is shed within [pos, pos + freed], so the
    block never claims space it did not already occupy,
  * end to end, the stage cannot make a layout infeasible or worse.
"""

import unittest

import numpy as np
import torch

from iccad2026_evaluate import check_overlap

from . import legalizer as lg


def _case(areas, cons, target=None, b2b=None):
    n = len(areas)
    if target is None:
        target = np.full((n, 4), -1.0)
    return {
        'area': torch.tensor(areas, dtype=torch.float64),
        'cons': torch.tensor(cons, dtype=torch.long),
        'target': torch.tensor(np.asarray(target, dtype=np.float64),
                               dtype=torch.float64),
        'gt': None,
        'b2b': torch.tensor(b2b if b2b is not None else np.zeros((0, 3)),
                            dtype=torch.float64),
        'p2b': torch.zeros((0, 3), dtype=torch.float64),
        'pins': torch.zeros((0, 2), dtype=torch.float64),
        'metrics': None,
    }


class ReshapeExtendTest(unittest.TestCase):
    def test_area_is_preserved_exactly(self):
        for need in (0.5, -0.5, 2.0, -3.0, 0.01):
            r = lg._reshape_extend(10.0, 4.0, 20.0, 6.0, need, 0.1)
            self.assertIsNotNone(r)
            _p, ws, _pp, hs = r
            self.assertAlmostEqual(ws * hs, 4.0 * 6.0, places=12)

    def test_far_edge_is_pinned(self):
        # need > 0: low edge held, block grows upward
        p, ws, _pp, _hs = lg._reshape_extend(10.0, 4.0, 20.0, 6.0, 1.5, 0.1)
        self.assertAlmostEqual(p, 10.0, places=12)
        self.assertAlmostEqual(ws, 5.5, places=12)
        # need < 0: high edge held, block grows downward
        p, ws, _pp, _hs = lg._reshape_extend(10.0, 4.0, 20.0, 6.0, -1.5, 0.1)
        self.assertAlmostEqual(p + ws, 14.0, places=12)
        self.assertAlmostEqual(ws, 5.5, places=12)

    def test_min_frac_guard(self):
        # growing the separation side 4x would leave 25% of the perpendicular
        self.assertIsNone(lg._reshape_extend(1.0, 1.0, 0.0, 4.0, 3.0, 0.3))
        self.assertIsNotNone(lg._reshape_extend(1.0, 1.0, 0.0, 4.0, 3.0, 0.2))

    def test_aspect_guard_never_blocks_an_improvement(self):
        # a block that is ALREADY worse than the cap, reshaped towards square:
        # sep 1 x perp 100 -> grow sep, aspect falls, must be allowed
        r = lg._reshape_extend(0.0, 1.0, 0.0, 100.0, 8.0, 0.01, aspect_cap=3.6)
        self.assertIsNotNone(r)
        _p, ws, _pp, hs = r
        self.assertLess(max(ws / hs, hs / ws), 100.0)
        # and one that would push aspect past the cap is refused
        self.assertIsNone(
            lg._reshape_extend(0.0, 10.0, 0.0, 10.0, 40.0, 0.01,
                               aspect_cap=3.6))

    def test_freed_width_stays_inside_the_old_footprint(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            ss, sp = rng.uniform(0.5, 8.0), rng.uniform(0.5, 8.0)
            pp = rng.uniform(-5, 5)
            need = rng.uniform(-3, 3)
            tgt = rng.uniform(-10, 10)
            r = lg._reshape_extend(0.0, ss, pp, sp, need, 0.05,
                                   perp_center_target=tgt)
            if r is None:
                continue
            _p, _ws, pp1, hs1 = r
            freed = sp - hs1
            self.assertGreaterEqual(pp1 + 1e-12, pp)
            self.assertLessEqual(pp1 - 1e-12, pp + freed)
            self.assertLessEqual(pp1 + hs1 - 1e-12, pp + sp)

    def test_hpwl_target_pulls_the_shed_width(self):
        # target far below -> shed from the high side, so the low edge stays
        r_lo = lg._reshape_extend(0.0, 4.0, 10.0, 4.0, 1.0, 0.1,
                                  perp_center_target=-100.0)
        # target far above -> shed from the low side, low edge moves up
        r_hi = lg._reshape_extend(0.0, 4.0, 10.0, 4.0, 1.0, 0.1,
                                  perp_center_target=+100.0)
        self.assertLess(r_lo[2], r_hi[2])
        self.assertAlmostEqual(r_lo[2], 10.0, places=12)

    def test_degenerate_inputs_return_none(self):
        self.assertIsNone(lg._reshape_extend(4.0, 2.0, 0.0, 2.0, 0.0, 0.1))


class PerpAlignCornerTouchTest(unittest.TestCase):
    """Two cluster members touching at a corner, with the mover's shift blocked
    by a preplaced neighbour on the shift axis. A shift cannot fix it; the
    reshape can, because it holds the pinned edge."""

    def _build(self):
        #  block 0: cluster member, x 0..4, y 0..4
        #  block 1: cluster member, x 4..8, y 4..8   (corner touch with 0)
        #  block 2: preplaced wall right under block 1, y 0..4 at x 4..8,
        #           so block 1 cannot be shifted down
        areas = [16.0, 16.0, 16.0]
        cons = [[0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0]]
        target = [[-1.0, -1, -1, -1], [-1.0, -1, -1, -1], [4.0, 0.0, 4.0, 4.0]]
        case = _case(areas, cons, target)
        sol = np.array([[0.0, 0.0, 4.0, 4.0],
                        [4.0, 4.0, 4.0, 4.0],
                        [4.0, 0.0, 4.0, 4.0]], dtype=np.float64)
        return case, sol

    def _run(self, cfg_over):
        case, sol = self._build()
        cons = case['cons'].numpy()
        pre = cons[:, 1] > 0
        keyx = sol[:, 0] + sol[:, 2] / 2
        keyy = sol[:, 1] + sol[:, 3] / 2
        H, V = lg.build_graph(sol, keyx, keyy)
        adjH, adjV = lg._adj_arrays(3, H), lg._adj_arrays(3, V)
        cfg = {**lg.DEFAULT_CFG, **cfg_over}
        S = float(np.sqrt(case['area'].numpy().sum()))
        n_soft = max(lg._n_soft_norm(cons), 1)
        hb, ab = 1.0, float(lg._bbox_area(torch.tensor(sol)))
        out = lg.cluster_perp_align(sol.copy(), case, adjH, adjV, pre, cfg, S,
                                    hb, ab, n_soft)
        return case, out, cons

    def test_reshape_can_run_without_breaking_anything(self):
        for over in ({'reshape_align': True}, {'reshape_align': False}):
            case, out, cons = self._run(over)
            self.assertTrue(lg.hard_feasibility(out, case)['feasible'], over)
            self.assertEqual(
                check_overlap([tuple(map(float, r)) for r in out]), 0)
            # the preplaced block must not have moved or changed shape
            np.testing.assert_allclose(out[2], [4.0, 0.0, 4.0, 4.0], atol=1e-12)
            # soft areas exact
            for i in (0, 1):
                self.assertAlmostEqual(out[i, 2] * out[i, 3], 16.0, places=9)

    def test_stage_never_worsens_the_proxy(self):
        rng = np.random.default_rng(3)
        for _ in range(12):
            n = 8
            areas = rng.uniform(4.0, 16.0, size=n)
            cons = np.zeros((n, 5), dtype=np.int64)
            cons[:4, 3] = 1                      # one cluster group
            cons[5, 4] = 1                       # a boundary bit
            case = _case(areas.tolist(), cons.tolist())
            sol = np.zeros((n, 4))
            x = 0.0
            for i in range(n):                   # a legal row, no overlap
                w = float(np.sqrt(areas[i]))
                sol[i] = [x, rng.uniform(0, 0.5), w, areas[i] / w]
                x += w + 0.01
            pre = np.zeros(n, dtype=bool)
            keyx = sol[:, 0] + sol[:, 2] / 2
            keyy = sol[:, 1] + sol[:, 3] / 2
            H, V = lg.build_graph(sol, keyx, keyy)
            adjH, adjV = lg._adj_arrays(n, H), lg._adj_arrays(n, V)
            n_soft = max(lg._n_soft_norm(cons), 1)
            hb, ab = 1.0, float(lg._bbox_area(torch.tensor(sol)))
            before = lg.proxy_cost(sol, case, hb, ab, n_soft)
            out = lg.cluster_perp_align(sol.copy(), case, adjH, adjV, pre,
                                        dict(lg.DEFAULT_CFG),
                                        float(np.sqrt(areas.sum())), hb, ab,
                                        n_soft)
            after = lg.proxy_cost(out, case, hb, ab, n_soft)
            self.assertLessEqual(after, before + 1e-12)
            self.assertTrue(lg.hard_feasibility(out, case)['feasible'])


if __name__ == '__main__':
    unittest.main()
