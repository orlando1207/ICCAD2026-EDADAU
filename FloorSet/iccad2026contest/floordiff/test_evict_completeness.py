"""Regression tests for stage E: eviction to anchor-consistency (the guarantee).

The claim under test (FEASIBILITY_ANALYSIS.md, Theorem 2 + Lemma 3): after
`evict_for_consistency`, `_find_conflict` reports no anchor conflict on either
axis, both relation graphs are acyclic, and therefore `assign_axis` produces an
overlap-free placement.
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


def _is_dag(n, edges):
    indeg = [0] * n
    succ = [[] for _ in range(n)]
    for i, j in edges:
        succ[i].append(j)
        indeg[j] += 1
    stack = [v for v in range(n) if indeg[v] == 0]
    seen = 0
    while stack:
        v = stack.pop()
        seen += 1
        for j in succ[v]:
            indeg[j] -= 1
            if indeg[j] == 0:
                stack.append(j)
    return seen == n


def _order_is_topological(order, edges):
    rank = {v: k for k, v in enumerate(order)}
    return all(rank[i] < rank[j] for i, j in edges)


class TrappedCorridorTest(unittest.TestCase):
    """Two anchors 2 units apart with three 3-wide soft blocks between them:
    no reshape or edge flip can fit them, so only eviction can restore
    consistency."""

    def _build(self):
        # blocks 0,1 preplaced at x=0..3 and x=5..8, same y band.
        # blocks 2,3,4 are soft, 3x3, all sitting inside the 2-wide corridor.
        areas = [9.0, 9.0, 9.0, 9.0, 9.0]
        cons = [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        target = [[0.0, 0.0, 3.0, 3.0], [5.0, 0.0, 3.0, 3.0],
                  [-1.0, -1, -1, -1], [-1.0, -1, -1, -1], [-1.0, -1, -1, -1]]
        case = _case(areas, cons, target)
        sol = np.array([[0.0, 0.0, 3.0, 3.0],
                        [5.0, 0.0, 3.0, 3.0],
                        [3.1, 0.0, 3.0, 3.0],
                        [3.3, 0.2, 3.0, 3.0],
                        [3.5, 0.4, 3.0, 3.0]], dtype=np.float64)
        return case, sol

    def test_conflict_is_detected(self):
        case, sol = self._build()
        n = len(sol)
        pre = case['cons'].numpy()[:, 1] > 0
        keyx = sol[:, 0] + sol[:, 2] / 2
        keyy = sol[:, 1] + sol[:, 3] / 2
        H, V = lg.build_graph(sol, keyx, keyy)
        ordx = np.lexsort((np.arange(n), keyx)).tolist()
        # the corridor is over-full, so an anchor lower bound must be violated
        self.assertIsNotNone(
            lg._find_conflict(n, H, sol[:, 0], sol[:, 2], pre, ordx))

    def test_eviction_restores_consistency_and_acyclicity(self):
        case, sol = self._build()
        n = len(sol)
        pre = case['cons'].numpy()[:, 1] > 0
        keyx = sol[:, 0] + sol[:, 2] / 2
        keyy = sol[:, 1] + sol[:, 3] / 2
        H, V = lg.build_graph(sol, keyx, keyy)
        score = lg._eviction_score(case)
        out = lg.evict_for_consistency(n, H, V, sol, pre, keyx, keyy, score)
        self.assertIsNotNone(out)
        H2, V2, ordx, ordy, evicted = out
        self.assertGreater(len(evicted), 0)
        self.assertTrue(all(not pre[m] for m in evicted))
        self.assertIsNone(
            lg._find_conflict(n, H2, sol[:, 0], sol[:, 2], pre, ordx))
        self.assertIsNone(
            lg._find_conflict(n, V2, sol[:, 1], sol[:, 3], pre, ordy))
        self.assertTrue(_is_dag(n, H2))
        self.assertTrue(_is_dag(n, V2))
        self.assertTrue(_order_is_topological(ordx, H2))
        self.assertTrue(_order_is_topological(ordy, V2))

    def test_assignment_after_eviction_is_overlap_free(self):
        case, sol = self._build()
        n = len(sol)
        cons = case['cons'].numpy()
        pre = cons[:, 1] > 0
        gt = lg.target_xywh(case).numpy().astype(np.float64)
        keyx = sol[:, 0] + sol[:, 2] / 2
        keyy = sol[:, 1] + sol[:, 3] / 2
        H, V = lg.build_graph(sol, keyx, keyy)
        score = lg._eviction_score(case)
        H2, V2, ordx, ordy, _ = lg.evict_for_consistency(
            n, H, V, sol, pre, keyx, keyy, score)
        out = sol.copy()
        out[:, 0] = lg.assign_axis(n, H2, sol[:, 0].copy(), sol[:, 2], pre,
                                   gt[:, 0], ordx)
        out[:, 1] = lg.assign_axis(n, V2, sol[:, 1].copy(), sol[:, 3], pre,
                                   gt[:, 1], ordy)
        self.assertLessEqual(lg.max_penetration(out), lg._EPS_OVL)
        self.assertEqual(check_overlap([tuple(map(float, r)) for r in out]), 0)
        # anchors must not have moved
        for i in np.nonzero(pre)[0]:
            self.assertAlmostEqual(out[i, 0], gt[i, 0], places=9)
            self.assertAlmostEqual(out[i, 1], gt[i, 1], places=9)

    def test_full_legalize_case_is_feasible(self):
        case, sol = self._build()
        pred = torch.tensor(sol, dtype=torch.float64)
        out, info = lg.legalize_case(pred, case)
        self.assertTrue(info['hard']['feasible'])
        self.assertTrue(info['graph']['final_assignment_ok'])
        self.assertEqual(
            check_overlap([tuple(map(float, r)) for r in out.numpy()]), 0)

    def test_disabling_eviction_keeps_the_floor_feasible(self):
        """With stage E off the relation set stays broken, so the guaranteed
        construction must catch it -- legalize_case may never return overlap."""
        case, sol = self._build()
        pred = torch.tensor(sol, dtype=torch.float64)
        out, info = lg.legalize_case(pred, case, {'evict_repair': False})
        self.assertTrue(info['hard']['feasible'])
        self.assertEqual(
            check_overlap([tuple(map(float, r)) for r in out.numpy()]), 0)


class GuaranteedConstructionTest(unittest.TestCase):
    def _random_case(self, rng, n=14):
        areas = rng.uniform(4.0, 40.0, size=n)
        cons = np.zeros((n, 5), dtype=np.int64)
        target = np.full((n, 4), -1.0)
        # a non-overlapping anchor field: a row of preplaced blocks
        x = 0.0
        for i in range(0, n, 3):
            w = float(np.sqrt(areas[i]))
            cons[i, 1] = 1
            target[i] = [x, 0.0, w, areas[i] / w]
            x += w + 0.5
        for i in range(1, n, 5):
            if cons[i, 1] == 0:
                cons[i, 0] = 1
                w = float(np.sqrt(areas[i]))
                target[i, 2:4] = [w, areas[i] / w]
        cons[2, 4] = 1
        cons[3, 3] = 1
        cons[4, 3] = 1
        case = _case(areas.tolist(), cons.tolist(), target)
        pred = np.zeros((n, 4))
        pred[:, 0] = rng.uniform(-5, 15, size=n)
        pred[:, 1] = rng.uniform(-5, 15, size=n)
        pred[:, 2] = np.sqrt(areas)
        pred[:, 3] = areas / pred[:, 2]
        for i in np.nonzero(cons[:, 1])[0]:
            pred[i] = target[i]
        for i in np.nonzero(cons[:, 0])[0]:
            pred[i, 2:4] = target[i, 2:4]
        return case, pred

    def test_always_feasible_on_random_anchor_fields(self):
        rng = np.random.default_rng(0)
        for _ in range(25):
            case, pred = self._random_case(rng)
            sol = lg.guaranteed_construction(pred, case)
            hard = lg.hard_feasibility(sol, case)
            self.assertTrue(hard['feasible'], hard)
            self.assertEqual(
                check_overlap([tuple(map(float, r)) for r in sol]), 0)

    def test_respects_immutable_geometry(self):
        rng = np.random.default_rng(3)
        case, pred = self._random_case(rng)
        sol = lg.guaranteed_construction(pred, case)
        cons = case['cons'].numpy()
        gt = lg.target_xywh(case).numpy()
        for i in np.nonzero(cons[:, 1])[0]:
            np.testing.assert_allclose(sol[i], gt[i], atol=1e-12)
        for i in np.nonzero(cons[:, 0])[0]:
            np.testing.assert_allclose(sol[i, 2:4], gt[i, 2:4], atol=1e-12)


class HoleRelocateTest(unittest.TestCase):
    def test_never_breaks_feasibility_and_never_worsens_proxy(self):
        rng = np.random.default_rng(7)
        n = 12
        areas = rng.uniform(4.0, 20.0, size=n)
        cons = np.zeros((n, 5), dtype=np.int64)
        case = _case(areas.tolist(), cons.tolist())
        # a legal shelf layout to start from
        sol = np.zeros((n, 4))
        x = 0.0
        for i in range(n):
            w = float(np.sqrt(areas[i]))
            sol[i] = [x, 0.0, w, areas[i] / w]
            x += w
        pre = np.zeros(n, dtype=bool)
        hb, ab = 1.0, float(lg._bbox_area(torch.tensor(sol)))
        n_soft = max(lg._n_soft_norm(cons), 1)
        before = lg.proxy_cost(sol, case, hb, ab, n_soft)
        out, stats = lg.hole_relocate(sol, case, list(range(n)), pre,
                                      dict(lg.DEFAULT_CFG), 10.0, hb, ab,
                                      n_soft)
        after = lg.proxy_cost(out, case, hb, ab, n_soft)
        self.assertTrue(lg.hard_feasibility(out, case)['feasible'])
        self.assertLessEqual(after, before + 1e-12)

    def test_refuses_to_touch_an_infeasible_layout(self):
        areas = [4.0, 4.0]
        cons = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        case = _case(areas, cons)
        sol = np.array([[0.0, 0.0, 2.0, 2.0], [0.5, 0.5, 2.0, 2.0]])
        out, stats = lg.hole_relocate(sol, case, [0, 1],
                                      np.zeros(2, dtype=bool),
                                      dict(lg.DEFAULT_CFG), 4.0, 1.0, 4.0, 1)
        np.testing.assert_allclose(out, sol)
        self.assertEqual(stats['moved'], 0)


if __name__ == '__main__':
    unittest.main()
