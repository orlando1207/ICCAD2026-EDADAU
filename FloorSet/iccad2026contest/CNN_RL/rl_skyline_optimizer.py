"""
RL-guided centers -> analytic_legalizer's skyline pipeline.

Replaces quadratic_placer.analytic_place()'s (cx, cy) output with centers
predicted by the trained GNN+CNN+RL policy (greedy rollout over
PlacementEnv), then runs the SAME validated legalization chain as
analytic_legalizer.optimizer.MyOptimizer:

    RL rollout -> (cx, cy)
        -> analytic_legalizer.constraints.parse_and_init / prepack_clusters
        -> analytic_legalizer.skyline_legalizer.skyline_legalize
        -> slide_boundary -> enforce_hard -> _detailed_place

Does not modify analytic_legalizer/ — only imports its pure functions.

Modes (constructor `center_source`):
  "model"     - greedy RL policy rollout centers (requires checkpoint;
                falls back to "quadratic" if no checkpoint or on error)
  "quadratic" - analytic_place() centers (ablation: sanity-checks that this
                transplant reproduces analytic_legalizer's own score)

Run:
    cd FloorSet/iccad2026contest
    python3 iccad2026_evaluate.py --evaluate CNN_RL/rl_skyline_optimizer.py --test-id 0 --verbose
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_CONTEST_DIR = Path(__file__).resolve().parent.parent
if str(_CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTEST_DIR))
_DL_DIR = Path(__file__).resolve().parent
if str(_DL_DIR) not in sys.path:
    sys.path.insert(0, str(_DL_DIR))

from iccad2026_evaluate import FloorplanOptimizer  # noqa: E402

from analytic_legalizer.constraints import (  # noqa: E402
    parse_and_init, prepack_clusters, slide_boundary, enforce_hard,
)
from analytic_legalizer.skyline_legalizer import (  # noqa: E402
    skyline_legalize, _detailed_place,
)
from analytic_legalizer.quadratic_placer import analytic_place  # noqa: E402
from ar_utils import ASPECT_BUCKETS  # noqa: E402

_DEFAULT_CKPT = _DL_DIR / "checkpoints" / "phase13_aspect.pt"


class RLSkylineOptimizer(FloorplanOptimizer):
    """RL-predicted (cx, cy) centers fed into analytic_legalizer's skyline
    legalization chain (skyline_legalize -> slide_boundary -> enforce_hard ->
    _detailed_place)."""

    def __init__(self, verbose: bool = False, checkpoint=None,
                 center_source: str = "model", compact_pass: bool = False,
                 aspect_pass: bool = True, shape_fit: bool = True,
                 b2_pass: bool = False):
        super().__init__(verbose)
        self.center_source = center_source
        self.compact_pass = compact_pass        # extra whitespace compaction (experiment)
        self.aspect_pass = aspect_pass          # predict soft-block aspect ratio (Phase 13)
        self.shape_fit = shape_fit              # contour-aware in-packer shaping (Phase 15/B1)
        self.b2_pass = b2_pass                  # critical-path slack shaping (Phase 15/B2)
        self._model = None
        if center_source == "model":
            ckpt_path = Path(checkpoint) if checkpoint else _DEFAULT_CKPT
            if ckpt_path.exists():
                try:
                    from gnn_encoder import GNNEncoder
                    from policy_net import PolicyValueNet
                    ck = torch.load(ckpt_path, map_location="cpu")
                    gnn = GNNEncoder(out_dim=ck["gnn_out"], hidden=ck["hidden"])
                    pol = PolicyValueNet(in_channels=ck.get("in_channels", 4),
                                         node_dim=ck["gnn_out"], hidden=ck["hidden"])
                    gnn.load_state_dict(ck["gnn"])
                    pol.load_state_dict(ck["policy"])
                    gnn.eval()
                    pol.eval()
                    self._model = (gnn, pol, int(ck["grid"]))
                    if verbose:
                        print(f"[RLSkyline] loaded model {ckpt_path} (grid={ck['grid']})")
                except Exception as e:  # pragma: no cover
                    if verbose:
                        print(f"[RLSkyline] checkpoint load failed ({e}); "
                              f"falling back to quadratic centers")

    def _rl_centers(self, area_targets, b2b, p2b, pins, constraints,
                    target_positions, block_count
                    ) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[int, float]]]:
        """Greedy RL rollout -> per-block (cx, cy) + (if `aspect_pass`) a
        predicted aspect ratio (w/h) for "free" blocks (no fixed/MIB/cluster
        shape constraint). None if no model loaded."""
        if self._model is None:
            return None
        from placement_env import PlacementEnv
        from canvas_raster import rasterize_env, CH_FEASIBILITY

        gnn, policy, grid = self._model
        env = PlacementEnv(grid=grid)
        env.reset(area_targets, b2b, p2b, pins, constraints, torch.ones(8),
                  target_positions=target_positions)
        node_emb, _ = gnn.encode_problem(area_targets, constraints, b2b, block_count)
        aspect_choice: Dict[int, float] = {}
        done = len(env.order) == 0
        while not done:
            st = env._build_state()
            cur = st["current_block"]
            canvas = rasterize_env(env)
            mask = canvas[CH_FEASIBILITY]
            is_free = (self.aspect_pass and cur not in env.fixed_dims
                       and cur not in env.mib_dims and int(constraints[cur, 3]) == 0)
            if is_free:
                a, aspect_idx, _lp, _v = policy.act_aspect(canvas, node_emb[cur], mask, greedy=True)
                aspect = ASPECT_BUCKETS[aspect_idx]
                aspect_choice[cur] = aspect
            else:
                a, _lp, _v = policy.act(canvas, node_emb[cur], mask, greedy=True)
                aspect = 1.0
            _s, _r, done, _i = env.step(a, aspect=aspect)

        cx = np.zeros(block_count, dtype=np.float64)
        cy = np.zeros(block_count, dtype=np.float64)
        for i in range(block_count):
            x, y, w, h = (float(t) for t in env.positions[i])
            cx[i] = x + w / 2.0
            cy[i] = y + h / 2.0
        return cx, cy, aspect_choice

    @staticmethod
    def _overlap_free(pos, tol=1e-3) -> bool:
        n = len(pos)
        for i in range(n):
            xi, yi, wi, hi = pos[i]
            for j in range(i + 1, n):
                xj, yj, wj, hj = pos[j]
                if (min(xi + wi, xj + wj) - max(xi, xj) > tol and
                        min(yi + hi, yj + hj) - max(yi, yj) > tol):
                    return False
        return True

    @staticmethod
    def _proxy(pos, b2b, p2b, pins) -> float:
        """bbox_area · raw_HPWL — the same quality drivers the contest cost uses
        (lower is better). Used to keep B2 only when it beats B1."""
        from analytic_legalizer.skyline_legalizer import _raw_hpwl
        xs = [p[0] for p in pos]; ys = [p[1] for p in pos]
        x2 = max(p[0] + p[2] for p in pos); y2 = max(p[1] + p[3] for p in pos)
        bbox = (x2 - min(xs)) * (y2 - min(ys))
        hp = _raw_hpwl(pos, b2b, p2b, pins)
        return bbox * (hp if hp > 1e-9 else 1.0)

    def _maybe_b2(self, pos_b1, blocks, super_blocks, cluster_groups,
                  area_targets, b2b, p2b, pins, finish):
        """Run the B2 critical-path shaping from B1's relative order; return
        whichever final layout (B1 or B2) is feasible and scores lower."""
        from topo_shape import legalize_b2
        pos_b2 = legalize_b2(pos_b1, blocks, super_blocks, cluster_groups,
                             area_targets, b2b=b2b, p2b=p2b, pins=pins)
        pos_b2 = finish(pos_b2)
        if (self._overlap_free(pos_b2) and
                self._proxy(pos_b2, b2b, p2b, pins) <
                self._proxy(pos_b1, b2b, p2b, pins) - 1e-6):
            return pos_b2
        return pos_b1

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None,
    ) -> List[Tuple[float, float, float, float]]:

        blocks, mib_groups, cluster_groups = parse_and_init(
            block_count, area_targets, constraints, target_positions
        )
        super_blocks = prepack_clusters(blocks, cluster_groups)

        cx = cy = None
        aspect_choice: Dict[int, float] = {}
        if self.center_source == "model":
            try:
                centers = self._rl_centers(
                    area_targets, b2b_connectivity, p2b_connectivity, pins_pos,
                    constraints, target_positions, block_count)
                if centers is not None:
                    cx, cy, aspect_choice = centers
            except Exception as e:  # pragma: no cover
                if self.verbose:
                    print(f"[RLSkyline] RL rollout failed ({e}); "
                          f"falling back to quadratic centers")

        if cx is None:
            cx, cy = analytic_place(
                blocks, super_blocks, cluster_groups,
                b2b_connectivity, p2b_connectivity, pins_pos,
                seed=0, noise_std=0.0,
            )

        # Phase 13: override the default square shape of "free" soft blocks
        # with the RL-predicted aspect ratio (area-preserving).
        for i, aspect in aspect_choice.items():
            area = max(float(area_targets[i]), 1e-6)
            blocks[i].w = math.sqrt(area * aspect)
            blocks[i].h = math.sqrt(area / aspect)

        if self.shape_fit:                          # Phase 15/B1: contour-aware shaping
            from skyline_shape import skyline_legalize_shaped
            pos, _ = skyline_legalize_shaped(
                blocks, super_blocks, cluster_groups, cx, cy, area_targets,
                b2b=b2b_connectivity, p2b=p2b_connectivity, pins=pins_pos,
            )
        else:
            pos, _ = skyline_legalize(
                blocks, super_blocks, cluster_groups, cx, cy, area_targets,
                b2b=b2b_connectivity, p2b=p2b_connectivity, pins=pins_pos,
            )
        if self.compact_pass:                       # squeeze residual whitespace
            from analytic_legalizer.topology import compact
            pos = compact(pos, blocks, super_blocks, cluster_groups)

        def _finish(p):
            p = slide_boundary(p, blocks, super_blocks, cluster_groups)
            p = enforce_hard(p, blocks, area_targets)
            return _detailed_place(p, blocks, b2b_connectivity,
                                   p2b_connectivity, pins_pos)

        pos = _finish(pos)

        if self.b2_pass:                             # Phase 15/B2: critical-path shaping
            try:
                pos = self._maybe_b2(
                    pos, blocks, super_blocks, cluster_groups, area_targets,
                    b2b_connectivity, p2b_connectivity, pins_pos, _finish)
            except Exception as e:  # pragma: no cover
                if self.verbose:
                    print(f"[RLSkyline] B2 pass failed ({e}); keeping B1 result")
        return pos


# The --evaluate loader matches by name across importlib's duplicate
# FloorplanOptimizer copy, so expose a fallback name (see CNN_RL/rl_optimizer.py).
Optimizer = RLSkylineOptimizer
