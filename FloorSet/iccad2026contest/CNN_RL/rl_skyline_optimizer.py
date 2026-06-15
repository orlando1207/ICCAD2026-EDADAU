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

import sys
from pathlib import Path
from typing import List, Optional, Tuple

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

_DEFAULT_CKPT = _DL_DIR / "checkpoints" / "phase10_soft.pt"


class RLSkylineOptimizer(FloorplanOptimizer):
    """RL-predicted (cx, cy) centers fed into analytic_legalizer's skyline
    legalization chain (skyline_legalize -> slide_boundary -> enforce_hard ->
    _detailed_place)."""

    def __init__(self, verbose: bool = False, checkpoint=None,
                 center_source: str = "model"):
        super().__init__(verbose)
        self.center_source = center_source
        self._model = None
        if center_source == "model":
            ckpt_path = Path(checkpoint) if checkpoint else _DEFAULT_CKPT
            if ckpt_path.exists():
                try:
                    from gnn_encoder import GNNEncoder
                    from policy_net import PolicyValueNet
                    ck = torch.load(ckpt_path, map_location="cpu")
                    gnn = GNNEncoder(out_dim=ck["gnn_out"], hidden=ck["hidden"])
                    pol = PolicyValueNet(in_channels=4, node_dim=ck["gnn_out"],
                                         hidden=ck["hidden"])
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
                    target_positions, block_count) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Greedy RL rollout -> per-block (cx, cy). None if no model loaded."""
        if self._model is None:
            return None
        from placement_env import PlacementEnv
        from canvas_raster import rasterize_env, CH_FEASIBILITY

        gnn, policy, grid = self._model
        env = PlacementEnv(grid=grid)
        env.reset(area_targets, b2b, p2b, pins, constraints, torch.ones(8),
                  target_positions=target_positions)
        node_emb, _ = gnn.encode_problem(area_targets, constraints, b2b, block_count)
        done = len(env.order) == 0
        while not done:
            st = env._build_state()
            cur = st["current_block"]
            canvas = rasterize_env(env)
            mask = canvas[CH_FEASIBILITY]
            a, _lp, _v = policy.act(canvas, node_emb[cur], mask, greedy=True)
            _s, _r, done, _i = env.step(a)

        cx = np.zeros(block_count, dtype=np.float64)
        cy = np.zeros(block_count, dtype=np.float64)
        for i in range(block_count):
            x, y, w, h = (float(t) for t in env.positions[i])
            cx[i] = x + w / 2.0
            cy[i] = y + h / 2.0
        return cx, cy

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
        if self.center_source == "model":
            try:
                centers = self._rl_centers(
                    area_targets, b2b_connectivity, p2b_connectivity, pins_pos,
                    constraints, target_positions, block_count)
                if centers is not None:
                    cx, cy = centers
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

        pos, _ = skyline_legalize(
            blocks, super_blocks, cluster_groups, cx, cy, area_targets,
            b2b=b2b_connectivity, p2b=p2b_connectivity, pins=pins_pos,
        )
        pos = slide_boundary(pos, blocks, super_blocks, cluster_groups)
        pos = enforce_hard(pos, blocks, area_targets)
        pos = _detailed_place(pos, blocks, b2b_connectivity, p2b_connectivity, pins_pos)
        return pos


# The --evaluate loader matches by name across importlib's duplicate
# FloorplanOptimizer copy, so expose a fallback name (see CNN_RL/rl_optimizer.py).
Optimizer = RLSkylineOptimizer
