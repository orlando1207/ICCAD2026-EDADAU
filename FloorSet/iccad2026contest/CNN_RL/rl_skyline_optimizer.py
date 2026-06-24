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

from legalizer.constraints import (  # noqa: E402
    parse_and_init, prepack_clusters, slide_boundary, enforce_hard,
)
from legalizer.skyline_legalizer import (  # noqa: E402
    skyline_legalize, _detailed_place,
)
from legalizer.quadratic_placer import analytic_place  # noqa: E402
from ar_utils import ASPECT_BUCKETS  # noqa: E402

_DEFAULT_CKPT = _DL_DIR / "checkpoints" / "phase13_aspect.pt"


class RLSkylineOptimizer(FloorplanOptimizer):
    """RL-predicted (cx, cy) centers fed into analytic_legalizer's skyline
    legalization chain (skyline_legalize -> slide_boundary -> enforce_hard ->
    _detailed_place)."""

    def __init__(self, verbose: bool = False, checkpoint=None,
                 center_source: str = "model", compact_pass: bool = False,
                 aspect_pass: bool = True, shape_fit: bool = True,
                 b2_pass: bool = False, boundary_pass: bool = False,
                 mib_shape: bool = False, cluster_shape: bool = False):
        super().__init__(verbose)
        import os
        self.center_source = center_source
        self.compact_pass = compact_pass        # extra whitespace compaction (experiment)
        # Phase 13 aspect head. Env override (ASPECT_PASS=0) lets the eval harness —
        # which only passes verbose — disable it for checkpoints trained with
        # --aspect-weight 0.0 (untrained head would otherwise pick garbage ratios).
        if "ASPECT_PASS" in os.environ:
            aspect_pass = os.environ["ASPECT_PASS"] == "1"
        self.aspect_pass = aspect_pass          # predict soft-block aspect ratio (Phase 13)
        self.shape_fit = shape_fit              # contour-aware in-packer shaping (Phase 15/B1)
        self.b2_pass = b2_pass                  # critical-path slack shaping (Phase 15/B2)
        self.boundary_pass = boundary_pass or os.environ.get("BOUNDARY_PASS", "0") == "1"  # TOP/RIGHT boundary reshape (Item A)
        self.mib_shape = mib_shape              # reshape free MIB groups (Item B-MIB, net loss)
        import os                               # re-pack cluster super-blocks (Item B-cluster)
        self.cluster_shape = cluster_shape or os.environ.get("CLUSTER_SHAPE", "0") == "1"
        # Real-cost gate (default ON): pick among reshape variants by the EXACT contest
        # cost form (1+0.5(area_gap+hpwl_gap))·e^{2V} — relative to a reference candidate
        # — instead of the bbox·HPWL proxy that omits the violation multiplier. Unlocks
        # cluster reshaping (1.8088→1.8032); short-circuited to the default single path
        # when the problem has no reshapeable cluster, so runtime is unchanged (~0.92s).
        self.real_cost_gate = os.environ.get("REAL_COST_GATE", "1") == "1"
        # Deformable-cluster candidate: place cluster members as individual reshapeable
        # blocks (drop the rigid super-block) so they conform to the contour; the
        # real-cost gate keeps it only if the area win beats any grouping violation.
        self.free_cluster = os.environ.get("FREE_CLUSTER", "1") == "1"
        self._model = None
        if center_source == "model":
            import os
            checkpoint = checkpoint or os.environ.get("RL_CKPT")   # eval-time override
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
                    target_positions, block_count, greedy: bool = True
                    ) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[int, float]]]:
        """Greedy (or, with greedy=False, sampled) RL rollout -> per-block (cx, cy)
        + (if `aspect_pass`) a
        predicted aspect ratio (w/h) for "free" blocks (no fixed/MIB/cluster
        shape constraint). None if no model loaded."""
        if self._model is None:
            return None
        from placement_env import PlacementEnv
        from canvas_raster import rasterize_env, CH_FEASIBILITY

        gnn, policy, grid = self._model
        env = PlacementEnv(grid=grid)
        env.skip_occupancy = True          # rollout uses rasterize_env, not the grid
        env.skip_terminal_cost = True      # rollout discards the reward
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
                a, aspect_idx, _lp, _v = policy.act_aspect(canvas, node_emb[cur], mask, greedy=greedy)
                aspect = ASPECT_BUCKETS[aspect_idx]
                aspect_choice[cur] = aspect
            else:
                a, _lp, _v = policy.act(canvas, node_emb[cur], mask, greedy=greedy)
                aspect = 1.0
            _s, _r, done, _i = env.step(a, aspect=aspect)

        cx = np.zeros(block_count, dtype=np.float64)
        cy = np.zeros(block_count, dtype=np.float64)
        dims = np.zeros((block_count, 2), dtype=np.float64)
        for i in range(block_count):
            x, y, w, h = (float(t) for t in env.positions[i])
            cx[i] = x + w / 2.0
            cy[i] = y + h / 2.0
            dims[i] = (w, h)
        # ePlace center refinement (EPLACE=1): analytical WL+density global placement
        # on the rollout centers before the skyline legalizer re-packs them. Off by
        # default; the legalizer/feasibility path is otherwise untouched.
        import os
        # ePlace center refinement (OPT-IN: EPLACE=1) — analytical WL+density refinement of
        # the rollout centers before skyline re-packs (win: phase13 1.6039→1.5919). Default
        # OFF so the scored pipeline stays phase13+skyline. EPLACE_MIN_N gates to count>=N.
        _emin = int(os.environ.get("EPLACE_MIN_N", "0"))
        if os.environ.get("EPLACE", "0") == "1" and block_count >= _emin:
            try:
                from eplace import eplace_refine
                areas = np.array([max(float(area_targets[i]), 1e-3)
                                  for i in range(block_count)])
                pmask = np.array([bool(float(constraints[i, 1]) > 0)
                                  for i in range(block_count)])
                cx, cy = eplace_refine(cx, cy, dims, areas, b2b, p2b, pins, pmask)
            except Exception as e:  # pragma: no cover
                print(f"[ePlace] refine failed ({e}); using rollout centers")
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
        from legalizer.skyline_legalizer import _raw_hpwl
        xs = [p[0] for p in pos]; ys = [p[1] for p in pos]
        x2 = max(p[0] + p[2] for p in pos); y2 = max(p[1] + p[3] for p in pos)
        bbox = (x2 - min(xs)) * (y2 - min(ys))
        hp = _raw_hpwl(pos, b2b, p2b, pins)
        return bbox * (hp if hp > 1e-9 else 1.0)

    def _gate_real_cost(self, cands, constraints, b2b, p2b, pins,
                        area_targets, target_positions):
        """Pick the candidate layout with the lowest EXACT contest cost
        `(1+0.5(area_gap+hpwl_gap))·e^{2V}`. The contest clamps each gap with
        `max(0,·)` (you can't beat the GT optimum), so to keep *improvements visible*
        we baseline against the BEST candidate on each axis — `min` bbox and `min`
        HPWL across candidates — making every gap ≥ 0 and the smallest layout the
        zero-gap reference. (Using candidate 0 as the baseline hid any candidate that
        was better than it: its negative gap clamped to 0, so only the violation term
        decided and the lowest-violation candidate always won.) Baselines use only the
        candidate layouts — no ground-truth — so this is valid at inference."""
        from iccad2026_evaluate import evaluate_solution, compute_cost

        ms = [evaluate_solution(
                  {"positions": p, "runtime": 1.0}, {}, constraints, b2b, p2b, pins,
                  area_targets, target_positions, median_runtime=1.0)
              for p in cands]
        a0 = min(m.bbox_area for m in ms)
        h0 = min(m.hpwl_total for m in ms)
        best_pos, best_cost = None, float("inf")
        for pos, m in zip(cands, ms):
            ag = (m.bbox_area - a0) / max(a0, 1e-9)
            hg = (m.hpwl_total - h0) / max(h0, 1e-9)
            cost = compute_cost(hg, ag, m.violations_relative, 1.0, m.is_feasible)
            if cost < best_cost - 1e-9:
                best_cost, best_pos = cost, pos
        return best_pos

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

    def _best_of_k_solve(self, K, block_count, area_targets, b2b, p2b, pins,
                         constraints, target_positions):
        """K rollouts (candidate 0 greedy, rest sampled) -> legalize each via the
        full solve() pipeline -> select the lowest real cost with the candidate-
        relative gate. Never worse than the greedy default (it is candidate 0)."""
        cands = []
        for k in range(K):
            centers = self._rl_centers(
                area_targets, b2b, p2b, pins, constraints,
                target_positions, block_count, greedy=(k == 0))
            if centers is None:                      # no model -> plain greedy path
                return self.solve(block_count, area_targets, b2b, p2b, pins,
                                  constraints, target_positions)
            cands.append(self.solve(
                block_count, area_targets, b2b, p2b, pins, constraints,
                target_positions, centers_override=centers))
        if len(cands) == 1:
            return cands[0]
        return self._gate_real_cost(cands, constraints, b2b, p2b, pins,
                                    area_targets, target_positions)

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None,
        centers_override: Optional[Tuple] = None,
    ) -> List[Tuple[float, float, float, float]]:

        # Best-of-K inference (RL_NSAMPLE>1): the policy's greedy argmax is not its
        # best output (poc_rl_leverage.py: K sampled rollouts beat greedy by ~0.03
        # weighted). Run K rollouts (1 greedy + K-1 sampled), legalize each through
        # the full pipeline, and keep the lowest real cost — selection uses the same
        # candidate-relative gate as reshape variants (no GT, inference-valid). Costs
        # K× runtime (runtime^0.3 is scored officially — watch the trade-off).
        import os
        K = int(os.environ.get("RL_NSAMPLE", "1"))
        if (K > 1 and centers_override is None and self.center_source == "model"
                and self._model is not None):
            return self._best_of_k_solve(
                K, block_count, area_targets, b2b_connectivity,
                p2b_connectivity, pins_pos, constraints, target_positions)

        blocks, mib_groups, cluster_groups = parse_and_init(
            block_count, area_targets, constraints, target_positions
        )
        super_blocks = prepack_clusters(blocks, cluster_groups)

        cx = cy = None
        aspect_choice: Dict[int, float] = {}
        if centers_override is not None:
            # External center/shape source (e.g. gnn_joint_placer) feeds the SAME
            # legalize tail that the grid+greedy RL rollout normally feeds.
            cx, cy, aspect_choice = centers_override
        elif self.center_source == "model":
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

        def _legalize(cluster_flag: bool, mib_flag: bool, sb=None, cg=None,
                      cohesion: bool = False):
            sb = super_blocks if sb is None else sb     # override → e.g. free clusters
            cg = cluster_groups if cg is None else cg
            if self.shape_fit:                      # Phase 15/B1: contour-aware shaping
                from skyline_shape import skyline_legalize_shaped
                p, _ = skyline_legalize_shaped(
                    blocks, sb, cg, cx, cy, area_targets,
                    b2b=b2b_connectivity, p2b=p2b_connectivity, pins=pins_pos,
                    mib_shape=mib_flag, cluster_shape=cluster_flag, cohesion=cohesion,
                )
            else:
                p, _ = skyline_legalize(
                    blocks, sb, cg, cx, cy, area_targets,
                    b2b=b2b_connectivity, p2b=p2b_connectivity, pins=pins_pos,
                )
            if self.compact_pass:                   # squeeze residual whitespace
                from legalizer.topology import compact
                p = compact(p, blocks, sb, cg)
            return p

        def _finish(p, sb=None, cg=None):
            sb = super_blocks if sb is None else sb
            cg = cluster_groups if cg is None else cg
            p = slide_boundary(p, blocks, sb, cg)
            p = enforce_hard(p, blocks, area_targets)
            return _detailed_place(p, blocks, b2b_connectivity,
                                   p2b_connectivity, pins_pos)

        # Experiment: topology-preserving legalize (TOPO_LEGALIZE=1). Instead of the
        # gravity-based skyline re-pack (which discards the rollout's relative order),
        # derive a cycle-free HCG/VCG *directly from the rollout centers* and pack via
        # longest-path — so block i stays left-of/below-of j exactly as the rollout
        # arranged them. Overlap-free by construction; no shaping/gate.
        import os as _os
        if _os.environ.get("TOPO_LEGALIZE", "0") == "1":
            from legalizer.topology import build_topology, longest_path_pack
            hcg, vcg = build_topology(np.asarray(cx, dtype=np.float64),
                                      np.asarray(cy, dtype=np.float64),
                                      blocks, super_blocks, cluster_groups)
            p = longest_path_pack(hcg, vcg, blocks, super_blocks, cluster_groups)
            return _finish(p)

        # The reshape variants only differ from the default when the problem has a
        # cluster that is actually reshapeable (plain shelf-packed: every member is
        # interior and non-preplaced). Skip the (2×-cost) gate otherwise so most
        # cases pay nothing extra.
        has_elig_cluster = any(
            members and all(blocks[m].boundary_code == 0 and not blocks[m].is_preplaced
                            for m in members)
            for members in cluster_groups.values())

        import os
        b2_in_gate = self.b2_pass or os.environ.get("RCG_B2", "0") == "1"
        free_cluster = self.free_cluster or os.environ.get("FREE_CLUSTER", "0") == "1"
        has_cluster = bool(cluster_groups)

        if self.real_cost_gate and self.shape_fit and (
                has_elig_cluster or b2_in_gate or (free_cluster and has_cluster)):
            # Build finished candidates from the reshape variants and pick the one
            # with the lowest EXACT-form contest cost, scored relative to a shared
            # reference (candidate 0 = current default: no cluster/MIB reshape). This
            # replaces the bbox·HPWL proxy gate, which omits the e^{2V} violation term.
            if has_elig_cluster:
                _vsel = os.environ.get("RCG_VARIANTS", "cluster")
                variants = {
                    "cluster": [(False, False), (True, False)],
                    "all": [(False, False), (True, False),
                            (False, True), (True, True)],
                    "clustermib": [(False, False), (True, True)],
                }[_vsel]
            else:
                variants = [(False, False)]
            cands = [_finish(_legalize(cf, mf)) for cf, mf in variants]
            if free_cluster and has_cluster:
                # Deformable-cluster candidate: drop the rigid super-block and place
                # each cluster member as an individual reshapeable block, so it can
                # conform to the contour / wrap around obstacles instead of being a
                # bottom-left-packed rectangle. Connectivity is NOT guaranteed, but
                # the real-cost gate scores any resulting grouping violation via e^{2V},
                # so this is kept only when the area/HPWL win outweighs it.
                saved_cg = {i: b.cluster_group for i, b in enumerate(blocks)
                            if b.cluster_group}
                try:
                    # Pack with cluster_group INTACT (cohesion reads it to grow
                    # connected blobs), then zero it so slide_boundary/_finish treat
                    # the now-deformed members individually.
                    fc_pos = _legalize(False, False, sb={}, cg={}, cohesion=True)
                    for i in saved_cg:
                        blocks[i].cluster_group = 0
                    fc = _finish(fc_pos, sb={}, cg={})
                    if self._overlap_free(fc):
                        cands.append(fc)
                except Exception as e:  # pragma: no cover
                    if self.verbose:
                        print(f"[RLSkyline] free-cluster candidate failed ({e})")
                finally:
                    for i, g in saved_cg.items():
                        blocks[i].cluster_group = g
            if b2_in_gate:                          # B2: critical-path slack shaping
                from topo_shape import legalize_b2
                for base in list(cands):            # try B2 on each base candidate
                    try:
                        b2pos = _finish(legalize_b2(
                            base, blocks, super_blocks, cluster_groups, area_targets,
                            b2b=b2b_connectivity, p2b=p2b_connectivity, pins=pins_pos))
                        if self._overlap_free(b2pos):
                            cands.append(b2pos)
                    except Exception as e:  # pragma: no cover
                        if self.verbose:
                            print(f"[RLSkyline] B2 candidate failed ({e})")
            pos = self._gate_real_cost(
                cands, constraints, b2b_connectivity, p2b_connectivity,
                pins_pos, area_targets, target_positions)
        else:
            pos = _finish(_legalize(self.cluster_shape, self.mib_shape))

        if self.boundary_pass:                       # Item A: TOP/RIGHT boundary reshape
            try:
                from boundary_shape import boundary_reshape
                pos = boundary_reshape(pos, blocks, constraints, area_targets)
            except Exception as e:  # pragma: no cover
                if self.verbose:
                    print(f"[RLSkyline] boundary reshape failed ({e})")

        if self.b2_pass and not b2_in_gate:          # legacy proxy-gated B2 path
            try:                                     # (skipped when the real-cost gate handles B2)
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
