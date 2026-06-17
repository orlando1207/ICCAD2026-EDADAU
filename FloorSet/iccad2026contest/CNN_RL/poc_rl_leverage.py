"""
Step 0 — RL leverage diagnostic (does RL-on-centers have anything to learn?).

Before building an RL fine-tune loop (see ALGORITHM.md §"Phase 16"), measure the
ONE thing that decides whether it can possibly work: given the *current* strong
legalizer (shaped + cohesion clusters + OBS_XCAND + real-cost gate, Total 1.6039),
how much does the final contest cost actually depend on the rollout's centers?

Method (reuses the production pipeline, no new logic):
  - greedy rollout  -> centers -> rl_skyline_optimizer.solve(centers_override=...)
    -> the SAME legalize chain inference uses -> real contest cost.
  - K stochastic rollouts of the same case -> K costs.
  - report greedy cost, best-of-K, mean, std per case + aggregate.

Reward / cost = the contest cost form WITHOUT the runtime term: with runtime=1.0
and median_runtime=1.0, evaluate_solution's runtime_adjustment = max(0.7, 1.0)=1.0,
so metrics.cost = (1 + 0.5*(hpwl_gap + area_gap)) * exp(2*V) — exactly the user's
requested (AREA+HPWL)*(VIOLATION) reward, no RuntimeFactor.

Interpretation:
  - best-of-K ~= greedy (tiny spread)  -> cost is insensitive to centers; RL on
    centers has no gradient to grab (Phase 12's likely true failure mode). STOP.
  - best-of-K clearly < greedy          -> real headroom; the gap is the ceiling an
    RL fine-tune could chase. PROCEED to the full loop.

Run (from FloorSet/iccad2026contest/):
    RL_CKPT=CNN_RL/checkpoints/phase13_aspect.pt ASPECT_PASS=1 \
        python3 CNN_RL/poc_rl_leverage.py --n-cases 20 --rollouts 16
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

_FLOORSET_ROOT = Path(__file__).resolve().parent.parent.parent
_CONTEST_DIR = Path(__file__).resolve().parent.parent      # FloorSet/iccad2026contest
if str(_FLOORSET_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOORSET_ROOT))
sys.path.insert(0, str(_CONTEST_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from iccad2026_evaluate import ContestEvaluator, evaluate_solution  # noqa: E402
from rl_skyline_optimizer import RLSkylineOptimizer  # noqa: E402
from placement_env import PlacementEnv  # noqa: E402
from canvas_raster import rasterize_env, CH_FEASIBILITY  # noqa: E402
from ar_utils import ASPECT_BUCKETS  # noqa: E402


def rollout_centers(opt, area, b2b, p2b, pins, cons, tp, bc, greedy):
    """Mirror RLSkylineOptimizer._rl_centers but with a greedy flag (greedy=False
    samples from the policy/aspect distributions)."""
    gnn, policy, grid = opt._model
    env = PlacementEnv(grid=grid)
    env.skip_occupancy = True
    env.skip_terminal_cost = True
    env.reset(area, b2b, p2b, pins, cons, torch.ones(8), target_positions=tp)
    node_emb, _ = gnn.encode_problem(area, cons, b2b, bc)
    aspect_choice = {}
    done = len(env.order) == 0
    while not done:
        st = env._build_state()
        cur = st["current_block"]
        canvas = rasterize_env(env)
        mask = canvas[CH_FEASIBILITY]
        is_free = (opt.aspect_pass and cur not in env.fixed_dims
                   and cur not in env.mib_dims and int(cons[cur, 3]) == 0)
        if is_free:
            a, aidx, _lp, _v = policy.act_aspect(canvas, node_emb[cur], mask, greedy=greedy)
            aspect = ASPECT_BUCKETS[aidx]
            aspect_choice[cur] = aspect
        else:
            a, _lp, _v = policy.act(canvas, node_emb[cur], mask, greedy=greedy)
            aspect = 1.0
        env.step(a, aspect=aspect)
        done = env.ptr >= len(env.order)
    cx = np.zeros(bc, dtype=np.float64)
    cy = np.zeros(bc, dtype=np.float64)
    for i in range(bc):
        x, y, w, h = (float(t) for t in env.positions[i])
        cx[i] = x + w / 2.0
        cy[i] = y + h / 2.0
    return cx, cy, aspect_choice


def cost_of(opt, bc, area, b2b, p2b, pins, cons, opt_tp, centers, baseline, target_pos):
    """Full current pipeline on given centers -> runtime-free contest cost."""
    positions = opt.solve(bc, area, b2b, p2b, pins, cons, opt_tp,
                          centers_override=centers)
    m = evaluate_solution({"positions": positions, "runtime": 1.0}, baseline,
                          cons, b2b, p2b, pins, area, target_pos, median_runtime=1.0)
    return float(m.cost), bool(m.is_feasible)


def _opt_target_pos(target_pos, constraints, bc):
    """Replicate the eval harness's opt_target_pos construction."""
    opt_tp = torch.full((bc, 4), -1.0)
    if target_pos is not None and constraints is not None:
        nc = constraints.shape[1] if constraints.dim() > 1 else 0
        for i in range(bc):
            is_fixed = nc > 0 and constraints[i, 0] != 0
            is_preplaced = nc > 1 and constraints[i, 1] != 0
            if is_preplaced:
                tx, ty, tw, th = target_pos[i]
                opt_tp[i] = torch.tensor([tx, ty, tw, th])
            elif is_fixed:
                _, _, tw, th = target_pos[i]
                opt_tp[i, 2] = tw
                opt_tp[i, 3] = th
    return opt_tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=20,
                    help="number of validation cases to probe")
    ap.add_argument("--start-idx", type=int, default=0,
                    help="first validation case index (use to probe large cases)")
    ap.add_argument("--rollouts", type=int, default=16,
                    help="K stochastic rollouts per case")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ev = ContestEvaluator(verbose=False)
    ev._load_dataset()
    opt = RLSkylineOptimizer(verbose=False)
    if opt._model is None:
        print("ERROR: no model loaded — set RL_CKPT to a valid checkpoint.")
        sys.exit(1)

    start = args.start_idx
    n = min(args.n_cases, len(ev.dataset) - start)
    print(f"[leverage] cases={start}..{start+n-1} rollouts/case={args.rollouts} "
          f"(reward = runtime-free contest cost, lower better)\n")
    hdr = f"{'case':>4} {'bc':>4} {'greedy':>8} {'best-K':>8} {'mean':>8} {'std':>7} {'g-best':>7} {'feas%':>6}"
    print(hdr)
    print("-" * len(hdr))

    g_list, b_list, gap_list = [], [], []
    t0 = time.time()
    for idx in range(start, start + n):
        sample = ev.dataset[idx]
        inputs, labels = sample["input"], sample["label"]
        area, b2b, p2b, pins, cons = inputs
        bc = int((area != -1).sum().item())
        baseline, target_pos = ev._extract_baseline(idx, labels, b2b, p2b, pins, bc)
        opt_tp = _opt_target_pos(target_pos, cons, bc)

        gc, gf = cost_of(opt, bc, area, b2b, p2b, pins, cons, opt_tp,
                         rollout_centers(opt, area, b2b, p2b, pins, cons, opt_tp, bc, True),
                         baseline, target_pos)
        costs, feas = [], []
        for _ in range(args.rollouts):
            c, f = cost_of(opt, bc, area, b2b, p2b, pins, cons, opt_tp,
                           rollout_centers(opt, area, b2b, p2b, pins, cons, opt_tp, bc, False),
                           baseline, target_pos)
            costs.append(c)
            feas.append(f)
        best = min(costs)
        mean = statistics.mean(costs)
        std = statistics.pstdev(costs)
        gap = gc - best                       # how much best-of-K beats greedy
        feas_pct = 100.0 * sum(feas) / len(feas)
        g_list.append(gc)
        b_list.append(best)
        gap_list.append(gap)
        print(f"{idx:>4} {bc:>4} {gc:>8.4f} {best:>8.4f} {mean:>8.4f} "
              f"{std:>7.4f} {gap:>7.4f} {feas_pct:>6.0f}")

    print("-" * len(hdr))
    print(f"\n[aggregate over {n} cases]")
    print(f"  mean greedy cost     : {statistics.mean(g_list):.4f}")
    print(f"  mean best-of-K cost  : {statistics.mean(b_list):.4f}")
    print(f"  mean (greedy-bestK)  : {statistics.mean(gap_list):.4f}  "
          f"<- RL headroom ceiling")
    print(f"  cases best-K beats greedy: {sum(1 for g in gap_list if g > 1e-4)}/{n}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    print("\nVerdict: large positive 'greedy-bestK' on many cases => RL has headroom.")
    print("         ~0 everywhere => cost is center-insensitive; RL-on-centers is a dead end.")


if __name__ == "__main__":
    main()
