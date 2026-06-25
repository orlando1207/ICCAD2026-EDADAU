"""
PoC trainer: GNN+attention joint placer trained DIRECTLY on the differentiable
contest cost (no behaviour cloning, no RL, no grid).

Question this answers: can "continuous + joint + cost-trained" drive the soft
quality terms (HPWL_gap, Area_gap, overlap) below grid+greedy, using only
gradient descent through the contest cost?

It splits the LiteTest cases into train/val, trains on mean diff_cost over the
train split, and reports mean diff_cost on BOTH splits before vs after. A held-
out val split gives an honest (small) generalization signal since the model
conditions only on the netlist (GNN), never on case identity.

NOTE this is a CEILING / FEASIBILITY probe, exactly like poc_slack_shaping.py:
hard constraints (boundary/MIB/cluster/preplaced) and the skyline legalize tail
are out of scope. The numbers here are the differentiable surrogate cost, not a
final feasible contest score. The bar to clear is the soft-cost the grid+greedy
pipeline reaches on the same surrogate (printed as `baseline (grid+greedy)`).

Run:
    cd FloorSet/iccad2026contest
    python3 CNN_RL/gnn_joint_placer/train_poc.py --n 120 --steps 400
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_CNN_RL = _HERE.parent
_CONTEST = _CNN_RL.parent
for p in (str(_HERE), str(_CNN_RL), str(_CONTEST)):
    if p not in sys.path:
        sys.path.insert(0, p)

from iccad2026_evaluate import FloorplanDatasetLiteTest  # noqa: E402
from model import JointPlacer  # noqa: E402
from diff_cost import diff_cost  # noqa: E402


def load_cases(ds, n):
    cases = []
    for idx in range(min(n, len(ds))):
        s = ds[idx]
        a, b2b, p2b, pins, c = s["input"]
        bc = int((a != -1).sum())
        metrics = s["label"][1]
        cases.append(dict(idx=idx, a=a, b2b=b2b, p2b=p2b, pins=pins, c=c,
                          bc=bc, metrics=metrics))
    return cases


def mean_cost(model, cases, scale_margin=1.4):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for cs in cases:
            scale = JointPlacer.canvas_scale(cs["a"], cs["bc"], scale_margin)
            raw = model(cs["a"], cs["c"], cs["b2b"], cs["bc"])
            pos = model.positions(raw, cs["a"], cs["bc"], scale)
            tot += float(diff_cost(pos, cs["b2b"], cs["p2b"], cs["pins"],
                                   cs["a"][:cs["bc"]], cs["metrics"]))
    return tot / max(len(cases), 1)


def mean_parts(model, cases, scale_margin=1.4):
    """Mean decomposition (hpwl_gap, area_gap, overlap_viol) over cases."""
    model.eval()
    acc = {"hpwl_gap": 0.0, "area_gap": 0.0, "overlap_viol": 0.0, "area_viol": 0.0}
    with torch.no_grad():
        for cs in cases:
            scale = JointPlacer.canvas_scale(cs["a"], cs["bc"], scale_margin)
            raw = model(cs["a"], cs["c"], cs["b2b"], cs["bc"])
            pos = model.positions(raw, cs["a"], cs["bc"], scale)
            _, parts = diff_cost(pos, cs["b2b"], cs["p2b"], cs["pins"],
                                 cs["a"][:cs["bc"]], cs["metrics"], return_parts=True)
            for k in acc:
                acc[k] += parts[k]
    return {k: v / max(len(cases), 1) for k, v in acc.items()}


def grid_greedy_baseline(cases):
    """Mean diff_cost of the existing RLSkylineOptimizer (the bar to beat)."""
    from rl_skyline_optimizer import RLSkylineOptimizer
    opt = RLSkylineOptimizer(verbose=False)
    tot = 0.0
    for cs in cases:
        otp = torch.full((cs["bc"], 4), -1.0)
        pos = opt.solve(cs["bc"], cs["a"], cs["b2b"], cs["p2b"], cs["pins"],
                        cs["c"], otp)
        P = torch.tensor(pos, dtype=torch.float32)
        tot += float(diff_cost(P, cs["b2b"], cs["p2b"], cs["pins"],
                               cs["a"][:cs["bc"]], cs["metrics"]))
    return tot / max(len(cases), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="cases to use (train+val)")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8, help="cases per grad step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--data-path", type=str, default="../")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the (slow) grid+greedy baseline measurement")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ds = FloorplanDatasetLiteTest(args.data_path)
    cases = load_cases(ds, args.n)
    n_val = max(1, int(len(cases) * args.val_frac))
    val, train = cases[:n_val], cases[n_val:]
    print(f"cases: {len(train)} train / {len(val)} val  "
          f"(blocks {min(c['bc'] for c in cases)}..{max(c['bc'] for c in cases)})")

    model = JointPlacer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e3:.1f}k")

    print(f"\n--- BEFORE training ---")
    print(f"untrained  train={mean_cost(model, train):.4f}  val={mean_cost(model, val):.4f}")
    if not args.no_baseline:
        t0 = time.time()
        gg_tr = grid_greedy_baseline(train); gg_va = grid_greedy_baseline(val)
        print(f"baseline (grid+greedy)  train={gg_tr:.4f}  val={gg_va:.4f}  "
              f"({time.time()-t0:.1f}s)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"\n--- training {args.steps} steps (batch {args.batch}) ---")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        perm = torch.randperm(len(train))[:args.batch]
        opt.zero_grad()
        loss = 0.0
        for k in perm.tolist():
            cs = train[k]
            scale = JointPlacer.canvas_scale(cs["a"], cs["bc"])
            raw = model(cs["a"], cs["c"], cs["b2b"], cs["bc"])
            pos = model.positions(raw, cs["a"], cs["bc"], scale)
            # Train in LOG space: log(cost) = log(quality) + β·V_soft. Same
            # minimizer as cost, but the exp(β·overlap) term becomes linear, so
            # a block pile-up gives a bounded gradient instead of exploding.
            loss = loss + torch.log(diff_cost(pos, cs["b2b"], cs["p2b"], cs["pins"],
                                              cs["a"][:cs["bc"]], cs["metrics"]))
        loss = loss / len(perm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % max(1, args.steps // 20) == 0 or step == 1:
            print(f"  step {step:4d}  train_batch_logcost={float(loss):.4f}  "
                  f"({time.time()-t0:.0f}s)")

    print(f"\n--- AFTER training ---")
    print(f"trained    train={mean_cost(model, train):.4f}  val={mean_cost(model, val):.4f}")
    pv = mean_parts(model, val)
    print(f"val decomposition: hpwl_gap={pv['hpwl_gap']:+.3f}  "
          f"area_gap={pv['area_gap']:+.3f}  overlap_viol={pv['overlap_viol']:.4f}  "
          f"area_viol={pv['area_viol']:.4f}")

    out = _HERE / "joint_placer_poc.pt"
    torch.save({"model": model.state_dict()}, out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
