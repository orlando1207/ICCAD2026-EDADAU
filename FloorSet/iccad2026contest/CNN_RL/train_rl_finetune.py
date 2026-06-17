"""
Phase 17 — RL fine-tune on the *current* pipeline's real contest cost.

Supersedes archive/train_ppo.py (Phase 12), which used: reward = legalized cost
through the OLD plain skyline_legalize, scored by compute_training_loss_differentiable
(not the real metric), warm-started from phase11, aspect head NOT in the action.
It failed. The pre-flight diagnostics (ALGORITHM.md §"RL pre-flight diagnostics")
established the genuinely-new bet:

  1. Reward = runtime-free real contest cost on the FULL current pipeline
     (RLSkylineOptimizer.solve through shaped + cohesion + real-cost gate). We just
     call solve(centers_override=...) so the reward IS exactly what inference scores.
     cost = evaluate_solution(..., runtime=1.0, median_runtime=1.0).cost
          = (1 + 0.5*(hpwl_gap+area_gap)) * e^{2V}   — no RuntimeFactor.
  2. Aspect head IS in the action: free soft blocks sample (position, aspect) and
     both heads' logprobs enter the PPO ratio + are updated.
  3. Warm-start phase13_aspect (the accumulated BC best; from-scratch is a dead end).
  4. Retained from Phase 12 (these were correct, not the failure cause): KL-anchor to
     the BC policy, per-case advantage baseline, frozen GNN, held-out greedy eval.

Target: best-of-K showed ~1.56 is reachable; RL aims to get greedy (1×, 0.83s) there.

Run (from FloorSet/iccad2026contest/):
    python3 CNN_RL/train_rl_finetune.py \
        --warmstart CNN_RL/checkpoints/phase13_aspect.pt \
        --iters 40 --batch-samples 8 --rollouts 8 --kl-coef 0.1 \
        --ckpt-name phase17_rl.pt
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_FLOORSET_ROOT = Path(__file__).resolve().parent.parent.parent
_CONTEST_DIR = Path(__file__).resolve().parent.parent
for _p in (_FLOORSET_ROOT, _CONTEST_DIR, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from iccad2026_evaluate import evaluate_solution  # noqa: E402
from rl_skyline_optimizer import RLSkylineOptimizer  # noqa: E402
from gnn_encoder import GNNEncoder  # noqa: E402
from policy_net import PolicyValueNet  # noqa: E402
from placement_env import PlacementEnv  # noqa: E402
from canvas_raster import rasterize_env, CH_FEASIBILITY, N_CHANNELS  # noqa: E402
from ar_utils import ASPECT_BUCKETS  # noqa: E402
from lite_dataset import FloorplanDatasetLite  # noqa: E402
from train_phase8 import boxes_from_fp_sol  # noqa: E402
from hard_constraints import make_target_positions_from_gt  # noqa: E402
from pretrain_bc import CKPT_DIR  # noqa: E402


def _opt_target_pos(target_pos, constraints, bc):
    """Eval-harness target_positions: -1 default; preplaced -> (x,y,w,h);
    fixed -> (w,h). This is exactly what inference (solve) consumes."""
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


def _gt_baseline(gt_boxes, cons, b2b, p2b, pins, area, opt_tp, bc):
    """GT bbox/HPWL as the contest baseline (constant per case, so it ranks the K
    rollouts correctly). Reuse evaluate_solution's own HPWL/bbox math on GT boxes."""
    gt_pos = [tuple(float(v) for v in gt_boxes[i]) for i in range(bc)]
    m = evaluate_solution({"positions": gt_pos, "runtime": 1.0}, {}, cons,
                          b2b, p2b, pins, area, opt_tp, median_runtime=1.0)
    return {"hpwl_baseline": m.hpwl_total, "area_baseline": m.bbox_area}


def _rollout(policy, gnn, env, node_emb, area, b2b, p2b, pins, cons, opt_tp, bc,
             aspect_pass, greedy=False):
    """One rollout. Returns (steps, centers). Each step is
    (canvas_cpu[C,G,G], block:int, pos_action:int, aspect_idx:int|-1, oldlp:float).
    aspect_idx=-1 means the block is not free (no aspect action)."""
    env.reset(area, b2b, p2b, pins, cons, torch.ones(8), target_positions=opt_tp)
    steps = []
    aspect_choice = {}
    done = len(env.order) == 0
    while not done:
        st = env._build_state()
        cur = st["current_block"]
        canvas_cpu = rasterize_env(env, device="cpu")
        canvas = canvas_cpu.to(node_emb.device)        # policy runs on node_emb's device
        mask = canvas[CH_FEASIBILITY]
        is_free = (aspect_pass and cur not in env.fixed_dims
                   and cur not in env.mib_dims and int(cons[cur, 3]) == 0)
        with torch.no_grad():
            if is_free:
                a, aidx, lp, _v = policy.act_aspect(canvas, node_emb[cur], mask, greedy=greedy)
                aspect = ASPECT_BUCKETS[aidx]
                aspect_choice[cur] = aspect
            else:
                a, lp, _v = policy.act(canvas, node_emb[cur], mask, greedy=greedy)
                aidx = -1
        steps.append((canvas_cpu.detach(), cur, a, aidx, float(lp)))
        env.step(a, aspect=(aspect_choice.get(cur, 1.0)))
        done = env.ptr >= len(env.order)
    cx = np.zeros(bc, dtype=np.float64)
    cy = np.zeros(bc, dtype=np.float64)
    for i in range(bc):
        x, y, w, h = (float(t) for t in env.positions[i])
        cx[i] = x + w / 2.0
        cy[i] = y + h / 2.0
    return steps, (cx, cy, aspect_choice)


def _real_cost(solver, bc, area, b2b, p2b, pins, cons, opt_tp, centers, baseline,
               target_pos):
    """Runtime-free real contest cost of the rollout's centers through the full
    current legalize pipeline — exactly what inference scores."""
    pos = solver.solve(bc, area, b2b, p2b, pins, cons, opt_tp, centers_override=centers)
    m = evaluate_solution({"positions": pos, "runtime": 1.0}, baseline, cons,
                          b2b, p2b, pins, area, target_pos, median_runtime=1.0)
    return float(m.cost)


def _prep_case(ds, idx):
    s = ds[idx % len(ds)]
    area, b2b, p2b, pins, cons = s["input"]
    _tree, fp_sol, _metrics = s["label"]
    bc = int((area != -1).sum())
    boxes = boxes_from_fp_sol(fp_sol, bc)
    tp = make_target_positions_from_gt(cons, boxes, bc)        # for GT baseline build
    opt_tp = _opt_target_pos(tp, cons, bc)
    baseline = _gt_baseline(boxes, cons, b2b, p2b, pins, area, opt_tp, bc)
    return area, b2b, p2b, pins, cons, bc, opt_tp, baseline


def _prep_val_case(ev, idx):
    """Validation case via the SCORING harness's own baseline/target_pos — so the
    eval number is directly comparable to the leaderboard Total Score (1.6039)."""
    s = ev.dataset[idx]
    area, b2b, p2b, pins, cons = s["input"]
    bc = int((area != -1).sum())
    baseline, target_pos = ev._extract_baseline(idx, s["label"], b2b, p2b, pins, bc)
    opt_tp = _opt_target_pos(target_pos, cons, bc)
    return area, b2b, p2b, pins, cons, bc, opt_tp, baseline, target_pos


def _greedy_eval(policy, gnn, solver, env, val_cases, device, aspect_pass):
    """Fixed validation-set greedy real-cost (mean) — low-variance progress signal,
    comparable to inference Total Score."""
    costs = []
    for (area, b2b, p2b, pins, cons, bc, opt_tp, baseline, target_pos) in val_cases:
        with torch.no_grad():
            node_emb, _ = gnn.encode_problem(area, cons, b2b, bc, device=device)
        _steps, centers = _rollout(policy, gnn, env, node_emb, area, b2b, p2b, pins,
                                   cons, opt_tp, bc, aspect_pass, greedy=True)
        costs.append(_real_cost(solver, bc, area, b2b, p2b, pins, cons, opt_tp,
                                centers, baseline, target_pos))
    return sum(costs) / len(costs)


def train(warmstart, iters=40, batch_samples=8, rollouts=8, lr=3e-5, clip=0.2,
          ppo_epochs=2, c_entropy=0.01, kl_coef=0.1, eval_every=5, eval_cases=10,
          seed=0, device=None, ckpt_name="phase17_rl.pt", root=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    root = root or (str(_FLOORSET_ROOT) + "/")

    ckpt = torch.load(warmstart, map_location="cpu")
    grid, gnn_out, hidden = ckpt["grid"], ckpt["gnn_out"], ckpt["hidden"]
    in_ch = ckpt.get("in_channels", N_CHANNELS)
    gnn = GNNEncoder(out_dim=gnn_out, hidden=hidden).to(device)
    policy = PolicyValueNet(in_channels=in_ch, node_dim=gnn_out, hidden=hidden).to(device)
    gnn.load_state_dict(ckpt["gnn"]); policy.load_state_dict(ckpt["policy"])
    gnn.eval()
    for p in gnn.parameters():
        p.requires_grad_(False)

    policy_ref = copy.deepcopy(policy).to(device)      # KL-anchor to BC policy
    policy_ref.eval()
    for p in policy_ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    env = PlacementEnv(grid=grid)
    env.skip_occupancy = True
    env.skip_terminal_cost = True
    # The reward solver: full inference pipeline, aspect head ON (centers carry aspect).
    os.environ.setdefault("ASPECT_PASS", "1")
    solver = RLSkylineOptimizer(verbose=False)
    if solver._model is None:           # solver loads its own ckpt; force greedy path off
        solver._model = (gnn, policy, grid)
    aspect_pass = solver.aspect_pass

    ds = FloorplanDatasetLite(root)                    # 1M training rollouts
    # Held-out eval on the VALIDATION set (the scored 100), spread across sizes so the
    # number tracks the real Total Score. Train never sees these (different dataset).
    from iccad2026_evaluate import ContestEvaluator
    ev = ContestEvaluator(verbose=False)
    ev._load_dataset()
    nval = len(ev.dataset)
    step = max(1, nval // eval_cases)
    val_idx = list(range(0, nval, step))[:eval_cases]
    val_cases = [_prep_val_case(ev, i) for i in val_idx]
    print(f"[train_rl] device={device} warmstart={Path(warmstart).name} grid={grid} "
          f"iters={iters} batch={batch_samples} rollouts={rollouts} lr={lr} "
          f"clip={clip} kl_coef={kl_coef} aspect_pass={aspect_pass} "
          f"eval={len(val_cases)} val cases {val_idx[:3]}..{val_idx[-1]}")

    t0 = time.time()
    base = _greedy_eval(policy, gnn, solver, env, val_cases, device, aspect_pass)
    print(f"  [eval] iter  -1  val greedy real_cost={base:.4f}  "
          f"({len(val_cases)} cases, {time.time()-t0:.0f}s)")

    idx = 0
    for it in range(iters):
        all_steps, all_adv = [], []
        iter_costs = []
        for _ in range(batch_samples):
            area, b2b, p2b, pins, cons, bc, opt_tp, baseline = _prep_case(ds, idx)
            idx += 1
            with torch.no_grad():
                node_emb, _ = gnn.encode_problem(area, cons, b2b, bc, device=device)
            eps = [_rollout(policy, gnn, env, node_emb, area, b2b, p2b, pins, cons,
                            opt_tp, bc, aspect_pass) for _ in range(rollouts)]
            costs = torch.tensor([
                _real_cost(solver, bc, area, b2b, p2b, pins, cons, opt_tp, c,
                           baseline, opt_tp) for _s, c in eps])
            iter_costs.append(float(costs.mean()))
            rewards = -costs
            adv = (rewards - rewards.mean()) / rewards.std().clamp_min(0.1)
            for (steps, _c), a in zip(eps, adv.tolist()):
                for (canvas, blk, action, aidx, oldlp) in steps:
                    all_steps.append((canvas, node_emb[blk].detach().cpu(),
                                      action, aidx, oldlp))
                    all_adv.append(a)

        if not all_steps:
            continue
        N = len(all_steps)
        adv_b = torch.tensor(all_adv)
        mb = 256
        kl_val = 0.0
        loss = torch.zeros(())
        for _ in range(ppo_epochs):
            perm = torch.randperm(N)
            for st in range(0, N, mb):
                ii = perm[st:st + mb]
                cb = torch.stack([all_steps[k][0] for k in ii]).to(device)
                eb = torch.stack([all_steps[k][1] for k in ii]).to(device)
                ab = adv_b[ii].to(device)
                actb = torch.tensor([all_steps[k][2] for k in ii], device=device)
                aidxb = torch.tensor([all_steps[k][3] for k in ii], device=device)
                obb = torch.tensor([all_steps[k][4] for k in ii], device=device)
                free = aidxb >= 0
                opt.zero_grad()
                logits, _v, aspect_logits = policy.forward_batch(cb, eb)
                logp_all = F.log_softmax(logits, dim=1)
                rows = torch.arange(len(ii), device=device)
                logp = logp_all[rows, actb]
                # add aspect logprob for free blocks (both heads were sampled)
                alp_all = F.log_softmax(aspect_logits, dim=1)
                alp = alp_all[rows, aidxb.clamp_min(0)]
                logp = logp + torch.where(free, alp, torch.zeros_like(alp))
                ratio = torch.exp(logp - obb)
                surr1 = ratio * ab
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * ab
                ent = -(logp_all.exp() * logp_all).sum(1).mean()
                with torch.no_grad():
                    ref_logits, _, _ = policy_ref.forward_batch(cb, eb)
                    ref_logp_all = F.log_softmax(ref_logits, dim=1)
                kl = (logp_all.exp() * (logp_all - ref_logp_all)).sum(1).mean()
                loss = -torch.min(surr1, surr2).mean() - c_entropy * ent + kl_coef * kl
                loss.backward()
                opt.step()
                kl_val = float(kl)

        mc = sum(iter_costs) / len(iter_costs)
        print(f"  iter {it:3d}  mean_real_cost={mc:.4f}  loss={float(loss):.4f}  "
              f"kl={kl_val:.4f}  ({(it+1)*batch_samples*rollouts} eps, "
              f"{time.time()-t0:.0f}s)")

        if (it + 1) % eval_every == 0 or it == iters - 1:
            evc = _greedy_eval(policy, gnn, solver, env, val_cases, device, aspect_pass)
            flag = "  <-- improved" if evc < base else ""
            print(f"  [eval] iter {it:3d}  val greedy real_cost={evc:.4f}  "
                  f"(baseline={base:.4f}, {time.time()-t0:.0f}s){flag}")
            _save(gnn, policy, grid, gnn_out, hidden, ckpt_name)

    return _save(gnn, policy, grid, gnn_out, hidden, ckpt_name)


def _save(gnn, policy, grid, gnn_out, hidden, ckpt_name):
    CKPT_DIR.mkdir(exist_ok=True)
    path = CKPT_DIR / ckpt_name
    torch.save({"gnn": gnn.state_dict(), "policy": policy.state_dict(),
                "grid": grid, "gnn_out": gnn_out, "hidden": hidden,
                "in_channels": N_CHANNELS}, path)
    print(f"  saved {path}")
    return str(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart", type=str,
                    default="CNN_RL/checkpoints/phase13_aspect.pt")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--batch-samples", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--c-entropy", type=float, default=0.01)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-cases", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--ckpt-name", type=str, default="phase17_rl.pt")
    args = ap.parse_args()
    train(warmstart=args.warmstart, iters=args.iters, batch_samples=args.batch_samples,
          rollouts=args.rollouts, lr=args.lr, clip=args.clip, ppo_epochs=args.ppo_epochs,
          c_entropy=args.c_entropy, kl_coef=args.kl_coef, eval_every=args.eval_every,
          eval_cases=args.eval_cases, seed=args.seed, device=args.device,
          ckpt_name=args.ckpt_name)
