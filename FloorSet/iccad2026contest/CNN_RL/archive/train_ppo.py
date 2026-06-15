"""
Phase 12 — PPO fine-tuning on the *legalized* contest cost, with a KL-anchor.

BC (train_fast.py, soft labels + pin feature) plateaus because it optimises a
proxy (match the GT cell), not the real objective. PPO fine-tunes the
BC-warm-started policy to DIRECTLY minimise the cost we actually score on: the
skyline-legalized layout's contest cost.

Design choices (the first PPO attempt, warm-started from phase10_soft,
diverged: mean_legalized_cost rose ~2.24 -> ~3.00 over 25 iters):

  1. **Reward = -(legalized cost)**, NOT the env's raw-placement cost. After the
     rollout we run the SAME analytic chain as rl_skyline_optimizer
     (skyline_legalize -> slide_boundary -> enforce_hard -> _detailed_place) and
     score that — so the policy is rewarded for centers that legalize well, which
     is exactly what inference does.
  2. **Per-case baseline** (mean reward over the K rollouts of the same case),
     not the value head: the BC checkpoint never trained a value head, and the
     per-case mean is a low-variance, bias-free baseline for a terminal reward.
     advantage = (reward - case_mean) / case_std (std floored at 0.1 so a
     near-zero spread across rollouts doesn't get amplified into a huge,
     noise-driven advantage).
  3. **Frozen GNN**: node embeddings are already detached before being stored
     for the update step, so the GNN never received gradients in attempt 1 —
     freezing it here is a no-op functionally, just makes that explicit and
     keeps node embeddings stable across iterations.
  4. **KL-anchor to the BC policy**: PPO's clip bounds the step size of a
     *single* update, but not cumulative drift over many iterations — which is
     how attempt 1 walked from cost 2.24 to 3.00 without any single step looking
     extreme. A frozen copy of the BC policy (`policy_ref`) is kept around, and
     `kl_coef * KL(policy || policy_ref)` is added to the loss every update,
     continuously pulling the policy back toward the known-good BC behaviour.
  5. **Fixed held-out eval set**: a small set of dataset indices from the *end*
     of the dataset (disjoint from the sequentially-consumed training indices
     for a while) is greedily rolled out and legalized every `--eval-every`
     iters. This is a much lower-variance signal than the per-iteration
     stochastic-rollout mean_legalized_cost, and is what should be watched for
     real progress vs regression.

Run:
    cd FloorSet/iccad2026contest
    python3 CNN_RL/train_ppo.py --warmstart CNN_RL/checkpoints/phase11_pin_soft.pt \
        --iters 60 --batch-samples 8 --rollouts 8 --kl-coef 0.1
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_FLOORSET_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_FLOORSET_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOORSET_ROOT))
_CONTEST = Path(__file__).resolve().parent.parent
if str(_CONTEST) not in sys.path:
    sys.path.insert(0, str(_CONTEST))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from iccad2026contest.iccad2026_evaluate import compute_training_loss_differentiable  # noqa: E402
from lite_dataset import FloorplanDatasetLite  # noqa: E402
from placement_env import PlacementEnv  # noqa: E402
from canvas_raster import rasterize_env, N_CHANNELS, CH_FEASIBILITY  # noqa: E402
from gnn_encoder import GNNEncoder  # noqa: E402
from policy_net import PolicyValueNet  # noqa: E402
from pretrain_bc import CKPT_DIR  # noqa: E402
from hard_constraints import make_target_positions_from_gt  # noqa: E402
from train_phase8 import boxes_from_fp_sol  # noqa: E402

from analytic_legalizer.constraints import (  # noqa: E402
    parse_and_init, prepack_clusters, slide_boundary, enforce_hard)
from analytic_legalizer.skyline_legalizer import skyline_legalize, _detailed_place  # noqa: E402


def _legalized_cost(env, cx, cy, area, cons, tp, b2b, p2b, pins, metrics,
                    blocks, supers, clusters):
    """Run the analytic skyline chain on the rollout's centers and score it with
    the official differentiable cost (lower is better)."""
    bc = env.block_count
    pos, _ = skyline_legalize(blocks, supers, clusters, cx, cy, area,
                              b2b=b2b, p2b=p2b, pins=pins)
    pos = slide_boundary(pos, blocks, supers, clusters)
    pos = enforce_hard(pos, blocks, area)
    pos = _detailed_place(pos, blocks, b2b, p2b, pins)
    pt = torch.tensor(pos, dtype=torch.float32)
    with torch.no_grad():
        c = compute_training_loss_differentiable(pt, b2b, p2b, pins, area[:bc], metrics)
    return float(c)


def _rollout(policy, gnn, env, node_emb, sample_tensors, tp, device,
             blocks, supers, clusters, greedy=False):
    """One rollout. Returns (steps, legalized_cost). Each step is
    (canvas[C,G,G] on CPU, block:int, action:int, old_logp:float)."""
    area, b2b, p2b, pins, cons, metrics = sample_tensors
    env.reset(area, b2b, p2b, pins, cons, metrics, target_positions=tp)
    steps = []
    done = len(env.order) == 0
    while not done:
        st = env._build_state()
        cur = st["current_block"]
        canvas_cpu = rasterize_env(env, device="cpu")     # CPU raster (GPU per-step is launch-bound)
        canvas = canvas_cpu.to(device)
        mask = canvas[CH_FEASIBILITY]
        with torch.no_grad():
            probs, _v, _ = policy.forward(canvas, node_emb[cur], mask)
            action = int(torch.argmax(probs)) if greedy else int(torch.multinomial(probs, 1))
            logp = float(torch.log(probs[action].clamp_min(1e-12)))
        steps.append((canvas_cpu.detach(), cur, action, logp))   # store CPU; stacked->GPU at update
        env.step(action)
        done = env.ptr >= len(env.order)
    cx = torch.tensor([float(env.positions[i][0] + env.positions[i][2] / 2) for i in range(env.block_count)])
    cy = torch.tensor([float(env.positions[i][1] + env.positions[i][3] / 2) for i in range(env.block_count)])
    cost = _legalized_cost(env, cx.numpy(), cy.numpy(), area, cons, tp, b2b, p2b,
                           pins, metrics, blocks, supers, clusters)
    return steps, cost


def _greedy_eval(policy, gnn, env, ds, indices, device):
    """Greedy-rollout legalized cost averaged over a fixed set of dataset
    indices. Low-variance signal for tracking real progress across iters."""
    costs = []
    for idx in indices:
        s = ds[idx]
        area, b2b, p2b, pins, cons = s["input"]
        _tree, fp_sol, metrics = s["label"]
        bc = int((area != -1).sum())
        boxes = boxes_from_fp_sol(fp_sol, bc)
        tp = make_target_positions_from_gt(cons, boxes, bc)
        blocks, _mib, clusters = parse_and_init(bc, area, cons, tp)
        supers = prepack_clusters(blocks, clusters)
        with torch.no_grad():
            node_emb, _ = gnn.encode_problem(area, cons, b2b, bc, device=device)
        tensors = (area, b2b, p2b, pins, cons, metrics)
        _steps, cost = _rollout(policy, gnn, env, node_emb, tensors, tp, device,
                                blocks, supers, clusters, greedy=True)
        costs.append(cost)
    return sum(costs) / len(costs)


def train(warmstart, iters=40, batch_samples=8, rollouts=8, lr=3e-5, clip=0.2,
          ppo_epochs=2, c_entropy=0.01, kl_coef=0.1, eval_every=5, eval_cases=10,
          seed=0, device=None, ckpt_name="phase12_ppo_kl.pt", root=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    root = root or (str(_FLOORSET_ROOT) + "/")

    ckpt = torch.load(warmstart, map_location="cpu")
    grid, gnn_out, hidden = ckpt["grid"], ckpt["gnn_out"], ckpt["hidden"]
    in_ch = ckpt.get("in_channels", N_CHANNELS)
    gnn = GNNEncoder(out_dim=gnn_out, hidden=hidden).to(device)
    policy = PolicyValueNet(in_channels=in_ch, node_dim=gnn_out, hidden=hidden).to(device)
    gnn.load_state_dict(ckpt["gnn"]); policy.load_state_dict(ckpt["policy"])

    # GNN encodes static netlist topology; node embeddings are detached before
    # being used in the update step regardless, so freezing it changes nothing
    # functionally — it just keeps embeddings stable across iterations.
    gnn.eval()
    for p in gnn.parameters():
        p.requires_grad_(False)

    # KL-anchor: frozen copy of the BC policy, used to penalise drift away from
    # the known-good starting behaviour every update (not just per-step).
    policy_ref = copy.deepcopy(policy).to(device)
    policy_ref.eval()
    for p in policy_ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    env = PlacementEnv(grid=grid)
    ds = FloorplanDatasetLite(root)
    eval_indices = list(range(max(0, len(ds) - eval_cases), len(ds)))
    print(f"[train_ppo] device={device} warmstart={Path(warmstart).name} "
          f"grid={grid} iters={iters} batch={batch_samples} rollouts={rollouts} "
          f"lr={lr} clip={clip} ppo_epochs={ppo_epochs} kl_coef={kl_coef}")

    t0 = time.time()
    base = _greedy_eval(policy, gnn, env, ds, eval_indices, device)
    print(f"  [eval] iter  -1  fixed-set greedy legalized_cost={base:.4f}  "
          f"({len(eval_indices)} cases, {time.time()-t0:.0f}s)")

    idx = 0
    for it in range(iters):
        all_steps, all_adv, all_oldlp = [], [], []
        iter_costs = []
        for _ in range(batch_samples):
            s = ds[idx % len(ds)]; idx += 1
            area, b2b, p2b, pins, cons = s["input"]
            _tree, fp_sol, metrics = s["label"]
            bc = int((area != -1).sum())
            boxes = boxes_from_fp_sol(fp_sol, bc)
            tp = make_target_positions_from_gt(cons, boxes, bc)
            blocks, _mib, clusters = parse_and_init(bc, area, cons, tp)
            supers = prepack_clusters(blocks, clusters)
            with torch.no_grad():
                node_emb, _ = gnn.encode_problem(area, cons, b2b, bc, device=device)
            tensors = (area, b2b, p2b, pins, cons, metrics)

            eps = [_rollout(policy, gnn, env, node_emb, tensors, tp, device,
                            blocks, supers, clusters) for _ in range(rollouts)]
            costs = torch.tensor([c for _s, c in eps])
            iter_costs.append(float(costs.mean()))
            # per-case baseline -> advantage (lower cost = higher reward).
            # std floored at 0.1: if the K rollouts of a case happen to land on
            # nearly the same cost, dividing by a tiny std would blow a small
            # difference up into a huge (noise-driven) advantage.
            rewards = -costs
            adv = (rewards - rewards.mean()) / rewards.std().clamp_min(0.1)
            for (steps, _c), a in zip(eps, adv.tolist()):
                for (canvas, blk, action, oldlp) in steps:
                    all_steps.append((canvas, node_emb[blk].detach().cpu(), action))
                    all_adv.append(a)
                    all_oldlp.append(oldlp)

        if not all_steps:
            continue
        N = len(all_steps)
        act_b = torch.tensor([a for _c, _e, a in all_steps])
        adv_b = torch.tensor(all_adv)
        oldlp_b = torch.tensor(all_oldlp)

        # Minibatched PPO update — forwarding all N steps at once OOMs the GPU
        # ([N,2*hidden,G,G] for the fuse cat); chunk into mb_size slices.
        mb = 256
        kl_val = 0.0
        for _ in range(ppo_epochs):
            perm = torch.randperm(N)
            for st in range(0, N, mb):
                ii = perm[st:st + mb]
                cb = torch.stack([all_steps[k][0] for k in ii]).to(device)
                eb = torch.stack([all_steps[k][1] for k in ii]).to(device)
                ab = adv_b[ii].to(device)
                ob = oldlp_b[ii].to(device)
                actb = act_b[ii].to(device)
                opt.zero_grad()
                logits, _v = policy.forward_batch(cb, eb)             # [mb,G*G]
                logp_all = F.log_softmax(logits, dim=1)
                logp = logp_all[torch.arange(len(ii), device=device), actb]
                ratio = torch.exp(logp - ob)
                surr1 = ratio * ab
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * ab
                ent = -(logp_all.exp() * logp_all).sum(1).mean()

                with torch.no_grad():
                    ref_logits, _ = policy_ref.forward_batch(cb, eb)
                    ref_logp_all = F.log_softmax(ref_logits, dim=1)
                kl = (logp_all.exp() * (logp_all - ref_logp_all)).sum(1).mean()

                loss = -torch.min(surr1, surr2).mean() - c_entropy * ent + kl_coef * kl
                loss.backward()
                opt.step()
                kl_val = float(kl)

        mc = sum(iter_costs) / len(iter_costs)
        print(f"  iter {it:3d}  mean_legalized_cost={mc:.4f}  "
              f"loss={float(loss):.4f}  kl={kl_val:.4f}  "
              f"({(it+1)*batch_samples*rollouts} eps, {time.time()-t0:.0f}s)")

        if (it + 1) % eval_every == 0 or it == iters - 1:
            ev = _greedy_eval(policy, gnn, env, ds, eval_indices, device)
            print(f"  [eval] iter {it:3d}  fixed-set greedy legalized_cost={ev:.4f}  "
                  f"(baseline={base:.4f}, {time.time()-t0:.0f}s)")
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
    ap.add_argument("--warmstart", type=str, default="CNN_RL/checkpoints/phase11_pin_soft.pt")
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
    ap.add_argument("--ckpt-name", type=str, default="phase12_ppo_kl.pt")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    train(warmstart=args.warmstart, iters=args.iters, batch_samples=args.batch_samples,
          rollouts=args.rollouts, lr=args.lr, clip=args.clip, ppo_epochs=args.ppo_epochs,
          c_entropy=args.c_entropy, kl_coef=args.kl_coef, eval_every=args.eval_every,
          eval_cases=args.eval_cases, seed=args.seed, ckpt_name=args.ckpt_name,
          device=args.device)
