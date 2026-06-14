"""
Phase 8 — scaled behaviour-cloning on the 1M training set
(VERSION_B_RL_PLACER.md §5 Phase 8).

BC is the stable, fast path to a *generalising* policy (Phase 4 showed pure RL
from scratch is too noisy; full PPO over 1M with the Python-loop reward is also
too slow for a first pass). We stream samples from get_training_dataloader,
teacher-force the env along GT, and do one online gradient step per sample so a
single policy learns to place across all block counts (21..120).

Training fp_sol is [N,4] = (w,h,x,y) (NOT the validation polygon format), so GT
boxes are read directly: positions = fp_sol[:, [2,3,0,1]] -> (x,y,w,h).

Run (after the 6GB LiteTensorData download is in place):
    cd FloorSet/iccad2026contest
    python3 DL_RL/train_phase8.py --num-samples 2000 --epochs 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_FLOORSET_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_FLOORSET_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOORSET_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from iccad2026contest.iccad2026_evaluate import get_training_dataloader  # noqa: E402
from placement_env import PlacementEnv  # noqa: E402
from canvas_raster import rasterize_env, N_CHANNELS, CH_FEASIBILITY  # noqa: E402
from gnn_encoder import GNNEncoder  # noqa: E402
from policy_net import PolicyValueNet  # noqa: E402
from pretrain_bc import _target_cell, CKPT_DIR  # noqa: E402
from hard_constraints import make_target_positions_from_gt  # noqa: E402


def boxes_from_fp_sol(fp_sol: torch.Tensor, bc: int) -> torch.Tensor:
    """Training fp_sol [N,4]=(w,h,x,y) -> GT boxes [bc,4]=(x,y,w,h)."""
    return fp_sol[:bc][:, [2, 3, 0, 1]].clone()


def build_bc_episode(env: PlacementEnv, tensors):
    """Teacher-force env along GT; collect (canvas, block, mask, target_cell)."""
    area, b2b, p2b, pins, cons, _tree, fp_sol, metrics = tensors
    bc = int((area != -1).sum())
    boxes = boxes_from_fp_sol(fp_sol, bc)
    tp = make_target_positions_from_gt(cons, boxes, bc)
    env.reset(area, b2b, p2b, pins, cons, metrics, target_positions=tp)

    steps = []
    done = len(env.order) == 0
    while not done:
        st = env._build_state()
        cur = st["current_block"]
        canvas = rasterize_env(env)
        mask = canvas[CH_FEASIBILITY]
        gx, gy, gw, gh = (float(v) for v in boxes[cur])
        target = _target_cell(env, gx, gy, gw, gh)
        steps.append((canvas.detach(), cur, mask.detach(), target))
        env.step(target)
        done = env.ptr >= len(env.order)
    return area, cons, b2b, bc, steps


def train(num_samples=2000, epochs=1, grid=64, gnn_out=128, hidden=64,
          lr=1e-3, seed=0, ckpt_name="phase8_bc.pt", log_every=100):
    torch.manual_seed(seed)
    gnn = GNNEncoder(out_dim=gnn_out, hidden=hidden)
    policy = PolicyValueNet(in_channels=N_CHANNELS, node_dim=gnn_out, hidden=hidden)
    opt = torch.optim.Adam(list(gnn.parameters()) + list(policy.parameters()), lr=lr)
    env = PlacementEnv(grid=grid)

    loader = get_training_dataloader(batch_size=1, num_samples=num_samples,
                                     shuffle=False)
    print(f"streaming {len(loader)} samples x {epochs} epoch(s), grid={grid}")

    seen = 0
    t0 = time.time()
    run_loss, run_acc, run_n = 0.0, 0, 0
    for ep in range(epochs):
        for batch in loader:
            tensors = [b.squeeze(0) for b in batch]
            area, cons, b2b, bc, steps = build_bc_episode(env, tensors)
            if not steps:
                continue
            opt.zero_grad()
            node_emb, _ = gnn.encode_problem(area, cons, b2b, bc)
            loss = torch.tensor(0.0)
            correct = 0
            for canvas, block, mask, target in steps:
                _p, _v, logits = policy(canvas, node_emb[block], mask)
                loss = loss + F.cross_entropy(logits.view(1, -1),
                                              torch.tensor([target]))
                correct += int(torch.argmax(logits) == target)
            loss = loss / len(steps)
            loss.backward()
            opt.step()

            seen += 1
            run_loss += float(loss)
            run_acc += correct
            run_n += len(steps)
            if seen % log_every == 0:
                rate = seen / (time.time() - t0)
                print(f"  [{seen}] loss={run_loss/log_every:.3f} "
                      f"cell_acc={run_acc/max(run_n,1):.3f} "
                      f"({rate:.1f} samples/s)")
                run_loss, run_acc, run_n = 0.0, 0, 0

    CKPT_DIR.mkdir(exist_ok=True)
    path = CKPT_DIR / ckpt_name
    torch.save({"gnn": gnn.state_dict(), "policy": policy.state_dict(),
                "grid": grid, "gnn_out": gnn_out, "hidden": hidden}, path)
    print(f"saved {path}  ({seen} samples in {time.time()-t0:.1f}s)")
    return str(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(num_samples=args.num_samples, epochs=args.epochs, grid=args.grid,
          seed=args.seed)
