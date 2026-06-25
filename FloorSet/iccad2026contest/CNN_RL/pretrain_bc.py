"""
Phase 4.5 — Behaviour-cloning warm-start (VERSION_B_RL_PLACER.md §5 Phase 4.5).

Pure RL from random barely learns (Phase 4 finding). We have ground-truth
placements, so we teacher-force the env along the GT layout and train the
policy to predict, at each step, the grid cell where the GT block's lower-left
corner lands. This warm-starts gnn+policy; Phase 4 then RL-fine-tunes from
the saved checkpoint.

GT comes from the validation set's polygon solution (`fp_sol` is [N, V, 2]);
for rectangular Lite blocks the bbox is exact, so no 6GB training download is
needed for this step.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from canvas_raster import rasterize_env, N_CHANNELS, CH_FEASIBILITY
from gnn_encoder import GNNEncoder
from placement_env import PlacementEnv
from policy_net import PolicyValueNet

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


def gt_boxes_from_fp(fp_sol: torch.Tensor, block_count: int) -> torch.Tensor:
    """[block_count, 4] = (x, y, w, h) lower-left bbox per block, from the GT
    polygon vertices (padding rows are < 0 and dropped)."""
    boxes = torch.zeros(block_count, 4)
    for i in range(block_count):
        poly = fp_sol[i]
        v = poly[(poly[:, 0] >= 0) & (poly[:, 1] >= 0)]
        x0, y0 = float(v[:, 0].min()), float(v[:, 1].min())
        x1, y1 = float(v[:, 0].max()), float(v[:, 1].max())
        boxes[i] = torch.tensor([x0, y0, x1 - x0, y1 - y0])
    return boxes


def _target_cell(env: PlacementEnv, x: float, y: float, w: float, h: float) -> int:
    """GT box (lower-left x,y + dims w,h) -> flat grid action, clamped to the
    feasible region. The reference point matches env.anchor (and hence
    _action_to_xy): "ll" snaps the GT lower-left corner to a cell; "center"
    snaps the GT CENTER to a cell (size-invariant target)."""
    G = env.grid
    cell_w = env.canvas_w / G
    cell_h = env.canvas_h / G
    if getattr(env, "anchor", "ll") == "center":
        # cell center = (col+0.5)*cell_w  ->  col = center/cell_w - 0.5
        col = int(round((x + w / 2) / cell_w - 0.5))
        row = int(round((y + h / 2) / cell_h - 0.5))
        # clamp so the centered block fits (matches the center-mode feasibility mask)
        min_col = max(int(math.ceil((w / 2) / cell_w - 0.5)), 0)
        min_row = max(int(math.ceil((h / 2) / cell_h - 0.5)), 0)
        max_col = min(int((env.canvas_w - w / 2) / cell_w - 0.5), G - 1)
        max_row = min(int((env.canvas_h - h / 2) / cell_h - 0.5), G - 1)
        col = min(max(col, min_col), max(max_col, min_col))
        row = min(max(row, min_row), max(max_row, min_row))
        return row * G + col
    col = int(round(x / cell_w))
    row = int(round(y / cell_h))
    # clamp so the block fits inside the canvas (matches feasibility mask)
    max_col = max(int((env.canvas_w - w) / cell_w), 0)
    max_row = max(int((env.canvas_h - h) / cell_h), 0)
    col = min(max(col, 0), min(max_col, G - 1))
    row = min(max(row, 0), min(max_row, G - 1))
    return row * G + col


def build_bc_episode(env: PlacementEnv, sample: dict):
    """Teacher-force the env along GT; return static inputs + per-step
    (canvas, block, mask, target_action)."""
    at, b2b, p2b, pins, cons = sample["input"]
    _fp, metrics = sample["label"]
    fp_sol = sample["label"][0]
    env.reset(at, b2b, p2b, pins, cons, metrics)
    bc = env.block_count
    boxes = gt_boxes_from_fp(fp_sol, bc)

    steps = []
    done = False
    while not done:
        state = env._build_state()
        cur = state["current_block"]
        canvas = rasterize_env(env)
        mask = canvas[CH_FEASIBILITY]
        gx, gy, gw, gh = (float(v) for v in boxes[cur])
        target = _target_cell(env, gx, gy, gw, gh)
        steps.append((canvas.detach(), cur, mask.detach(), target))
        env.step(target)  # teacher forcing: place at the GT-derived cell
        done = env.ptr >= bc
    return {"area": at, "cons": cons, "b2b": b2b, "bc": bc, "steps": steps}


def pretrain(samples: List[dict], epochs: int = 200, grid: int = 48,
             gnn_out: int = 64, hidden: int = 32, lr: float = 1e-3,
             seed: int = 0, save: bool = False, verbose: bool = False
             ) -> Tuple[GNNEncoder, PolicyValueNet, dict]:
    torch.manual_seed(seed)
    gnn = GNNEncoder(out_dim=gnn_out, hidden=hidden)
    policy = PolicyValueNet(in_channels=N_CHANNELS, node_dim=gnn_out, hidden=hidden)
    opt = torch.optim.Adam(list(gnn.parameters()) + list(policy.parameters()), lr=lr)

    env = PlacementEnv(grid=grid)
    episodes = [build_bc_episode(env, s) for s in samples]

    hist = {"loss": [], "acc": [], "near_acc": []}
    for ep in range(epochs):
        opt.zero_grad()
        total_loss = torch.tensor(0.0)
        correct = 0
        near = 0       # predicted cell within 1 cell (Chebyshev) of GT cell
        n = 0
        for e in episodes:
            node_emb, _ = gnn.encode_problem(e["area"], e["cons"], e["b2b"], e["bc"])
            for canvas, block, mask, target in e["steps"]:
                _probs, _value, logits = policy(canvas, node_emb[block], mask)
                total_loss = total_loss + F.cross_entropy(
                    logits.view(1, -1), torch.tensor([target]))
                pred = int(torch.argmax(logits))
                correct += int(pred == target)
                pr, pc = divmod(pred, grid)
                tr, tc = divmod(target, grid)
                near += int(abs(pr - tr) <= 1 and abs(pc - tc) <= 1)
                n += 1
        loss = total_loss / max(n, 1)
        loss.backward()
        opt.step()
        hist["loss"].append(float(loss))
        hist["acc"].append(correct / max(n, 1))
        hist["near_acc"].append(near / max(n, 1))
        if verbose and ep % 20 == 0:
            print(f"epoch {ep:3d}  bc_loss={float(loss):.4f}  "
                  f"cell_acc={correct/max(n,1):.3f}  near_acc(±1)={near/max(n,1):.3f}")

    if save:
        CKPT_DIR.mkdir(exist_ok=True)
        path = CKPT_DIR / "bc_warmstart.pt"
        torch.save({"gnn": gnn.state_dict(), "policy": policy.state_dict(),
                    "grid": grid, "gnn_out": gnn_out, "hidden": hidden}, path)
        hist["ckpt"] = str(path)
    return gnn, policy, hist


def load_warmstart(path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    gnn = GNNEncoder(out_dim=ckpt["gnn_out"], hidden=ckpt["hidden"])
    policy = PolicyValueNet(in_channels=N_CHANNELS, node_dim=ckpt["gnn_out"],
                            hidden=ckpt["hidden"])
    gnn.load_state_dict(ckpt["gnn"])
    policy.load_state_dict(ckpt["policy"])
    return gnn, policy, ckpt
