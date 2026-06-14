"""
Phase 5 acceptance test (VERSION_B_RL_PLACER.md §5 Phase 5).

  * env honours an aspect-ratio action (w/h correct, area still exact)
  * policy aspect head: distribution sums to 1, gradients flow, act_aspect works
  * oracle shape benefit: at GT positions, using GT shapes costs LESS than
    forcing squares -> proves the shape lever is worth learning

Run:
    cd FloorSet/iccad2026contest
    python3 DL_RL/test_ar_utils.py
"""

import math
import sys
from pathlib import Path

import torch

_FLOORSET_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_FLOORSET_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOORSET_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from iccad2026contest.iccad2026_evaluate import compute_training_loss_differentiable  # noqa: E402
from lite_dataset_test import FloorplanDatasetLiteTest  # noqa: E402
from placement_env import PlacementEnv  # noqa: E402
from canvas_raster import rasterize_env, N_CHANNELS, CH_FEASIBILITY  # noqa: E402
from gnn_encoder import GNNEncoder  # noqa: E402
from policy_net import PolicyValueNet  # noqa: E402
from pretrain_bc import gt_boxes_from_fp  # noqa: E402
from ar_utils import ASPECT_BUCKETS, N_ASPECT, gt_aspect_bucket  # noqa: E402


def test_env_aspect():
    ds = FloorplanDatasetLiteTest(str(_FLOORSET_ROOT))
    s = ds[0]
    at, b2b, p2b, pins, cons = s["input"]
    _fp, metrics = s["label"]
    env = PlacementEnv(grid=84)
    env.reset(at, b2b, p2b, pins, cons, metrics)
    # pick a block with no fixed/MIB shape so the aspect action actually applies
    free_idx = next(i for i, blk in enumerate(env.order)
                     if blk not in env.fixed_dims and blk not in env.mib_dims)
    for _ in range(free_idx):
        env.step(env.action_space_size // 2, aspect=1.0)
    blk = env.order[free_idx]
    area = float(at[blk])
    env.step(env.action_space_size // 2, aspect=2.0)
    x, y, w, h = (float(v) for v in env.positions[blk])
    assert abs((w / h) - 2.0) < 1e-3, w / h
    assert abs(w * h - area) / area < 1e-4, (w * h, area)
    print(f"env aspect PASS (w/h={w/h:.3f}, w*h={w*h:.1f} vs area {area:.1f}).")


def test_aspect_head():
    G, D = 16, 64
    net = PolicyValueNet(in_channels=N_CHANNELS, node_dim=D, hidden=16,
                         n_conv=2, n_aspect=N_ASPECT)
    canvas = torch.rand(N_CHANNELS, G, G)
    node = torch.randn(D)
    mask = torch.ones(G, G)
    probs, ar_probs, value, _ = net.forward_aspect(canvas, node, mask)
    assert ar_probs.shape == (N_ASPECT,)
    assert abs(float(ar_probs.sum()) - 1.0) < 1e-4
    loss = -torch.log(probs[0].clamp_min(1e-12)) - torch.log(ar_probs[0].clamp_min(1e-12)) + value ** 2
    loss.backward()
    assert net.aspect_head.weight.grad is not None and net.aspect_head.weight.grad.abs().sum() > 0
    a, ai, lp, v = net.act_aspect(canvas, node, mask, greedy=True)
    assert 0 <= ai < N_ASPECT and 0 <= a < G * G
    print(f"aspect head PASS (ar_probs sum=1, grad flows, act_aspect -> bucket {ai}).")


def test_oracle_shape_benefit():
    ds = FloorplanDatasetLiteTest(str(_FLOORSET_ROOT))
    worse_with_square = 0
    n = 4
    for idx in range(n):
        s = ds[idx]
        at, b2b, p2b, pins, cons = s["input"]
        _fp, metrics = s["label"]
        bc = int((at != -1).sum())
        boxes = gt_boxes_from_fp(s["label"][0], bc)  # [bc,4] = x,y,w,h

        gt_pos = boxes.clone()                                    # (x,y,w,h) GT
        sq_pos = boxes.clone()
        side = at[:bc].clamp_min(1e-6).sqrt()
        sq_pos[:, 2] = side
        sq_pos[:, 3] = side                                       # force squares

        with torch.no_grad():
            c_gt = float(compute_training_loss_differentiable(
                gt_pos, b2b, p2b, pins, at[:bc], metrics))
            c_sq = float(compute_training_loss_differentiable(
                sq_pos, b2b, p2b, pins, at[:bc], metrics))
        worse_with_square += int(c_sq > c_gt)
        print(f"  sample {idx}: GT-shape cost={c_gt:.4f}  square cost={c_sq:.4f}")

    assert worse_with_square >= 3, "GT shapes should usually beat forced squares"
    print(f"oracle shape benefit PASS ({worse_with_square}/{n} cases: GT shape < square).")


def main():
    torch.manual_seed(0)
    test_env_aspect()
    test_aspect_head()
    test_oracle_shape_benefit()
    print("\nPhase 5 PASS: aspect-ratio action wired (env + policy head); "
          "shape lever verified to reduce cost.")


if __name__ == "__main__":
    main()
