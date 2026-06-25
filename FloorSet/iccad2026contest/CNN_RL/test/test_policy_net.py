"""
Phase 3 acceptance test (VERSION_B_RL_PLACER.md §5 Phase 3).

  * masked distribution sums to 1 and puts ZERO prob on illegal cells
  * value is a scalar, gradients flow
  * end-to-end smoke: real env -> rasterizer -> GNN -> policy -> legal action

Run:
    cd FloorSet/iccad2026contest
    python3 DL_RL/test_policy_net.py
"""

import sys
from pathlib import Path

import torch

_FLOORSET_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_FLOORSET_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOORSET_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lite_dataset_test import FloorplanDatasetLiteTest  # noqa: E402
from placement_env import PlacementEnv  # noqa: E402
from canvas_raster import rasterize_env, N_CHANNELS, CH_FEASIBILITY  # noqa: E402
from gnn_encoder import GNNEncoder  # noqa: E402
from policy_net import PolicyValueNet  # noqa: E402


def test_masked_distribution():
    G, D = 16, 128
    net = PolicyValueNet(in_channels=N_CHANNELS, node_dim=D, hidden=32, n_conv=3)
    canvas = torch.rand(N_CHANNELS, G, G)
    node_emb = torch.randn(D)
    # legal region = lower-left 10x10 block
    mask = torch.zeros(G, G)
    mask[:10, :10] = 1.0

    probs, value, logits = net(canvas, node_emb, mask)
    assert probs.shape == (G * G,), probs.shape
    assert abs(float(probs.sum()) - 1.0) < 1e-4, float(probs.sum())
    # zero probability on illegal cells
    illegal = (mask.view(-1) < 0.5)
    assert float(probs[illegal].sum()) < 1e-6, float(probs[illegal].sum())
    assert value.dim() == 0, value.shape
    print("Masked-distribution PASS (sums to 1, zero on illegal, scalar value).")


def test_gradients():
    G, D = 12, 64
    net = PolicyValueNet(in_channels=N_CHANNELS, node_dim=D, hidden=16, n_conv=2)
    canvas = torch.rand(N_CHANNELS, G, G)
    node_emb = torch.randn(D)
    mask = torch.ones(G, G)
    probs, value, _ = net(canvas, node_emb, mask)
    # toy actor-critic-shaped loss
    loss = -torch.log(probs[0].clamp_min(1e-12)) + value.pow(2)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    print("Gradient-flow PASS.")


def test_end_to_end():
    ds = FloorplanDatasetLiteTest(str(_FLOORSET_ROOT))
    sample = ds[0]
    at, b2b, p2b, pins, cons = sample["input"]
    _fp, metrics = sample["label"]
    block_count = int((at != -1).sum().item())

    env = PlacementEnv(grid=84)
    env.reset(at, b2b, p2b, pins, cons, metrics)

    gnn = GNNEncoder(out_dim=128)
    node_emb, _graph = gnn.encode_problem(at, cons, b2b, block_count)
    policy = PolicyValueNet(in_channels=N_CHANNELS, node_dim=128)

    # one decision for the first block to be placed
    state = env._build_state()
    cur = state["current_block"]
    canvas = rasterize_env(env)
    mask = canvas[CH_FEASIBILITY]
    action, logprob, value = policy.act(canvas, node_emb[cur], mask, greedy=True)

    row, col = divmod(action, env.grid)
    assert mask[row, col] > 0.5, "chosen action is not in the legal region"
    print(f"End-to-end PASS: block {cur} -> action {action} (row {row}, col {col}), "
          f"legal=True, value={value:.3f}")


def main():
    torch.manual_seed(0)
    test_masked_distribution()
    test_gradients()
    test_end_to_end()
    print("\nPhase 3 PASS.")


if __name__ == "__main__":
    main()
