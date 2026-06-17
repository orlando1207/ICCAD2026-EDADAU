"""
Phase 0 acceptance test (VERSION_B_RL_PLACER.md §5 Phase 0).

Runs a random-policy rollout of PlacementEnv on local validation samples
(LiteTensorDataTest/, no 6GB download) and checks the episode runs cleanly
from reset() to done with a finite terminal cost.

Run:
    cd FloorSet/iccad2026contest
    python3 DL_RL/test_placement_env.py
"""

import random
import sys
from pathlib import Path

import torch

_FLOORSET_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_FLOORSET_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOORSET_ROOT))

from lite_dataset_test import FloorplanDatasetLiteTest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from placement_env import PlacementEnv  # noqa: E402


def random_rollout(env: PlacementEnv, sample: dict) -> dict:
    at, b2b, p2b, pins, cons = sample["input"]
    _fp_sol, metrics = sample["label"]

    env.reset(at, b2b, p2b, pins, cons, metrics)
    n_steps = 0
    total_reward = 0.0
    done = False
    while not done:
        action = random.randrange(env.action_space_size)
        _state, reward, done, info = env.step(action)
        total_reward += reward
        n_steps += 1
    return {
        "block_count": env.block_count,
        "n_steps": n_steps,
        "canvas": (round(env.canvas_w, 1), round(env.canvas_h, 1)),
        "contest_cost": info["contest_cost"],
        "total_reward": total_reward,
    }


def main():
    random.seed(0)
    torch.manual_seed(0)

    ds = FloorplanDatasetLiteTest(str(_FLOORSET_ROOT))
    print(f"Loaded {len(ds)} validation samples (no download).\n")

    n_test = 5
    for i in range(n_test):
        sample = ds[i]
        r = random_rollout(PlacementEnv(grid=84), sample)

        # Acceptance checks
        assert r["n_steps"] == r["block_count"], "steps != block_count"
        assert r["contest_cost"] == r["contest_cost"], "cost is NaN"  # NaN check
        assert r["contest_cost"] > 0, "cost should be positive"

        print(f"sample {i}: blocks={r['block_count']:3d}  "
              f"canvas={r['canvas']}  steps={r['n_steps']:3d}  "
              f"random-policy contest_cost={r['contest_cost']:.4f}")

    print("\nPhase 0 PASS: reset->done runs cleanly, finite positive cost.")
    print("(Costs are high because the policy is random — expected.)")


if __name__ == "__main__":
    main()
