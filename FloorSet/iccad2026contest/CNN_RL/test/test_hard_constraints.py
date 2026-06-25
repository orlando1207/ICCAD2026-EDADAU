"""
Phase 6 acceptance test (VERSION_B_RL_PLACER.md §5 Phase 6).

With target_positions supplied, the env must satisfy ALL hard constraints
(fixed dims, preplaced location+dims, MIB shared shape, boundary touch)
*regardless of what cells the policy picks* — they are enforced, not learned.
We drive it with RANDOM actions and assert zero violations.

Run:
    cd FloorSet/iccad2026contest
    python3 DL_RL/test_hard_constraints.py
"""

import random
import sys
from pathlib import Path

import torch

_FLOORSET_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_FLOORSET_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOORSET_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lite_dataset_test import FloorplanDatasetLiteTest  # noqa: E402
from placement_env import PlacementEnv  # noqa: E402
from pretrain_bc import gt_boxes_from_fp  # noqa: E402
from hard_constraints import (  # noqa: E402
    make_target_positions_from_gt, check_hard_constraints,
)


def run_one(ds, idx, grid=84, seed=0):
    random.seed(seed)
    s = ds[idx]
    at, b2b, p2b, pins, cons = s["input"]
    _fp, metrics = s["label"]
    bc = int((at != -1).sum())
    gt = gt_boxes_from_fp(s["label"][0], bc)
    tp = make_target_positions_from_gt(cons, gt, bc)

    env = PlacementEnv(grid=grid)
    env.reset(at, b2b, p2b, pins, cons, metrics, target_positions=tp)

    # preplaced blocks must be placed already and excluded from the RL order
    preplaced = [i for i in range(bc) if cons[i, 1] > 0]
    for i in preplaced:
        assert not torch.isnan(env.positions[i, 0]), f"preplaced {i} not placed"
        assert i not in env.order, f"preplaced {i} still in RL order"

    # random rollout over the remaining (non-preplaced) blocks
    done = len(env.order) == 0
    while not done:
        _s, _r, done, _info = env.step(random.randrange(env.action_space_size))

    v = check_hard_constraints(env.positions, cons, tp, bc,
                               (env.canvas_w, env.canvas_h))
    n_flags = {
        "fixed": int((cons[:bc, 0] > 0).sum()),
        "preplaced": len(preplaced),
        "mib": int((cons[:bc, 2] > 0).sum()),
        "boundary": int((cons[:bc, 4] > 0).sum()),
    }
    return v, n_flags, bc


def main():
    ds = FloorplanDatasetLiteTest(str(_FLOORSET_ROOT))

    # MIB members of a group should share the same area (same master cell)
    s0 = ds[0]
    cons0 = s0["input"][4]
    at0 = s0["input"][0]
    bc0 = int((at0 != -1).sum())
    mib_members = [i for i in range(bc0) if cons0[i, 2] == 1]
    areas = {round(float(at0[i]), 1) for i in mib_members}
    print(f"sample 0 MIB group 1 members={mib_members}, distinct areas={areas}")

    total_viol = {"fixed": 0, "preplaced": 0, "mib": 0, "boundary": 0}
    for idx in range(6):
        v, flags, bc = run_one(ds, idx)
        for k in total_viol:
            total_viol[k] += v[k]
        print(f"sample {idx}: blocks={bc} flags={flags} -> violations={v}")

    assert all(c == 0 for c in total_viol.values()), \
        f"hard-constraint violations remain: {total_viol}"
    print(f"\nPhase 6 PASS: all hard constraints satisfied under random actions "
          f"(totals {total_viol}).")


if __name__ == "__main__":
    main()
