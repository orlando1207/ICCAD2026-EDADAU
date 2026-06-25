"""
Phase 1 acceptance test (VERSION_B_RL_PLACER.md §5 Phase 1).

(a) Unit checks on a hand-crafted placement (occupancy / density / mask exact).
(b) Renders the 4 channels of a real mid-rollout state to a PNG for visual check.

Run:
    cd FloorSet/iccad2026contest
    python3 DL_RL/test_canvas_raster.py
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
from canvas_raster import (  # noqa: E402
    rasterize, rasterize_env, N_CHANNELS, CH_OCCUPANCY, CH_DENSITY, CH_WL_PULL,
    CH_FEASIBILITY, CH_PIN_PULL,
)


def test_units():
    # canvas 100x100, grid 10 -> cell 10x10
    G, canvas = 10, (100.0, 100.0)

    # one block (0,0,50,50) -> covers cols 0..4, rows 0..4 = 25 cells
    pos = torch.tensor([[0.0, 0.0, 50.0, 50.0]])
    img = rasterize(pos, canvas, G)
    assert img.shape == (N_CHANNELS, G, G), img.shape
    assert float(img[CH_OCCUPANCY].sum()) == 25.0, img[CH_OCCUPANCY].sum()
    assert float(img[CH_OCCUPANCY, :5, :5].sum()) == 25.0

    # add an overlapping block (30,30,40,40) -> overlap region density==2
    pos2 = torch.tensor([[0.0, 0.0, 50.0, 50.0], [30.0, 30.0, 40.0, 40.0]])
    img2 = rasterize(pos2, canvas, G)
    assert float(img2[CH_DENSITY].max()) == 2.0, img2[CH_DENSITY].max()

    # feasibility: current block 40x40, valid lower-left x+40<=100 -> cols 0..6
    img3 = rasterize(pos, canvas, G, current_block=0, current_dims=(40.0, 40.0))
    assert float(img3[CH_FEASIBILITY].sum()) == 49.0, img3[CH_FEASIBILITY].sum()  # 7x7

    # wl_pull in [0,1] and peaks near the single placed neighbour
    # place a neighbour (block 1) and ask pull for current block 0 connected to 1
    posn = torch.tensor([[float("nan")] * 4, [80.0, 80.0, 10.0, 10.0]])
    b2b = torch.tensor([[0.0, 1.0, 1.0]])
    img4 = rasterize(posn, canvas, G, current_block=0, current_dims=(10.0, 10.0), b2b=b2b)
    pull = img4[CH_WL_PULL]
    assert 0.0 <= float(pull.min()) and float(pull.max()) <= 1.0 + 1e-6
    # argmax cell should be in the top-right quadrant (near neighbour at 80,80)
    flat = int(pull.argmax())
    r, c = divmod(flat, G)
    assert r >= G // 2 and c >= G // 2, (r, c)

    # pin_pull: current block 0 connected via p2b to a pin at (80,80) -> pull
    # field peaks top-right, same as wl_pull but driven by the fixed pin.
    p2b = torch.tensor([[0.0, 0.0, 1.0]])          # (pin_idx=0, block_idx=0, w=1)
    pins = torch.tensor([[80.0, 80.0]])
    img5 = rasterize(pos, canvas, G, current_block=0, current_dims=(10.0, 10.0),
                     p2b=p2b, pins=pins)
    pin_pull = img5[CH_PIN_PULL]
    assert 0.0 <= float(pin_pull.min()) and float(pin_pull.max()) <= 1.0 + 1e-6
    pr, pc = divmod(int(pin_pull.argmax()), G)
    assert pr >= G // 2 and pc >= G // 2, (pr, pc)
    print("Unit checks PASS (occupancy/density/feasibility/wl_pull/pin_pull).")


def render_real_state():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = FloorplanDatasetLiteTest(str(_FLOORSET_ROOT))
    sample = ds[0]
    at, b2b, p2b, pins, cons = sample["input"]
    _fp, metrics = sample["label"]

    env = PlacementEnv(grid=84)
    env.reset(at, b2b, p2b, pins, cons, metrics)
    # place ~60% of the blocks greedily-ish (here: just sequential center cells)
    n_place = int(env.block_count * 0.6)
    for _ in range(n_place):
        env.step(env.action_space_size // 2)  # arbitrary fixed cell
    img = rasterize_env(env)

    titles = ["occupancy", "density", "wl_pull (b2b)", "feasibility", "pin_pull (p2b)"]
    fig, axes = plt.subplots(1, N_CHANNELS, figsize=(4 * N_CHANNELS, 4))
    for ch, (ax, t) in enumerate(zip(axes, titles)):
        # origin lower-left to match the coordinate convention
        ax.imshow(img[ch].cpu().numpy(), origin="lower", aspect="auto")
        ax.set_title(t)
        ax.set_xlabel("col (x)")
        ax.set_ylabel("row (y)")
    fig.suptitle(f"Phase 1 rasterizer — sample 0, {n_place}/{env.block_count} placed")
    out_path = Path(__file__).resolve().parent / "phase1_raster_preview.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    print(f"Saved visual preview -> {out_path}")
    print(f"  channels min/max: " + ", ".join(
        f"{titles[c]}=[{float(img[c].min()):.2f},{float(img[c].max()):.2f}]"
        for c in range(N_CHANNELS)))


def main():
    test_units()
    render_real_state()
    print("\nPhase 1 PASS.")


if __name__ == "__main__":
    main()
