"""
Phase 4.5 acceptance test (VERSION_B_RL_PLACER.md §5 Phase 4.5).

Behaviour-cloning warm-start: teacher-force the env along GT and train the
policy to predict the GT cell. Confirm the BC loss drops and cell-prediction
accuracy rises (policy is learning the GT layout), then save + reload the
checkpoint.

Run:
    cd FloorSet/iccad2026contest
    python3 DL_RL/test_pretrain_bc.py
"""

import sys
from pathlib import Path

import torch

_FLOORSET_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_FLOORSET_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOORSET_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lite_dataset_test import FloorplanDatasetLiteTest  # noqa: E402
from pretrain_bc import pretrain, load_warmstart, gt_boxes_from_fp  # noqa: E402


def test_gt_boxes():
    ds = FloorplanDatasetLiteTest(str(_FLOORSET_ROOT))
    s = ds[0]
    at = s["input"][0]
    bc = int((at != -1).sum())
    boxes = gt_boxes_from_fp(s["label"][0], bc)
    assert boxes.shape == (bc, 4)
    # area from bbox should match target within tolerance for rectangular blocks
    areas = boxes[:, 2] * boxes[:, 3]
    rel_err = ((areas - at[:bc]).abs() / at[:bc].clamp_min(1e-6))
    assert float(rel_err.max()) < 0.05, float(rel_err.max())
    print(f"GT-box reconstruction PASS (max area rel-err {float(rel_err.max()):.4f}).")


def main():
    torch.manual_seed(0)
    test_gt_boxes()

    ds = FloorplanDatasetLiteTest(str(_FLOORSET_ROOT))
    samples = [ds[0], ds[1]]

    _gnn, _policy, hist = pretrain(samples, epochs=300, grid=48, gnn_out=64,
                                   hidden=32, lr=1e-3, seed=0, save=True,
                                   verbose=True)

    init_loss = sum(hist["loss"][:5]) / 5
    final_loss = sum(hist["loss"][-5:]) / 5
    final_acc = sum(hist["acc"][-5:]) / 5
    final_near = sum(hist["near_acc"][-5:]) / 5
    print(f"\ninit bc_loss={init_loss:.4f}  final bc_loss={final_loss:.4f}  "
          f"final cell_acc={final_acc:.3f}  final near_acc(±1)={final_near:.3f}")

    assert final_loss < 0.6 * init_loss, "BC loss did not drop enough"
    # warm-start metric: predicted cell should land within 1 cell of GT
    assert final_near > 0.4, "policy did not learn GT placement well enough"

    # checkpoint round-trips
    g2, p2, ckpt = load_warmstart(hist["ckpt"])
    assert ckpt["grid"] == 48
    print(f"Checkpoint reload PASS ({hist['ckpt']}).")

    print("\nPhase 4.5 PASS: BC warm-start learns GT placement; checkpoint saved.")


if __name__ == "__main__":
    main()
