"""
Vectorised differentiable contest cost — fast training surrogate.

This mirrors iccad2026_evaluate.compute_training_loss_differentiable EXACTLY
(same ALPHA/BETA, same formula) but replaces its python double-loops over block
pairs and edges with tensor ops, so it can be called thousands of times per
training run. `check_matches_official()` validates the two agree on a real case.

  Cost = (1 + α·(HPWL_gap + Area_gap)) · exp(β·(overlap_viol + area_viol))
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_CONTEST = Path(__file__).resolve().parent.parent.parent
if str(_CONTEST) not in sys.path:
    sys.path.insert(0, str(_CONTEST))
from iccad2026_evaluate import ALPHA, BETA, AREA_TOLERANCE  # noqa: E402


def diff_cost(positions, b2b, p2b, pins, area_targets, baseline_metrics,
              return_parts: bool = False):
    """positions [N,4]=(x,y,w,h) (differentiable). Returns scalar cost tensor."""
    x, y, w, h = positions[:, 0], positions[:, 1], positions[:, 2], positions[:, 3]
    cx, cy = x + w / 2, y + h / 2
    N = positions.shape[0]

    # --- HPWL (b2b + p2b) --------------------------------------------------
    hpwl = positions.new_tensor(0.0)
    if b2b is not None and b2b.numel():
        e = b2b[b2b[:, 0] >= 0]
        i, j = e[:, 0].long(), e[:, 1].long()
        m = (i < N) & (j < N)
        i, j, wt = i[m], j[m], e[:, 2][m]
        hpwl = hpwl + (wt * ((cx[i] - cx[j]).abs() + (cy[i] - cy[j]).abs())).sum()
    if p2b is not None and p2b.numel() and pins is not None and pins.numel():
        e = p2b[p2b[:, 0] >= 0]
        pi, bj = e[:, 0].long(), e[:, 1].long()
        m = (pi < pins.shape[0]) & (bj < N)
        pi, bj, wt = pi[m], bj[m], e[:, 2][m]
        px, py = pins[pi, 0], pins[pi, 1]
        hpwl = hpwl + (wt * ((cx[bj] - px).abs() + (cy[bj] - py).abs())).sum()

    # --- bbox area ---------------------------------------------------------
    bbox = (x + w).max() - x.min()
    bbox = bbox * ((y + h).max() - y.min())

    # --- overlap (vectorised pairwise) ------------------------------------
    x2, y2 = x + w, y + h
    ox = (torch.minimum(x2[:, None], x2[None, :]) - torch.maximum(x[:, None], x[None, :])).clamp(min=0)
    oy = (torch.minimum(y2[:, None], y2[None, :]) - torch.maximum(y[:, None], y[None, :])).clamp(min=0)
    pair = ox * oy
    overlap_area = torch.triu(pair, diagonal=1).sum()
    overlap_viol = overlap_area / ((w * h).sum() + 1e-6)

    # --- area tolerance (0 when area is exact by construction) -------------
    valid = area_targets[:N] > 0
    err = torch.zeros_like(area_targets[:N])
    err[valid] = ((w * h)[valid] - area_targets[:N][valid]).abs() / area_targets[:N][valid]
    area_viol = torch.relu(err - AREA_TOLERANCE).sum() / (valid.sum() + 1e-6)

    # --- gaps vs baseline + cost ------------------------------------------
    base_area = baseline_metrics[0]
    base_hpwl = baseline_metrics[6] + baseline_metrics[7]
    hpwl_gap = torch.relu((hpwl - base_hpwl) / (base_hpwl + 1e-6))
    area_gap = torch.relu((bbox - base_area) / (base_area + 1e-6))

    v_soft = overlap_viol + area_viol
    cost = (1 + ALPHA * (hpwl_gap + area_gap)) * torch.exp(BETA * v_soft)
    if return_parts:
        return cost, {"hpwl_gap": float(hpwl_gap), "area_gap": float(area_gap),
                      "overlap_viol": float(overlap_viol), "area_viol": float(area_viol)}
    return cost


def check_matches_official(case: int = 0, data_path: str = "../"):
    """Sanity: our vectorised cost == the official python-loop cost on one case."""
    from iccad2026_evaluate import (FloorplanDatasetLiteTest,
                                     compute_training_loss_differentiable)
    ds = FloorplanDatasetLiteTest(data_path)
    s = ds[case]
    a, b2b, p2b, pins, c = s["input"]
    bc = int((a != -1).sum())
    poly = s["label"][0]
    metrics = s["label"][1]
    pos = []
    for i in range(bc):
        v = poly[i][poly[i][:, 0] != -1]
        x0, y0 = v.min(0).values
        x1, y1 = v.max(0).values
        pos.append([float(x0), float(y0), float(x1 - x0), float(y1 - y0)])
    P = torch.tensor(pos)
    mine = float(diff_cost(P, b2b, p2b, pins, a[:bc], metrics))
    off = float(compute_training_loss_differentiable(P, b2b, p2b, pins, a[:bc], metrics))
    print(f"case {case}: vectorised={mine:.6f}  official={off:.6f}  "
          f"diff={abs(mine-off):.2e}")
    return abs(mine - off)


if __name__ == "__main__":
    worst = max(check_matches_official(c) for c in range(5))
    print(f"max abs diff over 5 cases = {worst:.2e}")
