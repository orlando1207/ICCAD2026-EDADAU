"""Mini-ePlace center refinement — vectorised analytical global placement (torch).

Takes the RL rollout's per-block centers and runs WL + density-overflow gradient
descent (annealed) to produce a lower-HPWL, well-spread arrangement, then hands the
refined (cx, cy) to the existing skyline legalizer (which re-packs to exact feasible).
No FFT (n<=120 small). REAL units; preplaced pinned. See ALGORITHM.md / fd_proto.

Gated by EPLACE=1 in rl_skyline_optimizer._rl_centers — off by default.
"""
from __future__ import annotations
import math
import os
import numpy as np
import torch

# Hyperparams (env-tunable for sweeps; defaults are the validated phase13 setting).
_G        = int(os.environ.get("EPLACE_G", "40"))
_ITERS    = int(os.environ.get("EPLACE_ITERS", "250"))
_LR_FRAC  = float(os.environ.get("EPLACE_LR", "0.01"))
_LAM0     = float(os.environ.get("EPLACE_LAM0", "0.1"))
_LAM_GROW = float(os.environ.get("EPLACE_LAM_GROW", "1.012"))
_FILL     = float(os.environ.get("EPLACE_FILL", "1.15"))


def eplace_refine(cx, cy, dims, areas, b2b, p2b, pins, preplaced_mask,
                  G=_G, iters=_ITERS, lr_frac=_LR_FRAC, lam0=_LAM0,
                  lam_grow=_LAM_GROW, fill=_FILL):
    """Refine rollout centers (cx, cy) -> (cx', cy') by analytical global placement.
    dims=[n,2] (w,h), areas=[n]; b2b/p2b are the raw edge tensors; pins=[P,>=2].
    preplaced_mask=[n] bool (those centers are held fixed)."""
    n = len(cx)
    P0 = np.stack([np.asarray(cx, float), np.asarray(cy, float)], 1)
    D = np.asarray(dims, float)
    areas = np.asarray(areas, float)
    side = math.sqrt(max(areas.sum(), 1e-6) * fill)
    cen = P0.mean(0)
    span = P0.max(0) - P0.min(0); span[span < 1e-6] = 1.0
    P0 = cen + (P0 - cen) * (side / span.max()) * 0.9      # compress init into region
    org = (cen[0] - side / 2.0, cen[1] - side / 2.0)

    def _np(t):
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    b = _np(b2b)
    mb = (b[:, 0] >= 0) & (b[:, 1] >= 0) & (b[:, 0] < n) & (b[:, 1] < n)
    ei = torch.tensor(b[mb, 0].astype(np.int64)); ej = torch.tensor(b[mb, 1].astype(np.int64))
    pins_np = _np(pins)[:, :2].astype(float)
    p = _np(p2b)
    mp = (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] < len(pins_np)) & (p[:, 1] < n)
    pp = torch.tensor(p[mp, 0].astype(np.int64)); pb = torch.tensor(p[mp, 1].astype(np.int64))
    pinxy = torch.tensor(pins_np, dtype=torch.float64)

    P = torch.tensor(P0, dtype=torch.float64, requires_grad=True)
    Dt = torch.tensor(D, dtype=torch.float64)
    fixed = torch.tensor(np.asarray(preplaced_mask, bool))
    P0t = torch.tensor(P0, dtype=torch.float64)
    opt = torch.optim.Adam([P], lr=side * lr_frac)
    cellw = side / G; cap = cellw * cellw
    clx = torch.arange(G, dtype=torch.float64) * cellw + org[0]
    cly = torch.arange(G, dtype=torch.float64) * cellw + org[1]
    lam = lam0
    for _ in range(iters):
        opt.zero_grad()
        wl = (P[ei] - P[ej]).abs().sum() + (P[pb] - pinxy[pp]).abs().sum()
        lo = P[:, 0:1] - Dt[:, 0:1] / 2; hi = P[:, 0:1] + Dt[:, 0:1] / 2
        ox = (torch.minimum(hi, clx + cellw) - torch.maximum(lo, clx)).clamp(min=0)
        lo = P[:, 1:2] - Dt[:, 1:2] / 2; hi = P[:, 1:2] + Dt[:, 1:2] / 2
        oy = (torch.minimum(hi, cly + cellw) - torch.maximum(lo, cly)).clamp(min=0)
        dens = torch.einsum('bi,bj->ij', ox, oy)
        (wl + lam * ((dens - cap).clamp(min=0) ** 2).sum()).backward()
        if fixed.any():
            P.grad[fixed] = 0
        opt.step()
        if fixed.any():
            with torch.no_grad():
                P[fixed] = P0t[fixed]
        lam *= lam_grow
    Pf = P.detach().numpy()
    return Pf[:, 0], Pf[:, 1]
