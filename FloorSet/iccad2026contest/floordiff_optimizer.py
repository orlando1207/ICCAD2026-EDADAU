"""FloorDiff contest optimizer: diffusion prediction + EGL legalization.

Drop-in `FloorplanOptimizer` for the official evaluator:

    python iccad2026_evaluate.py --validate floordiff_optimizer.py
    python iccad2026_evaluate.py --evaluate floordiff_optimizer.py [--test-id N]

Pipeline per case (docs/superpowers/specs/2026-07-20-egl-legalizer-design.md):
  featurize -> batched diffusion sampling (N seeds, DDIM) -> decode top-k
  -> EGL legalization (ePlace-style gradient cleanup + constraint-graph
  minimal-movement assignment) per candidate -> best by official-cost proxy.

The model checkpoint loads in __init__ (not counted in the evaluator's per-case
runtime). Knobs via environment variables:
  FLOORDIFF_CKPT    checkpoint path   (default floordiff/checkpoints/myrun/last.pt)
  FLOORDIFF_DEVICE  torch device      (default cuda if available, else cpu)
  FLOORDIFF_SEEDS   sampled seeds     (default 32)
  FLOORDIFF_TOPK    legalized seeds   (default 6)
  FLOORDIFF_STEPS   DDIM steps        (default 50)
"""

import os
import sys
from pathlib import Path

import torch

# tiny per-op tensors + many-core hosts = OpenMP thrashing; a modest cap is
# faster in practice for both sampling and the LP solves
torch.set_num_threads(min(8, os.cpu_count() or 8))

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from iccad2026_evaluate import FloorplanOptimizer          # noqa: E402
from floordiff.data import featurize, decode               # noqa: E402
from floordiff.legalizer import legalize_best_of           # noqa: E402
from floordiff.sample import load_checkpoint, rank_cost    # noqa: E402


def _clean(t):
    """Drop -1 padding rows (batch collation artifacts)."""
    if t is None or len(t) == 0:
        return t
    keep = (t != -1).all(dim=1)
    return t[keep]


class MyOptimizer(FloorplanOptimizer):
    """Diffusion prediction + EGL (gradient + constraint-graph) legalizer."""

    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__()
        self.verbose = verbose
        ckpt = os.environ.get(
            'FLOORDIFF_CKPT', str(_HERE / 'floordiff/checkpoints/myrun/last.pt'))
        dev = os.environ.get(
            'FLOORDIFF_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(dev)
        self.diffusion = load_checkpoint(ckpt, self.device)
        self.n_seeds = int(os.environ.get('FLOORDIFF_SEEDS', 32))
        self.topk = int(os.environ.get('FLOORDIFF_TOPK', 6))
        self.steps = int(os.environ.get('FLOORDIFF_STEPS', 50))

    def solve(self, block_count, area_targets, b2b_connectivity,
              p2b_connectivity, pins_pos, constraints, target_positions=None):
        n = int(block_count)
        case = {
            'area': area_targets[:n].float(),
            'cons': constraints[:n].long(),
            'b2b': _clean(b2b_connectivity).float()
                   if b2b_connectivity is not None else torch.zeros(0, 3),
            'p2b': _clean(p2b_connectivity).float()
                   if p2b_connectivity is not None else torch.zeros(0, 3),
            'pins': _clean(pins_pos).float()
                    if pins_pos is not None else torch.zeros(0, 2),
            'gt': None,
            'metrics': None,
            'target': target_positions[:n].double()
                      if target_positions is not None else torch.zeros(n, 4),
        }

        # ---- sample N seeds as one batch, keep top-k by quick proxy
        tensors, meta = featurize(case)
        b = {k: v.unsqueeze(0).expand(self.n_seeds, *v.shape).contiguous()
             .to(self.device) for k, v in tensors.items() if k != 'z0'}
        with torch.no_grad():
            z = self.diffusion.sample(b['feat'], b['pair'], b['gfeat'],
                                      b['z_known'], b['freeze'],
                                      steps=self.steps, seed=0)
        cands = [decode(z[k].float().cpu(), meta) for k in range(self.n_seeds)]
        costs = [rank_cost(c, case) for c in cands]
        h = torch.tensor([c[0] for c in costs])
        a = torch.tensor([c[1] for c in costs])
        o = torch.tensor([c[2] for c in costs])
        score = h / h.min().clamp(min=1e-8) + a / a.min().clamp(min=1e-8) + 5.0 * o
        order = score.argsort().tolist()
        top = [cands[k].double() for k in order[:self.topk]]

        # ---- legalize (best-of-k, official-cost selector with pseudo-baselines)
        sol, _info = legalize_best_of(top, case)
        return [tuple(map(float, row)) for row in sol]
