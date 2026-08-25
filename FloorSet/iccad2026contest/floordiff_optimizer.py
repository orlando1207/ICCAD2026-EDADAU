"""FloorDiff contest optimizer: diffusion prediction + EGL legalization.

Drop-in `FloorplanOptimizer` for the official evaluator:

    python iccad2026_evaluate.py --validate floordiff_optimizer.py
    python iccad2026_evaluate.py --evaluate floordiff_optimizer.py [--test-id N]

Pipeline per case (docs/superpowers/specs/2026-07-20-egl-legalizer-design.md):
  featurize -> batched diffusion sampling (N seeds, DDIM) -> decode top-k
  -> EGL legalization (ePlace-style gradient cleanup + constraint-graph
  minimal-movement assignment) per candidate, all candidates CONCURRENTLY
  -> best by official-cost proxy.

Both the model checkpoint and the worker pool are built in __init__, which the
evaluator does not count toward per-case runtime. Knobs via environment variables:
  FLOORDIFF_CKPT     checkpoint path  (default floordiff/checkpoints/myrun/last.pt)
  FLOORDIFF_DEVICE   torch device     (default cuda if available, else cpu)
  FLOORDIFF_SEEDS    sampled seeds    (default 32)
  FLOORDIFF_TOPK     legalized seeds  (default = worker count)
  FLOORDIFF_STEPS    DDIM steps       (default 50)
  FLOORDIFF_WORKERS  pool size        (default min(24, cpu_count//2))
  FLOORDIFF_SERIAL   set to 1 to disable the pool (sequential fallback)
  FLOORDIFF_DEADLINE per-case legalization deadline in seconds (default off;
                     enabling it bounds wall-clock but is timing-dependent)
  FLOORDIFF_CFG      JSON overriding legalizer DEFAULT_CFG keys, for A/B runs,
                     e.g. FLOORDIFF_CFG='{"perp_align": false}'
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
from floordiff.legalizer import (DEFAULT_CFG, legalize_best_of,   # noqa: E402
                                 guaranteed_construction,
                                 hard_feasibility)
from floordiff.parallel import (legalize_parallel, make_pool,   # noqa: E402
                                resolve_workers)
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

        # --- worker pool first: spawn is immune to the CUDA context, but build
        # --- it before the model anyway so a fork-based fallback stays safe
        self.serial = os.environ.get('FLOORDIFF_SERIAL', '') not in ('', '0')
        self.pool, self.workers = (None, 1)
        if not self.serial:
            try:
                self.pool, self.workers = make_pool()
            except Exception as exc:            # no /dev/shm, no fork, etc.
                print(f'[floordiff] pool unavailable ({exc}); running sequentially')
                self.serial = True
        dl = os.environ.get('FLOORDIFF_DEADLINE')
        self.deadline = float(dl) if dl else None

        self.cfg = dict(DEFAULT_CFG)
        if os.environ.get('FLOORDIFF_CFG'):
            import json
            self.cfg.update(json.loads(os.environ['FLOORDIFF_CFG']))

        ckpt = os.environ.get(
            'FLOORDIFF_CKPT', str(_HERE / 'floordiff/checkpoints/myrun/last.pt'))
        dev = os.environ.get(
            'FLOORDIFF_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(dev)
        self.diffusion = load_checkpoint(ckpt, self.device)
        self.n_seeds = int(os.environ.get('FLOORDIFF_SEEDS', 32))
        # with the pool, extra candidates cost ~nothing in wall-clock (bounded by
        # the slowest one), so default top-k to the pool width
        self.topk = int(os.environ.get(
            'FLOORDIFF_TOPK', 6 if self.serial else self.workers))
        self.topk = max(1, min(self.topk, self.n_seeds))
        self.steps = int(os.environ.get('FLOORDIFF_STEPS', 50))
        if verbose:
            print(f'[floordiff] workers={self.workers} topk={self.topk} '
                  f'seeds={self.n_seeds} steps={self.steps} '
                  f'deadline={self.deadline} device={self.device}')
        self._warmup()

    def _warmup(self):
        """Pay every first-call cost here, where the evaluator does not bill it:
        CUDA context + kernel selection for both a small and a large token count,
        and one real legalization in every worker (scipy/shapely import, JIT).
        Without this the first scored case carries ~4 s of setup."""
        try:
            for n in (24, 120):
                case = {
                    'area': torch.full((n,), 100.0),
                    'cons': torch.zeros(n, 5, dtype=torch.long),
                    'b2b': torch.tensor([[float(i), float(i + 1), 1.0]
                                         for i in range(n - 1)]),
                    'p2b': torch.zeros(0, 3), 'pins': torch.zeros(0, 2),
                    'gt': None, 'metrics': None, 'target': torch.zeros(n, 4),
                }
                tensors, meta = featurize(case)
                b = {k: v.unsqueeze(0).expand(self.n_seeds, *v.shape).contiguous()
                     .to(self.device)
                     for k, v in tensors.items() if k != 'z0'}
                with torch.no_grad():
                    z = self.diffusion.sample(b['feat'], b['pair'], b['gfeat'],
                                              b['z_known'], b['freeze'],
                                              steps=4, seed=0)
                cands = [decode(z[k].float().cpu(), meta).double()
                         for k in range(min(self.topk, self.n_seeds))]
                if self.pool is not None:
                    legalize_parallel(self.pool, cands, case, self.cfg)
        except Exception as exc:
            print(f'[floordiff] warmup skipped: {exc}')

    def solve(self, block_count, area_targets, b2b_connectivity,
              p2b_connectivity, pins_pos, constraints, target_positions=None):
        """Never returns an infeasible placement: `legalize_case` carries its
        own hard gate plus a shelf-construction floor, and anything that escapes
        that (an exception anywhere in sampling or legalization) is caught here
        and answered with the same construction."""
        try:
            return self._solve(block_count, area_targets, b2b_connectivity,
                               p2b_connectivity, pins_pos, constraints,
                               target_positions)
        except Exception as exc:                       # never fail a case
            print(f'[floordiff] solve failed ({exc!r}); using construction')
            return self._rescue(block_count, area_targets, b2b_connectivity,
                                p2b_connectivity, pins_pos, constraints,
                                target_positions)

    def _case(self, block_count, area_targets, b2b_connectivity,
              p2b_connectivity, pins_pos, constraints, target_positions):
        n = int(block_count)
        return {
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

    def _rescue(self, block_count, area_targets, b2b_connectivity,
                p2b_connectivity, pins_pos, constraints, target_positions):
        """Model-free, search-free, guaranteed-feasible answer of last resort."""
        n = int(block_count)
        case = self._case(block_count, area_targets, b2b_connectivity,
                          p2b_connectivity, pins_pos, constraints,
                          target_positions)
        import numpy as np
        area = case['area'].numpy().astype('float64')
        side = np.sqrt(area)
        pred = np.stack([np.zeros(n), np.zeros(n), side, side], axis=1)
        sol = guaranteed_construction(pred, case, self.cfg)
        return [tuple(map(float, row)) for row in sol]

    def _solve(self, block_count, area_targets, b2b_connectivity,
               p2b_connectivity, pins_pos, constraints, target_positions=None):
        n = int(block_count)
        case = self._case(block_count, area_targets, b2b_connectivity,
                          p2b_connectivity, pins_pos, constraints,
                          target_positions)

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
        if self.pool is None:
            sol, info = legalize_best_of(top, case, self.cfg)
        else:
            sol, info = legalize_parallel(self.pool, top, case, self.cfg,
                                          deadline_s=self.deadline)
            if sol is None:                     # every worker failed: rescue
                sol, info = legalize_best_of(top[:2], case, self.cfg)
        if sol is None or not hard_feasibility(sol.numpy(), case)['feasible']:
            # legalize_case has its own floor, so this is unreachable in
            # principle; keep it anyway -- an infeasible return costs 10.0.
            print('[floordiff] no feasible candidate; using construction')
            return [tuple(map(float, row)) for row in
                    guaranteed_construction(top[0].numpy(), case, self.cfg)]
        if self.verbose:
            print(f"  n={n} proxy={info['proxy_cost']:.4f} "
                  f"seed={info.get('seed_rank')} "
                  f"cands={info.get('n_done', '?')}/{info.get('n_cands', '?')} "
                  f"t={info['runtime_s']:.2f}s")
        return [tuple(map(float, row)) for row in sol]
