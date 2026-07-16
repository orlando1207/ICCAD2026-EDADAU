"""Sampling: raw predictions on the validation cases (best-of-N, ranked by a
cost proxy — selection, not enforcement).

Run from iccad2026contest/:
  python -m floordiff.sample --ckpt floordiff/checkpoints/base/last.pt \
      --n-seeds 16 --steps 50 --out floordiff/out/preds.json
  python -m floordiff.evaluate --pred floordiff/out/preds.json
  python -m floordiff.visualize --pred floordiff/out/preds.json --cases 60,100,120
"""

import argparse
import json
import time
from pathlib import Path

import torch

from .data import VALIDATION_NS, decode, featurize, load_validation_case
from .diffusion import FloorDiffusion
from .evaluate import bbox_area, overlap_ratio, weighted_hpwl
from .model import FloorDiffNet, ModelConfig


def load_checkpoint(path, device, use_ema=True):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = FloorDiffNet(ModelConfig(**ck['config'])).to(device)
    model.load_state_dict(ck['ema'] if use_ema else ck['model'])
    model.eval()
    return FloorDiffusion(model, timesteps=ck['timesteps']).to(device)


def rank_cost(xywh, case):
    """Selection proxy across seeds of one case (relative scale; lower = better)."""
    return (weighted_hpwl(xywh, case), bbox_area(xywh), overlap_ratio(xywh))


@torch.no_grad()
def predict_case(diffusion, case, n_seeds, steps, device, seed0=0):
    tensors, meta = featurize(case)
    b = {k: v.unsqueeze(0).expand(n_seeds, *v.shape).contiguous().to(device)
         for k, v in tensors.items() if k != 'z0'}
    z = diffusion.sample(b['feat'], b['pair'], b['gfeat'], b['z_known'], b['freeze'],
                         steps=steps, seed=seed0)
    cands = [decode(z[k].float().cpu(), meta) for k in range(n_seeds)]
    costs = [rank_cost(c, case) for c in cands]
    h = torch.tensor([c[0] for c in costs])
    a = torch.tensor([c[1] for c in costs])
    o = torch.tensor([c[2] for c in costs])
    score = h / h.min().clamp(min=1e-8) + a / a.min().clamp(min=1e-8) + 5.0 * o
    best = int(score.argmin())
    return cands[best], best, score.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--cases', type=str, default='',
                    help='comma-separated n values (default: all 21..120)')
    ap.add_argument('--n-seeds', type=int, default=16)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--out', type=str, default='floordiff/out/preds.json')
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--no-ema', action='store_true')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    diffusion = load_checkpoint(args.ckpt, device, use_ema=not args.no_ema)
    ns = [int(x) for x in args.cases.split(',')] if args.cases else VALIDATION_NS

    out = {'cases': {}, 'meta': {'ckpt': args.ckpt, 'n_seeds': args.n_seeds,
                                 'steps': args.steps}}
    for n in ns:
        case = load_validation_case(n)
        t0 = time.time()
        xywh, best, scores = predict_case(diffusion, case, args.n_seeds,
                                          args.steps, device)
        dt = time.time() - t0
        out['cases'][str(n)] = {'n': n, 'positions': xywh.tolist(),
                                'best_seed': best, 'seed_scores': scores,
                                'runtime_s': dt}
        print(f'case n={n:>3}: {dt:.2f}s  best seed {best}  '
              f'overlap {overlap_ratio(xywh) * 100:.2f}%')

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    print(f'wrote {p}')


if __name__ == '__main__':
    main()
