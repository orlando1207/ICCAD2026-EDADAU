"""Dump top-k diffusion candidates for every case of a FloorSet-format kit, so
legalizer A/Bs on that kit become CPU-only and deterministic.

    python tools/dump_kit_preds.py <kit_path> <out.json> [--topk 24] [--seeds 32]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

CONTEST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTEST))
sys.path.insert(0, str(CONTEST.parent))

from floordiff.data import featurize, decode
from floordiff.sample import load_checkpoint, rank_cost
from tools.kitcase import kit_case, dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kit')
    ap.add_argument('out')
    ap.add_argument('--ckpt', default='floordiff/checkpoints/myrun/last.pt')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seeds', type=int, default=32)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--topk', type=int, default=24)
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    diffusion = load_checkpoint(a.ckpt, dev)
    ds, _ = dataset(a.kit)
    out = {'kit': str(Path(a.kit).resolve()), 'cases': {},
           'meta': {'seeds': a.seeds, 'steps': a.steps, 'topk': a.topk}}
    for idx in range(len(ds)):
        case, base, n = kit_case(a.kit, idx)
        tensors, meta = featurize(case)
        b = {k: v.unsqueeze(0).expand(a.seeds, *v.shape).contiguous().to(dev)
             for k, v in tensors.items() if k != 'z0'}
        t0 = time.time()
        with torch.no_grad():
            z = diffusion.sample(b['feat'], b['pair'], b['gfeat'], b['z_known'],
                                 b['freeze'], steps=a.steps, seed=0)
        cands = [decode(z[k].float().cpu(), meta) for k in range(a.seeds)]
        costs = [rank_cost(c, case) for c in cands]
        h = torch.tensor([c[0] for c in costs])
        ar = torch.tensor([c[1] for c in costs])
        o = torch.tensor([c[2] for c in costs])
        score = h / h.min().clamp(min=1e-8) + ar / ar.min().clamp(min=1e-8) + 5.0 * o
        order = score.argsort().tolist()
        out['cases'][str(idx)] = {
            'n': n,
            'candidates': [cands[k].tolist() for k in order[:a.topk]],
            'runtime_s': time.time() - t0}
        print(f'case {idx:4d} n={n:3d}  {time.time()-t0:.2f}s', flush=True)
    Path(a.out).write_text(json.dumps(out))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
