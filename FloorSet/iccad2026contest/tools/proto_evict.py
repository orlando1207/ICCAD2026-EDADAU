"""Prototype 'targeted eviction': keep the current legalized layout, move ONLY
the blocks that overlap an immutable anchor into guaranteed-free space.

Two eviction targets, tried in order, best-by-cost wins:
  (a) nearest free hole found by scanning candidate positions on the "corner"
      grid of the current layout (obstacle-aware, exact overlap test)
  (b) the strip above the bbox top (always available -> completeness)

Scored with the official formula (runtime-neutral).
"""
import sys, json
from pathlib import Path
import numpy as np, torch

C = Path('/home/b12901044/iccad-2026-C/FloorSet/iccad2026contest')
sys.path.insert(0, str(C)); sys.path.insert(0, str(C.parent))
import iccad2026_evaluate as ev
from lite_dataset_test import FloorplanDatasetLiteTest
from floordiff.data import featurize, decode
from floordiff import legalizer as lg
from floordiff.sample import load_checkpoint, rank_cost


def offenders(sol, case):
    """Soft (movable) blocks involved in an overlap; prefer moving the movable
    side of each offending pair."""
    hard = lg.hard_feasibility(sol, case)
    cons = case['cons'].numpy()
    pre = cons[:, 1] > 0
    bad = set()
    for (i, j) in hard['overlap_pairs']:
        if pre[i] and pre[j]:
            continue                      # unsolvable by moving (never seen)
        cand = [k for k in (i, j) if not pre[k]]
        # move the smaller-area one: less quality damage
        cand.sort(key=lambda k: sol[k, 2] * sol[k, 3])
        bad.add(int(cand[0]))
    return sorted(bad), hard


def free_at(sol, keep, i, x, y, eps=1e-6):
    w, h = sol[i, 2], sol[i, 3]
    for k in keep:
        ox = min(x + w, sol[k, 0] + sol[k, 2]) - max(x, sol[k, 0])
        oy = min(y + h, sol[k, 1] + sol[k, 3]) - max(y, sol[k, 1])
        if ox > eps and oy > eps:
            return False
    return True


def evict(sol, case, hpwl_base, area_base, n_soft, nbrs):
    sol = sol.copy()
    bad, hard = offenders(sol, case)
    if not bad:
        return sol, 0
    n = len(sol)
    keep = [k for k in range(n) if k not in set(bad)]
    # candidate corner grid: block corners of the kept layout
    xs = sorted({float(sol[k, 0]) for k in keep} |
                {float(sol[k, 0] + sol[k, 2]) for k in keep})
    ys = sorted({float(sol[k, 1]) for k in keep} |
                {float(sol[k, 1] + sol[k, 3]) for k in keep})
    for i in bad:
        cx0 = sol[i, 0] + sol[i, 2] / 2
        cy0 = sol[i, 1] + sol[i, 3] / 2
        best, bpos = None, None
        for x in xs:
            for y in ys:
                if not free_at(sol, keep, i, x, y):
                    continue
                d = abs(x + sol[i, 2] / 2 - cx0) + abs(y + sol[i, 3] / 2 - cy0)
                if best is None or d < best:
                    best, bpos = d, (x, y)
        if bpos is None:                       # completeness: strip above
            y_top = float((sol[keep, 1] + sol[keep, 3]).max())
            bpos = (float(sol[keep, 0].min()), y_top)
        sol[i, 0], sol[i, 1] = bpos
        keep.append(i)
    return sol, len(bad)


def main():
    DP = '/home/b12901044/iccad-2026-C/cadc1111/floorset_testkit'
    ids = [int(x) for x in sys.argv[1].split(',')]
    ds = FloorplanDatasetLiteTest(DP)
    diff = load_checkpoint(str(C / 'floordiff/checkpoints/myrun/last.pt'),
                           torch.device('cuda:0'))
    evl = ev.ContestEvaluator(DP, verbose=False)
    rows = []
    for idx in ids:
        s = ds[idx]
        area, b2b, p2b, pins, cons = s['input']
        n = int((area != -1).sum().item())
        base, tgt = evl._extract_baseline(idx, s['label'], b2b, p2b, pins, n)
        otp = torch.full((n, 4), -1.0)
        for i in range(n):
            if cons[i, 1] != 0: otp[i] = torch.tensor(list(tgt[i]))
            elif cons[i, 0] != 0: otp[i, 2], otp[i, 3] = tgt[i][2], tgt[i][3]
        cl = lambda t: t[(t != -1).all(dim=1)]
        case = {'area': area[:n].float(), 'cons': cons[:n].long(),
                'b2b': cl(b2b).float(), 'p2b': cl(p2b).float(),
                'pins': cl(pins).float(), 'gt': None, 'metrics': None,
                'target': otp.double()}
        tensors, meta = featurize(case)
        NS = 16
        b = {k: v.unsqueeze(0).expand(NS, *v.shape).contiguous().cuda()
             for k, v in tensors.items() if k != 'z0'}
        with torch.no_grad():
            z = diff.sample(b['feat'], b['pair'], b['gfeat'], b['z_known'],
                            b['freeze'], steps=50, seed=0)
        cands = [decode(z[k].float().cpu(), meta) for k in range(NS)]
        cst = [rank_cost(c, case) for c in cands]
        h = torch.tensor([c[0] for c in cst]); a = torch.tensor([c[1] for c in cst])
        o = torch.tensor([c[2] for c in cst])
        sc = h / h.min().clamp(min=1e-8) + a / a.min().clamp(min=1e-8) + 5.0 * o
        order = sc.argsort().tolist()
        hb, ab = base['hpwl_baseline'], base['area_baseline']
        n_soft = lg._n_soft_norm(cons[:n].numpy())
        nbrs = lg._nbr_lists(case, n)
        best = None
        for rank in range(6):
            sol0, info = lg.legalize_case(cands[order[rank]].double(), case)
            s0 = sol0.numpy()
            base_cost = lg.proxy_cost(s0, case, hb, ab, n_soft)
            s1, nmoved = evict(s0, case, hb, ab, n_soft, nbrs)
            hard1 = lg.hard_feasibility(s1, case)
            c1 = lg.proxy_cost(s1, case, hb, ab, n_soft)
            rec = dict(rank=rank, base_cost=base_cost, base_feas=info['hard']['feasible'],
                       moved=nmoved, evict_feas=hard1['feasible'], evict_cost=c1)
            if best is None or (hard1['feasible'], -c1) > (best['evict_feas'], -best['evict_cost']):
                best = rec
        t = torch.tensor(s1, dtype=torch.float64)
        print(f"{idx:3d} n={n:3d} pre={int((cons[:n,1]!=0).sum()):3d} | "
              f"before: feas={best['base_feas']} cost={best['base_cost']:.3f} | "
              f"evicted {best['moved']} blocks -> feas={best['evict_feas']} "
              f"cost={best['evict_cost']:.3f}", flush=True)
        rows.append(dict(id=idx, n=n, **best))
    json.dump(rows, open(sys.argv[2] if len(sys.argv) > 2 else 'evict.json', 'w'), indent=1)
    ok = sum(r['evict_feas'] for r in rows)
    cc = [r['evict_cost'] for r in rows]
    mv = [r['moved'] for r in rows]
    print(f"\nfeasible after eviction: {ok}/{len(rows)}   "
          f"cost mean {np.mean(cc):.3f} median {np.median(cc):.3f} max {max(cc):.3f}   "
          f"blocks moved mean {np.mean(mv):.1f} max {max(mv)}")


if __name__ == '__main__':
    main()
