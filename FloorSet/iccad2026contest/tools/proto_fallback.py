"""Prototype guaranteed-feasible fallbacks, scored with the official formula.

Two constructions, both overlap-free BY CONSTRUCTION with preplaced obstacles:
  A) shelf-above:  movable blocks shelf-packed strictly above every obstacle top
  B) free-rect:    maximal-free-rectangle first-fit over the obstacle field,
                   overflow goes to the shelf strip above (so still total)

Both stamp preplaced (x,y,w,h) and fixed (w,h) exactly and give soft blocks
area = 0.995 * target (inside the 1% band).
Scored with legalizer.proxy_cost against evaluator-equivalent baselines.
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

AREA_SCALE = 0.995
ASPECT_CAP = 3.0


def _dims(pred, case, pre, fix, gt):
    n = len(pred)
    w = pred[:, 2].copy(); h = pred[:, 3].copy()
    area = case['area'].numpy().astype(np.float64)
    soft = (~pre) & (~fix)
    # legal soft dims: keep predicted aspect (capped), area = 0.995 * target
    ar = np.clip(w[soft] / np.maximum(h[soft], 1e-9), 1.0 / ASPECT_CAP, ASPECT_CAP)
    a = area[soft] * AREA_SCALE
    w[soft] = np.sqrt(a * ar); h[soft] = a / w[soft]
    w[fix], h[fix] = gt[fix, 2], gt[fix, 3]
    w[pre], h[pre] = gt[pre, 2], gt[pre, 3]
    return w, h


def shelf_above(pred, case, gt, W_mult=1.0):
    cons = case['cons'].numpy()
    pre = cons[:, 1] > 0; fix = cons[:, 0] > 0
    n = len(pred)
    w, h = _dims(pred, case, pre, fix, gt)
    sol = np.zeros((n, 4)); sol[:, 2] = w; sol[:, 3] = h
    sol[pre, 0] = gt[pre, 0]; sol[pre, 1] = gt[pre, 1]
    mov = np.nonzero(~pre)[0]
    if pre.any():
        x_org = float(gt[pre, 0].min())
        y_base = float((gt[pre, 1] + gt[pre, 3]).max())
        obs_w = float((gt[pre, 0] + gt[pre, 2]).max()) - x_org
    else:
        x_org, y_base, obs_w = 0.0, 0.0, 0.0
    A = float((w[mov] * h[mov]).sum())
    W = max(obs_w * W_mult, np.sqrt(A) * W_mult, float(w[mov].max()) if len(mov) else 1.0)
    # keep predicted topology: rows by predicted y, blocks left-to-right by x
    order = sorted(mov.tolist(), key=lambda i: (pred[i, 1], pred[i, 0]))
    cx, cy, rowh = x_org, y_base, 0.0
    for i in order:
        if cx > x_org and cx + w[i] > x_org + W:
            cx = x_org; cy += rowh; rowh = 0.0
        sol[i, 0], sol[i, 1] = cx, cy
        cx += w[i]; rowh = max(rowh, h[i])
    return sol


def free_rect(pred, case, gt):
    """First-fit into maximal free rectangles of the obstacle field; overflow
    to the shelf strip above.  Guillotine-style split, so no overlap ever."""
    cons = case['cons'].numpy()
    pre = cons[:, 1] > 0; fix = cons[:, 0] > 0
    n = len(pred)
    w, h = _dims(pred, case, pre, fix, gt)
    sol = np.zeros((n, 4)); sol[:, 2] = w; sol[:, 3] = h
    sol[pre, 0] = gt[pre, 0]; sol[pre, 1] = gt[pre, 1]
    mov = np.nonzero(~pre)[0]
    A_all = float((w * h).sum())
    side = np.sqrt(A_all) * 1.05
    if pre.any():
        x0 = float(gt[pre, 0].min()); y0 = float(gt[pre, 1].min())
        x1 = max(float((gt[pre, 0] + gt[pre, 2]).max()), x0 + side)
        y1 = max(float((gt[pre, 1] + gt[pre, 3]).max()), y0 + side)
    else:
        x0 = y0 = 0.0; x1 = y1 = side
    free = [(x0, y0, x1 - x0, y1 - y0)]
    # carve out every obstacle (guillotine subtract)
    for i in np.nonzero(pre)[0]:
        ox, oy, ow, oh = gt[i, 0], gt[i, 1], gt[i, 2], gt[i, 3]
        out = []
        for (fx, fy, fw, fh) in free:
            if ox >= fx + fw or ox + ow <= fx or oy >= fy + fh or oy + oh <= fy:
                out.append((fx, fy, fw, fh)); continue
            if oy > fy: out.append((fx, fy, fw, oy - fy))
            if oy + oh < fy + fh: out.append((fx, oy + oh, fw, fy + fh - oy - oh))
            if ox > fx: out.append((fx, max(fy, oy), ox - fx,
                                    min(fy + fh, oy + oh) - max(fy, oy)))
            if ox + ow < fx + fw:
                out.append((ox + ow, max(fy, oy), fx + fw - ox - ow,
                            min(fy + fh, oy + oh) - max(fy, oy)))
        free = [r for r in out if r[2] > 1e-9 and r[3] > 1e-9]
    # place biggest-first into the free rect nearest the prediction
    order = sorted(mov.tolist(), key=lambda i: -(w[i] * h[i]))
    leftover = []
    for i in order:
        best, bk = None, None
        for k, (fx, fy, fw, fh) in enumerate(free):
            if fw + 1e-9 >= w[i] and fh + 1e-9 >= h[i]:
                d = abs(fx - pred[i, 0]) + abs(fy - pred[i, 1])
                if best is None or d < best:
                    best, bk = d, k
        if bk is None:
            leftover.append(i); continue
        fx, fy, fw, fh = free.pop(bk)
        sol[i, 0], sol[i, 1] = fx, fy
        # guillotine split of the remainder
        if fw - w[i] > 1e-9: free.append((fx + w[i], fy, fw - w[i], h[i]))
        if fh - h[i] > 1e-9: free.append((fx, fy + h[i], fw, fh - h[i]))
    if leftover:
        y_top = max(float((sol[:, 1] + sol[:, 3]).max()), y0)
        x_org = float(sol[:, 0].min())
        W = max(np.sqrt(sum(w[i] * h[i] for i in leftover)) * 1.2,
                max(w[i] for i in leftover))
        cx, cy, rowh = x_org, y_top, 0.0
        for i in sorted(leftover, key=lambda i: (pred[i, 1], pred[i, 0])):
            if cx > x_org and cx + w[i] > x_org + W:
                cx = x_org; cy += rowh; rowh = 0.0
            sol[i, 0], sol[i, 1] = cx, cy
            cx += w[i]; rowh = max(rowh, h[i])
    return sol


def main():
    DP = '/home/b12901044/iccad-2026-C/cadc1111/floorset_testkit'
    ids = [int(x) for x in sys.argv[1].split(',')]
    ds = FloorplanDatasetLiteTest(DP)
    diff = load_checkpoint(str(C / 'floordiff/checkpoints/myrun/last.pt'),
                           torch.device('cuda:0'))
    evl = ev.ContestEvaluator(DP, verbose=False)
    out = []
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
        b = {k: v.unsqueeze(0).expand(8, *v.shape).contiguous().cuda()
             for k, v in tensors.items() if k != 'z0'}
        with torch.no_grad():
            z = diff.sample(b['feat'], b['pair'], b['gfeat'], b['z_known'],
                            b['freeze'], steps=50, seed=0)
        pred = decode(z[0].float().cpu(), meta).double().numpy()
        gt_t = np.asarray(otp.double().numpy())
        # preplaced/fixed targets come from otp (cols 2,3 for fixed)
        hb, ab = base['hpwl_baseline'], base['area_baseline']
        n_soft = lg._n_soft_norm(cons[:n].numpy())
        row = {'id': idx, 'n': n,
               'pre': int((cons[:n, 1] != 0).sum())}
        for tag, fn in (('shelf', shelf_above), ('freerect', free_rect)):
            sol = fn(pred, case, gt_t)
            hard = lg.hard_feasibility(sol, case)
            cost = lg.proxy_cost(sol, case, hb, ab, n_soft)
            t = torch.tensor(sol, dtype=torch.float64)
            hg = (float(lg._whpwl(t, case)) - hb) / hb
            ag = (float(lg._bbox_area(t)) - ab) / ab
            vb, vg, vm = lg._violations_official(sol, cons[:n].numpy())
            row[tag] = dict(feasible=hard['feasible'], cost=cost,
                            hpwl_gap=hg, area_gap=ag,
                            vrel=(vb + vg + vm) / max(n_soft, 1),
                            ovl=hard['overlap_violations'],
                            area_bad=hard['area_violations'],
                            dim_bad=hard['dimension_violations'])
        print(f"{idx:3d} n={n:3d} pre={row['pre']:3d} | "
              + "  ".join(f"{t}: feas={row[t]['feasible']} cost={row[t]['cost']:.3f} "
                          f"(hpwl {row[t]['hpwl_gap']:+.2f} area {row[t]['area_gap']:+.2f} "
                          f"vrel {row[t]['vrel']:.3f})" for t in ('shelf', 'freerect')),
              flush=True)
        out.append(row)
    json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else 'proto.json', 'w'), indent=1)
    for t in ('shelf', 'freerect'):
        c = [r[t]['cost'] for r in out]
        nf = sum(r[t]['feasible'] for r in out)
        print(f"{t}: feasible {nf}/{len(out)}  mean cost {np.mean(c):.3f} "
              f"median {np.median(c):.3f} max {max(c):.3f}")


if __name__ == '__main__':
    main()
