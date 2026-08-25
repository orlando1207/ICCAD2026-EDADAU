"""Per-case hard-constraint failure taxonomy on a FloorSet-format data root.

Usage (from FloorSet/iccad2026contest):
    python diag_testkit.py <data_path> [--opt floordiff_optimizer.py] [--ids 0,1,2]
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CONTEST = Path('/home/b12901044/iccad-2026-C/FloorSet/iccad2026contest')
sys.path.insert(0, str(CONTEST))
sys.path.insert(0, str(CONTEST.parent))

import iccad2026_evaluate as ev
from lite_dataset_test import FloorplanDatasetLiteTest


def load_opt(path):
    spec = importlib.util.spec_from_file_location('optimizer_module', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for name in dir(m):
        o = getattr(m, name)
        if isinstance(o, type) and issubclass(o, ev.FloorplanOptimizer) \
                and o.__name__ != 'FloorplanOptimizer':
            return o(verbose=False)
    raise SystemExit('no optimizer class')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_path')
    ap.add_argument('--opt', default=str(CONTEST / 'floordiff_optimizer.py'))
    ap.add_argument('--ids', default=None)
    ap.add_argument('--out', default='diag.json')
    a = ap.parse_args()

    ds = FloorplanDatasetLiteTest(a.data_path)
    man = Path(a.data_path) / 'manifest.json'
    names = {}
    if man.exists():
        for c in json.load(open(man))['cases']:
            names[c['test_id']] = c['name']
    ids = ([int(x) for x in a.ids.split(',')] if a.ids
           else list(range(len(ds))))
    opt = load_opt(a.opt)
    rows = []
    for idx in ids:
        s = ds[idx]
        area, b2b, p2b, pins, cons = s['input']
        polys, metrics = s['label']
        n = int((area != -1).sum().item())
        base, tgt = ev.ContestEvaluator(a.data_path, verbose=False)._extract_baseline(
            idx, s['label'], b2b, p2b, pins, n)
        otp = torch.full((n, 4), -1.0)
        nc = cons.shape[1]
        for i in range(n):
            if nc > 1 and cons[i, 1] != 0:
                otp[i] = torch.tensor(list(tgt[i]))
            elif nc > 0 and cons[i, 0] != 0:
                otp[i, 2], otp[i, 3] = tgt[i][2], tgt[i][3]
        t0 = time.time()
        pos = opt.solve(n, area, b2b, p2b, pins, cons, otp)
        rt = time.time() - t0
        fp = {i for i in range(n) if cons[i, 0] != 0 or cons[i, 1] != 0}
        ovl = ev.check_overlap(pos)
        arv = ev.check_area_tolerance(pos, area, skip_indices=fp)
        dim = ev.check_dimension_hard_constraints(pos, tgt, cons, n)
        # attribute dimension failures: fixed vs preplaced-dims vs preplaced-pos
        dfix = dpre_dim = dpre_pos = 0
        for i in range(n):
            isf = cons[i, 0] != 0
            isp = cons[i, 1] != 0
            if not (isf or isp):
                continue
            px, py, pw, ph = pos[i]
            tx, ty, tw, th = tgt[i]
            if abs(pw - tw) > 1e-4 or abs(ph - th) > 1e-4:
                dpre_dim += int(bool(isp)); dfix += int(not isp)
            elif isp and (abs(px - tx) > 1e-4 or abs(py - ty) > 1e-4):
                dpre_pos += 1
        # which pairs overlap, and whether an immutable block is involved
        ovl_pairs, ovl_imm = [], 0
        maxpen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, w1, h1 = pos[i]
                x2, y2, w2, h2 = pos[j]
                ox = min(x1 + w1, x2 + w2) - max(x1, x2)
                oy = min(y1 + h1, y2 + h2) - max(y1, y2)
                if ox > 1e-6 and oy > 1e-6:
                    ovl_pairs.append((i, j))
                    maxpen = max(maxpen, min(ox, oy))
                    if i in fp or j in fp:
                        ovl_imm += 1
        rows.append(dict(
            id=idx, name=names.get(idx, ''), n=n, runtime=rt,
            n_pre=int((cons[:n, 1] != 0).sum()), n_fix=int((cons[:n, 0] != 0).sum()),
            n_bnd=int((cons[:n, 4] != 0).sum()),
            n_mib=int(cons[:n, 2].max()), n_clu=int(cons[:n, 3].max()),
            feasible=(ovl == 0 and arv == 0 and dim == 0),
            overlap=ovl, overlap_immutable=ovl_imm, max_pen=maxpen,
            area_bad=arv, dim_bad=dim,
            dim_fixed=dfix, dim_pre_dim=dpre_dim, dim_pre_pos=dpre_pos,
            pairs=ovl_pairs[:12]))
        r = rows[-1]
        print(f"{idx:3d} n={r['n']:3d} pre={r['n_pre']:3d} fix={r['n_fix']:3d} "
              f"bnd={r['n_bnd']:3d} mib={r['n_mib']:2d} | "
              f"{'OK ' if r['feasible'] else 'BAD'} ovl={ovl}({ovl_imm} imm) "
              f"pen={maxpen:.3g} area={arv} dim={dim}"
              f"(f{dfix}/pd{dpre_dim}/pp{dpre_pos}) t={rt:.2f}s  {r['name']}",
              flush=True)
    json.dump(rows, open(a.out, 'w'), indent=1)
    nf = sum(1 for r in rows if r['feasible'])
    print(f"\nfeasible {nf}/{len(rows)}")
    for k in ('overlap', 'area_bad', 'dim_bad'):
        bad = [r['id'] for r in rows if r[k]]
        print(f"  {k}: {len(bad)} cases {bad}")


if __name__ == '__main__':
    main()
