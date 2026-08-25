"""Constraint-density distribution: official validation (100) vs training shards."""
import sys, glob, random
from pathlib import Path
import numpy as np, torch

CONTEST = Path('/home/b12901044/iccad-2026-C/FloorSet/iccad2026contest')
sys.path.insert(0, str(CONTEST)); sys.path.insert(0, str(CONTEST.parent))
from floordiff.data import load_validation_case, load_shard, case_from_shard

def stats(cons, area):
    cons = cons.numpy() if torch.is_tensor(cons) else cons
    area = area.numpy() if torch.is_tensor(area) else area
    n = len(cons)
    pre = int((cons[:, 1] != 0).sum()); fix = int((cons[:, 0] != 0).sum())
    bnd = int((cons[:, 4] != 0).sum())
    # MIB groups whose members have heterogeneous target areas (>1% apart)
    het = 0; ngrp = 0
    for g in np.unique(cons[:, 2]):
        if g == 0: continue
        ngrp += 1
        m = np.nonzero(cons[:, 2] == g)[0]
        a = area[m]
        if a.max() / max(a.min(), 1e-9) > 1.0201: het += 1
    nclu = len([g for g in np.unique(cons[:, 3]) if g > 0])
    return dict(n=n, pre=pre, fix=fix, bnd=bnd, mib=ngrp, mib_het=het, clu=nclu,
                pre_f=pre / n, fix_f=fix / n, bnd_f=bnd / n)

def summarize(tag, rows):
    def q(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return f"mean {v.mean():6.3f}  p50 {np.percentile(v,50):6.3f}  p90 {np.percentile(v,90):6.3f}  max {v.max():6.3f}"
    print(f"\n--- {tag} ({len(rows)} cases) ---")
    for k in ('pre_f', 'fix_f', 'bnd_f', 'mib', 'mib_het', 'clu'):
        print(f"  {k:8s} {q(k)}")
    for thr in (0.05, 0.15, 0.25, 0.40):
        c = sum(1 for r in rows if r['pre_f'] > thr)
        print(f"  preplaced fraction > {thr:.0%}: {c}/{len(rows)} ({100*c/len(rows):.0f}%)")
    c = sum(1 for r in rows if r['mib_het'] > 0)
    print(f"  has heterogeneous-area MIB group: {c}/{len(rows)} ({100*c/len(rows):.0f}%)")

val = []
for nb in range(21, 121):
    try:
        c = load_validation_case(nb)
    except Exception as e:
        continue
    val.append(stats(c['cons'], c['area']))
summarize('official validation LiteTensorDataTest', val)

files = sorted(glob.glob('/home/b12901044/iccad-2026-C/FloorSet/floorset_lite/worker_*/layouts_*.th'))
print(f"\n{len(files)} training shards on disk")
random.seed(0)
pick = random.sample(files, min(40, len(files)))
tr = []
for f in pick:
    try:
        sh = load_shard(f)
    except Exception:
        continue
    for li in range(0, 112, 16):
        c = case_from_shard(sh, li)
        a = c['area']
        n = int((a != -1).sum())
        tr.append(stats(c['cons'][:n], a[:n]))
summarize('training shards (sampled)', tr)
