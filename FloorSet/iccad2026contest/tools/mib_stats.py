import sys, glob, random
from pathlib import Path
import numpy as np, torch
C=Path('/home/b12901044/iccad-2026-C/FloorSet/iccad2026contest')
sys.path.insert(0,str(C)); sys.path.insert(0,str(C.parent))
from floordiff.data import load_shard, case_from_shard, load_validation_case
random.seed(0)
files=sorted(glob.glob('/home/b12901044/iccad-2026-C/FloorSet/floorset_lite/worker_*/layouts_*.th'))
rows=[]
for f in random.sample(files,30):
    sh=load_shard(f)
    for li in range(0,112,8):
        c=case_from_shard(sh,li); a=c['area'].numpy(); cons=c['cons'].numpy()
        n=int((c['area']!=-1).sum()); a=a[:n]; cons=cons[:n]
        worst=0.0; nviol=0; gsz=0
        for g in np.unique(cons[:,2]):
            if g==0: continue
            m=np.nonzero(cons[:,2]==g)[0]; gsz=max(gsz,len(m))
            am=a[m]; rep=am[0]
            rel=np.abs(am-rep)/am
            worst=max(worst,float(rel.max()))
            nviol+=int((rel>0.01).sum())          # blocks old code would break
        rows.append((n,worst,nviol,gsz))
w=np.array([r[1] for r in rows]); v=np.array([r[2] for r in rows]); g=np.array([r[3] for r in rows])
print(f'training layouts sampled: {len(rows)}')
print(f'  MIB group size: mean {g.mean():.2f} max {g.max()}')
print(f'  worst intra-group rel area error vs representative: mean {w.mean():.3f} p50 {np.percentile(w,50):.3f} p90 {np.percentile(w,90):.3f} max {w.max():.3f}')
print(f'  layouts where blind tying breaks >=1 block area (old beta code): {int((v>0).sum())}/{len(rows)} = {100*(v>0).mean():.0f}%')
print(f'  blocks broken per such layout: mean {v[v>0].mean():.2f} max {v.max()}')
