"""Instrumented single-case legalization: where does the preplaced overlap come from?"""
import sys, json
from pathlib import Path
import numpy as np, torch

CONTEST = Path('/home/b12901044/iccad-2026-C/FloorSet/iccad2026contest')
sys.path.insert(0, str(CONTEST)); sys.path.insert(0, str(CONTEST.parent))
import iccad2026_evaluate as ev
from lite_dataset_test import FloorplanDatasetLiteTest
from floordiff.data import featurize, decode
from floordiff import legalizer as lg
from floordiff.sample import load_checkpoint, rank_cost

DP = '/home/b12901044/iccad-2026-C/cadc1111/floorset_testkit'
ids = [int(x) for x in sys.argv[1].split(',')]
ds = FloorplanDatasetLiteTest(DP)
diff = load_checkpoint(str(CONTEST / 'floordiff/checkpoints/myrun/last.pt'), torch.device('cuda:0'))

for idx in ids:
    s = ds[idx]
    area, b2b, p2b, pins, cons = s['input']
    n = int((area != -1).sum().item())
    base, tgt = ev.ContestEvaluator(DP, verbose=False)._extract_baseline(
        idx, s['label'], b2b, p2b, pins, n)
    otp = torch.full((n, 4), -1.0)
    for i in range(n):
        if cons[i, 1] != 0:
            otp[i] = torch.tensor(list(tgt[i]))
        elif cons[i, 0] != 0:
            otp[i, 2], otp[i, 3] = tgt[i][2], tgt[i][3]

    def clean(t):
        return t[(t != -1).all(dim=1)]
    case = {'area': area[:n].float(), 'cons': cons[:n].long(),
            'b2b': clean(b2b).float(), 'p2b': clean(p2b).float(),
            'pins': clean(pins).float(), 'gt': None, 'metrics': None,
            'target': otp.double()}
    tensors, meta = featurize(case)
    NS = 32
    b = {k: v.unsqueeze(0).expand(NS, *v.shape).contiguous().cuda()
         for k, v in tensors.items() if k != 'z0'}
    with torch.no_grad():
        z = diff.sample(b['feat'], b['pair'], b['gfeat'], b['z_known'], b['freeze'],
                        steps=50, seed=0)
    cands = [decode(z[k].float().cpu(), meta) for k in range(NS)]
    costs = [rank_cost(c, case) for c in cands]
    h = torch.tensor([c[0] for c in costs]); a = torch.tensor([c[1] for c in costs])
    o = torch.tensor([c[2] for c in costs])
    sc = h / h.min().clamp(min=1e-8) + a / a.min().clamp(min=1e-8) + 5.0 * o
    order = sc.argsort().tolist()

    print(f'\n===== case {idx}  n={n}  pre={int((cons[:n,1]!=0).sum())} '
          f'fix={int((cons[:n,0]!=0).sum())} =====')
    S = float(np.sqrt(case['area'].numpy().sum()))
    print(f'S={S:.2f}')
    nfeas = 0
    for rank in range(6):
        sol, info = lg.legalize_case(cands[order[rank]].double(), case)
        hard = info['hard']
        g = info['graph']
        nfeas += int(hard['feasible'])
        print(f" seed{rank}: feasible={hard['feasible']} proxy={info['proxy_cost']:.3f} "
              f"ovl={hard['overlap_violations']} area={hard['area_violations']} "
              f"dim={hard['dimension_violations']} pen={hard['max_penetration']:.4g} "
              f"repair_fail={g['repair_failures']} assign_fail={g['assign_failures']} "
              f"final_ok={g['final_assignment_ok']}")
        if not hard['feasible'] and rank == 0:
            sn = sol.numpy()
            pre = (case['cons'].numpy()[:, 1] > 0)
            fix = (case['cons'].numpy()[:, 0] > 0)
            for (i, j) in hard['overlap_pairs']:
                print(f"    pair ({i},{j}) pre={pre[i]},{pre[j]} fix={fix[i]},{fix[j]} "
                      f"boxes {np.round(sn[i],2)} {np.round(sn[j],2)}")
    print(f' feasible seeds: {nfeas}/6')
