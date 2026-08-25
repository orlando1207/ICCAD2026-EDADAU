"""Does stage E fire on the official validation set, and is it inert there?

One candidate per case (deterministic), so the two arms below are directly
comparable per case:

    python tools/val_evict.py /tmp/on.json  cuda:0
    CFG='{"evict_repair": false, "guaranteed_floor": false, "reclaim": false}' \
        python tools/val_evict.py /tmp/off.json cuda:0

Identical per-case costs across the two arms is the no-regression proof for
FEASIBILITY_ANALYSIS.md 7 ("Official 100").
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

CONTEST = Path('/home/b12901044/iccad-2026-C/FloorSet/iccad2026contest')
sys.path.insert(0, str(CONTEST))
sys.path.insert(0, str(CONTEST.parent))

from floordiff.data import featurize, decode, load_validation_case
from floordiff import legalizer as lg
from floordiff.sample import load_checkpoint

CFG = json.loads(os.environ.get('CFG', '{}')) or None


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'val_evict.json'
    dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else 'cuda:0')
    diff = load_checkpoint(
        str(CONTEST / 'floordiff/checkpoints/myrun/last.pt'), dev)
    rows = []
    for nb in range(21, 121):
        case = load_validation_case(nb)
        tensors, meta = featurize(case)
        b = {k: v.unsqueeze(0).to(dev)
             for k, v in tensors.items() if k != 'z0'}
        with torch.no_grad():
            z = diff.sample(b['feat'], b['pair'], b['gfeat'], b['z_known'],
                            b['freeze'], steps=50, seed=0)
        pred = decode(z[0].float().cpu(), meta).double()
        t0 = time.time()
        sol, info = lg.legalize_case(pred, case, CFG)
        dt = time.time() - t0
        e = info['evict']
        rows.append(dict(n=nb, feas=info['hard']['feasible'],
                         cost=info['proxy_cost'],
                         evict_rounds=e['evict_rounds'],
                         evicted=e['evicted_total'],
                         floor=info['floor_used'],
                         reclaimed=info['reclaim']['moved'],
                         repair_fail=info['graph']['repair_failures'],
                         assign_fail=info['graph']['assign_failures'],
                         t=dt))
        r = rows[-1]
        print(f"  n={nb:3d}: feas={r['feas']} evict_rounds={r['evict_rounds']} "
              f"evicted={r['evicted']} floor={r['floor']} "
              f"rf={r['repair_fail']} af={r['assign_fail']} "
              f"cost={r['cost']:.4f} t={dt:.2f}s", flush=True)
    json.dump(rows, open(out_path, 'w'), indent=1)
    print(f"\ncases {len(rows)}  infeasible {sum(1 for r in rows if not r['feas'])}"
          f"  evict fired {sum(1 for r in rows if r['evict_rounds'])}"
          f"  floor used {sum(r['floor'] for r in rows)}"
          f"  reclaim moves {sum(r['reclaimed'] for r in rows)}")
    print(f"legalize time: mean {np.mean([r['t'] for r in rows]):.3f}s "
          f"max {max(r['t'] for r in rows):.3f}s")
    print(f"rung failures: repair {sum(r['repair_fail'] for r in rows)}  "
          f"assign {sum(r['assign_fail'] for r in rows)} "
          f"(cases with any: "
          f"{sum(1 for r in rows if r['repair_fail'] or r['assign_fail'])})")


if __name__ == '__main__':
    main()
