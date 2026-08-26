"""Render witness vs solver layout side by side for kit cases.

    python tools/render_kit_case.py <kit_path> <results.json> <ids> <out_dir>

Blocks are coloured by constraint role; violated soft constraints are outlined.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

CONTEST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTEST))
sys.path.insert(0, str(CONTEST.parent))

import iccad2026_evaluate as ev
from lite_dataset_test import FloorplanDatasetLiteTest
from floordiff import legalizer as lg

FILL = {'pre': '#c94f4f', 'fix': '#d98c3f', 'mib': '#4f7fc9',
        'clu': '#4fa06b', 'bnd': '#8a6fc4', 'soft': '#d8d8d8'}


def role(c):
    if c[1] != 0:
        return 'pre'
    if c[0] != 0:
        return 'fix'
    if c[2] != 0:
        return 'mib'
    if c[3] != 0:
        return 'clu'
    if c[4] != 0:
        return 'bnd'
    return 'soft'


def draw(ax, sol, cons, title):
    sol = np.asarray(sol, dtype=np.float64)
    x0, y0 = sol[:, 0].min(), sol[:, 1].min()
    x1 = (sol[:, 0] + sol[:, 2]).max()
    y1 = (sol[:, 1] + sol[:, 3]).max()
    for i in range(len(sol)):
        r = role(cons[i])
        ax.add_patch(Rectangle((sol[i, 0], sol[i, 1]), sol[i, 2], sol[i, 3],
                               facecolor=FILL[r], edgecolor='white',
                               linewidth=0.6, zorder=2))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor='#222', linewidth=1.4, zorder=3))
    # overlaps in red hatch
    hard = lg.hard_feasibility(sol, {'cons': torch.tensor(cons)})
    for (i, j) in hard['overlap_pairs']:
        ix0 = max(sol[i, 0], sol[j, 0]); iy0 = max(sol[i, 1], sol[j, 1])
        ix1 = min(sol[i, 0] + sol[i, 2], sol[j, 0] + sol[j, 2])
        iy1 = min(sol[i, 1] + sol[i, 3], sol[j, 1] + sol[j, 3])
        ax.add_patch(Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0,
                               facecolor='none', edgecolor='red',
                               hatch='///', linewidth=1.2, zorder=4))
    pad = 0.03 * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def main():
    kit, res_path, ids, out_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ds = FloorplanDatasetLiteTest(kit)
    evl = ev.ContestEvaluator(kit, verbose=False)
    got = {r['test_id']: r for r in json.load(open(res_path))['test_results']}
    for idx in [int(x) for x in ids.split(',')]:
        s = ds[idx]
        area, b2b, p2b, pins, cons_t = s['input']
        n = int((area != -1).sum().item())
        cons = cons_t[:n].numpy()
        base, tgt = evl._extract_baseline(idx, s['label'], b2b, p2b, pins, n)
        ours = got[idx]['positions'][:n]
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), dpi=150)
        vb, vg, vm = lg._violations_official(np.asarray(tgt, dtype=np.float64), cons)
        ns = max(lg._n_soft_norm(cons), 1)
        draw(axes[0], tgt, cons,
             f'witness (reference layout)   V_rel={(vb+vg+vm)/ns:.3f}')
        vb, vg, vm = lg._violations_official(np.asarray(ours, dtype=np.float64), cons)
        draw(axes[1], ours, cons,
             f"ours   cost={got[idx]['cost']:.3f}  V_rel={(vb+vg+vm)/ns:.3f}  "
             f"area_gap={got[idx]['area_gap']:+.2f}")
        pre_f = (cons[:, 1] != 0).mean()
        fig.suptitle(f'case {idx}  n={n}  preplaced {pre_f:.0%}  '
                     f'fixed {(cons[:,0]!=0).mean():.0%}  '
                     f'boundary {(cons[:,4]!=0).mean():.0%}', fontsize=10)
        handles = [Rectangle((0, 0), 1, 1, facecolor=FILL[k]) for k in FILL]
        fig.legend(handles, list(FILL), loc='lower center', ncol=6,
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        out = Path(out_dir) / f'case_{idx:04d}.png'
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print('wrote', out)


if __name__ == '__main__':
    main()
