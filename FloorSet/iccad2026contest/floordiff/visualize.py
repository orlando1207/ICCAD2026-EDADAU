"""Visualizer for raw predictions: ground truth vs. model output, side by side.

Blocks are colored by constraint type (same palette priority as the root repo's
`get_hard_color`): cluster > fixed > preplaced > MIB > boundary > none. A boundary
block's *required* side(s) are drawn as thick blue edge segments so near-misses are
visible. Terminals are black '+'. Optionally overlays the strongest b2b nets.

Run from iccad2026contest/:
  python -m floordiff.visualize --pred floordiff/out/preds.json --cases 60,100,120
  python -m floordiff.visualize --gt-only --cases 21          # inspect GT alone
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from .data import VALIDATION_NS, gt_xywh, load_validation_case
from .evaluate import evaluate_case

CLUSTER_CMAP = ['#e74c3c', '#ff8c69', '#c0392b', '#ff5533', '#d35400']


def block_style(cons_row):
    fixed, pre, mib, clu, bnd = [int(v) for v in cons_row]
    if clu > 0:
        return CLUSTER_CMAP[(clu - 1) % len(CLUSTER_CMAP)], f'cluster{clu}'
    if fixed:
        return 'violet', 'fixed'
    if pre:
        return 'dimgray', 'preplaced'
    if mib > 0:
        return 'darkgreen', f'mib{mib}'
    if bnd > 0:
        return 'olive', 'boundary'
    return 'silver', ''


def draw_layout(ax, xywh, case, title, show_nets=0):
    cons, pins = case['cons'], case['pins']
    n = xywh.shape[0]
    for i in range(n):
        x, y, w, h = [float(v) for v in xywh[i]]
        color, _ = block_style(cons[i])
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor='black',
                               linewidth=0.5, alpha=0.75))
        if n <= 130:
            ax.text(x + w / 2, y + h / 2, str(i), ha='center', va='center',
                    fontsize=5 if n > 60 else 7)
    # required boundary sides in blue
    x0s, y0s = xywh[:, 0], xywh[:, 1]
    x1s, y1s = x0s + xywh[:, 2], y0s + xywh[:, 3]
    for i in range(n):
        bits = int(cons[i, 4])
        if bits == 0:
            continue
        x0, y0, x1, y1 = float(x0s[i]), float(y0s[i]), float(x1s[i]), float(y1s[i])
        if bits & 1:
            ax.plot([x0, x0], [y0, y1], color='blue', lw=2)
        if bits & 2:
            ax.plot([x1, x1], [y0, y1], color='blue', lw=2)
        if bits & 4:
            ax.plot([x0, x1], [y1, y1], color='blue', lw=2)
        if bits & 8:
            ax.plot([x0, x1], [y0, y0], color='blue', lw=2)
    # solution bounding box
    ax.add_patch(Rectangle((float(x0s.min()), float(y0s.min())),
                           float(x1s.max() - x0s.min()), float(y1s.max() - y0s.min()),
                           fill=False, edgecolor='steelblue', linestyle='--', lw=0.8))
    ax.plot(pins[:, 0], pins[:, 1], '+', color='black', markersize=3, alpha=0.6)
    if show_nets > 0 and len(case['b2b']):
        b2b = case['b2b']
        top = b2b[b2b[:, 2].argsort(descending=True)[:show_nets]]
        cx = xywh[:, 0] + xywh[:, 2] / 2
        cy = xywh[:, 1] + xywh[:, 3] / 2
        wmax = float(top[:, 2].max())
        for i, j, w in top.tolist():
            ax.plot([cx[int(i)], cx[int(j)]], [cy[int(i)], cy[int(j)]],
                    color='teal', lw=0.5 + 1.5 * w / wmax, alpha=0.35)
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    ax.autoscale_view()


def render_case(case, pred_xywh, out_path, show_nets=0):
    gt = gt_xywh(case)
    if pred_xywh is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        draw_layout(ax, gt, case, f'ground truth (n={gt.shape[0]})', show_nets)
    else:
        r = evaluate_case(pred_xywh, case)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        draw_layout(axes[0], gt, case, f'ground truth (n={r["n"]})', show_nets)
        draw_layout(axes[1], pred_xywh, case,
                    f'prediction  disp={r["disp_mean"]:.3f}  '
                    f'ovl={100 * r["overlap_ratio"]:.1f}%  '
                    f'hpwl+{100 * r["hpwl_gap"]:.0f}%  '
                    f'area+{100 * r["area_gap"]:.0f}%', show_nets)
        # shared extents so sizes are comparable
        xlim = (min(a.get_xlim()[0] for a in axes), max(a.get_xlim()[1] for a in axes))
        ylim = (min(a.get_ylim()[0] for a in axes), max(a.get_ylim()[1] for a in axes))
        for a in axes:
            a.set_xlim(xlim)
            a.set_ylim(ylim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', type=str, default='',
                    help='predictions JSON from floordiff.sample')
    ap.add_argument('--gt-only', action='store_true', help='render ground truth only')
    ap.add_argument('--cases', type=str, default='',
                    help='comma-separated n values (default: all in --pred, or 21..120)')
    ap.add_argument('--out', type=str, default='floordiff/out/viz')
    ap.add_argument('--nets', type=int, default=0,
                    help='overlay the K highest-weight b2b nets (0 = off)')
    args = ap.parse_args()

    preds = None
    if args.pred:
        preds = json.loads(Path(args.pred).read_text())['cases']
    if args.cases:
        ns = [int(x) for x in args.cases.split(',')]
    elif preds:
        ns = sorted(int(k) for k in preds)
    else:
        ns = VALIDATION_NS

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in ns:
        case = load_validation_case(n)
        pred = None
        if not args.gt_only:
            if preds is None or str(n) not in preds:
                raise SystemExit(f'no prediction for case n={n}; pass --pred or --gt-only')
            pred = torch.tensor(preds[str(n)]['positions'], dtype=torch.float32)
        path = out_dir / f'case_{n:03d}.png'
        render_case(case, pred, path, show_nets=args.nets)
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
