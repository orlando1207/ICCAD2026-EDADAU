"""Draw a FloorSet case: constraints, ground truth, and any solution.

`visualize.py` (the file shipped with the dataset) draws polygons but calls
plt.show() and knows nothing about violations, so it is unusable over SSH and
useless for debugging an infeasible layout.  This wraps the same colour
convention -- get_hard_color(), so the legend means what the team already
expects -- and adds the parts that matter: it writes a PNG, and it outlines
every block the rules say is broken.

    python3 viz.py --data-path DIR --id 0                      # GT only
    python3 viz.py --data-path DIR --id 0 --solve              # GT | our solver
    python3 viz.py --data-path DIR --id 0 --sol sols.json      # GT | saved
    python3 viz.py --data-path DIR --all --outdir png/         # batch
    python3 viz.py --data-path DIR --id 0 --official           # the shipped one

Solution JSON is the format `iccad2026_evaluate.py --save-solutions` writes:
    {"solutions": [{"test_id": 0, "block_count": 40, "positions": [[x,y,w,h],...]}]}

Colours (from visualize.py):
    red cluster   violet fixed   gray preplaced   darkgreen MIB
    olive boundary   silver unconstrained
Violations are drawn on top: thick solid = hard (red overlap, orange area out
of tolerance, magenta fixed/preplaced moved); dashed = soft (deepskyblue
boundary bit not touching the solution's own bbox, blue block sitting in a
split-off part of its cluster group, goldenrod block whose shape doesn't
match the rest of its MIB group). A block showing both gets the hard style --
soft is drawn first and hard overwrites it, same "worst wins" rule as before.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")                      # headless: write files, do not block
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np

import iccad2026_evaluate as off
import rules
from visualize import get_hard_color

VIOL_STYLE = {
    "boundary":  dict(edgecolor="deepskyblue",  lw=1.6, ls="--", label="boundary miss"),
    "grouping":  dict(edgecolor="blue",         lw=1.6, ls="--", label="grouping split"),
    "mib":       dict(edgecolor="darkgoldenrod", lw=1.6, ls="--", label="mib shape mismatch"),
    "overlap":   dict(edgecolor="red",     lw=2.2, label="overlap"),
    "area":      dict(edgecolor="orange",  lw=2.2, label="area >1%"),
    "dimension": dict(edgecolor="magenta", lw=2.2, label="fixed/preplaced moved"),
}


def load_case(data_path: str, test_id: int):
    ev = off.ContestEvaluator(data_path=data_path, verbose=False)
    ev._load_dataset()
    s = ev.dataset[test_id]
    area, b2b, p2b, pins, cons = s["input"]
    n = int((area != -1).sum().item())
    bl, tgt = ev._extract_baseline(test_id, s["label"], b2b, p2b, pins, n)
    gt = np.array(tgt, dtype=float)[:n]
    spec = rules.CaseSpec.from_evaluator(n, area, cons, gt)
    return dict(n=n, area=area, b2b=b2b, p2b=p2b, pins=pins, cons=cons,
                spec=spec, gt=gt, baseline=bl, ev=ev, sample=s)


def violation_map(pos: np.ndarray, spec: rules.CaseSpec) -> Dict[int, str]:
    """block -> which rule it broke (worst first, so the outline is honest)."""
    rep = rules.check_hard(pos, spec)
    out: Dict[int, str] = {}
    for i, *_ in rep.fixed_blocks + rep.preplaced_blocks:
        out[i] = "dimension"
    for i, _ in rep.area_blocks:
        out[i] = "area"
    for i, j, *_ in rep.overlap_pairs:
        out[i] = out[j] = "overlap"
    return out


def _component_labels(x0, y0, x1, y1) -> np.ndarray:
    """Same adjacency rule as rules._components, but returns a label per
    block instead of just a count, so the *minority* pieces can be outlined."""
    m = len(x0)
    parent = list(range(m))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(m):
        for j in range(i + 1, m):
            ox = min(x1[i], x1[j]) - max(x0[i], x0[j])
            oy = min(y1[i], y1[j]) - max(y0[i], y0[j])
            if (ox >= 0 and oy > 0) or (oy >= 0 and ox > 0):
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    return np.array([find(i) for i in range(m)])


def soft_violation_map(pos: np.ndarray, spec: rules.CaseSpec) -> Dict[int, str]:
    """block -> which soft rule (S1-S3) it broke.  Unlike the aggregate
    counts in rules.SoftReport, this points at the actual offending blocks:
    the ones missing a boundary bit, the minority piece(s) of a split cluster
    group, and the off-majority shapes inside a MIB group."""
    rep = rules.check_soft(pos, spec)
    out: Dict[int, str] = {}
    for i in rep.boundary_blocks:
        out[i] = "boundary"

    x0, y0, w, h = pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3]
    x1, y1 = x0 + w, y0 + h
    for mem in spec.cluster_groups:
        mem = mem[mem < spec.n]
        if len(mem) < 2:
            continue
        labels = _component_labels(x0[mem], y0[mem], x1[mem], y1[mem])
        if len(set(labels)) < 2:
            continue
        majority, _ = collections.Counter(labels.tolist()).most_common(1)[0]
        for local_i, lab in zip(mem, labels):
            if lab != majority:
                out[int(local_i)] = "grouping"

    for mem in spec.mib_groups:
        mem = mem[mem < spec.n]
        if len(mem) < 2:
            continue
        shapes = [(round(float(w[i]), rules.MIB_ROUND), round(float(h[i]), rules.MIB_ROUND))
                  for i in mem]
        if len(set(shapes)) < 2:
            continue
        majority, _ = collections.Counter(shapes).most_common(1)[0]
        for local_i, shp in zip(mem, shapes):
            if shp != majority:
                out[int(local_i)] = "mib"
    return out


def draw(ax, pos: np.ndarray, spec: rules.CaseSpec, pins, title: str,
         show_ids: bool = True, viol: Optional[Dict[int, str]] = None):
    seen = set()
    for i in range(len(pos)):
        x, y, w, h = pos[i]
        if not np.isfinite([x, y, w, h]).all():
            continue
        face, label = get_hard_color(spec.cons[i])
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor="black",
                               lw=0.4, alpha=0.35,
                               label=label if label not in seen else None))
        seen.add(label)
        if viol and i in viol:
            ax.add_patch(Rectangle((x, y), w, h, fill=False, zorder=5,
                                   **{k: v for k, v in VIOL_STYLE[viol[i]].items()
                                      if k != "label"}))
        if show_ids:
            ax.annotate(str(i), (x + w / 2, y + h / 2), ha="center", va="center",
                        fontsize=5, color="black")

    if pins is not None and len(pins):
        pn = np.asarray(pins, dtype=float).reshape(-1, 2)
        ax.scatter(pn[:, 0], pn[:, 1], s=4, c="green", zorder=6, label="pin")

    ok = np.isfinite(pos).all(axis=1)
    if ok.any():
        p = pos[ok]
        lo = p[:, :2].min(axis=0)
        hi = (p[:, :2] + p[:, 2:]).max(axis=0)
        pad = 0.05 * max(hi - lo).max()
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9)


def legend(fig, ax, viol_kinds):
    h, l = ax.get_legend_handles_labels()
    by = dict(zip(l, h))
    for k in viol_kinds:
        st = VIOL_STYLE[k]
        by[st["label"]] = Patch(facecolor="none", edgecolor=st["edgecolor"],
                                lw=st["lw"], label=st["label"])
    fig.legend(by.values(), by.keys(), loc="lower center", ncol=5, fontsize=7,
               frameon=False)


def verdict_line(pos, case) -> str:
    v = rules.evaluate(pos, case["spec"], case["b2b"], case["p2b"], case["pins"],
                       case["baseline"]["hpwl_baseline"],
                       case["baseline"]["area_baseline"], 1.0)
    tag = "FEASIBLE" if v.feasible else "INFEASIBLE"
    s = (f"{tag}  cost={v.cost:.3f}  hpwl_gap={v.hpwl_gap:+.3f} "
         f"area_gap={v.area_gap:+.3f}  V_rel={v.soft.relative:.3f} "
         f"(bnd={v.soft.boundary} grp={v.soft.grouping} mib={v.soft.mib})")
    if not v.feasible:
        s += "\n" + v.hard.summary()
    return s


def solve_case(case, opt=None) -> np.ndarray:
    """`opt`: reuse one MyOptimizer across several calls (checkpoint load is
    the expensive part) instead of paying warmup again per case."""
    if opt is None:
        from op_src import MyOptimizer
        opt = MyOptimizer(verbose=False)
    tgt = np.full((case["n"], 4), -1.0, dtype=np.float32)
    spec = case["spec"]
    for i in range(case["n"]):
        if spec.preplaced_mask[i]:
            tgt[i] = case["gt"][i]
        elif spec.fixed_mask[i]:
            tgt[i, 2:4] = case["gt"][i, 2:4]
    import torch
    pos = opt.solve(case["n"], case["area"], case["b2b"], case["p2b"],
                    case["pins"], case["cons"], torch.from_numpy(tgt))
    return np.asarray(pos, dtype=float)[:case["n"]]


def load_solution(path: str, test_id: int, n: int) -> np.ndarray:
    sols = json.load(open(path)).get("solutions", [])
    for s in sols:
        if int(s["test_id"]) == test_id:
            return np.asarray(s["positions"], dtype=float)[:n]
    raise SystemExit(f"no solution for test_id {test_id} in {path}")


def all_violations(pos: np.ndarray, spec: rules.CaseSpec) -> Dict[int, str]:
    """Soft first, hard overwrites -- a block with both shows the hard style,
    same 'worst wins' convention violation_map already used on its own."""
    return {**soft_violation_map(pos, spec), **violation_map(pos, spec)}


def render(case, test_id: int, out: Path, sol: Optional[np.ndarray], label: str):
    panels = [("ground truth (witness)", case["gt"],
              all_violations(case["gt"], case["spec"]))]
    if sol is not None:
        panels.append((label, sol, all_violations(sol, case["spec"])))

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 7.4))
    axes = np.atleast_1d(axes)
    kinds = set()
    for ax, (t, p, viol) in zip(axes, panels):
        draw(ax, p, case["spec"], case["pins"],
             f"case {test_id} - {t}\n{verdict_line(p, case)}", viol=viol)
        kinds |= set((viol or {}).values())
    legend(fig, axes[0], kinds)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", default="/home/lovet/ICCAD2026-EDADAU/FloorSet/")
    ap.add_argument("--id", type=int, nargs="+", help="test_id(s) to draw")
    ap.add_argument("--all", action="store_true", help="draw every case")
    ap.add_argument("--solve", action="store_true", help="run our optimizer too")
    ap.add_argument("--sol", help="solutions JSON from --save-solutions")
    ap.add_argument("--out", help="output PNG (single id)")
    ap.add_argument("--outdir", default="./viz", help="output dir (--all)")
    ap.add_argument("--official", action="store_true",
                    help="use the dataset's own visualize_lite() instead")
    args = ap.parse_args()

    if args.all:
        ev = off.ContestEvaluator(data_path=args.data_path, verbose=False)
        ev._load_dataset()
        ids = range(len(ev.dataset))
    elif args.id is not None:
        ids = args.id
    else:
        raise SystemExit("give --id N or --all")

    opt = None
    if args.solve:
        from op_src import MyOptimizer
        opt = MyOptimizer(verbose=False)   # loaded once, reused across --id N M ...

    for tid in ids:
        case = load_case(args.data_path, tid)
        if args.official:
            from visualize import visualize_lite
            g = case["gt"]
            fp = np.column_stack([g[:, 2], g[:, 3], g[:, 0], g[:, 1]])  # w,h,x,y
            visualize_lite(fp, case["b2b"], case["p2b"], case["pins"],
                           case["cons"], tid)
            continue

        sol, label = None, ""
        if args.sol:
            sol, label = load_solution(args.sol, tid, case["n"]), f"saved: {Path(args.sol).name}"
        elif args.solve:
            sol, label = solve_case(case, opt), "our solver"

        multi = args.all or (isinstance(ids, list) and len(ids) > 1)
        out = Path(args.out) if (args.out and not multi) \
            else Path(args.outdir) / f"case_{tid:03d}.png"
        render(case, tid, out, sol, label)


if __name__ == "__main__":
    main()
