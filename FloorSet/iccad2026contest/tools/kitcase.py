"""Shared kit-case plumbing: build a floordiff `case` dict for any FloorSet-format
data root (the official validation set or a generated kit), exactly as the
official evaluator would present it to `solve()`.

Used by the prediction dumper and the A/B harness so a kit can go through the
same cached sample -> re-legalize -> score loop as the validation set.
"""

import sys
from pathlib import Path

import torch

CONTEST = Path(__file__).resolve().parents[1]
if str(CONTEST) not in sys.path:
    sys.path.insert(0, str(CONTEST))
if str(CONTEST.parent) not in sys.path:
    sys.path.insert(0, str(CONTEST.parent))

import iccad2026_evaluate as ev
from lite_dataset_test import FloorplanDatasetLiteTest

_CACHE = {}


def dataset(kit):
    if kit not in _CACHE:
        _CACHE[kit] = (FloorplanDatasetLiteTest(kit),
                       ev.ContestEvaluator(kit, verbose=False))
    return _CACHE[kit]


def _clean(t):
    if t is None or len(t) == 0:
        return t
    return t[(t != -1).all(dim=1)]


def kit_case(kit, idx):
    """(case, baselines, n) for one test id of the kit at `kit`."""
    ds, evl = dataset(kit)
    s = ds[idx]
    area, b2b, p2b, pins, cons = s['input']
    n = int((area != -1).sum().item())
    base, tgt = evl._extract_baseline(idx, s['label'], b2b, p2b, pins, n)
    otp = torch.full((n, 4), -1.0)
    for i in range(n):
        if cons[i, 1] != 0:
            otp[i] = torch.tensor(list(tgt[i]))
        elif cons[i, 0] != 0:
            otp[i, 2], otp[i, 3] = tgt[i][2], tgt[i][3]
    case = {
        'area': area[:n].float(),
        'cons': cons[:n].long(),
        'b2b': _clean(b2b).float() if b2b is not None else torch.zeros(0, 3),
        'p2b': _clean(p2b).float() if p2b is not None else torch.zeros(0, 3),
        'pins': _clean(pins).float() if pins is not None else torch.zeros(0, 2),
        'gt': None,
        # the kit's stored metrics are -1 so the evaluator recomputes baselines
        # from the reference polygons; carry those recomputed values instead of
        # letting the legalizer fall back to pseudo-baselines off the prediction
        'metrics': torch.tensor([base['area_baseline'], -1, -1, -1, -1, -1,
                                 base['hpwl_baseline'], 0.0]),
        'target': otp.double(),
    }
    return case, base, n


def official_metrics(kit, idx, sol):
    """Score a solution with the official evaluator's own baselines."""
    ds, evl = dataset(kit)
    s = ds[idx]
    area, b2b, p2b, pins, cons = s['input']
    n = int((area != -1).sum().item())
    base, tgt = evl._extract_baseline(idx, s['label'], b2b, p2b, pins, n)
    positions = [tuple(map(float, r)) for r in sol]
    return ev.evaluate_solution(
        {'positions': positions, 'runtime': 1.0}, base, cons,
        b2b, p2b, pins, area, target_positions=tgt, median_runtime=1.0)
