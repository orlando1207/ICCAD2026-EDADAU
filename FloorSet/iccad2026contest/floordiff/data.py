"""Data pipeline: shard/validation loading, featurization, augmentation, decode.

Conventions (verified against the raw tensors, see design doc Part B):
  - training shard `layouts_*.th` = list of 7 tensors, leading dim 112 (same n per file):
      [0] (112, n, 6)  = [area, fixed, preplaced, mib_gid, cluster_gid, boundary_bits]
      [1] (112, E, 3)  = b2b edges (i, j, weight)
      [2] (112, P, 3)  = p2b edges (pin_idx, block_idx, weight)
      [3] (112, r, 2)  = terminal positions
      [4] tree_sol, [5] fp_sol (112, n, 4) = (w, h, x, y) with (x, y) = LOWER-LEFT corner,
      [6] metrics (112, 8)
  - validation: LiteTensorDataTest/config_{n}/litedata_1.pth (+ litelabel_1.pth with a
    (n, 5, 2) closed-polygon solution).
  - boundary bits: LEFT=1, RIGHT=2, TOP=4, BOTTOM=8 (corners are ORs).

Latent per block: z = (cx_n, cy_n, s_n) where
  cx_n = (cx - ox) / S * COORD_SCALE      (ox, oy) = terminal-bbox center, S = sqrt(sum areas)
  s_n  = s * S_SCALE,  s = 0.5 * log(w / h)
Decode: w = sqrt(a) * exp(s), h = sqrt(a) * exp(-s)  -> w * h == a exactly by construction.
"""

import glob
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

FLOORSET_ROOT = Path(__file__).resolve().parents[2]

# Latent scaling: makes each z channel ~unit variance over the training data
# (measured: center std 0.341, s std 0.263 across 20 workers).
COORD_SCALE = 2.9
S_SCALE = 3.8
Z_CLAMP = 3.5          # |z| never exceeds ~3.1 in data; sampler clamps x0-hat here
Z_DIM = 3

N_FEAT = 24            # per-block static features
N_GLOBAL = 12          # per-case global features
N_PAIR = 3             # pairwise (attention-bias) features


# --------------------------------------------------------------------------- raw cases

def _case_dict(area, cons, b2b, p2b, pins, gt, metrics):
    return {
        'area': area.float(),          # (n,)
        'cons': cons.long(),           # (n, 5) [fixed, preplaced, mib, cluster, boundary]
        'b2b': b2b.float(),            # (E, 3) (i, j, w)
        'p2b': p2b.float(),            # (P, 3) (pin, block, w)
        'pins': pins.float(),          # (r, 2)
        'gt': gt.float() if gt is not None else None,      # (n, 4) (w, h, x, y-lower-left)
        'metrics': metrics.float() if metrics is not None else None,  # (8,)
    }


def load_shard(path):
    return torch.load(path, weights_only=False)


def case_from_shard(shard, li):
    inp, b2b, p2b, pins, _tree, fp, met = shard
    return _case_dict(inp[li, :, 0], inp[li, :, 1:], b2b[li], p2b[li], pins[li],
                      fp[li], met[li])


def load_validation_case(n_blocks, root=FLOORSET_ROOT):
    """One of the 100 official validation cases (n_blocks in 21..120)."""
    cdir = Path(root) / 'LiteTensorDataTest' / f'config_{n_blocks}'
    inp, b2b, p2b, pins = torch.load(cdir / 'litedata_1.pth', weights_only=False)[0]
    metrics, sol = torch.load(cdir / 'litelabel_1.pth', weights_only=False)[0]
    xs, ys = sol[:, :, 0], sol[:, :, 1]
    w = xs.max(1).values - xs.min(1).values
    h = ys.max(1).values - ys.min(1).values
    gt = torch.stack([w, h, xs.min(1).values, ys.min(1).values], dim=1)
    return _case_dict(inp[:, 0], inp[:, 1:], b2b, p2b, pins, gt, metrics)


VALIDATION_NS = list(range(21, 121))


# --------------------------------------------------------------------------- augmentation

def augment_case(case, op):
    """op in {0: id, 1: flip-x, 2: flip-y, 3: flip both}. Orientation-preserving only
    (no 90-degree rotations), so fixed/preplaced dims never swap. Applied to raw coords;
    the frame recenters on the flipped terminals downstream, so the sign flip is enough."""
    if op == 0:
        return case
    c = dict(case)
    flipx, flipy = bool(op & 1), bool(op & 2)
    gt = case['gt'].clone()
    pins = case['pins'].clone()
    cons = case['cons'].clone()
    if flipx:
        gt[:, 2] = -(gt[:, 2] + gt[:, 0])          # x' = -(x + w)
        pins[:, 0] = -pins[:, 0]
        b = cons[:, 4]
        left, right = (b & 1).bool(), (b & 2).bool()
        b = b & ~torch.tensor(3)                    # clear L/R bits
        b[left] |= 2
        b[right] |= 1
        cons[:, 4] = b
    if flipy:
        gt[:, 3] = -(gt[:, 3] + gt[:, 1])
        pins[:, 1] = -pins[:, 1]
        b = cons[:, 4]
        top, bottom = (b & 4).bool(), (b & 8).bool()
        b = b & ~torch.tensor(12)                   # clear T/B bits
        b[top] |= 8
        b[bottom] |= 4
        cons[:, 4] = b
    c['gt'], c['pins'], c['cons'] = gt, pins, cons
    return c


# --------------------------------------------------------------------------- featurization

def group_lists(gids):
    """gids (n,) long, 0 = no group. Returns list of LongTensor member indices."""
    out = []
    for g in torch.unique(gids):
        if g > 0:
            out.append(torch.nonzero(gids == g).flatten())
    return out


def featurize(case):
    """Returns (tensors, meta). `tensors` is everything the model/loss consumes
    (fixed shapes given n); `meta` carries decode/eval bookkeeping."""
    area, cons = case['area'], case['cons']
    b2b, p2b, pins, gt = case['b2b'], case['p2b'], case['pins'], case['gt']
    n = area.shape[0]
    S = area.sum().sqrt()
    ox = (pins[:, 0].min() + pins[:, 0].max()) / 2
    oy = (pins[:, 1].min() + pins[:, 1].max()) / 2

    fixed = cons[:, 0] > 0
    pre = cons[:, 1] > 0
    frozen_shape = fixed | pre

    # --- known values (inputs by contest definition; from gt here, target_positions at test)
    wh_known = torch.zeros(n, 2)
    xy_known = torch.zeros(n, 2)     # lower-left, raw frame
    if gt is not None:
        wh_known[frozen_shape] = gt[frozen_shape][:, 0:2]
        xy_known[pre] = gt[pre][:, 2:4]
    s_known = torch.zeros(n)
    s_known[frozen_shape] = 0.5 * torch.log(wh_known[frozen_shape, 0] /
                                            wh_known[frozen_shape, 1])
    c_known = torch.zeros(n, 2)
    c_known[pre] = xy_known[pre] + wh_known[pre] / 2

    z_known = torch.zeros(n, Z_DIM)
    z_known[:, 0] = (c_known[:, 0] - ox) / S * COORD_SCALE
    z_known[:, 1] = (c_known[:, 1] - oy) / S * COORD_SCALE
    z_known[:, 2] = s_known * S_SCALE
    z_known[~pre, 0:2] = 0.0
    z_known[~frozen_shape, 2] = 0.0

    freeze = torch.zeros(n, Z_DIM, dtype=torch.bool)
    freeze[pre] = True
    freeze[frozen_shape, 2] = True

    # --- ground-truth latent (training / eval only)
    z0 = None
    if gt is not None:
        cx = gt[:, 2] + gt[:, 0] / 2
        cy = gt[:, 3] + gt[:, 1] / 2
        s = 0.5 * torch.log(gt[:, 0] / gt[:, 1])
        z0 = torch.stack([(cx - ox) / S * COORD_SCALE,
                          (cy - oy) / S * COORD_SCALE,
                          s * S_SCALE], dim=1)

    # --- connectivity summaries
    w_mean = b2b[:, 2].mean() if len(b2b) else torch.tensor(1.0)
    wdeg = torch.zeros(n)
    deg = torch.zeros(n)
    if len(b2b):
        i, j, w = b2b[:, 0].long(), b2b[:, 1].long(), b2b[:, 2]
        wdeg.index_add_(0, i, w)
        wdeg.index_add_(0, j, w)
        ones = torch.ones_like(w)
        deg.index_add_(0, i, ones)
        deg.index_add_(0, j, ones)
    p_wsum = torch.zeros(n)
    pull = torch.zeros(n, 2)
    disp = torch.zeros(n)
    if len(p2b):
        pi, bi, pw = p2b[:, 0].long(), p2b[:, 1].long(), p2b[:, 2]
        p_wsum.index_add_(0, bi, pw)
        pull.index_add_(0, bi, pw[:, None] * pins[pi])
        has = p_wsum > 0
        pull[has] = pull[has] / p_wsum[has, None]
        d = (pins[pi] - pull[bi]).norm(dim=1) * pw
        disp.index_add_(0, bi, d)
        disp[has] = disp[has] / p_wsum[has]
    has_pin = (p_wsum > 0).float()

    boundary_bits = torch.stack([(cons[:, 4] & b).bool().float()
                                 for b in (1, 2, 4, 8)], dim=1)   # (n, 4) L R T B
    mib_g = group_lists(cons[:, 2])
    clu_g = group_lists(cons[:, 3])
    mib_size = torch.zeros(n)
    clu_size = torch.zeros(n)
    for g in mib_g:
        mib_size[g] = float(len(g))
    for g in clu_g:
        clu_size[g] = float(len(g))

    feat = torch.zeros(n, N_FEAT)
    feat[:, 0] = torch.log(area / area.mean())
    feat[:, 1] = area.sqrt() / S
    feat[:, 2] = fixed.float()
    feat[:, 3] = pre.float()
    feat[:, 4] = s_known * S_SCALE
    feat[:, 5] = wh_known[:, 0] / S
    feat[:, 6] = wh_known[:, 1] / S
    feat[:, 7] = z_known[:, 0]
    feat[:, 8] = z_known[:, 1]
    feat[:, 9:13] = boundary_bits
    feat[:, 13] = (cons[:, 2] > 0).float()
    feat[:, 14] = mib_size / 8.0
    feat[:, 15] = (cons[:, 3] > 0).float()
    feat[:, 16] = clu_size / 12.0
    feat[:, 17] = torch.log1p(wdeg / w_mean)
    feat[:, 18] = deg / n
    feat[:, 19] = torch.log1p(p_wsum / w_mean)
    feat[:, 20] = (pull[:, 0] - ox) / S * COORD_SCALE * has_pin
    feat[:, 21] = (pull[:, 1] - oy) / S * COORD_SCALE * has_pin
    feat[:, 22] = disp / S * COORD_SCALE
    feat[:, 23] = has_pin

    pair = torch.zeros(n, n, N_PAIR)
    if len(b2b):
        i, j, w = b2b[:, 0].long(), b2b[:, 1].long(), b2b[:, 2]
        lw = torch.log1p(w / w_mean)
        pair[i, j, 0] += lw
        pair[j, i, 0] += lw
    for g in mib_g:
        pair[g[:, None], g[None, :], 1] = 1.0
    for g in clu_g:
        pair[g[:, None], g[None, :], 2] = 1.0

    tw = pins[:, 0].max() - pins[:, 0].min()
    th = pins[:, 1].max() - pins[:, 1].min()
    gfeat = torch.tensor([
        n / 120.0,
        pins.shape[0] / 400.0,
        math.log(float(S)) / 6.0,
        tw / S, th / S,
        fixed.float().mean(), pre.float().mean(),
        (cons[:, 2] > 0).float().mean(), (cons[:, 3] > 0).float().mean(),
        (cons[:, 4] > 0).float().mean(),
        min(1.0, 2.0 * len(b2b) / max(1, n * (n - 1))),
        (area.mean() * n) / (S * S),
    ])

    tensors = {'feat': feat, 'pair': pair, 'gfeat': gfeat,
               'z_known': z_known, 'freeze': freeze}
    if z0 is not None:
        tensors['z0'] = z0
    meta = {'n': n, 'S': S, 'ox': ox, 'oy': oy, 'area': area,
            'fixed': fixed, 'pre': pre, 'wh_known': wh_known, 'xy_known': xy_known,
            'mib_groups': mib_g, 'clu_groups': clu_g, 'boundary_bits': boundary_bits,
            'case': case}
    return tensors, meta


def decode(z, meta):
    """Latent (n, 3) -> (n, 4) rows (x, y, w, h), lower-left corner, raw frame.
    Applies: exact-area shape decode, fixed/preplaced exact values, MIB post-snap."""
    area, S = meta['area'], meta['S']
    fixed, pre = meta['fixed'], meta['pre']
    frozen_shape = fixed | pre
    s = z[:, 2] / S_SCALE
    s[frozen_shape] = 0.5 * torch.log(meta['wh_known'][frozen_shape, 0] /
                                      meta['wh_known'][frozen_shape, 1])
    # MIB post-snap: identical dims within group (mean s keeps area exact; a fixed/
    # preplaced member dictates the whole group)
    for g in meta['mib_groups']:
        fz = g[frozen_shape[g]]
        s[g] = s[fz[0]].clone() if len(fz) else s[g].mean()
    w = area.sqrt() * torch.exp(s)
    h = area / w
    w[frozen_shape] = meta['wh_known'][frozen_shape, 0]
    h[frozen_shape] = meta['wh_known'][frozen_shape, 1]
    cx = z[:, 0] / COORD_SCALE * S + meta['ox']
    cy = z[:, 1] / COORD_SCALE * S + meta['oy']
    x = cx - w / 2
    y = cy - h / 2
    x[pre] = meta['xy_known'][pre, 0]
    y[pre] = meta['xy_known'][pre, 1]
    return torch.stack([x, y, w, h], dim=1)


def gt_xywh(case):
    """Ground truth as (x, y, w, h) rows (matching decode()'s output order)."""
    g = case['gt']
    return torch.stack([g[:, 2], g[:, 3], g[:, 0], g[:, 1]], dim=1)


# --------------------------------------------------------------------------- training set

def build_shard_index(root=FLOORSET_ROOT, cache=None, max_files=None, verbose=True):
    """[(relpath, n_blocks)] for every training shard; cached as JSON (scan loads
    every file once, ~9k files)."""
    root = Path(root)
    cache = Path(cache) if cache else Path(__file__).parent / 'cache' / 'shard_index.json'
    if cache.exists():
        entries = json.loads(cache.read_text())
        if max_files:
            entries = entries[:max_files]
    else:
        files = sorted(glob.glob(str(root / 'floorset_lite' / 'worker_*' / 'layouts_*.th')))
        if max_files:
            files = files[:max_files]
        entries = []
        for k, f in enumerate(files):
            n = torch.load(f, weights_only=False)[0].shape[1]
            entries.append([os.path.relpath(f, root), int(n)])
            if verbose and (k + 1) % 500 == 0:
                print(f'  indexed {k + 1}/{len(files)} shards')
        if not max_files:      # only cache a full scan
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(entries))
    return [(str(root / p), n) for p, n in entries]


class ShardDataset(Dataset):
    """Index = (file_idx, layout_idx). Featurizes on the fly (with augmentation)."""

    LAYOUTS_PER_FILE = 112

    def __init__(self, index, augment=True):
        self.index = index
        self.augment = augment

    @staticmethod
    @lru_cache(maxsize=4)   # per worker process
    def _shard(path):
        return load_shard(path)

    def __len__(self):
        return len(self.index) * self.LAYOUTS_PER_FILE

    def __getitem__(self, key):
        fi, li = key
        case = case_from_shard(self._shard(self.index[fi][0]), li)
        if self.augment:
            case = augment_case(case, int(torch.randint(0, 4, ())))
        tensors, _meta = featurize(case)
        return tensors


class BucketBatchSampler(Sampler):
    """Each batch is drawn from a single shard file (uniform n -> zero padding).
    Files are sampled with weight exp(n / temp), echoing the contest's exp(n/12)."""

    def __init__(self, index, batch_size, steps, temp=24.0, seed=0):
        self.index = index
        self.batch_size = min(batch_size, ShardDataset.LAYOUTS_PER_FILE)
        self.steps = steps
        w = torch.tensor([math.exp(n / temp) for _, n in index], dtype=torch.float64)
        self.weights = w / w.sum()
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            fi = int(torch.multinomial(self.weights, 1, generator=self.gen))
            lis = torch.randperm(ShardDataset.LAYOUTS_PER_FILE,
                                 generator=self.gen)[:self.batch_size]
            yield [(fi, int(li)) for li in lis]


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
