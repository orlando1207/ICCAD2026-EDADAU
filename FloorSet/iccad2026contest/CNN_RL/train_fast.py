"""
Phase 10 — fast batched behaviour-cloning trainer.

Fixes the bottleneck that made `train_phase8.py` CPU-bound at ~1 sample/s (GPU
idle). Three changes:

  1. **Vectorised rasterizer** (`canvas_raster.rasterize`: integral-image
     occupancy, broadcast wl_pull) — no per-block Python loops.
  2. **One batched CNN forward per sample**: stack the sample's per-step canvases
     into `[B,4,G,G]`, run the GNN once (it encodes the *static* netlist, reused
     across steps), and do a single `forward_batch` + cross-entropy + backward
     instead of B tiny batch-1 forwards.
  3. **Parallel episode building** via a `DataLoader` with `num_workers`: the
     per-step rasterization (the dominant cost, ~4 ms/step on CPU and *worse* on
     GPU because tiny [G,G] kernels are launch-bound) is the producer side and is
     spread across CPU cores, so the GPU consumer stays fed. This is the lever
     that actually scales — measured ~1.4 -> ~10+ samples/s on a 24-core box.

Gradient is accumulated over `--accum` samples before each optimiser step.
Checkpoints are written every `--save-every` samples (survives interruption).

NOTE: the policy net was redesigned this phase (dilated convs + global branch),
so old `phase8_bc.pt` checkpoints are NOT compatible — retrain here.

Run:
    cd FloorSet/iccad2026contest
    python3 CNN_RL/train_fast.py --num-samples 20000 --epochs 1 --grid 64 --workers 12
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_FLOORSET_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_FLOORSET_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOORSET_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lite_dataset import FloorplanDatasetLite  # noqa: E402
from placement_env import PlacementEnv  # noqa: E402
from canvas_raster import rasterize_env, N_CHANNELS  # noqa: E402
from gnn_encoder import GNNEncoder  # noqa: E402
from policy_net import PolicyValueNet  # noqa: E402
from pretrain_bc import _target_cell, CKPT_DIR  # noqa: E402
from hard_constraints import make_target_positions_from_gt  # noqa: E402
from train_phase8 import boxes_from_fp_sol  # noqa: E402


class BCEpisodeDataset(Dataset):
    """Wraps the 1M training set; __getitem__ teacher-forces one sample along GT
    and returns its stacked per-step canvases + targets. The heavy per-step
    rasterization happens here, so it parallelises across DataLoader workers.
    Rasters are built on CPU (workers can't share a CUDA context); the main
    process moves them to the GPU for the batched forward."""

    def __init__(self, root: str, grid: int, num_samples: int = None):
        self.base = FloorplanDatasetLite(root)
        self.grid = grid
        self.n = len(self.base) if num_samples is None else min(num_samples, len(self.base))
        self._env = None

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self._env is None:                       # one env per worker process
            self._env = PlacementEnv(grid=self.grid)
        env = self._env
        s = self.base[idx]
        area, b2b, p2b, pins, cons = s["input"]
        _tree, fp_sol, metrics = s["label"]
        bc = int((area != -1).sum())
        boxes = boxes_from_fp_sol(fp_sol, bc)
        tp = make_target_positions_from_gt(cons, boxes, bc)
        env.reset(area, b2b, p2b, pins, cons, metrics, target_positions=tp)

        canvases, blocks, targets = [], [], []
        done = len(env.order) == 0
        while not done:
            st = env._build_state()
            cur = st["current_block"]
            canvases.append(rasterize_env(env).detach().half())   # fp16 -> small IPC
            gx, gy, gw, gh = (float(v) for v in boxes[cur])
            targets.append(_target_cell(env, gx, gy, gw, gh))
            blocks.append(cur)
            env.step(targets[-1])
            done = env.ptr >= len(env.order)

        if not canvases:                            # all-preplaced -> skip
            return None
        return {"canvas": torch.stack(canvases),                       # [B,4,G,G] fp16
                "blocks": torch.tensor(blocks, dtype=torch.long),      # [B]
                "target": torch.tensor(targets, dtype=torch.long),     # [B]
                "area": area, "cons": cons, "b2b": b2b, "bc": bc}


def _soft_labels(targets: torch.Tensor, G: int, sigma: float) -> torch.Tensor:
    """[B] target cell indices -> [B, G*G] Gaussian soft labels centred on each
    target cell (std `sigma` in cells), row-normalised to a probability. Softens
    the exact-cell target so neighbouring cells also get gradient -> a stabler,
    less-noisy signal than one-hot cross-entropy (cell_acc stops being all-or-
    nothing; the model learns 'be near here', which is what the legalizer needs)."""
    B = targets.shape[0]
    dev = targets.device
    tr = (targets // G).float().view(B, 1, 1)        # target row
    tc = (targets % G).float().view(B, 1, 1)         # target col
    rows = torch.arange(G, device=dev).float().view(1, G, 1)
    cols = torch.arange(G, device=dev).float().view(1, 1, G)
    d2 = (rows - tr) ** 2 + (cols - tc) ** 2          # [B,G,G]
    w = torch.exp(-d2 / (2.0 * sigma * sigma)).view(B, -1)
    return w / w.sum(1, keepdim=True).clamp_min(1e-12)


def train(num_samples=20000, epochs=1, grid=64, gnn_out=128, hidden=64,
          lr=1e-3, accum=8, workers=12, seed=0, ckpt_name="phase10_bc.pt",
          log_every=100, save_every=2000, device=None, root=None, soft_sigma=0.0):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    root = root or (str(_FLOORSET_ROOT) + "/")
    gnn = GNNEncoder(out_dim=gnn_out, hidden=hidden).to(device)
    policy = PolicyValueNet(in_channels=N_CHANNELS, node_dim=gnn_out,
                            hidden=hidden).to(device)
    opt = torch.optim.Adam(list(gnn.parameters()) + list(policy.parameters()), lr=lr)

    ds = BCEpisodeDataset(root, grid=grid, num_samples=num_samples)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers,
                        collate_fn=lambda b: b[0],
                        persistent_workers=(workers > 0),
                        prefetch_factor=(4 if workers > 0 else None))
    mode = f"soft(sigma={soft_sigma})" if soft_sigma > 0 else "hard(cross-entropy)"
    print(f"[train_fast] device={device} workers={workers} "
          f"samples={len(ds)} x {epochs} epoch(s) grid={grid} accum={accum} loss={mode}")

    seen, t0 = 0, time.time()
    run_loss, run_correct, run_near, run_steps = 0.0, 0, 0, 0
    opt.zero_grad()
    for ep in range(epochs):
        for sample in loader:
            if sample is None:
                continue
            canvas = sample["canvas"].to(device).float()    # [B,4,G,G]
            blocks = sample["blocks"].to(device)
            target = sample["target"].to(device)

            node_emb, _ = gnn.encode_problem(
                sample["area"], sample["cons"], sample["b2b"], sample["bc"],
                device=device)
            logits, _value = policy.forward_batch(canvas, node_emb[blocks])
            if soft_sigma > 0:
                soft = _soft_labels(target, grid, soft_sigma)        # [B,G*G]
                loss = -(soft * F.log_softmax(logits, dim=1)).sum(1).mean()
            else:
                loss = F.cross_entropy(logits, target)
            (loss / accum).backward()

            seen += 1
            if seen % accum == 0:
                opt.step()
                opt.zero_grad()

            pred = logits.argmax(dim=1)
            run_correct += int((pred == target).sum())
            # near_acc: predicted cell within Chebyshev +-1 of the GT cell
            pr, pc, tr2, tc2 = pred // grid, pred % grid, target // grid, target % grid
            run_near += int((((pr - tr2).abs() <= 1) & ((pc - tc2).abs() <= 1)).sum())
            run_loss += float(loss)
            run_steps += target.numel()
            if seen % log_every == 0:
                rate = seen / (time.time() - t0)
                print(f"  [{seen}] loss={run_loss/log_every:.3f} "
                      f"cell_acc={run_correct/max(run_steps,1):.3f} "
                      f"near_acc={run_near/max(run_steps,1):.3f} "
                      f"({rate:.1f} samples/s)")
                run_loss, run_correct, run_near, run_steps = 0.0, 0, 0, 0
            if seen % save_every == 0:
                _save(gnn, policy, grid, gnn_out, hidden, ckpt_name, seen, t0)

    opt.step()                                       # flush remaining grads
    return _save(gnn, policy, grid, gnn_out, hidden, ckpt_name, seen, t0)


def _save(gnn, policy, grid, gnn_out, hidden, ckpt_name, seen, t0):
    CKPT_DIR.mkdir(exist_ok=True)
    path = CKPT_DIR / ckpt_name
    torch.save({"gnn": gnn.state_dict(), "policy": policy.state_dict(),
                "grid": grid, "gnn_out": gnn_out, "hidden": hidden}, path)
    print(f"  saved {path}  ({seen} samples, {time.time()-t0:.1f}s)")
    return str(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-name", type=str, default="phase10_bc.pt")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--soft-sigma", type=float, default=0.0,
                    help="Gaussian soft-label std in cells (0 = hard cross-entropy)")
    args = ap.parse_args()
    train(num_samples=args.num_samples, epochs=args.epochs, grid=args.grid,
          accum=args.accum, workers=args.workers, lr=args.lr, seed=args.seed,
          ckpt_name=args.ckpt_name, device=args.device, soft_sigma=args.soft_sigma)
