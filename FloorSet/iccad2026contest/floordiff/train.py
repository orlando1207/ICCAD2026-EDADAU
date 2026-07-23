"""FloorDiff training.

Run from iccad2026contest/:
  # smoke test (few shards, small model, few steps)
  python -m floordiff.train --max-files 20 --steps 300 --batch-size 32 \
      --d-model 128 --n-layers 4 --run smoke

  # real training (indexes all ~9k shards on first run, cached)
  python -m floordiff.train --steps 200000 --batch-size 64 --run base

Checkpoints -> floordiff/checkpoints/<run>/step_*.pt  (model + EMA + optimizer + config).
Quick-val (every --val-every steps): 1-seed 20-step samples on a few validation cases,
reports mean center displacement — the stage-1 headline metric.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import (BucketBatchSampler, ShardDataset, build_shard_index, collate,
                   decode, featurize, gt_xywh, load_validation_case)
from .diffusion import EMA, FloorDiffusion
from .model import FloorDiffNet, ModelConfig


def quick_val(diffusion, device, ns=(60, 100, 120), steps=20):
    """Mean normalized center displacement over a few validation cases (1 seed)."""
    diffusion.eval()
    disps = []
    for n in ns:
        case = load_validation_case(n)
        tensors, meta = featurize(case)
        b = {k: v.unsqueeze(0).to(device) for k, v in tensors.items()}
        z = diffusion.sample(b['feat'], b['pair'], b['gfeat'],
                             b['z_known'], b['freeze'], steps=steps, seed=0)
        xywh = decode(z[0].cpu(), meta)
        gt = gt_xywh(case)
        pc = xywh[:, 0:2] + xywh[:, 2:4] / 2
        gc = gt[:, 0:2] + gt[:, 2:4] / 2
        disps.append(float((pc - gc).norm(dim=1).mean() / meta['S']))
    diffusion.train()
    return sum(disps) / len(disps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=str, default='base')
    ap.add_argument('--steps', type=int, default=200_000)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--d-model', type=int, default=384)
    ap.add_argument('--n-layers', type=int, default=12)
    ap.add_argument('--n-heads', type=int, default=8)
    ap.add_argument('--ffn', type=str, default='swiglu', choices=['swiglu', 'gelu'])
    ap.add_argument('--no-qk-norm', action='store_true')
    ap.add_argument('--edge-loss-weight', type=float, default=0.25,
                    help='connectivity-weighted relative-position aux loss (0=off)')
    ap.add_argument('--timesteps', type=int, default=1000)
    ap.add_argument('--max-files', type=int, default=0, help='limit shards (smoke)')
    ap.add_argument('--no-augment', action='store_true')
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--bf16', action='store_true', default=True)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--val-every', type=int, default=2000)
    ap.add_argument('--save-every', type=int, default=5000)
    ap.add_argument('--resume', type=str, default='')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ckpt_dir = Path(__file__).parent / 'checkpoints' / args.run
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print('indexing shards...')
    index = build_shard_index(max_files=args.max_files or None)
    print(f'{len(index)} shard files, n in '
          f'[{min(n for _, n in index)}, {max(n for _, n in index)}]')

    ds = ShardDataset(index, augment=not args.no_augment)
    sampler = BucketBatchSampler(index, args.batch_size, args.steps, seed=args.seed)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    cfg = ModelConfig(d_model=args.d_model, n_layers=args.n_layers,
                      n_heads=args.n_heads, ffn=args.ffn,
                      qk_norm=not args.no_qk_norm)
    model = FloorDiffNet(cfg).to(device)
    diffusion = FloorDiffusion(model, timesteps=args.timesteps).to(device)
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'model: {n_params / 1e6:.2f}M params')

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        ema.shadow.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt'])
        start_step = ck['step']
        print(f'resumed from {args.resume} @ step {start_step}')

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    def save(step, tag):
        torch.save({'model': model.state_dict(), 'ema': ema.shadow.state_dict(),
                    'opt': opt.state_dict(), 'step': step,
                    'config': cfg.to_dict(), 'timesteps': args.timesteps},
                   ckpt_dir / f'{tag}.pt')

    autocast = torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                              enabled=args.bf16 and device.type == 'cuda')
    log = {'loss': 0.0, 'k': 0}
    t0 = time.time()
    model.train()
    for step, batch in enumerate(loader, start=start_step):
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast:
            loss = diffusion.loss(batch, edge_weight=args.edge_loss_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

        log['loss'] += loss.item()
        log['k'] += 1
        if (step + 1) % args.log_every == 0:
            rate = log['k'] * batch['z0'].shape[0] / (time.time() - t0)
            print(f"step {step + 1:>7} | loss {log['loss'] / log['k']:.4f} | "
                  f"lr {lr_at(step):.2e} | {rate:.0f} samples/s")
            log = {'loss': 0.0, 'k': 0}
            t0 = time.time()
        if (step + 1) % args.val_every == 0:
            ema_diff = FloorDiffusion(ema.shadow, timesteps=args.timesteps).to(device)
            d = quick_val(ema_diff, device)
            print(f"step {step + 1:>7} | quick-val mean displacement {d:.4f} (xS)")
        if (step + 1) % args.save_every == 0:
            save(step + 1, f'step_{step + 1}')
            save(step + 1, 'last')
    save(args.steps, 'last')
    print(f'done. checkpoints in {ckpt_dir}')


if __name__ == '__main__':
    main()
