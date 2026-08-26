"""Diffusion wrapper: cosine schedule, x0-prediction training loss (min-SNR weighted,
frozen channels masked), DDIM sampler with inpainting of known channels, EMA.

Per the imitation-first design (doc C.0/C.4/C.5): the primary loss is masked MSE to
ground truth; sampling is a plain conditional DDIM plus inpainting — no physics
guidance, no polish.

Optional auxiliary term (`edge_weight` > 0): a connectivity-weighted relative-position
loss on b2b-connected block pairs. This is still pure imitation — it supervises the
predicted *center differences* of connected blocks toward the GT center differences,
weighted by net strength — but it focuses model capacity on exactly the relative
arrangement that weighted HPWL rewards, rather than treating every block's absolute
position as equally important.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Z_CLAMP


def cosine_alphas_bar(T, s=0.008):
    t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-8, 1.0)
    return ab[1:].float()          # (T,) alphas_bar at t = 1..T (index t-1)


class FloorDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, min_snr_gamma=5.0, self_cond_p=0.5):
        super().__init__()
        self.model = model
        self.T = timesteps
        self.min_snr_gamma = min_snr_gamma
        self.self_cond_p = self_cond_p
        ab = cosine_alphas_bar(timesteps)
        self.register_buffer('alphas_bar', ab)
        self.register_buffer('sqrt_ab', ab.sqrt())
        self.register_buffer('sqrt_1mab', (1 - ab).sqrt())

    def q_sample(self, x0, t, noise):
        """t: (B,) int in [0, T)."""
        a = self.sqrt_ab[t][:, None, None]
        b = self.sqrt_1mab[t][:, None, None]
        return a * x0 + b * noise

    # ------------------------------------------------------------------ training

    def loss(self, batch, edge_weight=0.0):
        z0, feat, pair, gfeat = batch['z0'], batch['feat'], batch['pair'], batch['gfeat']
        freeze = batch['freeze']                                  # (B, N, 3) bool
        B = z0.shape[0]
        t = torch.randint(0, self.T, (B,), device=z0.device)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)

        self_cond = None
        if self.self_cond_p > 0 and torch.rand(()) < self.self_cond_p:
            with torch.no_grad():
                self_cond = self.model(z_t, t, feat, pair, gfeat).detach()
        x0_pred = self.model(z_t, t, feat, pair, gfeat, self_cond=self_cond)

        snr = self.alphas_bar[t] / (1 - self.alphas_bar[t])
        wsnr = snr.clamp(max=self.min_snr_gamma)                  # min-SNR for x0-param
        w = wsnr[:, None, None]
        mask = (~freeze).float()
        per = F.mse_loss(x0_pred, z0, reduction='none') * mask * w
        main = per.sum() / (mask * w).sum().clamp(min=1e-8)
        if edge_weight <= 0:
            return main

        # connectivity-weighted relative-position term (center channels only).
        # pair[..., 0] is the symmetric log-b2b-weight matrix built in featurize,
        # so it doubles as a dense per-edge weight -> no extra data plumbing.
        W = pair[..., 0]                                          # (B, N, N) >= 0
        cp = x0_pred[..., :2]                                     # (B, N, 2)
        cg = z0[..., :2]
        dpred = cp[:, :, None, :] - cp[:, None, :, :]             # (B, N, N, 2)
        dgt = cg[:, :, None, :] - cg[:, None, :, :]
        se = ((dpred - dgt) ** 2).sum(-1)                         # (B, N, N)
        Ws = W * wsnr[:, None, None]
        edge = (Ws * se).sum() / Ws.sum().clamp(min=1e-8)
        return main + edge_weight * edge

    # ------------------------------------------------------------------ sampling

    @torch.no_grad()
    def sample(self, feat, pair, gfeat, z_known, freeze, steps=50, seed=None,
               device=None):
        """DDIM (eta=0) with inpainting: frozen channels are re-imposed at every step
        as q_sample(z_known, t) and set exactly at the end.
        All inputs batched (B leading dim; replicate a case B times for best-of-N).
        Returns x0-hat (B, N, 3)."""
        device = device or feat.device
        B, N, _ = z_known.shape
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(seed)
        z_t = torch.randn(B, N, 3, device=device, generator=gen)

        ts = torch.linspace(self.T - 1, 0, steps, device=device).long()
        self_cond = None
        for k in range(steps):
            t = ts[k].expand(B)
            # inpaint known channels at the current noise level
            noise = torch.randn(B, N, 3, device=device, generator=gen)
            zk_t = self.q_sample(z_known, t, noise)
            z_t = torch.where(freeze, zk_t, z_t)

            x0 = self.model(z_t, t, feat, pair, gfeat, self_cond=self_cond)
            x0 = x0.clamp(-Z_CLAMP, Z_CLAMP)
            self_cond = x0

            ab_t = self.alphas_bar[t][:, None, None]
            eps = (z_t - ab_t.sqrt() * x0) / (1 - ab_t).sqrt().clamp(min=1e-8)
            if k + 1 < steps:
                t_prev = ts[k + 1].expand(B)
                ab_p = self.alphas_bar[t_prev][:, None, None]
                z_t = ab_p.sqrt() * x0 + (1 - ab_p).sqrt() * eps
            else:
                z_t = x0
        return torch.where(freeze, z_known, z_t)


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p, 1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)
