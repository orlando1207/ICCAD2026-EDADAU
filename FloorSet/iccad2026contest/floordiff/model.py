"""FloorDiff denoiser: DiT-style Transformer with AdaLN-Zero conditioning and
connectivity-biased attention (Graphormer-style additive bias from pairwise features).

Input per block token: [z_t (3) | self-cond x0-hat (3) | static features (24)].
Conditioning vector: timestep embedding + global-feature embedding -> AdaLN-Zero.
Output: x0-hat (3 per block). All batches have uniform n (bucketed data), no padding mask.
"""

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_FEAT, N_GLOBAL, N_PAIR, Z_DIM


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    def to_dict(self):
        return asdict(self)


class TimestepEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class BiasedAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, bias):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                      # (B, H, N, hd)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = BiasedAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_model))
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)

    def forward(self, x, c, bias):
        sh1, sc1, g1, sh2, sc2, g2 = self.adaln(c)[:, None, :].chunk(6, dim=-1)
        x = x + g1 * self.attn(self.norm1(x) * (1 + sc1) + sh1, bias)
        x = x + g2 * self.mlp(self.norm2(x) * (1 + sc2) + sh2)
        return x


class FloorDiffNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.token_in = nn.Linear(Z_DIM + Z_DIM + N_FEAT, d)   # z_t | self-cond | static
        self.t_embed = TimestepEmbed(d)
        self.g_embed = nn.Sequential(nn.Linear(N_GLOBAL, d), nn.SiLU(), nn.Linear(d, d))
        self.bias_mlp = nn.Sequential(nn.Linear(N_PAIR, 32), nn.SiLU(),
                                      nn.Linear(32, cfg.n_heads))
        self.blocks = nn.ModuleList([
            DiTBlock(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_layers)])
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False)
        self.adaln_out = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.head = nn.Linear(d, Z_DIM)
        nn.init.zeros_(self.adaln_out[1].weight)
        nn.init.zeros_(self.adaln_out[1].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z_t, t, feat, pair, gfeat, self_cond=None):
        """z_t (B,N,3), t (B,), feat (B,N,F), pair (B,N,N,P), gfeat (B,G) -> x0-hat (B,N,3)."""
        if self_cond is None:
            self_cond = torch.zeros_like(z_t)
        x = self.token_in(torch.cat([z_t, self_cond, feat], dim=-1))
        c = self.t_embed(t) + self.g_embed(gfeat)
        bias = self.bias_mlp(pair).permute(0, 3, 1, 2)          # (B, H, N, N)
        for blk in self.blocks:
            x = blk(x, c, bias)
        sh, sc = self.adaln_out(c)[:, None, :].chunk(2, dim=-1)
        return self.head(self.norm_out(x) * (1 + sc) + sh)
