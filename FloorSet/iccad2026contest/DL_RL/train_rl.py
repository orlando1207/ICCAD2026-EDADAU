"""
Phase 4 — PPO training loop (VERSION_B_RL_PLACER.md §5 Phase 4).

Ties PlacementEnv + GNNEncoder + PolicyValueNet into a PPO actor-critic:
  collect episodes -> GAE advantages -> clipped PPO update.

The netlist embedding (GNN) is static per problem, so it is encoded once per
episode at reset and reused for every block decision. The canvas raster is
rebuilt each step (it depends on the partial placement). Reward is terminal
only for now (-contest cost); per-step shaping is a later add (§4).

This module is import-safe and the acceptance test (test_train_rl.py)
overfits a tiny fixed sample set to confirm the mean cost goes DOWN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from canvas_raster import rasterize_env, N_CHANNELS, CH_FEASIBILITY
from gnn_encoder import GNNEncoder
from placement_env import PlacementEnv
from policy_net import PolicyValueNet


@dataclass
class Step:
    canvas: torch.Tensor          # [C,G,G] fixed input (detached)
    block: int                    # current block id
    mask: torch.Tensor           # [G,G] feasibility (detached)
    action: int
    old_logp: float
    value: float
    reward: float
    advantage: float = 0.0
    ret: float = 0.0


@dataclass
class Episode:
    area: torch.Tensor
    constraints: torch.Tensor
    b2b: torch.Tensor
    block_count: int
    steps: List[Step] = field(default_factory=list)
    terminal_cost: float = 0.0


class PPOAgent:
    def __init__(self, gnn: GNNEncoder, policy: PolicyValueNet,
                 device: str = "cpu"):
        self.gnn = gnn.to(device)
        self.policy = policy.to(device)
        self.device = device

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def collect_episode(self, env: PlacementEnv, sample: dict) -> Episode:
        at, b2b, p2b, pins, cons = sample["input"]
        _fp, metrics = sample["label"]
        env.reset(at, b2b, p2b, pins, cons, metrics)
        bc = env.block_count

        node_emb, _ = self.gnn.encode_problem(at, cons, b2b, bc, self.device)
        ep = Episode(area=at, constraints=cons, b2b=b2b, block_count=bc)

        done = False
        while not done:
            state = env._build_state()
            cur = state["current_block"]
            canvas = rasterize_env(env)
            mask = canvas[CH_FEASIBILITY]
            probs, value, _ = self.policy(canvas, node_emb[cur], mask)
            action = int(torch.multinomial(probs, 1))
            logp = float(torch.log(probs[action].clamp_min(1e-12)))
            _s, reward, done, info = env.step(action)
            ep.steps.append(Step(canvas=canvas.detach(), block=cur,
                                 mask=mask.detach(), action=action,
                                 old_logp=logp, value=float(value),
                                 reward=float(reward)))
        ep.terminal_cost = info["contest_cost"]
        return ep

    @staticmethod
    def compute_gae(ep: Episode, gamma: float, lam: float) -> None:
        adv = 0.0
        next_value = 0.0  # terminal
        for t in reversed(range(len(ep.steps))):
            s = ep.steps[t]
            delta = s.reward + gamma * next_value - s.value
            adv = delta + gamma * lam * adv
            s.advantage = adv
            s.ret = adv + s.value
            next_value = s.value

    # ------------------------------------------------------------------ #
    def update(self, episodes: List[Episode], optimizer: torch.optim.Optimizer,
               clip: float = 0.2, epochs: int = 4, c_value: float = 0.5,
               c_entropy: float = 0.01) -> dict:
        # flatten advantages for normalization
        advs = torch.tensor([s.advantage for ep in episodes for s in ep.steps])
        adv_mean, adv_std = advs.mean(), advs.std().clamp_min(1e-6)

        last = {}
        for _ in range(epochs):
            optimizer.zero_grad()
            pol_loss = torch.tensor(0.0, device=self.device)
            val_loss = torch.tensor(0.0, device=self.device)
            ent_sum = torch.tensor(0.0, device=self.device)
            n = 0
            for ep in episodes:
                node_emb, _ = self.gnn.encode_problem(
                    ep.area, ep.constraints, ep.b2b, ep.block_count, self.device)
                for s in ep.steps:
                    probs, value, _ = self.policy(s.canvas, node_emb[s.block], s.mask)
                    logp = torch.log(probs[s.action].clamp_min(1e-12))
                    ratio = torch.exp(logp - s.old_logp)
                    a = (s.advantage - adv_mean) / adv_std
                    surr1 = ratio * a
                    surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * a
                    pol_loss = pol_loss - torch.min(surr1, surr2)
                    val_loss = val_loss + (value - s.ret) ** 2
                    ent_sum = ent_sum - (probs * torch.log(probs.clamp_min(1e-12))).sum()
                    n += 1
            loss = (pol_loss + c_value * val_loss - c_entropy * ent_sum) / max(n, 1)
            loss.backward()
            optimizer.step()
            last = {"loss": float(loss), "pol": float(pol_loss / max(n, 1)),
                    "val": float(val_loss / max(n, 1))}
        return last


def train(samples: List[dict], iters: int = 30, grid: int = 32,
          gnn_out: int = 64, hidden: int = 32, lr: float = 3e-4,
          gamma: float = 0.99, lam: float = 0.95, rollouts: int = 8,
          warmstart_ckpt: Optional[str] = None,
          seed: int = 0, verbose: bool = True) -> List[float]:
    """Overfit/train on a fixed set of samples. Returns mean terminal cost per iter.

    `rollouts`: episodes collected per sample per iteration. PPO is high-variance
    with a tiny batch — a single rollout per sample barely learns; ~8 makes the
    advantage estimate usable and the mean cost drops reliably.
    `warmstart_ckpt`: path to a Phase 4.5 BC checkpoint to RL-fine-tune from.
    When given, gnn/policy arch + grid are taken from the checkpoint so the
    weights load and the canvas semantics match what BC trained on.
    """
    torch.manual_seed(seed)
    if warmstart_ckpt is not None:
        ckpt = torch.load(warmstart_ckpt, map_location="cpu")
        gnn_out, hidden, grid = ckpt["gnn_out"], ckpt["hidden"], ckpt["grid"]
        gnn = GNNEncoder(out_dim=gnn_out, hidden=hidden)
        policy = PolicyValueNet(in_channels=N_CHANNELS, node_dim=gnn_out, hidden=hidden)
        gnn.load_state_dict(ckpt["gnn"])
        policy.load_state_dict(ckpt["policy"])
        if verbose:
            print(f"[warmstart] loaded {warmstart_ckpt} "
                  f"(grid={grid}, gnn_out={gnn_out}, hidden={hidden})")
    else:
        gnn = GNNEncoder(out_dim=gnn_out, hidden=hidden)
        policy = PolicyValueNet(in_channels=N_CHANNELS, node_dim=gnn_out, hidden=hidden)
    agent = PPOAgent(gnn, policy)
    optimizer = torch.optim.Adam(
        list(gnn.parameters()) + list(policy.parameters()), lr=lr)
    env = PlacementEnv(grid=grid)

    history = []
    for it in range(iters):
        episodes = [agent.collect_episode(env, s)
                    for _ in range(rollouts) for s in samples]
        for ep in episodes:
            agent.compute_gae(ep, gamma, lam)
        stats = agent.update(episodes, optimizer)
        mean_cost = sum(ep.terminal_cost for ep in episodes) / len(episodes)
        history.append(mean_cost)
        if verbose:
            print(f"iter {it:3d}  mean_cost={mean_cost:.4f}  loss={stats['loss']:.4f}")
    return history
