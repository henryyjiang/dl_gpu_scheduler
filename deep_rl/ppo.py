"""
PPO trainer with per-arrival-rate Welford reward normalisation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))
from environment import SchedulingEnv


class RewardNormalizer:
    """Welford online normaliser keyed by arrival rate.

    λ=0.5 and λ=1.0 produce ~78× different reward magnitudes; normalising
    per-bucket keeps value-loss O(1) regardless of load level.
    """

    def __init__(self, clip: float = 10.0) -> None:
        self._clip  = clip
        self._stats: dict[float, dict] = {}

    def _key(self, rate: float) -> float:
        return round(rate, 2)

    def update_and_normalize(self, reward: float, rate: float) -> float:
        k = self._key(rate)
        if k not in self._stats:
            self._stats[k] = {"mean": 0.0, "m2": 0.0, "n": 0}
        s = self._stats[k]
        s["n"] += 1
        delta     = reward - s["mean"]
        s["mean"] += delta / s["n"]
        s["m2"]   += delta * (reward - s["mean"])  # Welford

        if s["n"] < 2:
            return 0.0  # not enough history to estimate std
        std = max((s["m2"] / (s["n"] - 1)) ** 0.5, 1e-8)
        return float(np.clip((reward - s["mean"]) / std, -self._clip, self._clip))


class RolloutBuffer:
    """One episode of (state, action, reward, …) transitions."""

    def __init__(self) -> None:
        self.job_feats: list[np.ndarray] = []
        self.gpu_feats: list[np.ndarray] = []
        self.masks:     list[np.ndarray] = []
        self.actions:   list[int]        = []
        self.log_probs: list[float]      = []
        self.rewards:   list[float]      = []
        self.values:    list[float]      = []
        self.dones:     list[bool]       = []

    def add(
        self,
        job_feats: np.ndarray,
        gpu_feats: np.ndarray,
        mask:      np.ndarray,
        action:    int,
        log_prob:  float,
        reward:    float,
        value:     float,
        done:      bool,
    ) -> None:
        self.job_feats.append(job_feats)
        self.gpu_feats.append(gpu_feats)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.rewards)


class PPOTrainer:
    def __init__(
        self,
        network:        nn.Module,
        lr:             float = 3e-4,
        clip_eps:       float = 0.2,
        vf_coef:        float = 0.5,
        ent_coef:       float = 0.01,
        gamma:          float = 0.99,
        gae_lambda:     float = 0.95,
        n_epochs:       int   = 4,
        batch_size:     int   = 256,
        max_grad_norm:  float = 0.5,
        device:         str   = "cpu",
    ):
        self.network        = network
        self.optimizer      = optim.Adam(network.parameters(), lr=lr)
        self.clip_eps       = clip_eps
        self.vf_coef        = vf_coef
        self.ent_coef       = ent_coef
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda
        self.n_epochs       = n_epochs
        self.batch_size     = batch_size
        self.max_grad_norm  = max_grad_norm
        self.device         = device
        self.buffer             = RolloutBuffer()
        self._reward_normalizer = RewardNormalizer()

    def collect_episode(
        self,
        env:          SchedulingEnv,
        arrival_rate: float        = 0.5,
        seed:         Optional[int] = None,
    ) -> int:
        obs = env.reset(arrival_rate=arrival_rate, seed=seed)
        if obs[0] is None:
            return 0

        job_feats, gpu_feats, mask = obs
        self.network.eval()
        steps = 0

        while True:
            jf = torch.FloatTensor(job_feats).unsqueeze(0).to(self.device)
            gf = torch.FloatTensor(gpu_feats).unsqueeze(0).to(self.device)
            mk = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, value = self.network.act(jf, gf, mk)

            next_jf, next_gf, next_mk, reward, done = env.step(action.item())
            reward_norm = self._reward_normalizer.update_and_normalize(reward, arrival_rate)
            self.buffer.add(
                job_feats, gpu_feats, mask,
                action.item(), log_prob.item(), reward_norm, value.item(), done,
            )
            steps += 1

            if done:
                break
            job_feats, gpu_feats, mask = next_jf, next_gf, next_mk

        return steps

    def _compute_gae(self) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        values  = np.array(self.buffer.values,  dtype=np.float32)
        dones   = np.array(self.buffer.dones,   dtype=np.float32)
        N       = len(rewards)

        advantages = np.zeros(N, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(N)):
            not_done = 1.0 - dones[t]
            next_val = values[t + 1] if t + 1 < N else 0.0
            delta    = rewards[t] + self.gamma * not_done * next_val - values[t]
            gae      = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self) -> dict[str, float]:
        advantages, returns = self._compute_gae()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        jf   = torch.FloatTensor(np.array(self.buffer.job_feats)).to(self.device)
        gf   = torch.FloatTensor(np.array(self.buffer.gpu_feats)).to(self.device)
        mk   = torch.BoolTensor (np.array(self.buffer.masks    )).to(self.device)
        acts = torch.LongTensor (np.array(self.buffer.actions  )).to(self.device)
        olp  = torch.FloatTensor(np.array(self.buffer.log_probs)).to(self.device)
        adv  = torch.FloatTensor(advantages                     ).to(self.device)
        ret  = torch.FloatTensor(returns                        ).to(self.device)

        N = len(acts)
        self.network.train()
        total_p = total_v = total_e = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            perm = np.random.permutation(N)
            for start in range(0, N, self.batch_size):
                idx = torch.LongTensor(perm[start : start + self.batch_size]).to(self.device)

                log_probs, values, entropy = self.network.evaluate_actions(
                    jf[idx], gf[idx], mk[idx], acts[idx]
                )

                ratio  = (log_probs - olp[idx]).exp()
                surr1  = ratio * adv[idx]
                surr2  = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv[idx]
                p_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(values, ret[idx])
                e_loss = -entropy.mean()

                loss = p_loss + self.vf_coef * v_loss + self.ent_coef * e_loss
                self.optimizer.zero_grad()
                loss.backward()
                total_norm = nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                if torch.isfinite(total_norm):
                    self.optimizer.step()

                total_p   += p_loss.item()
                total_v   += v_loss.item()
                total_e   += entropy.mean().item()
                n_updates += 1

        self.buffer.clear()
        return {
            "policy_loss": total_p / n_updates,
            "value_loss":  total_v / n_updates,
            "entropy":     total_e / n_updates,
        }
