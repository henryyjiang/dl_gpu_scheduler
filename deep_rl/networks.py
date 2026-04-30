"""
Pointer-network Actor-Critic for the GPU scheduling agent.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent))
from environment import JOB_FEAT, GPU_FEAT


def _mlp(dims: list[int], activate_last: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or activate_last:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, num_gpus: int = 10, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self._scale    = math.sqrt(embed_dim)

        self.job_encoder = nn.Sequential(
            nn.Linear(JOB_FEAT, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.cluster_encoder = nn.Sequential(
            nn.Linear(num_gpus * GPU_FEAT, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.value_head = _mlp([embed_dim * 2, embed_dim, 1])

    def forward(
        self,
        job_feats: torch.Tensor,  # [B, Q, JOB_FEAT]
        gpu_feats: torch.Tensor,  # [B, num_gpus, GPU_FEAT]
        mask:      torch.Tensor,  # [B, Q]  True = valid action
    ) -> tuple[torch.Tensor, torch.Tensor]:
        job_embeds  = self.job_encoder(job_feats)
        cluster_ctx = self.cluster_encoder(gpu_feats.flatten(1))

        # Clamp prevents ±inf overflow from poisoning the PPO ratio backward.
        # -1e9 instead of -inf: MPS exp(-inf) returns NaN in logsumexp backward.
        logits = (
            torch.bmm(job_embeds, cluster_ctx.unsqueeze(-1)).squeeze(-1) / self._scale
        ).clamp(min=-50.0, max=50.0)
        logits = logits.masked_fill(~mask, -1e9)

        valid_f        = mask.unsqueeze(-1).float()
        denom          = valid_f.sum(dim=1).clamp(min=1.0)
        mean_job_embed = (job_embeds * valid_f).sum(dim=1) / denom
        value = self.value_head(
            torch.cat([cluster_ctx, mean_job_embed], dim=-1)
        ).squeeze(-1)

        return logits, value

    def act(
        self,
        job_feats: torch.Tensor,
        gpu_feats: torch.Tensor,
        mask:      torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(job_feats, gpu_feats, mask)
        dist   = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(
        self,
        job_feats: torch.Tensor,
        gpu_feats: torch.Tensor,
        mask:      torch.Tensor,
        actions:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(job_feats, gpu_feats, mask)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()
