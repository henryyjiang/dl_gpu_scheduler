"""
Inference wrapper: loads a checkpoint and implements SchedulerInterface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))
from models import Job, Cluster
from scheduler_interface import SchedulerInterface

sys.path.insert(0, str(Path(__file__).parent))
from environment import MAX_QUEUE, JOB_FEAT, GPU_FEAT, JOB_NORM, GPU_NORM
from networks import ActorCritic


class RLScheduler(SchedulerInterface):
    def __init__(
        self,
        checkpoint_path: str | Path,
        num_gpus: int = 10,
        embed_dim: int = 64,
        device: str = "cpu",
    ):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self._network = ActorCritic(num_gpus=num_gpus, embed_dim=embed_dim)
        self._network.load_state_dict(ckpt["state_dict"])
        self._network.eval()
        self._num_gpus = num_gpus
        self._device   = device

    @property
    def name(self) -> str:
        return "RL-PPO"

    def on_job_arrival(self, job, queue, cluster, current_time):
        return self._schedule(queue, cluster)

    def on_job_completion(self, completed_job, queue, cluster, current_time):
        return self._schedule(queue, cluster)

    def _schedule(
        self, queue: list[Job], cluster: Cluster
    ) -> list[tuple[Job, int]]:
        if not queue:
            return []

        assignments: list[tuple[Job, int]] = []
        working    = list(queue)
        temp_used: dict[int, float] = {g.gpu_id: 0.0 for g in cluster.gpus}

        while True:
            working.sort(key=lambda j: 0 if any(
                g.free_memory - temp_used[g.gpu_id] >= j.gpu_mem_required
                for g in cluster.gpus
            ) else 1)

            if not working or not any(
                g.free_memory - temp_used[g.gpu_id] >= working[0].gpu_mem_required
                for g in cluster.gpus
            ):
                break

            jf, gf, mask = self._build_obs(working, cluster, temp_used)
            jf_t = torch.FloatTensor(jf).unsqueeze(0).to(self._device)
            gf_t = torch.FloatTensor(gf).unsqueeze(0).to(self._device)
            mk_t = torch.BoolTensor(mask).unsqueeze(0).to(self._device)

            with torch.no_grad():
                action, _, _ = self._network.act(jf_t, gf_t, mk_t, deterministic=True)

            chosen = working[action.item()]
            best_gpu = max(
                [g for g in cluster.gpus
                 if g.free_memory - temp_used[g.gpu_id] >= chosen.gpu_mem_required],
                key=lambda g: g.free_memory - temp_used[g.gpu_id],
            )
            assignments.append((chosen, best_gpu.gpu_id))
            temp_used[best_gpu.gpu_id] += chosen.gpu_mem_required
            working.remove(chosen)

        return assignments

    def _build_obs(
        self,
        working:   list[Job],
        cluster:   Cluster,
        temp_used: dict[int, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        job_feats = np.zeros((MAX_QUEUE, JOB_FEAT), dtype=np.float32)
        mask      = np.zeros(MAX_QUEUE,              dtype=bool)

        for i, job in enumerate(working[:MAX_QUEUE]):
            raw = np.array([
                job.model_size, float(job.batch_size), float(job.seq_len),
                job.true_latency, job.gpu_mem_required, job.gpu_util_intensity,
            ], dtype=np.float32)
            job_feats[i] = raw / JOB_NORM
            mask[i] = any(
                g.free_memory - temp_used[g.gpu_id] >= job.gpu_mem_required
                for g in cluster.gpus
            )

        gpu_feats = np.zeros((self._num_gpus, GPU_FEAT), dtype=np.float32)
        for i, gpu in enumerate(cluster.gpus):
            eff_free = gpu.free_memory - temp_used[gpu.gpu_id]
            raw = np.array([eff_free, gpu.current_util, float(gpu.num_running)],
                           dtype=np.float32)
            gpu_feats[i] = raw / GPU_NORM

        return job_feats, gpu_feats, mask
