"""
Step-based scheduling environment over the discrete-event simulator.
"""

from __future__ import annotations

import copy
import heapq
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))
from models import Job, Cluster, Event, EventType
from data_loader import assign_poisson_arrivals

MAX_QUEUE = 32   # queue slots exposed to the agent; shorter queues are zero-padded
JOB_FEAT  = 6    # model_size, batch_size, seq_len, latency, mem, util
GPU_FEAT  = 3    # free_mem, current_util, num_running

JOB_NORM = np.array([70.0, 128.0, 4096.0, 300.0, 78.0, 100.0], dtype=np.float32)
GPU_NORM = np.array([80.0, 100.0, 10.0],                         dtype=np.float32)

REWARD_SCALE = 100.0  # seconds → O(1) reward scale
TAIL_EXP     = 1.5    # convex JCT penalty; >1 weights outliers heavier than avg
WAIT_COEF    = 0.5    # anti-starvation weight


class SchedulingEnv:
    def __init__(
        self,
        jobs: list[Job],
        num_gpus: int = 10,
        gpu_memory: float = 80.0,
        interference_alpha: float = 0.0,
        tail_exp: float = TAIL_EXP,
        wait_coef: float = WAIT_COEF,
        completion_bonus: float = 0.0,
    ):
        self.all_jobs           = jobs
        self.num_gpus           = num_gpus
        self.gpu_memory         = gpu_memory
        self.interference_alpha = interference_alpha
        self.tail_exp           = tail_exp
        self.wait_coef          = wait_coef
        self.completion_bonus   = completion_bonus

    def reset(
        self,
        arrival_rate: float = 0.5,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if seed is None:
            seed = int(np.random.randint(0, 100_000))

        jobs = copy.deepcopy(self.all_jobs)
        jobs = assign_poisson_arrivals(jobs, arrival_rate, seed=seed)
        for i, j in enumerate(jobs):
            j.job_id = i

        self._clock:          float     = 0.0
        self._tb:             int       = 0
        self._cluster:        Cluster   = Cluster.create(self.num_gpus, self.gpu_memory)
        self._job_queue:      list[Job] = []
        self._completed_jobs: list[Job] = []
        self._event_queue:    list      = []
        self._obs_order:      list[Job] = []  # set by _obs(); used by step()

        for job in jobs:
            self._push(Event(
                time=job.arrival_time,
                tiebreaker=self._next_tb(),
                etype=EventType.JOB_ARRIVAL,
                payload={"job": job},
            ))

        _, done = self._advance_to_next_decision()
        if done:
            return None, None, None
        return self._obs()

    def step(self, action: int) -> tuple:
        job           = self._obs_order[action]
        feasible_gpus = self._cluster.gpus_with_capacity(job)
        if not feasible_gpus:
            # FP rounding on used_memory can make a job appear infeasible at
            # assignment time despite being feasible at observation time.
            for fallback in self._obs_order:
                feasible_gpus = self._cluster.gpus_with_capacity(fallback)
                if feasible_gpus:
                    job = fallback
                    break
        best_gpu = max(feasible_gpus, key=lambda g: g.free_memory)

        job.start_time   = self._clock
        job.assigned_gpu = best_gpu.gpu_id
        best_gpu.allocate(job, self._clock)
        self._job_queue.remove(job)

        other_util = max(0.0, best_gpu.current_util - job.gpu_util_intensity)
        slowdown   = 1.0 + self.interference_alpha * (other_util / 100.0)
        self._push(Event(
            time=self._clock + job.true_latency * slowdown,
            tiebreaker=self._next_tb(),
            etype=EventType.JOB_COMPLETION,
            payload={"job": job, "gpu_id": best_gpu.gpu_id},
        ))

        if self._can_assign_any():
            return *self._obs(), 0.0, False
        reward, done = self._advance_to_next_decision()
        if done:
            return None, None, None, reward, True
        return *self._obs(), reward, False

    def _advance_to_next_decision(self) -> tuple[float, bool]:
        reward = 0.0
        while self._event_queue:
            event       = heapq.heappop(self._event_queue)
            self._clock = event.time

            if event.etype == EventType.JOB_ARRIVAL:
                self._job_queue.append(event.payload["job"])

            elif event.etype == EventType.JOB_COMPLETION:
                job: Job            = event.payload["job"]
                job.completion_time = self._clock
                self._cluster.gpus[event.payload["gpu_id"]].release(job, self._clock)
                self._completed_jobs.append(job)
                reward += self.completion_bonus
                reward -= self.wait_coef * job.wait_time / REWARD_SCALE
                reward -= (job.turnaround_time / REWARD_SCALE) ** self.tail_exp

            if self._can_assign_any():
                return reward, False

        return reward, True

    def _can_assign_any(self) -> bool:
        return any(bool(self._cluster.gpus_with_capacity(j)) for j in self._job_queue)

    def _obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Feasible jobs first so the mask always has ≥1 True within MAX_QUEUE.
        feasible = [j for j in self._job_queue if self._cluster.gpus_with_capacity(j)]
        blocked  = [j for j in self._job_queue if not self._cluster.gpus_with_capacity(j)]
        self._obs_order = (feasible + blocked)[:MAX_QUEUE]

        job_feats = np.zeros((MAX_QUEUE, JOB_FEAT), dtype=np.float32)
        mask      = np.zeros(MAX_QUEUE,              dtype=bool)

        for i, job in enumerate(self._obs_order):
            raw = np.array([
                job.model_size, float(job.batch_size), float(job.seq_len),
                job.true_latency, job.gpu_mem_required, job.gpu_util_intensity,
            ], dtype=np.float32)
            job_feats[i] = raw / JOB_NORM
            mask[i]      = bool(self._cluster.gpus_with_capacity(job))

        gpu_feats = np.zeros((self.num_gpus, GPU_FEAT), dtype=np.float32)
        for i, gpu in enumerate(self._cluster.gpus):
            raw = np.array([gpu.free_memory, gpu.current_util, float(gpu.num_running)],
                           dtype=np.float32)
            gpu_feats[i] = raw / GPU_NORM

        return job_feats, gpu_feats, mask

    def _push(self, event: Event) -> None:
        heapq.heappush(self._event_queue, event)

    def _next_tb(self) -> int:
        self._tb += 1
        return self._tb
