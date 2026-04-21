"""
Core data structures for the GPU task scheduling simulator.

Defines Jobs, GPU state, simulation events, and metrics collection.
"""

from __future__ import annotations
import dataclasses
import enum
from typing import Optional


class EventType(enum.Enum):
    JOB_ARRIVAL = "job_arrival"
    JOB_COMPLETION = "job_completion"


@dataclasses.dataclass(order=True)
class Event:
    time: float
    tiebreaker: int
    etype: EventType = dataclasses.field(compare=False)
    payload: dict = dataclasses.field(default_factory=dict, compare=False)


@dataclasses.dataclass
class Job:
    job_id: int

    gpu_mem_required: float    # GB of GPU memory needed
    gpu_util_intensity: float  # expected GPU utilisation (0-100)
    model_size: float          # model parameter size (GB)
    batch_size: int
    seq_len: int
    true_latency: float        # seconds
    arrival_time: float = 0.0
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    assigned_gpu: Optional[int] = None

    @property
    def wait_time(self) -> Optional[float]:
        if self.start_time is None:
            return None
        return self.start_time - self.arrival_time

    @property
    def turnaround_time(self) -> Optional[float]:
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time

    @property
    def is_completed(self) -> bool:
        return self.completion_time is not None

    @property
    def is_running(self) -> bool:
        return self.start_time is not None and self.completion_time is None


@dataclasses.dataclass
class GPU:
    gpu_id: int
    total_memory: float = 80.0
    used_memory: float = 0.0
    current_util: float = 0.0
    running_jobs: list = dataclasses.field(default_factory=list)
    _util_area: float = 0.0
    _last_update_time: float = 0.0

    @property
    def free_memory(self) -> float:
        return self.total_memory - self.used_memory

    @property
    def num_running(self) -> int:
        return len(self.running_jobs)

    def can_fit(self, job: Job) -> bool:
        return job.gpu_mem_required <= self.free_memory

    def allocate(self, job: Job, current_time: float) -> None:
        self._update_util_area(current_time)
        self.used_memory += job.gpu_mem_required
        self.current_util = min(100.0, self.current_util + job.gpu_util_intensity)
        self.running_jobs.append(job)

    def release(self, job: Job, current_time: float) -> None:
        self._update_util_area(current_time)
        self.used_memory = max(0.0, self.used_memory - job.gpu_mem_required)
        self.current_util = max(0.0, self.current_util - job.gpu_util_intensity)
        self.running_jobs = [j for j in self.running_jobs if j.job_id != job.job_id]

    def _update_util_area(self, current_time: float) -> None:
        dt = current_time - self._last_update_time
        if dt > 0:
            self._util_area += self.current_util * dt
            self._last_update_time = current_time

    def avg_utilisation(self, current_time: float) -> float:
        self._update_util_area(current_time)
        if current_time <= 0:
            return 0.0
        return self._util_area / current_time


@dataclasses.dataclass
class Cluster:
    gpus: list

    @staticmethod
    def create(num_gpus: int = 10, memory_per_gpu: float = 80.0) -> "Cluster":
        return Cluster(
            gpus=[GPU(gpu_id=i, total_memory=memory_per_gpu) for i in range(num_gpus)]
        )

    def idle_gpus(self) -> list:
        return [g for g in self.gpus if g.num_running == 0]

    def gpus_with_capacity(self, job: Job) -> list:
        return [g for g in self.gpus if g.can_fit(job)]

    def avg_cluster_util(self, current_time: float) -> float:
        if not self.gpus:
            return 0.0
        return sum(g.avg_utilisation(current_time) for g in self.gpus) / len(self.gpus)

@dataclasses.dataclass
class SimulationMetrics:
    scheduler_name: str
    num_jobs: int
    num_gpus: int
    makespan: float                # wall-clock from first arrival to last completion
    avg_job_completion_time: float
    avg_wait_time: float
    median_jct: float
    p95_jct: float
    p99_jct: float
    throughput: float
    avg_gpu_utilisation: float     # time-weighted mean across GPUs
    avg_queue_length: float        # time-weighted average queue depth
    max_queue_length: int

    def summary(self) -> str:
        lines = [
            f"=== {self.scheduler_name} | {self.num_jobs} jobs on {self.num_gpus} GPUs ===",
            f"  Makespan           : {self.makespan:>10.2f} s",
            f"  Avg JCT            : {self.avg_job_completion_time:>10.2f} s",
            f"  Median JCT         : {self.median_jct:>10.2f} s",
            f"  P95 JCT            : {self.p95_jct:>10.2f} s",
            f"  P99 JCT            : {self.p99_jct:>10.2f} s",
            f"  Avg Wait           : {self.avg_wait_time:>10.2f} s",
            f"  Throughput         : {self.throughput:>10.4f} jobs/s",
            f"  Avg GPU Util       : {self.avg_gpu_utilisation:>10.2f} %",
            f"  Avg Queue Depth    : {self.avg_queue_length:>10.2f}",
            f"  Max Queue Depth    : {self.max_queue_length:>10d}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
