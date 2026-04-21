"""
Dataset loader: reads the CSV, converts rows to Job objects,
and generates Poisson-process arrival times with a fixed seed.
"""

from __future__ import annotations
import csv
import io
import numpy as np
from typing import Optional
from models import Job

def _estimate_memory_gb(model_size: float, batch_size: int, seq_len: int) -> float:
    weights_mem = model_size
    activation_mem = (batch_size * seq_len) / 5000
    kv_overhead = model_size * 0.3
    total = weights_mem + activation_mem + kv_overhead
    return float(np.clip(total, 2.0, 78.0))

def load_jobs_from_csv(
    csv_path: str,
    max_jobs: Optional[int] = None,
) -> list[Job]:
    jobs: list[Job] = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if max_jobs is not None and idx >= max_jobs:
                break

            model_size = float(row["model_size"])
            batch_size = int(round(float(row["batch_size"])))
            seq_len = int(round(float(row["seq_len"])))
            latency = float(row["latency"])
            gpu_util = float(row["gpu_util"])

            mem_required = _estimate_memory_gb(model_size, batch_size, seq_len)

            jobs.append(Job(
                job_id=idx,
                gpu_mem_required=mem_required,
                gpu_util_intensity=min(gpu_util, 100.0),
                model_size=model_size,
                batch_size=batch_size,
                seq_len=seq_len,
                true_latency=latency,
            ))

    return jobs

def assign_poisson_arrivals(
    jobs: list[Job],
    arrival_rate: float,
    seed: int = 42,
) -> list[Job]:
    rng = np.random.default_rng(seed)

    inter_arrivals = rng.exponential(1.0 / arrival_rate, size=len(jobs))
    arrival_times = np.cumsum(inter_arrivals)

    for job, t in zip(jobs, arrival_times):
        job.arrival_time = float(t)

    jobs.sort(key=lambda j: j.arrival_time)
    return jobs
