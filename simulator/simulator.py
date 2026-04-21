"""
Discrete Event Simulator for GPU task scheduling.
"""

from __future__ import annotations

import copy
import heapq
import numpy as np
from typing import Optional

from models import (
    Event,
    EventType,
    Job,
    GPU,
    Cluster,
    SimulationMetrics,
)
from scheduler_interface import SchedulerInterface


class Simulator:
    def __init__(
        self,
        num_gpus: int = 10,
        gpu_memory: float = 80.0,
        scheduler: Optional[SchedulerInterface] = None,
        interference_alpha: float = 0.0,
    ):
        self.num_gpus = num_gpus
        self.gpu_memory = gpu_memory
        self.scheduler = scheduler
        self.interference_alpha = interference_alpha

        self._cluster: Optional[Cluster] = None
        self._event_queue: list[Event] = []
        self._job_queue: list[Job] = []
        self._completed_jobs: list[Job] = []
        self._all_jobs: list[Job] = []
        self._clock: float = 0.0
        self._event_counter: int = 0

        self._queue_depth_area: float = 0.0
        self._last_queue_update_time: float = 0.0
        self._max_queue_depth: int = 0


    def run(self, jobs: list[Job], scheduler: Optional[SchedulerInterface] = None) -> SimulationMetrics:
        if scheduler is not None:
            self.scheduler = scheduler
        if self.scheduler is None:
            raise ValueError("No scheduler provided.")

        self._reset()
        self._all_jobs = copy.deepcopy(jobs)

        self._cluster = Cluster.create(self.num_gpus, self.gpu_memory)
        self.scheduler.on_simulation_start(self._cluster)

        for job in self._all_jobs:
            self._push_event(Event(
                time=job.arrival_time,
                tiebreaker=self._next_tiebreaker(),
                etype=EventType.JOB_ARRIVAL,
                payload={"job": job},
            ))

        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            self._clock = event.time

            if event.etype == EventType.JOB_ARRIVAL:
                self._handle_arrival(event)
            elif event.etype == EventType.JOB_COMPLETION:
                self._handle_completion(event)

        self.scheduler.on_simulation_end(self._cluster, self._clock)

        return self._compute_metrics()

    def _handle_arrival(self, event: Event) -> None:
        job: Job = event.payload["job"]

        self._update_queue_depth_area()
        self._job_queue.append(job)
        self._max_queue_depth = max(self._max_queue_depth, len(self._job_queue))

        assignments = self.scheduler.on_job_arrival(
            job=job,
            queue=self._job_queue,
            cluster=self._cluster,
            current_time=self._clock,
        )

        self._execute_assignments(assignments)

    def _handle_completion(self, event: Event) -> None:
        job: Job = event.payload["job"]
        gpu_id: int = event.payload["gpu_id"]

        job.completion_time = self._clock

        gpu = self._cluster.gpus[gpu_id]
        gpu.release(job, self._clock)

        self._completed_jobs.append(job)

        self._update_queue_depth_area()
        assignments = self.scheduler.on_job_completion(
            completed_job=job,
            queue=self._job_queue,
            cluster=self._cluster,
            current_time=self._clock,
        )

        self._execute_assignments(assignments)


    def _execute_assignments(self, assignments: list[tuple[Job, int]]) -> None:

        assigned_ids = set()

        for job, gpu_id in assignments:
            if job.job_id in assigned_ids:
                continue
            assigned_ids.add(job.job_id)

            self._update_queue_depth_area()
            self._job_queue = [j for j in self._job_queue if j.job_id != job.job_id]

            gpu = self._cluster.gpus[gpu_id]
            job.start_time = self._clock
            job.assigned_gpu = gpu_id
            gpu.allocate(job, self._clock)

            # Compute effective latency with co-location interference.
            other_util = max(0.0, gpu.current_util - job.gpu_util_intensity)
            slowdown = 1.0 + self.interference_alpha * (other_util / 100.0)
            effective_latency = job.true_latency * slowdown

            completion_time = self._clock + effective_latency
            self._push_event(Event(
                time=completion_time,
                tiebreaker=self._next_tiebreaker(),
                etype=EventType.JOB_COMPLETION,
                payload={"job": job, "gpu_id": gpu_id},
            ))

    def _compute_metrics(self) -> SimulationMetrics:
        if not self._completed_jobs:
            raise RuntimeError("No jobs completed — check your setup.")

        jcts = [j.turnaround_time for j in self._completed_jobs]
        waits = [j.wait_time for j in self._completed_jobs]
        jcts_arr = np.array(jcts)

        first_arrival = min(j.arrival_time for j in self._all_jobs)
        last_completion = max(j.completion_time for j in self._completed_jobs)
        makespan = last_completion - first_arrival

        self._update_queue_depth_area()

        return SimulationMetrics(
            scheduler_name=self.scheduler.name,
            num_jobs=len(self._completed_jobs),
            num_gpus=self.num_gpus,
            makespan=makespan,
            avg_job_completion_time=float(np.mean(jcts_arr)),
            avg_wait_time=float(np.mean(waits)),
            median_jct=float(np.median(jcts_arr)),
            p95_jct=float(np.percentile(jcts_arr, 95)),
            p99_jct=float(np.percentile(jcts_arr, 99)),
            throughput=len(self._completed_jobs) / makespan if makespan > 0 else 0.0,
            avg_gpu_utilisation=self._cluster.avg_cluster_util(self._clock),
            avg_queue_length=self._queue_depth_area / self._clock if self._clock > 0 else 0.0,
            max_queue_length=self._max_queue_depth,
        )

    def _reset(self) -> None:
        self._cluster = None
        self._event_queue = []
        self._job_queue = []
        self._completed_jobs = []
        self._all_jobs = []
        self._clock = 0.0
        self._event_counter = 0
        self._queue_depth_area = 0.0
        self._last_queue_update_time = 0.0
        self._max_queue_depth = 0

    def _push_event(self, event: Event) -> None:
        heapq.heappush(self._event_queue, event)

    def _next_tiebreaker(self) -> int:
        self._event_counter += 1
        return self._event_counter

    def _update_queue_depth_area(self) -> None:
        dt = self._clock - self._last_queue_update_time
        if dt > 0:
            self._queue_depth_area += len(self._job_queue) * dt
            self._last_queue_update_time = self._clock