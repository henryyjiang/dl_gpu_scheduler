"""
Heuristic scheduling policies: FIFO and Shortest Job First (SJF).
"""

from __future__ import annotations

from models import Job, Cluster
from scheduler_interface import SchedulerInterface

def _greedy_assign(
    queue: list[Job],
    cluster: Cluster,
) -> list[tuple[Job, int]]:
    assignments: list[tuple[Job, int]] = []
    assigned_job_ids: set[int] = set()

    for job in queue:
        candidates = cluster.gpus_with_capacity(job)
        if not candidates:
            continue

        best_gpu = max(candidates, key=lambda g: g.free_memory)
        assignments.append((job, best_gpu.gpu_id))
        assigned_job_ids.add(job.job_id)

        # NOTE: not actually allocating memory here, simulator does
        # still count memory as to not double-hook
        best_gpu.used_memory += job.gpu_mem_required

    for job, gpu_id in assignments:
        gpu = cluster.gpus[gpu_id]
        gpu.used_memory -= job.gpu_mem_required

    return assignments

class FIFOScheduler(SchedulerInterface):
    @property
    def name(self) -> str:
        return "FIFO"

    def on_job_arrival(self, job, queue, cluster, current_time):
        return self._try_schedule(queue, cluster)

    def on_job_completion(self, completed_job, queue, cluster, current_time):
        return self._try_schedule(queue, cluster)

    def _try_schedule(self, queue, cluster):
        assignments = []
        while queue:
            head = queue[0]
            candidates = cluster.gpus_with_capacity(head)
            if not candidates:
                break 
            best_gpu = max(candidates, key=lambda g: g.free_memory)
            assignments.append((head, best_gpu.gpu_id))
            best_gpu.used_memory += head.gpu_mem_required
            queue.pop(0)

        for job, gpu_id in assignments:
            cluster.gpus[gpu_id].used_memory -= job.gpu_mem_required

        return assignments

class SJFScheduler(SchedulerInterface):
    @property
    def name(self) -> str:
        return "SJF"

    def on_job_arrival(self, job, queue, cluster, current_time):
        return self._try_schedule(queue, cluster)

    def on_job_completion(self, completed_job, queue, cluster, current_time):
        return self._try_schedule(queue, cluster)

    def _try_schedule(self, queue, cluster):
        queue.sort(key=lambda j: j.true_latency)
        return _greedy_assign(queue, cluster)
