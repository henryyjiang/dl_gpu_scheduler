import torch
import copy
import random

from simulator.scheduler_interface import SchedulerInterface
from simulator.schedulers import SJFScheduler
from simulator.data_loader import assign_poisson_arrivals
from simulator.simulator import Simulator


class NeuralScheduler(SchedulerInterface):
    def __init__(self, model):
        self.model = model
        self.model.eval()  

    @property
    def name(self):
        return "Neural"

    @staticmethod
    def _build_features(queue, cluster):
        """
        Convert simulator state → tensors
        """

        job_feats = []
        for j in queue:
            job_feats.append([
                j.model_size,
                j.batch_size,
                j.seq_len,
                j.gpu_util_intensity,
                j.gpu_mem_required
            ])

        gpu_feats = []
        for g in cluster.gpus:
            gpu_feats.append([
                g.free_memory,
                g.current_util,
                g.num_running
            ])

        job_feats = torch.tensor(job_feats, dtype=torch.float32)
        gpu_feats = torch.tensor(gpu_feats, dtype=torch.float32)

        if job_feats.numel() > 0:
            job_feats[:, 0] /= 20.0
            job_feats[:, 1] /= 32.0
            job_feats[:, 2] /= 2048.0
            job_feats[:, 3] /= 100.0
            job_feats[:, 4] /= 80.0

        if gpu_feats.numel() > 0:
            gpu_feats[:, 0] /= 80.0
            gpu_feats[:, 1] /= 100.0
            gpu_feats[:, 2] /= 10.0

        return job_feats, gpu_feats

    def _schedule(self, queue, cluster):
        if len(queue) == 0:
            return []

        job_feats, gpu_feats = self._build_features(queue, cluster)

        with torch.no_grad():
            scores = self.model(job_feats, gpu_feats)  # (N_jobs, K_gpus)

        assignments = []
        used_jobs = set()

        while True:
            best = None
            best_score = -1e9

            for j_idx, job in enumerate(queue):
                if job.job_id in used_jobs:
                    continue

                for g_idx, g in enumerate(cluster.gpus):

                    if not g.can_fit(job):
                        continue

                    s = scores[j_idx, g_idx].item()

                    if s > best_score:
                        best_score = s
                        best = (j_idx, g_idx)

            if best is None:
                break

            j_idx, g_idx = best
            job = queue[j_idx]

            assignments.append((job, g_idx))
            used_jobs.add(job.job_id)

            cluster.gpus[g_idx].used_memory += job.gpu_mem_required

        # undo fake memory
        for job, g_idx in assignments:
            cluster.gpus[g_idx].used_memory -= job.gpu_mem_required

        return assignments

    def on_job_arrival(self, job, queue, cluster, current_time):
        return self._schedule(queue, cluster)

    def on_job_completion(self, completed_job, queue, cluster, current_time):
        return self._schedule(queue, cluster)


class _CollectionComplete(Exception):
    pass


class _SJFDataCollectorScheduler(SchedulerInterface):
    def __init__(self, num_samples, shuffle_queue=False):
        self.num_samples = num_samples
        self.shuffle_queue = shuffle_queue
        self.teacher = SJFScheduler()
        self.samples = []

    @property
    def name(self):
        return "SJFDataCollector"

    def _collect_one(self, queue, cluster):
        if len(self.samples) >= self.num_samples:
            raise _CollectionComplete()

        if not queue or len(queue) < 2:
            return

        feature_queue = list(queue)
        if self.shuffle_queue:
            random.shuffle(feature_queue)

        job_id_to_idx = {job.job_id: idx for idx, job in enumerate(feature_queue)}
        num_jobs = len(feature_queue)
        num_gpus = len(cluster.gpus)

        queue_copy = copy.deepcopy(feature_queue)
        cluster_copy = copy.deepcopy(cluster)

        teacher_assignments = self.teacher._try_schedule(queue_copy, cluster_copy)
        if not teacher_assignments:
            return

        labels = torch.zeros((num_jobs, num_gpus), dtype=torch.float32)
        for job, gpu_id in teacher_assignments:
            j_idx = job_id_to_idx.get(job.job_id)
            if j_idx is not None:
                labels[j_idx, gpu_id] = 1.0

        if labels.sum().item() <= 0:
            return

        job_feats, gpu_feats = NeuralScheduler._build_features(feature_queue, cluster)

        self.samples.append({
            "job_feats": job_feats,
            "gpu_feats": gpu_feats,
            "labels": labels,
        })

        if len(self.samples) >= self.num_samples:
            raise _CollectionComplete()

    def on_job_arrival(self, job, queue, cluster, current_time):
        self._collect_one(queue, cluster)
        return self.teacher.on_job_arrival(job, queue, cluster, current_time)

    def on_job_completion(self, completed_job, queue, cluster, current_time):
        self._collect_one(queue, cluster)
        return self.teacher.on_job_completion(completed_job, queue, cluster, current_time)


def collect_training_data(simulator, jobs, arrival_rate, num_samples):
    original_scheduler = simulator.scheduler
    collected = []
    rates = arrival_rate if isinstance(arrival_rate, (list, tuple)) else [arrival_rate]

    try:
        for rate in rates:
            remaining = num_samples - len(collected)
            if remaining <= 0:
                break

            collector = _SJFDataCollectorScheduler(num_samples=remaining, shuffle_queue=False)
            jobs_with_arrivals = assign_poisson_arrivals(
                copy.deepcopy(jobs),
                arrival_rate=rate,
            )
            simulator.scheduler = collector

            try:
                simulator.run(jobs_with_arrivals)
            except _CollectionComplete:
                pass

            collected.extend(collector.samples)
    finally:
        simulator.scheduler = original_scheduler

    return collected


def collect_rich_training_data(simulator, jobs, target_samples):
    arrival_rates = [0.25, 0.5, 1.0]
    min_assigned_jobs = 2
    rich_dataset = []
    preferred_samples = []
    fallback_samples = []

    if len(jobs) < 100:
        print(
            f"[collect_rich_training_data] Warning: job pool is small ({len(jobs)} jobs). "
            "Use a larger dataset for richer queue states."
        )

    base_memory = getattr(simulator, "gpu_memory", 80.0)
    memory_settings = sorted({base_memory, max(base_memory, 100.0), 120.0})
    max_rounds = 8

    for _ in range(max_rounds):
        if len(rich_dataset) >= target_samples:
            break

        made_progress = False

        for memory in memory_settings:
            for rate in arrival_rates:
                remaining = target_samples - len(rich_dataset)
                if remaining <= 0:
                    break

                # Over-sample per pass so we can keep richer states.
                request_count = max(128, remaining * 3)
                temp_simulator = Simulator(
                    num_gpus=simulator.num_gpus,
                    gpu_memory=memory,
                    interference_alpha=simulator.interference_alpha,
                )
                batch = collect_training_data(
                    simulator=temp_simulator,
                    jobs=jobs,
                    arrival_rate=rate,
                    num_samples=request_count,
                )

                if batch:
                    made_progress = True

                for sample in batch:
                    assigned_jobs = int((sample["labels"].sum(dim=1) > 0).sum().item())
                    if assigned_jobs >= min_assigned_jobs:
                        preferred_samples.append(sample)
                    else:
                        fallback_samples.append(sample)

                while preferred_samples and len(rich_dataset) < target_samples:
                    rich_dataset.append(preferred_samples.pop(0))

        if not made_progress:
            break

    # If rich states are insufficient, backfill to hit the requested size.
    while fallback_samples and len(rich_dataset) < target_samples:
        rich_dataset.append(fallback_samples.pop(0))

    if rich_dataset:
        avg_queue_size = sum(s["job_feats"].shape[0] for s in rich_dataset) / len(rich_dataset)
        avg_assigned_jobs = sum(
            int((s["labels"].sum(dim=1) > 0).sum().item()) for s in rich_dataset
        ) / len(rich_dataset)
        pct_multi_assigned = (
            sum(
                1
                for s in rich_dataset
                if int((s["labels"].sum(dim=1) > 0).sum().item()) >= min_assigned_jobs
            )
            / len(rich_dataset)
        )
    else:
        avg_queue_size = 0.0
        avg_assigned_jobs = 0.0
        pct_multi_assigned = 0.0

    print("=== RICH DATASET SUMMARY ===")
    print(f"target samples: {target_samples}")
    print(f"collected samples: {len(rich_dataset)}")
    print(f"average queue size: {avg_queue_size:.2f}")
    print(f"average assigned jobs per sample: {avg_assigned_jobs:.2f}")
    print(f"fraction with >= {min_assigned_jobs} assigned jobs: {pct_multi_assigned:.3f}")

    return rich_dataset