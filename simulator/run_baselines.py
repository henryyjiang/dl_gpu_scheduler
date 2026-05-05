"""
Usage: python run_baselines.py --csv /gpu_datasets/test.csv  # your real data

Results are saved to results/ as both CSV and JSON for downstream analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from models import SimulationMetrics
from simulator import Simulator
from schedulers import FIFOScheduler, SJFScheduler
from model.neural_scheduler import NeuralScheduler
from model.neural_scheduler_model import NeuralSchedulerModel
from data_loader import (
    load_jobs_from_csv,
    assign_poisson_arrivals,
)


DEFAULT_SEED = 42
NUM_GPUS = 10
GPU_MEMORY_GB = 80.0

ARRIVAL_RATES = {
    "light":    0.10,
    "moderate": 0.25,
    "heavy":    0.50,
    "extreme":  1.0,
}

SCHEDULERS = [
    FIFOScheduler,
    SJFScheduler,
]

def run_experiment(
    jobs_template: list,
    arrival_rate: float,
    load_label: str,
    seed: int,
    num_gpus: int = NUM_GPUS,
    gpu_memory: float = GPU_MEMORY_GB,
    interference_alpha: float = 0.0,
) -> list[SimulationMetrics]:
    results = []

    for sched_cls in SCHEDULERS:
        import copy
        jobs = copy.deepcopy(jobs_template)
        jobs = assign_poisson_arrivals(jobs, arrival_rate=arrival_rate, seed=seed)

        scheduler = sched_cls()
        sim = Simulator(
            num_gpus=num_gpus,
            gpu_memory=gpu_memory,
            scheduler=scheduler,
            interference_alpha=interference_alpha,
        )

        t0 = time.perf_counter()
        metrics = sim.run(jobs)
        wall_time = time.perf_counter() - t0

        print(f"\n{metrics.summary()}")
        print(f"  Load level         : {load_label} (λ={arrival_rate})")
        print(f"  Sim wall-clock     : {wall_time:.3f} s")

        results.append(metrics)

    return results


def save_results(
    all_results: list[dict],
    output_dir: str,
) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "baseline_results.csv")
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    json_path = os.path.join(output_dir, "baseline_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="GPU Scheduler Baseline Runner")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to test.csv dataset")
    parser.add_argument("--max-jobs", type=int, default=None,
                        help="Limit number of jobs (useful for quick tests)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for Poisson arrivals")
    parser.add_argument("--num-gpus", type=int, default=NUM_GPUS,
                        help="Number of GPUs in the cluster")
    parser.add_argument("--gpu-memory", type=float, default=GPU_MEMORY_GB,
                        help="Memory per GPU in GB")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for output files")
    parser.add_argument("--interference", type=float, default=0.0,
                        help="Co-location interference factor alpha (0=none, 0.3=moderate)")
    args = parser.parse_args()

    num_gpus = args.num_gpus
    gpu_memory = args.gpu_memory
    interference_alpha = args.interference

    jobs_template = load_jobs_from_csv(args.csv, max_jobs=args.max_jobs)

    all_results = []

    for load_label, arrival_rate in ARRIVAL_RATES.items():

        metrics_list = run_experiment(
            jobs_template=jobs_template,
            arrival_rate=arrival_rate,
            load_label=load_label,
            seed=args.seed,
            num_gpus=num_gpus,
            gpu_memory=gpu_memory,
            interference_alpha=interference_alpha,
        )

        for m in metrics_list:
            row = m.to_dict()
            row["load_level"] = load_label
            row["arrival_rate"] = arrival_rate
            all_results.append(row)

    save_results(all_results, args.output_dir)


if __name__ == "__main__":
    main()