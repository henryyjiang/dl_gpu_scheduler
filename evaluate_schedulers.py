import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIM_DIR = ROOT / "simulator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SIM_DIR) not in sys.path:
    sys.path.append(str(SIM_DIR))

from simulator.data_loader import load_jobs_from_csv, assign_poisson_arrivals
from simulator.simulator import Simulator
from simulator.schedulers import FIFOScheduler, SJFScheduler
from model.neural_scheduler_model import NeuralSchedulerModel
from model.neural_scheduler import NeuralScheduler


def main():
    parser = argparse.ArgumentParser(description="Evaluate FIFO, SJF, and Neural schedulers.")
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="model/neural_scheduler_imitation.pt",
        help="Path to trained neural scheduler checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for arrivals.")
    parser.add_argument("--num-gpus", type=int, default=10, help="Number of GPUs.")
    parser.add_argument("--gpu-memory", type=float, default=80.0, help="GPU memory (GB).")
    parser.add_argument(
        "--interference",
        type=float,
        default=0.0,
        help="Interference alpha used by simulator.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Optional job cap for quick evaluation.",
    )
    args = parser.parse_args()

    arrival_rates = [0.25, 0.5, 1.0]

    jobs_template = load_jobs_from_csv(args.csv, max_jobs=args.max_jobs)

    model = NeuralSchedulerModel()
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    schedulers = [
        ("FIFO", FIFOScheduler()),
        ("SJF", SJFScheduler()),
        ("Neural", NeuralScheduler(model)),
    ]

    for rate in arrival_rates:
        print(f"\n=== LOAD {rate} ===")

        base_jobs = assign_poisson_arrivals(
            copy.deepcopy(jobs_template),
            arrival_rate=rate,
            seed=args.seed,
        )

        for name, scheduler in schedulers:
            sim = Simulator(
                num_gpus=args.num_gpus,
                gpu_memory=args.gpu_memory,
                scheduler=scheduler,
                interference_alpha=args.interference,
            )
            jobs_for_run = copy.deepcopy(base_jobs)
            metrics = sim.run(jobs_for_run)

            print(f"Scheduler: {name}")
            print(f"Avg JCT: {metrics.avg_job_completion_time:.4f}")
            print(f"P99 JCT: {metrics.p99_jct:.4f}")
            print(f"Throughput: {metrics.throughput:.4f}")
            print(f"Avg GPU Utilisation: {metrics.avg_gpu_utilisation:.4f}")
            print(f"Max Queue Length: {metrics.max_queue_length}")
            print()


if __name__ == "__main__":
    import torch

    main()
