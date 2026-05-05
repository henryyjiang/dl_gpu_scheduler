import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIM_DIR = ROOT / "simulator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SIM_DIR) not in sys.path:
    sys.path.append(str(SIM_DIR))

try:
    from simulator.simulator import Simulator
    from simulator.data_loader import load_jobs_from_csv
    from model.neural_scheduler import collect_training_data
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", "unknown")
    raise SystemExit(
        f"Missing dependency/module: {missing}\n"
        "Install requirements in your active environment and retry."
    ) from exc


def main():
    parser = argparse.ArgumentParser(description="Verify imitation-learning dataset.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to dataset CSV (e.g., gpu_dataset/test.csv)",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Optional limit for faster checks.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=10,
        help="Number of GPUs in simulator cluster.",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=80.0,
        help="Memory per GPU in GB.",
    )
    args = parser.parse_args()

    jobs = load_jobs_from_csv(args.csv, max_jobs=args.max_jobs)
    simulator = Simulator(num_gpus=args.num_gpus, gpu_memory=args.gpu_memory)

    dataset = collect_training_data(
        simulator=simulator,
        jobs=jobs,
        arrival_rate=0.5,
        num_samples=1000,
    )

    print("=== DATASET CHECK ===")
    print(f"total samples: {len(dataset)}")
    if not dataset:
        print("No samples collected.")
        return

    sample0 = dataset[0]
    print(f"first sample keys: {list(sample0.keys())}")
    print(f"job_feats shape: {tuple(sample0['job_feats'].shape)}")
    print(f"gpu_feats shape: {tuple(sample0['gpu_feats'].shape)}")
    print(f"labels shape: {tuple(sample0['labels'].shape)}")

    print("\n=== SAMPLE 0 ===")
    print("labels tensor:")
    print(sample0["labels"])
    print("labels.argmax(dim=1):")
    print(sample0["labels"].argmax(dim=1))

    assigned_per_job = sample0["labels"].sum(dim=1) > 0
    num_assigned_jobs = int(assigned_per_job.sum().item())
    print(f"jobs with assignments in sample 0: {num_assigned_jobs}/{sample0['labels'].shape[0]}")
    assert sample0["labels"].sum(dim=1).max() <= 1, "A job has more than one assignment."

    print("\n=== PER-SAMPLE CONSISTENCY ===")
    for i, sample in enumerate(dataset):
        labels = sample["labels"]
        assigned = labels.sum(dim=1) > 0
        print(f"sample {i}: assigned_jobs={int(assigned.sum().item())}/{labels.shape[0]}")
        assert labels.sum(dim=1).max() <= 1, f"Sample {i}: a job has more than one assignment."

    print("\n=== RANDOM SAMPLE CHECKS ===")
    num_random = min(3, len(dataset))
    indices = random.sample(range(len(dataset)), k=num_random)
    for idx in indices:
        s = dataset[idx]
        labels = s["labels"]
        assigned = labels.sum(dim=1) > 0
        print(f"\n=== SAMPLE {idx} ===")
        print(f"job_feats shape: {tuple(s['job_feats'].shape)}")
        print(f"gpu_feats shape: {tuple(s['gpu_feats'].shape)}")
        print(f"labels shape: {tuple(labels.shape)}")
        print(f"jobs with assignments: {int(assigned.sum().item())}/{labels.shape[0]}")
        print(f"max assignments per job: {float(labels.sum(dim=1).max().item())}")

    print("\nDataset checks passed.")


if __name__ == "__main__":
    main()
