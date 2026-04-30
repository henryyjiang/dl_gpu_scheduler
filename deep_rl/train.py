"""
PPO training loop for the GPU scheduler.
"""

from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))
from data_loader import load_jobs_from_csv, assign_poisson_arrivals
from simulator import Simulator
from schedulers import FIFOScheduler, SJFScheduler

sys.path.insert(0, str(Path(__file__).parent))
from environment import SchedulingEnv
from networks import ActorCritic
from ppo import PPOTrainer

DATA_ROOT = Path(__file__).parent.parent / "gpu_dataset"
CKPT_DIR  = Path(__file__).parent / "checkpoints"

NUM_GPUS       = 10
EMBED_DIM      = 64
TOTAL_EPISODES = 500
EVAL_EVERY     = 50
MAX_TRAIN_JOBS = 5000  # large enough for queue saturation at λ=1.0
MAX_VAL_JOBS   = 500

# λ=0.25 excluded: queue never builds, contributing zero gradient signal.
ARRIVAL_RATES = [0.5, 1.0]

PPO_KWARGS = dict(
    lr            = 3e-4,
    clip_eps      = 0.2,
    vf_coef       = 0.5,
    ent_coef      = 0.001,  # was 0.01; entropy overpowered gradient at high load
    gamma         = 0.95,   # was 0.99; γ^500≈0 starves early-episode assignments
    gae_lambda    = 0.95,
    n_epochs      = 4,
    batch_size    = 256,
    max_grad_norm = 0.5,
)


def _eval_baseline(scheduler, jobs: list, arrival_rate: float, seed: int = 999) -> dict:
    j = copy.deepcopy(jobs)
    j = assign_poisson_arrivals(j, arrival_rate, seed=seed)
    m = Simulator(num_gpus=NUM_GPUS, scheduler=scheduler).run(j)
    return {
        "avg_jct":    m.avg_job_completion_time,
        "p95_jct":    m.p95_jct,
        "p99_jct":    m.p99_jct,
        "throughput": m.throughput,
    }


def _eval_rl(
    network:      ActorCritic,
    jobs:         list,
    arrival_rate: float,
    seed:         int = 999,
    device:       str = "cpu",
) -> dict:
    env = SchedulingEnv(jobs, num_gpus=NUM_GPUS)
    obs = env.reset(arrival_rate=arrival_rate, seed=seed)
    if obs[0] is None:
        return {"avg_jct": float("inf"), "n_completed": 0}

    job_feats, gpu_feats, mask = obs
    network.eval()

    while True:
        jf = torch.FloatTensor(job_feats).unsqueeze(0).to(device)
        gf = torch.FloatTensor(gpu_feats).unsqueeze(0).to(device)
        mk = torch.BoolTensor(mask).unsqueeze(0).to(device)

        with torch.no_grad():
            action, _, _ = network.act(jf, gf, mk, deterministic=True)

        next_jf, next_gf, next_mk, _, done = env.step(action.item())
        if done:
            break
        job_feats, gpu_feats, mask = next_jf, next_gf, next_mk

    completed = env._completed_jobs
    if not completed:
        return {"avg_jct": float("inf"), "p95_jct": float("inf"),
                "p99_jct": float("inf"), "throughput": 0.0, "n_completed": 0}

    jcts     = np.array([j.turnaround_time for j in completed])
    makespan = max(j.completion_time for j in completed)
    return {
        "avg_jct":    float(np.mean(jcts)),
        "p95_jct":    float(np.percentile(jcts, 95)),
        "p99_jct":    float(np.percentile(jcts, 99)),
        "throughput": float(len(completed) / max(1.0, makespan)),
        "n_completed": len(completed),
    }


def _print_eval(network: ActorCritic, val_jobs: list, device: str) -> float:
    heavy_p95 = float("inf")
    for rate, label in [(0.25, "moderate"), (0.5, "heavy"), (1.0, "extreme")]:
        fifo = _eval_baseline(FIFOScheduler(), val_jobs, rate)
        sjf  = _eval_baseline(SJFScheduler(),  val_jobs, rate)
        rl   = _eval_rl(network, val_jobs, rate, device=device)
        print(f"  {label} (λ={rate})")
        print(f"    {'metric':12s}  {'FIFO':>9s}  {'SJF':>9s}  {'RL-PPO':>9s}")
        print(f"    {'avg_jct':12s}  {fifo['avg_jct']:>8.2f}s  {sjf['avg_jct']:>8.2f}s  {rl['avg_jct']:>8.2f}s")
        print(f"    {'p95_jct':12s}  {fifo['p95_jct']:>8.2f}s  {sjf['p95_jct']:>8.2f}s  {rl['p95_jct']:>8.2f}s")
        print(f"    {'p99_jct':12s}  {fifo['p99_jct']:>8.2f}s  {sjf['p99_jct']:>8.2f}s  {rl['p99_jct']:>8.2f}s")
        print(f"    {'throughput':12s}  {fifo['throughput']:>8.4f}   {sjf['throughput']:>8.4f}   {rl['throughput']:>8.4f}  (j/s)  n={rl['n_completed']}")
        if rate == 0.5:
            heavy_p95 = rl["p95_jct"]
    return heavy_p95


def main() -> None:
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device : {device}")
    CKPT_DIR.mkdir(exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    print("Loading data...")
    train_jobs = load_jobs_from_csv(str(DATA_ROOT / "train.csv"), max_jobs=MAX_TRAIN_JOBS)
    val_jobs   = load_jobs_from_csv(str(DATA_ROOT / "val.csv"),   max_jobs=MAX_VAL_JOBS)
    print(f"  train={len(train_jobs)} jobs   val={len(val_jobs)} jobs")

    env     = SchedulingEnv(train_jobs, num_gpus=NUM_GPUS)
    network = ActorCritic(num_gpus=NUM_GPUS, embed_dim=EMBED_DIM).to(device)
    trainer = PPOTrainer(network, device=device, **PPO_KWARGS)
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")

    best_val_p95 = float("inf")
    history: list[dict] = []
    t0 = time.perf_counter()

    for ep in range(1, TOTAL_EPISODES + 1):
        rate = random.choice(ARRIVAL_RATES)
        seed = random.randint(0, 1_000_000)

        steps  = trainer.collect_episode(env, arrival_rate=rate, seed=seed)
        ep_rew = sum(trainer.buffer.rewards)

        if steps == 0:
            trainer.buffer.clear()
            continue

        losses  = trainer.update()
        elapsed = time.perf_counter() - t0
        row = {"episode": ep, "steps": steps, "total_reward": ep_rew,
               "arrival_rate": rate, "elapsed_s": round(elapsed, 1), **losses}
        history.append(row)

        print(
            f"ep {ep:4d} | λ={rate:.2f} | steps={steps:5d} | "
            f"rew={ep_rew:9.2f} | p={losses['policy_loss']:6.4f} | "
            f"v={losses['value_loss']:6.4f} | ent={losses['entropy']:5.3f}"
        )

        if ep % EVAL_EVERY == 0:
            print(f"\n{'─'*62}")
            print(f"Evaluation at episode {ep}  (elapsed {elapsed:.0f}s)")
            val_p95 = _print_eval(network, val_jobs, device)

            if val_p95 < best_val_p95:
                best_val_p95 = val_p95
                ckpt_name = f"best_ep{ep}_p95_{val_p95:.0f}s.pt"
                torch.save(
                    {"episode": ep, "state_dict": network.state_dict(),
                     "p95_jct": best_val_p95},
                    CKPT_DIR / ckpt_name,
                )
                print(f"  → New best model  p95_jct(heavy)={best_val_p95:.2f}s  → {ckpt_name}")
            print(f"{'─'*62}\n")

    torch.save({"episode": TOTAL_EPISODES, "state_dict": network.state_dict()},
               CKPT_DIR / "final.pt")
    (CKPT_DIR / "history.json").write_text(json.dumps(history, indent=2))

    print("\n=== Final evaluation ===")
    _print_eval(network, val_jobs, device)
    print(f"\nCheckpoints saved to {CKPT_DIR}")


if __name__ == "__main__":
    main()
