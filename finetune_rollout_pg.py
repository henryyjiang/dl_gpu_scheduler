import argparse
import copy
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parent
SIM_DIR = ROOT / "simulator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SIM_DIR) not in sys.path:
    sys.path.append(str(SIM_DIR))

from model.neural_scheduler import NeuralScheduler
from model.neural_scheduler_model import NeuralSchedulerModel
from simulator.data_loader import assign_poisson_arrivals, load_jobs_from_csv
from simulator.scheduler_interface import SchedulerInterface
from simulator.schedulers import SJFScheduler
from simulator.simulator import Simulator


class StochasticNeuralScheduler(SchedulerInterface):
    def __init__(self, model):
        self.model = model
        self.teacher = SJFScheduler()
        self.log_probs = []
        self.bc_losses = []
        self.entropies = []

    @property
    def name(self):
        return "NeuralStochastic"

    def on_simulation_start(self, cluster):
        self.log_probs = []
        self.bc_losses = []
        self.entropies = []

    def _teacher_ce_loss(self, queue, cluster, scores):
        queue_copy = copy.deepcopy(queue)
        cluster_copy = copy.deepcopy(cluster)
        teacher_assignments = self.teacher._try_schedule(queue_copy, cluster_copy)
        if not teacher_assignments:
            return None

        job_id_to_idx = {job.job_id: idx for idx, job in enumerate(queue)}
        target_jobs = []
        target_gpus = []
        for job, gpu_id in teacher_assignments:
            if job.job_id in job_id_to_idx:
                target_jobs.append(job_id_to_idx[job.job_id])
                target_gpus.append(gpu_id)

        if not target_jobs:
            return None

        target_tensor = torch.tensor(target_gpus, dtype=torch.long, device=scores.device)
        target_scores = scores[target_jobs]
        return F.cross_entropy(target_scores, target_tensor)

    def _sample_schedule(self, queue, cluster):
        if len(queue) == 0:
            return []

        job_feats, gpu_feats = NeuralScheduler._build_features(queue, cluster)
        scores = self.model(job_feats, gpu_feats)  # (N_jobs, N_gpus), logits

        ce_loss = self._teacher_ce_loss(queue, cluster, scores)
        if ce_loss is not None:
            self.bc_losses.append(ce_loss)

        assignments = []
        used_jobs = set()

        while True:
            feasible_pairs = []
            pair_logits = []

            for j_idx, job in enumerate(queue):
                if job.job_id in used_jobs:
                    continue
                for g_idx, gpu in enumerate(cluster.gpus):
                    if not gpu.can_fit(job):
                        continue
                    feasible_pairs.append((j_idx, g_idx))
                    pair_logits.append(scores[j_idx, g_idx])

            if not feasible_pairs:
                break

            logits = torch.stack(pair_logits, dim=0)
            dist = Categorical(logits=logits)
            sampled_idx = dist.sample()
            self.log_probs.append(dist.log_prob(sampled_idx))
            self.entropies.append(dist.entropy())

            j_idx, g_idx = feasible_pairs[sampled_idx.item()]
            job = queue[j_idx]
            assignments.append((job, g_idx))
            used_jobs.add(job.job_id)

            cluster.gpus[g_idx].used_memory += job.gpu_mem_required

        for job, g_idx in assignments:
            cluster.gpus[g_idx].used_memory -= job.gpu_mem_required

        return assignments

    def on_job_arrival(self, job, queue, cluster, current_time):
        return self._sample_schedule(queue, cluster)

    def on_job_completion(self, completed_job, queue, cluster, current_time):
        return self._sample_schedule(queue, cluster)


def _evaluate_greedy_model(
    model,
    jobs,
    seed,
    num_gpus,
    gpu_memory,
    beta,
    gamma,
    arrival_rates,
):
    model.eval()
    rewards = []
    metrics_by_rate = {}
    with torch.no_grad():
        for rate in arrival_rates:
            rollout_jobs = assign_poisson_arrivals(
                copy.deepcopy(jobs),
                arrival_rate=rate,
                seed=seed,
            )
            scheduler = NeuralScheduler(model)
            sim = Simulator(
                num_gpus=num_gpus,
                gpu_memory=gpu_memory,
                scheduler=scheduler,
                interference_alpha=0.0,
            )
            metrics = sim.run(rollout_jobs)
            reward = -(
                metrics.avg_job_completion_time
                + beta * metrics.p99_jct
                - gamma * metrics.avg_gpu_utilisation
            )
            rewards.append(reward)
            metrics_by_rate[rate] = metrics
    model.train()
    return sum(rewards) / len(rewards), metrics_by_rate


def finetune_with_rollouts(
    jobs,
    model_path="model/neural_scheduler_imitation.pt",
    num_updates=20,
    rollouts_per_update=4,
    arrival_rates=(0.25, 0.5, 1.0),
    arrival_rate_weights=(0.1, 0.3, 0.6),
    beta=0.2,
    gamma=1.0,
    learning_rate=5e-6,
    bc_weight_start=0.2,
    bc_weight_end=0.05,
    entropy_coef=0.001,
    seed=42,
    num_gpus=10,
    gpu_memory=80.0,
    eval_every=5,
    best_model_path="model/neural_scheduler_pg_best.pt",
):
    model = NeuralSchedulerModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_heavy_score = float("-inf")
    global_rollout_idx = 0

    for update_idx in range(num_updates):
        rollout_records = []
        progress = update_idx / max(1, num_updates - 1)
        bc_weight = bc_weight_start + (bc_weight_end - bc_weight_start) * progress

        for _ in range(rollouts_per_update):
            rate = random.choices(arrival_rates, weights=arrival_rate_weights, k=1)[0]
            rollout_jobs = assign_poisson_arrivals(
                copy.deepcopy(jobs),
                arrival_rate=rate,
                seed=seed + global_rollout_idx,
            )
            global_rollout_idx += 1

            scheduler = StochasticNeuralScheduler(model)
            sim = Simulator(
                num_gpus=num_gpus,
                gpu_memory=gpu_memory,
                scheduler=scheduler,
                interference_alpha=0.0,
            )
            metrics = sim.run(rollout_jobs)
            if not scheduler.log_probs:
                continue

            reward = -(
                metrics.avg_job_completion_time
                + beta * metrics.p99_jct
                - gamma * metrics.avg_gpu_utilisation
            )
            log_prob_sum = torch.stack(scheduler.log_probs).sum()
            if scheduler.bc_losses:
                rollout_bc_loss = torch.stack(scheduler.bc_losses).mean()
            else:
                rollout_bc_loss = torch.tensor(0.0, dtype=log_prob_sum.dtype, device=log_prob_sum.device)
            if scheduler.entropies:
                rollout_entropy = torch.stack(scheduler.entropies).mean()
            else:
                rollout_entropy = torch.tensor(0.0, dtype=log_prob_sum.dtype, device=log_prob_sum.device)

            rollout_records.append((reward, log_prob_sum, rollout_bc_loss, rollout_entropy, metrics, rate))

        if not rollout_records:
            print(f"Update {update_idx + 1}: no valid rollouts, skipping.")
            continue

        rewards_tensor = torch.tensor([r[0] for r in rollout_records], dtype=torch.float32)
        advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-6)

        pg_terms = []
        bc_terms = []
        entropy_terms = []
        for adv, (_, log_prob_sum, rollout_bc_loss, rollout_entropy, _, _) in zip(advantages, rollout_records):
            pg_terms.append(-adv * log_prob_sum)
            bc_terms.append(rollout_bc_loss)
            entropy_terms.append(rollout_entropy)

        pg_loss = torch.stack(pg_terms).mean()
        bc_loss = torch.stack(bc_terms).mean()
        entropy_bonus = torch.stack(entropy_terms).mean()
        loss = pg_loss + bc_weight * bc_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        avg_reward = rewards_tensor.mean().item()
        avg_jct = sum(r[4].avg_job_completion_time for r in rollout_records) / len(rollout_records)
        avg_p99 = sum(r[4].p99_jct for r in rollout_records) / len(rollout_records)
        avg_util = sum(r[4].avg_gpu_utilisation for r in rollout_records) / len(rollout_records)
        avg_rate = sum(r[5] for r in rollout_records) / len(rollout_records)

        print(
            f"Update {update_idx + 1}/{num_updates} | "
            f"rollouts={len(rollout_records)} | "
            f"avg_rate={avg_rate:.2f} | "
            f"reward={avg_reward:.4f} | "
            f"avg_jct={avg_jct:.4f} | "
            f"p99={avg_p99:.4f} | "
            f"util={avg_util:.4f} | "
            f"pg_loss={pg_loss.item():.4f} | "
            f"bc_loss={bc_loss.item():.4f} | "
            f"entropy={entropy_bonus.item():.4f} | "
            f"bc_w={bc_weight:.3f}"
        )

        if (update_idx + 1) % eval_every == 0 or (update_idx + 1) == num_updates:
            eval_reward, metrics_by_rate = _evaluate_greedy_model(
                model=model,
                jobs=jobs,
                seed=seed,
                num_gpus=num_gpus,
                gpu_memory=gpu_memory,
                beta=beta,
                gamma=gamma,
                arrival_rates=arrival_rates,
            )
            heavy_rate = max(arrival_rates)
            heavy_metrics = metrics_by_rate[heavy_rate]
            heavy_score = -(heavy_metrics.avg_job_completion_time + 0.5 * heavy_metrics.p99_jct)
            print(
                f"Greedy eval @ update {update_idx + 1}: "
                f"reward={eval_reward:.4f}, "
                f"load={heavy_rate} avg_jct={heavy_metrics.avg_job_completion_time:.4f}, "
                f"p99={heavy_metrics.p99_jct:.4f}"
            )
            if heavy_score > best_heavy_score:
                best_heavy_score = heavy_score
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved new best model to: {best_model_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Rollout-based policy-gradient fine-tuning.")
    parser.add_argument("--csv", type=str, required=True, help="Path to jobs CSV.")
    parser.add_argument("--model-path", type=str, default="model/neural_scheduler_imitation.pt")
    parser.add_argument("--output-path", type=str, default="model/neural_scheduler_pg.pt")
    parser.add_argument("--best-model-path", type=str, default="model/neural_scheduler_pg_best.pt")
    parser.add_argument("--num-updates", type=int, default=20)
    parser.add_argument("--rollouts-per-update", type=int, default=8)
    parser.add_argument("--arrival-rates", type=str, default="0.25,0.5,1.0")
    parser.add_argument("--arrival-rate-weights", type=str, default="0.1,0.3,0.6")
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--bc-weight-start", type=float, default=0.2)
    parser.add_argument("--bc-weight-end", type=float, default=0.05)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-gpus", type=int, default=10)
    parser.add_argument("--gpu-memory", type=float, default=80.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args()

    jobs = load_jobs_from_csv(args.csv, max_jobs=args.max_jobs)
    arrival_rates = tuple(float(x.strip()) for x in args.arrival_rates.split(",") if x.strip())
    arrival_rate_weights = tuple(float(x.strip()) for x in args.arrival_rate_weights.split(",") if x.strip())
    if len(arrival_rate_weights) != len(arrival_rates):
        raise ValueError("arrival_rate_weights must match arrival_rates length.")
    model = finetune_with_rollouts(
        jobs=jobs,
        model_path=args.model_path,
        num_updates=args.num_updates,
        rollouts_per_update=args.rollouts_per_update,
        arrival_rates=arrival_rates,
        arrival_rate_weights=arrival_rate_weights,
        beta=args.beta,
        gamma=args.gamma,
        learning_rate=args.lr,
        bc_weight_start=args.bc_weight_start,
        bc_weight_end=args.bc_weight_end,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
        num_gpus=args.num_gpus,
        gpu_memory=args.gpu_memory,
        eval_every=args.eval_every,
        best_model_path=args.best_model_path,
    )

    torch.save(model.state_dict(), args.output_path)
    print(f"Saved fine-tuned model to: {args.output_path}")


if __name__ == "__main__":
    main()
