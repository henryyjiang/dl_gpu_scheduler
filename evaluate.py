"""
Evaluation and visualisation for the GPU scheduler comparison.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "simulator"))
sys.path.insert(0, str(ROOT / "rl"))

from data_loader import load_jobs_from_csv, assign_poisson_arrivals
from simulator import Simulator
from schedulers import FIFOScheduler, SJFScheduler
from rl_scheduler import RLScheduler

TEST_CSV     = ROOT / "gpu_dataset" / "test.csv"
HISTORY_PATH = ROOT / "rl" / "checkpoints" / "history.json"

LOAD_LEVELS = {"light": 0.10, "moderate": 0.25, "heavy": 0.50, "extreme": 1.00}
LOAD_NAMES  = list(LOAD_LEVELS.keys())

_BASE_COLORS = {"FIFO": "#4477AA", "SJF": "#EE6677"}
_RL_COLORS   = ["#228833", "#EE8800", "#AA44FF", "#00AACC"]
_RL_HATCHES  = ["xx", "oo", "++", ".."]

COLORS:  dict[str, str] = {}
HATCHES: dict[str, str] = {}

NUM_GPUS = 10
SEED     = 42

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})


def _init_styles(sched_names: list[str]) -> None:
    COLORS.update(_BASE_COLORS)
    HATCHES.update({"FIFO": "", "SJF": "//"})
    rl_names = [n for n in sched_names if n not in _BASE_COLORS]
    for i, name in enumerate(rl_names):
        COLORS[name]  = _RL_COLORS[i % len(_RL_COLORS)]
        HATCHES[name] = _RL_HATCHES[i % len(_RL_HATCHES)]


def run_single(scheduler, jobs, arrival_rate: float, seed: int = SEED):
    j = copy.deepcopy(jobs)
    j = assign_poisson_arrivals(j, arrival_rate, seed=seed)
    sim = Simulator(num_gpus=NUM_GPUS, scheduler=scheduler)
    metrics = sim.run(j)
    jcts  = [job.turnaround_time for job in sim._completed_jobs]
    waits = [job.wait_time       for job in sim._completed_jobs]
    return metrics, jcts, waits


def run_all(test_jobs: list, rl_checkpoints: dict[str, Path]) -> dict:
    """results[scheduler_name][load_level] = (SimulationMetrics, jcts, waits)"""
    schedulers: dict = {"FIFO": FIFOScheduler(), "SJF": SJFScheduler()}
    for label, ckpt_path in rl_checkpoints.items():
        if ckpt_path.exists():
            schedulers[label] = RLScheduler(ckpt_path, num_gpus=NUM_GPUS)
        else:
            print(f"[warn] Checkpoint not found at {ckpt_path} — skipping {label}.")

    results: dict = {}
    for name, sched in schedulers.items():
        results[name] = {}
        for load, rate in LOAD_LEVELS.items():
            print(f"  {name:8s}  {load:8s}  λ={rate}", end=" … ", flush=True)
            metrics, jcts, waits = run_single(sched, test_jobs, rate)
            results[name][load] = (metrics, jcts, waits)
            print(f"avg_jct={metrics.avg_job_completion_time:.2f}s")
    return results


STAT_FIELDS = [
    ("avg_jct_s",        "Avg JCT (s)"),
    ("median_jct_s",     "Median JCT (s)"),
    ("p95_jct_s",        "P95 JCT (s)"),
    ("p99_jct_s",        "P99 JCT (s)"),
    ("avg_wait_s",       "Avg Wait (s)"),
    ("makespan_s",       "Makespan (s)"),
    ("throughput",       "Throughput (j/s)"),
    ("avg_gpu_util_pct", "Avg GPU Util (%)"),
    ("avg_queue_depth",  "Avg Queue Depth"),
    ("max_queue_depth",  "Max Queue Depth"),
]


def _metrics_to_row(name: str, load: str, metrics) -> dict:
    return {
        "scheduler":        name,
        "load_level":       load,
        "arrival_rate":     LOAD_LEVELS[load],
        "num_jobs":         metrics.num_jobs,
        "avg_jct_s":        round(metrics.avg_job_completion_time, 3),
        "median_jct_s":     round(metrics.median_jct,              3),
        "p95_jct_s":        round(metrics.p95_jct,                 3),
        "p99_jct_s":        round(metrics.p99_jct,                 3),
        "avg_wait_s":       round(metrics.avg_wait_time,           3),
        "makespan_s":       round(metrics.makespan,                3),
        "throughput":       round(metrics.throughput,              5),
        "avg_gpu_util_pct": round(metrics.avg_gpu_utilisation,    3),
        "avg_queue_depth":  round(metrics.avg_queue_length,       3),
        "max_queue_depth":  metrics.max_queue_length,
    }


def save_stats_table(results: dict, out_dir: Path) -> list[dict]:
    rows = [
        _metrics_to_row(name, load, metrics)
        for name in results
        for load, (metrics, _, __) in results[name].items()
    ]

    col_w       = 16
    sched_names = list(results.keys())
    sep         = "─" * (14 + col_w * len(sched_names))

    for label, display in STAT_FIELDS:
        print(f"\n{display}")
        print(sep[:len(display) + 2 + col_w * len(sched_names)])
        print(f"  {'':12s}" + "".join(f"{n:>{col_w}}" for n in sched_names))
        for load in LOAD_NAMES:
            line = f"  {load:12s}"
            for name in sched_names:
                row = next(r for r in rows if r["scheduler"] == name and r["load_level"] == load)
                val = row[label]
                line += f"{val:>{col_w}.3f}" if isinstance(val, float) else f"{val:>{col_w}}"
            print(line)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stats_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    json_path = out_dir / "stats_table.json"
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"\nStats saved → {csv_path.relative_to(ROOT)}  &  {json_path.relative_to(ROOT)}")
    return rows


def _save(fig: plt.Figure, name: str, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.relative_to(ROOT)}")


def _grouped_bars(ax, data: dict[str, list], load_names: list[str],
                  sched_names: list[str], ylabel: str, log: bool = False) -> None:
    n_groups = len(load_names)
    n_bars   = len(sched_names)
    width    = 0.22
    x        = np.arange(n_groups)
    for i, name in enumerate(sched_names):
        offset = (i - (n_bars - 1) / 2) * width
        ax.bar(
            x + offset, data[name], width,
            label=name, color=COLORS[name], hatch=HATCHES[name],
            edgecolor="white", linewidth=0.5, alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(load_names, fontsize=11)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.legend(framealpha=0.6)


def fig_avg_jct(results: dict, fig_dir: Path) -> None:
    sched_names = list(results.keys())
    data = {
        name: [results[name][load][0].avg_job_completion_time for load in LOAD_NAMES]
        for name in sched_names
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bars(ax, data, LOAD_NAMES, sched_names, ylabel="Average JCT (seconds)", log=True)
    ax.set_title("Average Job Completion Time by Scheduler and Load Level")
    ax.set_xlabel("Load Level")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()
    _save(fig, "fig1_avg_jct_bars.png", fig_dir)


def fig_jct_boxplots(results: dict, fig_dir: Path) -> None:
    sched_names = list(results.keys())
    fig, axes   = plt.subplots(2, 2, figsize=(12, 9))
    axes        = axes.flatten()
    for ax, load in zip(axes, LOAD_NAMES):
        rate     = LOAD_LEVELS[load]
        all_jcts = [results[name][load][1] for name in sched_names]
        bp = ax.boxplot(
            all_jcts,
            tick_labels = sched_names,
            patch_artist= True,
            showfliers  = False,
            medianprops = {"color": "black", "linewidth": 2},
            whiskerprops= {"linewidth": 1.2},
            capprops    = {"linewidth": 1.2},
        )
        for patch, name in zip(bp["boxes"], sched_names):
            patch.set_facecolor(COLORS[name])
            patch.set_alpha(0.75)
        ax.set_title(f"{load.capitalize()} load  (λ={rate})")
        ax.set_ylabel("JCT (seconds)")
        if load == "extreme":
            ax.set_yscale("log")
            ax.set_ylabel("JCT (seconds, log scale)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.suptitle("JCT Distribution by Scheduler and Load Level\n"
                 "(boxes show IQR, whiskers 5–95th pct, outliers hidden)",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "fig2_jct_boxplots.png", fig_dir)


def fig_cdf(results: dict, fig_dir: Path) -> None:
    sched_names = list(results.keys())
    fig, axes   = plt.subplots(1, 2, figsize=(12, 5))
    for ax, load, log_x in zip(axes, ["heavy", "extreme"], [False, True]):
        rate = LOAD_LEVELS[load]
        for name in sched_names:
            jcts = np.sort(results[name][load][1])
            cdf  = np.arange(1, len(jcts) + 1) / len(jcts)
            ax.plot(jcts, cdf, label=name, color=COLORS[name], linewidth=2)
        ax.axhline(0.50, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.axhline(0.95, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.text(ax.get_xlim()[0] if not log_x else 1, 0.51, "P50", fontsize=9, color="gray")
        ax.text(ax.get_xlim()[0] if not log_x else 1, 0.96, "P95", fontsize=9, color="gray")
        ax.set_title(f"CDF of JCT — {load} load  (λ={rate})")
        ax.set_xlabel("JCT (seconds" + (", log scale)" if log_x else ")"))
        ax.set_ylabel("Cumulative fraction of jobs")
        ax.set_ylim(0, 1)
        if log_x:
            ax.set_xscale("log")
        ax.legend(framealpha=0.6)
    fig.suptitle("Cumulative Distribution of Job Completion Time", fontsize=13)
    fig.tight_layout()
    _save(fig, "fig3_cdf_heavy_extreme.png", fig_dir)


def fig_wait_and_queue(results: dict, fig_dir: Path) -> None:
    sched_names = list(results.keys())
    rates       = list(LOAD_LEVELS.values())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name in sched_names:
        waits  = [results[name][load][0].avg_wait_time    for load in LOAD_NAMES]
        queues = [results[name][load][0].avg_queue_length for load in LOAD_NAMES]
        ax1.plot(rates, waits,  marker="o", label=name, color=COLORS[name], linewidth=2)
        ax2.plot(rates, queues, marker="o", label=name, color=COLORS[name], linewidth=2)
    ax1.set_title("Average Wait Time vs Arrival Rate")
    ax1.set_xlabel("Arrival rate λ (jobs/s)")
    ax1.set_ylabel("Avg wait time (seconds)")
    ax1.set_yscale("symlog", linthresh=1)
    ax1.legend(framealpha=0.6)
    ax2.set_title("Average Queue Depth vs Arrival Rate")
    ax2.set_xlabel("Arrival rate λ (jobs/s)")
    ax2.set_ylabel("Avg queue depth (jobs)")
    ax2.set_yscale("symlog", linthresh=1)
    ax2.legend(framealpha=0.6)
    fig.suptitle("Queue Behaviour Under Increasing Load", fontsize=13)
    fig.tight_layout()
    _save(fig, "fig4_wait_and_queue.png", fig_dir)


def fig_tail_latency(results: dict, fig_dir: Path) -> None:
    sched_names = list(results.keys())
    fig, axes   = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pct, field in zip(axes, ["P95", "P99"], ["p95_jct", "p99_jct"]):
        data = {
            name: [getattr(results[name][load][0], field) for load in LOAD_NAMES]
            for name in sched_names
        }
        _grouped_bars(ax, data, LOAD_NAMES, sched_names, ylabel=f"{pct} JCT (seconds)", log=True)
        ax.set_title(f"{pct} Job Completion Time")
        ax.set_xlabel("Load Level")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.suptitle("Tail Latency (P95 and P99 JCT) by Scheduler", fontsize=13)
    fig.tight_layout()
    _save(fig, "fig5_tail_latency.png", fig_dir)


def fig_training_curve(history_path: Path, fig_dir: Path) -> None:
    if not history_path.exists():
        print("  [skip] history.json not found — skipping training curve")
        return

    history = json.loads(history_path.read_text())
    eps     = [h["episode"]      for h in history]
    rewards = [h["total_reward"] for h in history]
    entropy = [h["entropy"]      for h in history]
    v_loss  = [h["value_loss"]   for h in history]
    rates   = [h["arrival_rate"] for h in history]

    window = 10
    def smooth(xs):
        return np.convolve(xs, np.ones(window) / window, mode="valid")

    rate_colors = {0.25: "#AACCFF", 0.5: "#FFAA77", 1.0: "#FF6677"}
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

    ax = axes[0]
    for ep, rew, rate in zip(eps, rewards, rates):
        ax.scatter(ep, rew, c=rate_colors[rate], s=8, alpha=0.5, zorder=2)
    for rate, c in rate_colors.items():
        ax.scatter([], [], c=c, label=f"λ={rate}", s=30)
    ax.plot(eps[window-1:], smooth(rewards), color="black", linewidth=1.5,
            label="10-ep rolling mean", zorder=3)
    ax.set_ylabel("Total episode reward")
    ax.set_title("Training Reward (coloured by arrival rate)")
    ax.legend(fontsize=9, framealpha=0.6)

    ax = axes[1]
    ax.plot(eps, entropy, color="#9966CC", linewidth=1, alpha=0.6)
    ax.plot(eps[window-1:], smooth(entropy), color="#9966CC", linewidth=2)
    ax.set_ylabel("Policy entropy (nats)")
    ax.set_title("Policy Entropy (higher = more exploratory)")

    ax = axes[2]
    ax.plot(eps, v_loss, color="#EE6677", linewidth=1, alpha=0.6)
    ax.plot(eps[window-1:], smooth(v_loss), color="#EE6677", linewidth=2)
    ax.set_ylabel("Value loss (MSE)")
    ax.set_xlabel("Training episode")
    ax.set_title("Value Network Loss")
    ax.set_yscale("log")

    fig.suptitle("PPO Training Diagnostics", fontsize=13)
    fig.tight_layout()
    _save(fig, "fig6_training_curve.png", fig_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GPU scheduler checkpoints.")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to first .pt checkpoint. Defaults to most-recently-modified "
             "best_ep*.pt in rl/checkpoints/, falling back to rl/checkpoints/best.pt.",
    )
    parser.add_argument(
        "--checkpoint2", type=str, default=None,
        help="Optional second .pt checkpoint. Both RL models appear side-by-side "
             "in every figure and the stats table.",
    )
    parser.add_argument("--label1", type=str, default="RL-PPO",
                        help="Legend label for --checkpoint (default: RL-PPO).")
    parser.add_argument("--label2", type=str, default="RL-PPO-2",
                        help="Legend label for --checkpoint2 (default: RL-PPO-2).")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Subdirectory under analysis/. Defaults to run_YYYYMMDD_HHMMSS.")
    args = parser.parse_args()

    ckpt_dir = ROOT / "rl" / "checkpoints"
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
    else:
        candidates = sorted(ckpt_dir.glob("best_ep*.pt"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        ckpt_path = candidates[0] if candidates else ckpt_dir / "best.pt"

    rl_checkpoints: dict[str, Path] = {args.label1: ckpt_path}
    if args.checkpoint2:
        rl_checkpoints[args.label2] = Path(args.checkpoint2).resolve()

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir  = ROOT / "analysis" / run_name
    fig_dir  = out_dir / "figures"

    for label, path in rl_checkpoints.items():
        print(f"Checkpoint ({label}): {path}")
    print(f"Output dir : {out_dir.relative_to(ROOT)}/")
    print(f"Loading test set from {TEST_CSV.relative_to(ROOT)} …")
    test_jobs = load_jobs_from_csv(str(TEST_CSV))
    print(f"  {len(test_jobs)} jobs loaded\n")

    print("Running simulations …")
    results = run_all(test_jobs, rl_checkpoints)
    _init_styles(list(results.keys()))

    print("\n── Statistics table ──────────────────────────────────────────────")
    save_stats_table(results, out_dir)

    history_path = ckpt_path.parent / "history.json"
    if not history_path.exists():
        history_path = HISTORY_PATH

    print("\n── Generating figures ────────────────────────────────────────────")
    fig_avg_jct(results, fig_dir)
    fig_jct_boxplots(results, fig_dir)
    fig_cdf(results, fig_dir)
    fig_wait_and_queue(results, fig_dir)
    fig_tail_latency(results, fig_dir)
    fig_training_curve(history_path, fig_dir)

    print(f"\nDone. All outputs in {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
