"""Publication-quality figures for the RL GPU scheduler paper."""

from __future__ import annotations

import argparse
import copy
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

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        200,
})

COLORS  = {"FIFO": "#4477AA", "SJF": "#EE6677", "RL-PPO": "#228833"}
MARKERS = {"FIFO": "o",       "SJF": "s",        "RL-PPO": "^"}
HATCHES = {"FIFO": "",        "SJF": "//",        "RL-PPO": "xx"}

BIN_COLORS = ["#88CCEE", "#DDAA33", "#CC3311"]

NUM_GPUS    = 10
SEED        = 42
LOAD_LEVELS = {"light": 0.10, "moderate": 0.25, "heavy": 0.50, "extreme": 1.00}
LOAD_NAMES  = list(LOAD_LEVELS.keys())
RATES       = list(LOAD_LEVELS.values())

HISTORY_PATH = ROOT / "rl" / "checkpoints" / "history.json"


def _run_single(scheduler, jobs: list, rate: float, seed: int = SEED):
    j = copy.deepcopy(jobs)
    j = assign_poisson_arrivals(j, rate, seed=seed)
    sim = Simulator(num_gpus=NUM_GPUS, scheduler=scheduler)
    metrics = sim.run(j)
    return metrics, list(sim._completed_jobs)


def run_all(test_jobs: list, ckpt_path: Path) -> dict:
    """results[scheduler_name][load_name] = {"metrics": SimulationMetrics, "jobs": list[Job]}"""
    schedulers: dict = {"FIFO": FIFOScheduler(), "SJF": SJFScheduler()}
    if ckpt_path.exists():
        schedulers["RL-PPO"] = RLScheduler(ckpt_path, num_gpus=NUM_GPUS)
    else:
        print(f"[warn] Checkpoint not found at {ckpt_path} — RL-PPO skipped.")

    results: dict = {}
    for name, sched in schedulers.items():
        results[name] = {}
        for load, rate in LOAD_LEVELS.items():
            print(f"  {name:8s}  {load:8s}  λ={rate:.2f}", end=" … ", flush=True)
            metrics, jobs = _run_single(sched, test_jobs, rate)
            results[name][load] = {"metrics": metrics, "jobs": jobs}
            print(f"n={len(jobs)}  avg_jct={metrics.avg_job_completion_time:.1f}s")
    return results


def _save(fig: plt.Figure, name: str, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


def _grouped_bars(
    ax, data: dict, x_labels: list, sched_names: list,
    ylabel: str, log: bool = False,
) -> None:
    n, nb, w = len(x_labels), len(sched_names), 0.22
    xs = np.arange(n)
    for i, name in enumerate(sched_names):
        offset = (i - (nb - 1) / 2) * w
        ax.bar(
            xs + offset, data[name], w,
            label=name, color=COLORS[name], hatch=HATCHES[name],
            edgecolor="white", linewidth=0.5, alpha=0.9,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(framealpha=0.6, fontsize=9)


def _job_tertile_bins(test_jobs: list) -> tuple[float, float]:
    lats = [j.true_latency for j in test_jobs]
    return float(np.percentile(lats, 33)), float(np.percentile(lats, 66))


def _bin_label(lo: float, hi: float) -> list[str]:
    return [f"Short\n(<{lo:.0f}s)", f"Medium\n({lo:.0f}–{hi:.0f}s)", f"Long\n(>{hi:.0f}s)"]


def _jcts_by_bin(jobs: list, lo: float, hi: float) -> tuple[list, list, list]:
    short  = [j.turnaround_time for j in jobs if j.true_latency <  lo]
    medium = [j.turnaround_time for j in jobs if lo <= j.true_latency < hi]
    long_  = [j.turnaround_time for j in jobs if j.true_latency >= hi]
    return short, medium, long_


def _slowdowns(jobs: list) -> list[float]:
    return [j.turnaround_time / j.true_latency for j in jobs if j.true_latency > 0]


def fig01_jct_percentiles(results: dict, fig_dir: Path) -> None:
    sched_names = list(results.keys())
    cfg = [
        ("avg_job_completion_time", "Avg JCT (s)"),
        ("p95_jct",                 "P95 JCT (s)"),
        ("p99_jct",                 "P99 JCT (s)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (field, ylabel) in zip(axes, cfg):
        data = {
            name: [getattr(results[name][load]["metrics"], field) for load in LOAD_NAMES]
            for name in sched_names
        }
        _grouped_bars(ax, data, LOAD_NAMES, sched_names, ylabel, log=True)
        ax.set_xlabel("Load Level")
        ax.set_title(ylabel.replace(" (s)", ""))
    fig.suptitle("Job Completion Time Percentiles by Scheduler and Load", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig01_jct_percentiles.png", fig_dir)


def fig02_jct_cdf(results: dict, fig_dir: Path) -> None:
    sched_names = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, load in zip(axes, LOAD_NAMES):
        rate  = LOAD_LEVELS[load]
        log_x = (load == "extreme")

        for name in sched_names:
            jcts = np.sort([j.turnaround_time for j in results[name][load]["jobs"]])
            cdf  = np.arange(1, len(jcts) + 1) / len(jcts)
            ax.plot(jcts, cdf, label=name, color=COLORS[name], linewidth=2)

        for pct, frac in [(0.50, 0.50), (0.95, 0.95), (0.99, 0.99)]:
            ax.axhline(frac, color="gray", linestyle=":", linewidth=0.9, alpha=0.6)
            ax.text(ax.get_xlim()[1] * 0.02 if not log_x else 1,
                    frac + 0.01, f"P{int(pct*100)}",
                    fontsize=8, color="gray", va="bottom")

        ax.set_title(f"{load.capitalize()} load  (λ={rate})")
        ax.set_xlabel("JCT" + (" (s, log scale)" if log_x else " (s)"))
        ax.set_ylabel("Cumulative fraction of jobs")
        ax.set_ylim(0, 1.05)
        if log_x:
            ax.set_xscale("log")
        ax.legend(framealpha=0.6, fontsize=9)

    fig.suptitle("Cumulative Distribution of Job Completion Time", fontsize=13)
    fig.tight_layout()
    _save(fig, "fig02_jct_cdf.png", fig_dir)


def fig03_tail_scaling(results: dict, fig_dir: Path) -> None:
    sched_names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    cfg = [
        ("median_jct", "Median (P50) JCT (s)"),
        ("p95_jct",    "P95 JCT (s)"),
        ("p99_jct",    "P99 JCT (s)"),
    ]
    for ax, (field, ylabel) in zip(axes, cfg):
        for name in sched_names:
            vals = [getattr(results[name][load]["metrics"], field) for load in LOAD_NAMES]
            ax.plot(RATES, vals, marker=MARKERS[name], label=name,
                    color=COLORS[name], linewidth=2, markersize=7)
        ax.set_xlabel("Arrival rate λ (jobs/s)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.replace(" (s)", ""))
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.legend(framealpha=0.6, fontsize=9)

    fig.suptitle("Tail Latency Scaling with Arrival Rate", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig03_tail_scaling.png", fig_dir)


def fig04_throughput_utilization(results: dict, fig_dir: Path) -> None:
    sched_names = list(results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    tput = {
        name: [results[name][load]["metrics"].throughput for load in LOAD_NAMES]
        for name in sched_names
    }
    util = {
        name: [results[name][load]["metrics"].avg_gpu_utilisation for load in LOAD_NAMES]
        for name in sched_names
    }

    _grouped_bars(ax1, tput, LOAD_NAMES, sched_names, "Throughput (jobs/s)")
    ax1.set_xlabel("Load Level")
    ax1.set_title("Scheduler Throughput")

    _grouped_bars(ax2, util, LOAD_NAMES, sched_names, "Avg GPU Utilization (%)")
    ax2.set_xlabel("Load Level")
    ax2.set_title("Average GPU Utilization")

    fig.suptitle("Resource Efficiency by Scheduler and Load", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig04_throughput_utilization.png", fig_dir)


def fig05_wait_time(results: dict, fig_dir: Path) -> None:
    sched_names = list(results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for name in sched_names:
        waits = np.sort([j.wait_time for j in results[name]["extreme"]["jobs"]])
        cdf   = np.arange(1, len(waits) + 1) / len(waits)
        ax1.plot(waits, cdf, label=name, color=COLORS[name], linewidth=2)
    ax1.axhline(0.95, color="gray", linestyle=":", linewidth=0.9, alpha=0.6)
    ax1.text(ax1.get_xlim()[1] * 0.02, 0.96, "P95", fontsize=8, color="gray")
    ax1.set_xscale("log")
    ax1.set_xlabel("Wait time (s, log scale)")
    ax1.set_ylabel("Cumulative fraction of jobs")
    ax1.set_title("Wait Time CDF — Extreme Load (λ=1.0)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(framealpha=0.6, fontsize=9)

    for name in sched_names:
        avgs = [results[name][load]["metrics"].avg_wait_time for load in LOAD_NAMES]
        ax2.plot(RATES, avgs, marker=MARKERS[name], label=name,
                 color=COLORS[name], linewidth=2, markersize=7)
    ax2.set_xlabel("Arrival rate λ (jobs/s)")
    ax2.set_ylabel("Avg wait time (s)")
    ax2.set_title("Average Wait Time vs Arrival Rate")
    ax2.set_yscale("symlog", linthresh=1)
    ax2.legend(framealpha=0.6, fontsize=9)

    fig.suptitle("Queue Wait Time — Starvation Analysis", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig05_wait_time.png", fig_dir)


def fig06_fairness_by_job_size(results: dict, test_jobs: list, fig_dir: Path) -> None:
    lo, hi = _job_tertile_bins(test_jobs)
    labels  = _bin_label(lo, hi)
    sched_names = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, load in zip(axes, ["heavy", "extreme"]):
        rate = LOAD_LEVELS[load]
        binned: dict[str, list] = {name: [] for name in sched_names}
        for name in sched_names:
            bins = _jcts_by_bin(results[name][load]["jobs"], lo, hi)
            binned[name] = bins

        n_bins  = 3
        n_sched = len(sched_names)
        group_w = 0.8
        bar_w   = group_w / n_sched

        for b in range(n_bins):
            for s_idx, name in enumerate(sched_names):
                x    = b * (n_sched + 1) + s_idx
                data = binned[name][b]
                if data:
                    bp = ax.boxplot(
                        data, positions=[x], widths=bar_w * 0.9,
                        patch_artist=True, showfliers=False,
                        medianprops={"color": "black", "linewidth": 1.5},
                        whiskerprops={"linewidth": 1.0},
                        capprops={"linewidth": 1.0},
                    )
                    bp["boxes"][0].set_facecolor(COLORS[name])
                    bp["boxes"][0].set_alpha(0.75)

        group_centres = [
            b * (n_sched + 1) + (n_sched - 1) / 2 for b in range(n_bins)
        ]
        ax.set_xticks(group_centres)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("JCT (s)")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_title(f"{load.capitalize()} load  (λ={rate})")

        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[n], alpha=0.75, label=n)
            for n in sched_names
        ]
        ax.legend(handles=handles, framealpha=0.6, fontsize=9)

    fig.suptitle(
        "JCT by Job Execution-Time Tertile  "
        "(whiskers = 5th–95th pct, outliers hidden)",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "fig06_fairness_by_job_size.png", fig_dir)


def fig07_slowdown(results: dict, test_jobs: list, fig_dir: Path) -> None:
    sched_names = list(results.keys())
    lo, _ = _job_tertile_bins(test_jobs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for name in sched_names:
        sd   = np.sort(_slowdowns(results[name]["extreme"]["jobs"]))
        cdf  = np.arange(1, len(sd) + 1) / len(sd)
        ax1.plot(sd, cdf, label=name, color=COLORS[name], linewidth=2)
    ax1.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Ideal (×1)")
    ax1.axhline(0.95, color="gray", linestyle=":", linewidth=0.9, alpha=0.5)
    ax1.text(ax1.get_xlim()[1] * 0.98, 0.96, "P95",
             fontsize=8, color="gray", ha="right")
    ax1.set_xscale("log")
    ax1.set_xlabel("Slowdown (JCT / exec time, log scale)")
    ax1.set_ylabel("Cumulative fraction of jobs")
    ax1.set_title("Slowdown CDF — Extreme Load (λ=1.0)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(framealpha=0.6, fontsize=9)

    for name in sched_names:
        jobs  = results[name]["extreme"]["jobs"]
        execs = [j.true_latency    for j in jobs]
        jcts  = [j.turnaround_time for j in jobs]
        ax2.scatter(execs, jcts, s=4, alpha=0.25, color=COLORS[name], label=name)

    xlim = ax2.get_xlim()
    lo_v = max(0.01, xlim[0])
    hi_v = xlim[1]
    diag = np.logspace(np.log10(lo_v), np.log10(hi_v), 100)
    ax2.plot(diag, diag, "k--", linewidth=1, alpha=0.5, label="Ideal (zero wait)")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Execution time (s)")
    ax2.set_ylabel("JCT (s)")
    ax2.set_title("Execution Time vs JCT — Extreme Load (λ=1.0)")
    ax2.legend(framealpha=0.6, fontsize=9, markerscale=4)

    fig.suptitle("Job Slowdown Analysis", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig07_slowdown.png", fig_dir)


def fig08_improvement_heatmap(results: dict, fig_dir: Path) -> None:
    if "RL-PPO" not in results:
        print("  [skip] RL-PPO not available — skipping heatmap")
        return

    lower_is_better = [
        ("avg_job_completion_time", "Avg JCT"),
        ("p95_jct",                 "P95 JCT"),
        ("p99_jct",                 "P99 JCT"),
        ("avg_wait_time",           "Avg Wait"),
        ("makespan",                "Makespan"),
    ]
    higher_is_better = [
        ("throughput",          "Throughput"),
        ("avg_gpu_utilisation", "GPU Util"),
    ]

    all_metrics   = lower_is_better + higher_is_better
    metric_labels = [m[1] for m in all_metrics]
    baselines     = [b for b in ["SJF", "FIFO"] if b in results]

    fig, axes = plt.subplots(1, len(baselines), figsize=(6 * len(baselines), 5))
    if len(baselines) == 1:
        axes = [axes]

    for ax, baseline in zip(axes, baselines):
        matrix = np.zeros((len(all_metrics), len(LOAD_NAMES)))

        for r_idx, (field, _) in enumerate(all_metrics):
            for c_idx, load in enumerate(LOAD_NAMES):
                base_val = getattr(results[baseline][load]["metrics"], field)
                rl_val   = getattr(results["RL-PPO"][load]["metrics"],  field)
                if base_val == 0:
                    matrix[r_idx, c_idx] = 0.0
                elif (field, _) in lower_is_better:
                    matrix[r_idx, c_idx] = (base_val - rl_val) / base_val * 100
                else:
                    matrix[r_idx, c_idx] = (rl_val - base_val) / base_val * 100

        vmax = max(abs(matrix.max()), abs(matrix.min()), 1.0)
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(LOAD_NAMES)))
        ax.set_xticklabels(LOAD_NAMES, fontsize=9)
        ax.set_yticks(range(len(metric_labels)))
        ax.set_yticklabels(metric_labels, fontsize=9)
        ax.set_xlabel("Load Level")
        ax.set_title(f"RL-PPO % improvement over {baseline}")

        for r in range(len(all_metrics)):
            for c in range(len(LOAD_NAMES)):
                val = matrix[r, c]
                ax.text(c, r, f"{val:+.1f}%", ha="center", va="center",
                        fontsize=8, color="black" if abs(val) < vmax * 0.7 else "white")

        fig.colorbar(im, ax=ax, label="% improvement (green = RL better)")

    fig.suptitle("RL-PPO Improvement Over Baselines", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig08_improvement_heatmap.png", fig_dir)


def fig09_queue_and_makespan(results: dict, fig_dir: Path) -> None:
    sched_names = list(results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for name in sched_names:
        avg_q = [results[name][load]["metrics"].avg_queue_length for load in LOAD_NAMES]
        ax1.plot(RATES, avg_q, marker=MARKERS[name], label=f"{name} avg",
                 color=COLORS[name], linewidth=2, markersize=6)
        max_q = [results[name][load]["metrics"].max_queue_length for load in LOAD_NAMES]
        ax1.plot(RATES, max_q, marker=MARKERS[name], linestyle="--",
                 color=COLORS[name], linewidth=1, markersize=5, alpha=0.5,
                 label=f"{name} max")

    ax1.set_xlabel("Arrival rate λ (jobs/s)")
    ax1.set_ylabel("Queue depth (jobs)")
    ax1.set_title("Queue Depth vs Arrival Rate")
    ax1.set_yscale("symlog", linthresh=1)
    ax1.legend(framealpha=0.6, fontsize=8, ncol=2)

    for name in sched_names:
        spans = [results[name][load]["metrics"].makespan for load in LOAD_NAMES]
        ax2.plot(RATES, spans, marker=MARKERS[name], label=name,
                 color=COLORS[name], linewidth=2, markersize=7)

    ax2.set_xlabel("Arrival rate λ (jobs/s)")
    ax2.set_ylabel("Makespan (s)")
    ax2.set_title("Makespan vs Arrival Rate")
    ax2.legend(framealpha=0.6, fontsize=9)

    fig.suptitle("Queue Behaviour and Makespan", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig09_queue_and_makespan.png", fig_dir)


def fig10_training_curve(fig_dir: Path) -> None:
    if not HISTORY_PATH.exists():
        print("  [skip] history.json not found — skipping training curve")
        return

    history = json.loads(HISTORY_PATH.read_text())
    if not history:
        print("  [skip] history.json is empty")
        return

    eps     = [h["episode"]      for h in history]
    rewards = [h["total_reward"] for h in history]
    entropy = [h["entropy"]      for h in history]
    v_loss  = [h["value_loss"]   for h in history]
    rates   = [h["arrival_rate"] for h in history]

    window = min(10, len(history))

    def smooth(xs: list) -> np.ndarray:
        return np.convolve(xs, np.ones(window) / window, mode="valid")

    rate_colors = {0.25: "#AACCFF", 0.50: "#FFAA77", 1.00: "#FF6677"}
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

    ax = axes[0]
    for ep, rew, rate in zip(eps, rewards, rates):
        ax.scatter(ep, rew, c=rate_colors.get(rate, "#AAAAAA"), s=8, alpha=0.5, zorder=2)
    for rate, c in rate_colors.items():
        ax.scatter([], [], c=c, label=f"λ={rate}", s=30)
    ax.plot(eps[window - 1:], smooth(rewards), color="black", linewidth=1.5,
            label=f"{window}-ep rolling mean", zorder=3)
    ax.set_ylabel("Total episode reward")
    ax.set_title("Training Reward (coloured by arrival rate)")
    ax.legend(fontsize=9, framealpha=0.6)

    ax = axes[1]
    ax.plot(eps, entropy, color="#9966CC", linewidth=1, alpha=0.5)
    ax.plot(eps[window - 1:], smooth(entropy), color="#9966CC", linewidth=2)
    ax.set_ylabel("Policy entropy (nats)")
    ax.set_title("Policy Entropy  (higher = more exploratory)")

    ax = axes[2]
    ax.plot(eps, v_loss, color="#EE6677", linewidth=1, alpha=0.5)
    ax.plot(eps[window - 1:], smooth(v_loss), color="#EE6677", linewidth=2)
    ax.set_ylabel("Value loss (MSE)")
    ax.set_xlabel("Training episode")
    ax.set_title("Value Network Loss")
    ax.set_yscale("log")

    fig.suptitle("PPO Training Diagnostics", fontsize=13)
    fig.tight_layout()
    _save(fig, "fig10_training_curve.png", fig_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper figures for the RL GPU scheduler."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help=(
            "Path to a .pt checkpoint.  Defaults to the most-recently-modified "
            "best_ep*.pt in rl/checkpoints/, falling back to rl/checkpoints/best.pt."
        ),
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Output subdirectory name under analysis/. Defaults to paper_YYYYMMDD_HHMMSS.",
    )
    args = parser.parse_args()

    ckpt_dir = ROOT / "rl" / "checkpoints"
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        candidates = sorted(
            ckpt_dir.glob("best_ep*.pt"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        ckpt_path = candidates[0] if candidates else ckpt_dir / "best.pt"

    run_name = args.run_name or datetime.now().strftime("paper_%Y%m%d_%H%M%S")
    out_dir  = ROOT / "analysis" / run_name
    fig_dir  = out_dir / "figures"

    print(f"Checkpoint : {ckpt_path.relative_to(ROOT)}")
    print(f"Output dir : {out_dir.relative_to(ROOT)}/")

    test_csv  = ROOT / "gpu_dataset" / "test.csv"
    print(f"\nLoading test set from {test_csv.relative_to(ROOT)} …")
    test_jobs = load_jobs_from_csv(str(test_csv))
    print(f"  {len(test_jobs)} jobs  |  "
          f"exec time range {min(j.true_latency for j in test_jobs):.1f}s – "
          f"{max(j.true_latency for j in test_jobs):.1f}s\n")

    print("Running simulations …")
    results = run_all(test_jobs, ckpt_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "checkpoint":    str(ckpt_path),
        "run_name":      run_name,
        "generated_at":  datetime.now().isoformat(),
        "num_test_jobs": len(test_jobs),
        "schedulers":    list(results.keys()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print("\nGenerating figures …")
    fig01_jct_percentiles(results, fig_dir)
    fig02_jct_cdf(results, fig_dir)
    fig03_tail_scaling(results, fig_dir)
    fig04_throughput_utilization(results, fig_dir)
    fig05_wait_time(results, fig_dir)
    fig06_fairness_by_job_size(results, test_jobs, fig_dir)
    fig07_slowdown(results, test_jobs, fig_dir)
    fig08_improvement_heatmap(results, fig_dir)
    fig09_queue_and_makespan(results, fig_dir)
    fig10_training_curve(fig_dir)

    print(f"\nDone — 10 figures in {out_dir.relative_to(ROOT)}/figures/")


if __name__ == "__main__":
    main()
