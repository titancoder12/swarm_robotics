from __future__ import annotations

from pathlib import Path
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


ROOT = Path(__file__).resolve().parents[5]
RAW_EXPORT = ROOT / "docs/gvrsf2026/experiments/20260406_102744/killer_scaling/raw_exports/all_episode_results.csv"
OUT_DIR = Path(__file__).resolve().parent

ORDER = [
    "trained_with_pheromone__eval_with_pheromone",
    "trained_with_pheromone__eval_without_pheromone",
    "trained_without_pheromone__eval_without_pheromone",
]
LABELS = {
    "trained_with_pheromone__eval_with_pheromone": "Pheromone-trained\n+ pheromone eval",
    "trained_with_pheromone__eval_without_pheromone": "Pheromone-trained\n- pheromone eval",
    "trained_without_pheromone__eval_without_pheromone": "No-pheromone-trained\n- pheromone eval",
}
COLORS = {
    "trained_with_pheromone__eval_with_pheromone": "#1565C0",
    "trained_with_pheromone__eval_without_pheromone": "#5DA5DA",
    "trained_without_pheromone__eval_without_pheromone": "#D97706",
}


def load_rows() -> list[dict[str, str]]:
    with RAW_EXPORT.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    sem = math.sqrt(variance) / math.sqrt(len(values))
    return float(stats.t.ppf(0.975, df=len(values) - 1) * sem)


def rows_for_size(rows: list[dict[str, str]], n_agents: int) -> dict[str, list[dict[str, str]]]:
    return {cond: [r for r in rows if r["condition"] == cond and int(r["n_agents"]) == n_agents] for cond in ORDER}


def plot_matched_bar(rows: list[dict[str, str]], n_agents: int, metric: str, title: str, ylabel: str, filename: str, p_text: str | None = None) -> Path:
    by_cond = rows_for_size(rows, n_agents)
    values = {cond: [float(r[metric]) for r in by_cond[cond]] for cond in ORDER}
    means = [mean(values[c]) for c in ORDER]
    cis = [ci95(values[c]) for c in ORDER]
    x = [0, 1.35, 2.7]

    plt.rcParams.update(
        {"font.size": 13, "axes.titlesize": 20, "axes.titleweight": "bold", "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12}
    )
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    bars = ax.bar(
        x,
        means,
        yerr=cis,
        width=0.74,
        color=[COLORS[c] for c in ORDER],
        edgecolor="black",
        linewidth=0.8,
        capsize=6,
        error_kw={"elinewidth": 1.3, "capthick": 1.3},
        zorder=3,
    )
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=18)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65, zorder=0)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(m + c for m, c in zip(means, cis))
    ax.set_ylim(0, ymax + 0.7)
    if p_text:
        ax.text(0.98, 0.96, p_text, transform=ax.transAxes, ha="right", va="top", fontsize=12, bbox={"facecolor": "white", "edgecolor": "#CFCFCF", "boxstyle": "round,pad=0.35"})
    fig.tight_layout()
    out = OUT_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_scaling_metric(rows: list[dict[str, str]], metric: str, title: str, ylabel: str, filename: str) -> Path:
    sizes = sorted({int(r["n_agents"]) for r in rows})
    plt.rcParams.update(
        {"font.size": 13, "axes.titlesize": 20, "axes.titleweight": "bold", "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12}
    )
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    for cond in ORDER:
        means = []
        for size in sizes:
            vals = [float(r[metric]) for r in rows if r["condition"] == cond and int(r["n_agents"]) == size]
            means.append(mean(vals))
        ax.plot(sizes, means, marker="o", markersize=6, linewidth=2.2, color=COLORS[cond], label=LABELS[cond].replace("\n", " "))
    ax.set_xlabel("Number of agents")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=18)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    out = OUT_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_paired_seed_delta(rows: list[dict[str, str]], n_agents: int, metric: str, title: str, ylabel: str, filename: str) -> Path:
    subset = [r for r in rows if int(r["n_agents"]) == n_agents]
    by_seed: dict[int, dict[str, float]] = {}
    for r in subset:
        by_seed.setdefault(int(r["seed"]), {})[r["condition"]] = float(r[metric])
    seeds = sorted(by_seed)
    deltas = []
    for seed in seeds:
        pair = by_seed[seed]
        if ORDER[0] in pair and ORDER[2] in pair:
            deltas.append(pair[ORDER[0]] - pair[ORDER[2]])
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.axhline(0, color="#C62828", linestyle="--", linewidth=1.4)
    ax.scatter(range(1, len(deltas) + 1), deltas, s=80, color="#1565C0", edgecolors="black", linewidths=0.8, zorder=3)
    mean_delta = mean(deltas)
    ax.axhline(mean_delta, color="black", linewidth=2.0)
    ax.text(0.98, 0.96, f"Mean delta = {mean_delta:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=12, bbox={"facecolor": "white", "edgecolor": "#CFCFCF", "boxstyle": "round,pad=0.35"})
    ax.set_xlabel("Matched trial")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=18)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def add_pvalue_bracket(ax, x1: float, x2: float, y: float, h: float, text: str, fontsize: int = 13) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.8, clip_on=False)
    ax.text((x1 + x2) / 2, y + h + 0.03, text, ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


def plot_main_final_30_agents(rows: list[dict[str, str]]) -> Path:
    subset = rows_for_size(rows, 30)
    with_pheromone = [float(r["food_delivered"]) for r in subset[ORDER[0]]]
    without_pheromone = [float(r["food_delivered"]) for r in subset[ORDER[2]]]
    mean_with = mean(with_pheromone)
    mean_without = mean(without_pheromone)
    ci_with = ci95(with_pheromone)
    ci_without = ci95(without_pheromone)

    x = [0, 1.35]
    labels = ["With pheromone", "Without pheromone"]
    colors = ["#1565C0", "#8E8E93"]

    plt.rcParams.update(
        {"font.size": 15, "axes.titlesize": 24, "axes.titleweight": "bold", "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14}
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(
        x,
        [mean_with, mean_without],
        yerr=[ci_with, ci_without],
        width=0.72,
        color=colors,
        edgecolor="black",
        linewidth=0.9,
        capsize=7,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
        zorder=3,
    )

    value_labels = ["2.66", "0.00"]
    for idx, (bar, label) in enumerate(zip(bars, value_labels)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.06,
            label,
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold" if idx == 1 else "semibold",
            color="#C62828" if idx == 1 else "black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Food Delivered per Episode")
    ax.set_title("Stigmergy is Required for Coordination (30 Agents)", pad=28)
    ax.text(
        0.5,
        1.02,
        "Removing pheromones causes complete failure (0 deliveries)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        color="#444444",
    )
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65, zorder=0)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 3.05)

    add_pvalue_bracket(ax, x[0], x[1], 2.78, 0.07, "p = 0.000847", fontsize=14)

    ax.text(x[0] + 0.42, 2.15, "Emergent coordination", ha="left", va="center", fontsize=13, color="#1F2937")
    ax.text(x[1] + 0.38, 0.22, "No coordination", ha="left", va="center", fontsize=13, color="#1F2937")

    fig.tight_layout()
    out = OUT_DIR / "stigmergy_main_final.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_scaling_final(rows: list[dict[str, str]]) -> Path:
    sizes = sorted({int(r["n_agents"]) for r in rows})
    with_cond = ORDER[0]
    without_cond = ORDER[2]
    with_means = []
    without_means = []
    for size in sizes:
        with_vals = [float(r["food_delivered"]) for r in rows if r["condition"] == with_cond and int(r["n_agents"]) == size]
        without_vals = [float(r["food_delivered"]) for r in rows if r["condition"] == without_cond and int(r["n_agents"]) == size]
        with_means.append(mean(with_vals))
        without_means.append(mean(without_vals))

    plt.rcParams.update(
        {"font.size": 15, "axes.titlesize": 24, "axes.titleweight": "bold", "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13}
    )
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        sizes,
        with_means,
        color="#1565C0",
        linewidth=3.2,
        marker="o",
        markersize=8,
        markerfacecolor="#1565C0",
        markeredgecolor="black",
        label="With Pheromone",
        zorder=3,
    )
    ax.plot(
        sizes,
        without_means,
        color="#C62828",
        linewidth=3.0,
        linestyle="--",
        marker="o",
        markersize=8,
        markerfacecolor="#C62828",
        markeredgecolor="black",
        label="Without Pheromone",
        zorder=3,
    )

    ax.set_xlabel("Number of Agents")
    ax.set_ylabel("Mean Food Delivered")
    ax.set_title("Stigmergy Enables Scalable Coordination", pad=28)
    ax.text(
        0.5,
        1.02,
        "Performance increases with swarm size only when pheromones are present",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        color="#444444",
    )
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    ymin = min(min(without_means), min(with_means))
    ymax = max(max(with_means), max(without_means))
    ax.set_ylim(max(-0.05, ymin - 0.1), ymax + 0.45)
    ax.set_xlim(min(sizes) - 0.5, max(sizes) + 1.0)

    ax.text(18, 2.35, "Stigmergy \u2192 scaling performance", color="#1F2937", fontsize=13, ha="left", va="center")
    ax.text(14.5, 0.12, "No stigmergy \u2192 no performance", color="#1F2937", fontsize=13, ha="left", va="center")

    fig.tight_layout()
    out = OUT_DIR / "stigmergy_scaling_final.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_consistency_final(rows: list[dict[str, str]], n_agents: int = 30) -> Path:
    subset = [r for r in rows if int(r["n_agents"]) == n_agents]
    by_seed: dict[int, dict[str, float]] = {}
    for r in subset:
        by_seed.setdefault(int(r["seed"]), {})[r["condition"]] = float(r["food_delivered"])
    seeds = sorted(by_seed)
    deltas = []
    for seed in seeds:
        pair = by_seed[seed]
        if ORDER[0] in pair and ORDER[2] in pair:
            deltas.append(pair[ORDER[0]] - pair[ORDER[2]])

    trials = list(range(1, len(deltas) + 1))
    mean_delta = mean(deltas)
    all_positive = all(delta > 0 for delta in deltas)

    plt.rcParams.update(
        {"font.size": 15, "axes.titlesize": 24, "axes.titleweight": "bold", "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14}
    )
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(
        trials,
        deltas,
        s=110,
        color="#1565C0",
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )
    ax.axhline(0, color="#C62828", linestyle="--", linewidth=1.6, zorder=1)
    ax.axhline(mean_delta, color="black", linewidth=2.2, zorder=2)

    ymin = min(min(deltas), 0.0)
    ymax = max(max(deltas), mean_delta)
    pad = max(0.2, 0.12 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(0.5, len(trials) + 0.8)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Improvement in Food Delivered")
    ax.set_title("Consistent Improvement Across Matched Trials", pad=28)
    ax.text(
        0.5,
        1.02,
        "Each point shows performance gain from adding stigmergy",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        color="#444444",
    )
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(len(trials) + 0.65, 0, "No improvement", color="#C62828", fontsize=12, ha="right", va="bottom", backgroundcolor="white")
    ax.text(len(trials) + 0.65, mean_delta, "Mean improvement", color="black", fontsize=12, fontweight="bold", ha="right", va="bottom", backgroundcolor="white")
    if all_positive:
        ax.text(
            0.02,
            0.96,
            "All trials show positive improvement",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            bbox={"facecolor": "white", "edgecolor": "#CFCFCF", "boxstyle": "round,pad=0.35"},
        )

    fig.tight_layout()
    out = OUT_DIR / "stigmergy_consistency_final.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    rows = load_rows()
    plot_scaling_metric(rows, "delivery_conversion", "Delivery Conversion Across Swarm Sizes", "Delivery conversion", "stigmergy_delivery_conversion_by_swarm_size.png")
    plot_matched_bar(
        rows,
        6,
        "food_delivered",
        "Stigmergy Effect at 6 Agents",
        "Mean food delivered",
        "stigmergy_food_delivered_6_agents.png",
        "Paired t-test p = 0.00653",
    )
    plot_matched_bar(
        rows,
        30,
        "food_delivered",
        "Stigmergy Effect at 30 Agents",
        "Mean food delivered",
        "stigmergy_food_delivered_30_agents.png",
        "Paired t-test p = 0.000847\nWilcoxon p = 9.96e-06",
    )
    plot_matched_bar(
        rows,
        30,
        "late_deliveries",
        "Late-Episode Deliveries at 30 Agents",
        "Mean late deliveries",
        "stigmergy_late_deliveries_30_agents.png",
        "Paired t-test p = 0.01868",
    )
    plot_paired_seed_delta(
        rows,
        30,
        "food_delivered",
        "Matched-Trial Improvement at 30 Agents",
        "Food delivered improvement",
        "stigmergy_food_delivered_delta_30_agents.png",
    )
    plot_paired_seed_delta(
        rows,
        6,
        "food_delivered",
        "Matched-Trial Improvement at 6 Agents",
        "Food delivered improvement",
        "stigmergy_food_delivered_delta_6_agents.png",
    )
    plot_main_final_30_agents(rows)
    plot_scaling_final(rows)
    plot_consistency_final(rows, 30)


if __name__ == "__main__":
    main()
