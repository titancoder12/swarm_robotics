from __future__ import annotations

from pathlib import Path
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


ROOT = Path(__file__).resolve().parents[5]
RAW_EXPORT = ROOT / "docs/gvrsf2026/experiments/20260406_102744/raw_exports/all_episode_results.csv"
OUT_DIR = Path(__file__).resolve().parent

FAMILY = "baseline_comparison"
ORDER = ["current_mappo", "rule_based", "random"]
LABELS = {
    "current_mappo": "MAPPO",
    "rule_based": "Rule-based",
    "random": "Random",
}
COLORS = {
    "current_mappo": "#1565C0",
    "rule_based": "#D97706",
    "random": "#6B7280",
}


def load_rows() -> list[dict[str, str]]:
    with RAW_EXPORT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["family"] == FAMILY and r["condition"] in ORDER]


def by_condition(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {cond: [r for r in rows if r["condition"] == cond] for cond in ORDER}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    sample_mean = mean(values)
    variance = sum((v - sample_mean) ** 2 for v in values) / (len(values) - 1)
    sem = math.sqrt(variance) / math.sqrt(len(values))
    return float(stats.t.ppf(0.975, df=len(values) - 1) * sem)


def grouped_means(rows_by_cond: dict[str, list[dict[str, str]]], columns: list[str]) -> dict[str, dict[str, float]]:
    data: dict[str, dict[str, float]] = {}
    for cond, rows in rows_by_cond.items():
        data[cond] = {col: mean([float(r[col]) for r in rows]) for col in columns}
    return data


def plot_pickups_vs_deliveries(means: dict[str, dict[str, float]]) -> Path:
    x = [0.0, 1.25, 2.5]
    width = 0.28
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    light = {
        "current_mappo": "#6BAED6",
        "rule_based": "#FDBA74",
        "random": "#B0B7C3",
    }

    for idx, cond in enumerate(ORDER):
        picked = means[cond]["food_picked_up"]
        delivered = means[cond]["food_delivered"]
        ax.bar(x[idx] - width / 2, picked, width=width, color=light[cond], edgecolor="black", linewidth=0.8, zorder=3)
        ax.bar(x[idx] + width / 2, delivered, width=width, color=COLORS[cond], edgecolor="black", linewidth=0.8, zorder=3)
        ax.text(x[idx] - width / 2, picked + 0.06, f"{picked:.2f}", ha="center", va="bottom", fontsize=11)
        ax.text(x[idx] + width / 2, delivered + 0.06, f"{delivered:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in ORDER])
    ax.set_ylabel("Mean per episode")
    ax.set_title("Baseline Pickup vs Delivery", pad=18)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(max(means[c]["food_picked_up"], means[c]["food_delivered"]) for c in ORDER) + 0.75)
    fig.tight_layout()
    out = OUT_DIR / "baseline_pickups_vs_deliveries_grouped.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pickup_delivery_final(means: dict[str, dict[str, float]]) -> Path:
    labels = [LABELS[c] for c in ORDER]
    picked_vals = [means[c]["food_picked_up"] for c in ORDER]
    delivered_vals = [means[c]["food_delivered"] for c in ORDER]
    conversions = [
        0.0 if picked == 0 else delivered / picked
        for picked, delivered in zip(picked_vals, delivered_vals)
    ]

    x = [0.0, 1.4, 2.8]
    width = 0.32
    picked_colors = ["#7FB3D5", "#A7D7A7", "#E0B3B3"]
    delivered_colors = ["#1F5A94", "#2F7D32", "#A94442"]

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(11.2, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    picked_bars = []
    delivered_bars = []
    for i, cond in enumerate(ORDER):
        pb = ax.bar(x[i] - width / 2, picked_vals[i], width=width, color=picked_colors[i], edgecolor="black", linewidth=0.8, zorder=3)
        db = ax.bar(x[i] + width / 2, delivered_vals[i], width=width, color=delivered_colors[i], edgecolor="black", linewidth=0.8, zorder=3)
        picked_bars.extend(pb)
        delivered_bars.extend(db)

    for bar, val in zip(picked_bars, picked_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    for bar, val in zip(delivered_bars, delivered_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    for i, xi in enumerate(x):
        ax.text(
            xi,
            max(picked_vals[i], delivered_vals[i]) + 0.38,
            f"Conversion = {round(conversions[i] * 100):.0f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean per episode")
    ax.set_title("Pickup vs Delivery Efficiency", pad=28)
    ax.text(
        0.5,
        1.02,
        "MAPPO converts pickups into successful deliveries",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="#444444",
    )
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65, zorder=0)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(max(picked_vals), max(delivered_vals)) + 0.85)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#BFCFE0", edgecolor="black", linewidth=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor="#4E6B8A", edgecolor="black", linewidth=0.8),
    ]
    ax.legend(legend_handles, ["Picked up", "Delivered"], frameon=False, loc="upper right")

    fig.tight_layout()
    out = OUT_DIR / "baseline_pickup_delivery_final.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_food_delivered_paired_by_seed(rows_by_cond: dict[str, list[dict[str, str]]]) -> Path:
    seed_map: dict[int, dict[str, float]] = {}
    for cond, rows in rows_by_cond.items():
        for row in rows:
            seed_map.setdefault(int(row["seed"]), {})[cond] = float(row["food_delivered"])

    seeds = sorted(seed_map)
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    xpos = [0, 1, 2]
    for seed in seeds:
        vals = [seed_map[seed].get(cond, 0.0) for cond in ORDER]
        ax.plot(xpos, vals, color="#A3A3A3", alpha=0.55, linewidth=1)
        for i, cond in enumerate(ORDER):
            ax.scatter(xpos[i], vals[i], color=COLORS[cond], s=28, zorder=3)

    means = [mean([seed_map[s].get(cond, 0.0) for s in seeds]) for cond in ORDER]
    ax.plot(xpos, means, color="black", linewidth=2.2, marker="o", markersize=6, label="Mean", zorder=4)
    ax.set_xticks(xpos)
    ax.set_xticklabels([LABELS[c] for c in ORDER])
    ax.set_ylabel("Food delivered")
    ax.set_title("Food Delivered Across Matched Trials", pad=18)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = OUT_DIR / "baseline_food_delivered_paired_by_seed.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_boxplot(rows_by_cond: dict[str, list[dict[str, str]]], column: str, ylabel: str, title: str, filename: str) -> Path:
    data = [[float(r[column]) for r in rows_by_cond[cond]] for cond in ORDER]
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=[LABELS[c] for c in ORDER], widths=0.55)
    for patch, cond in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COLORS[cond])
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=18)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def add_pvalue_bracket(ax, x1: float, x2: float, y: float, h: float, text: str, fontsize: int = 12) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.4, clip_on=False)
    ax.text((x1 + x2) / 2, y + h + 0.03, text, ha="center", va="bottom", fontsize=fontsize)


def plot_food_delivered_final(rows_by_cond: dict[str, list[dict[str, str]]]) -> Path:
    values = {cond: [float(r["food_delivered"]) for r in rows_by_cond[cond]] for cond in ORDER}
    means = [mean(values[cond]) for cond in ORDER]
    cis = [ci95(values[cond]) for cond in ORDER]
    x = [0, 1.35, 2.7]
    colors = ["#1565C0", "#2E8B57", "#9E9E9E"]

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(
        x,
        means,
        yerr=cis,
        width=0.72,
        color=colors,
        edgecolor="black",
        linewidth=0.9,
        capsize=7,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
        zorder=3,
    )

    for bar, label in zip(bars, ["1.25", "0.30", "0.05"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["MAPPO", "Rule-Based", "Random"])
    ax.set_ylabel("Mean food delivered per episode")
    ax.set_title("MAPPO Outperforms Baselines", pad=18)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.6, zorder=0)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymax = max(m + c for m, c in zip(means, cis))
    ax.set_ylim(0, ymax + 0.55)
    add_pvalue_bracket(ax, x[0], x[1], ymax + 0.05, 0.05, "p = 0.0123", fontsize=12)
    add_pvalue_bracket(ax, x[0], x[2], ymax + 0.20, 0.05, "p = 0.0012", fontsize=12)

    fig.tight_layout()
    out = OUT_DIR / "baseline_food_delivered_final.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    rows = load_rows()
    rows_by_cond = by_condition(rows)
    means = grouped_means(rows_by_cond, ["food_picked_up", "food_delivered", "exploration_coverage", "mean_episode_reward", "time_to_first_discovery"])
    plot_pickups_vs_deliveries(means)
    plot_pickup_delivery_final(means)
    plot_food_delivered_final(rows_by_cond)
    plot_food_delivered_paired_by_seed(rows_by_cond)
    plot_boxplot(
        rows_by_cond,
        "time_to_first_discovery",
        "Steps to first discovery",
        "Time to First Discovery by Baseline",
        "baseline_time_to_first_discovery_boxplot.png",
    )
    plot_boxplot(
        rows_by_cond,
        "mean_episode_reward",
        "Mean episode reward",
        "Mean Episode Reward by Baseline",
        "baseline_mean_episode_reward_boxplot.png",
    )
    plot_boxplot(
        rows_by_cond,
        "exploration_coverage",
        "Exploration coverage",
        "Exploration Coverage by Baseline",
        "baseline_exploration_coverage_boxplot.png",
    )


if __name__ == "__main__":
    main()
