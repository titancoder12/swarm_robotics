from __future__ import annotations

from collections import defaultdict
import csv
import math
from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


ROOT = Path(__file__).resolve().parents[5]
RAW_EXPORT = ROOT / "docs/gvrsf2026/experiments/20260406_102744/killer_scaling/raw_exports/all_episode_results.csv"
OUT_DIR = Path(__file__).resolve().parent

COND_WITH = "trained_with_pheromone__eval_with_pheromone"
COND_WITHOUT = "trained_with_pheromone__eval_without_pheromone"
ORDER = [COND_WITH, COND_WITHOUT]
LABELS = {
    COND_WITH: "Train with pheromone\nEvaluate with pheromone",
    COND_WITHOUT: "Train with pheromone\nEvaluate without pheromone",
}
SHORT_LABELS = {
    COND_WITH: "With pheromone",
    COND_WITHOUT: "Without pheromone",
}
COLORS = {
    COND_WITH: "#1565C0",
    COND_WITHOUT: "#8E8E93",
}


def load_rows() -> list[dict[str, str]]:
    with RAW_EXPORT.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["condition"] in ORDER]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    sem = math.sqrt(variance) / math.sqrt(len(values))
    return float(stats.t.ppf(0.975, df=len(values) - 1) * sem)


def paired_values(rows: list[dict[str, str]], n_agents: int, metric: str) -> tuple[list[float], list[float]]:
    by_seed: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if int(row["n_agents"]) == n_agents:
            by_seed[int(row["seed"])][row["condition"]] = row
    a: list[float] = []
    b: list[float] = []
    for seed in sorted(by_seed):
        pair = by_seed[seed]
        if COND_WITH in pair and COND_WITHOUT in pair:
            a.append(float(pair[COND_WITH][metric]))
            b.append(float(pair[COND_WITHOUT][metric]))
    return a, b


def pvalue_text(a: list[float], b: list[float]) -> str:
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b, zero_method="wilcox")
        w_text = f"Wilcoxon p = {w.pvalue:.4f}"
    except ValueError:
        w_text = "Wilcoxon p = n/a"
    return f"Paired t-test p = {t.pvalue:.4f}\n{w_text}"


def add_pvalue_bracket(ax, x1: float, x2: float, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.6, clip_on=False)
    ax.text((x1 + x2) / 2, y + h + 0.03, text, ha="center", va="bottom", fontsize=12, fontweight="bold")


def plot_scaling(rows: list[dict[str, str]], metric: str, title: str, subtitle: str, ylabel: str, filename: str) -> None:
    sizes = sorted({int(r["n_agents"]) for r in rows})
    plt.rcParams.update(
        {"font.size": 14, "axes.titlesize": 22, "axes.titleweight": "bold", "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12}
    )
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for cond in ORDER:
        means = []
        for size in sizes:
            vals = [float(r[metric]) for r in rows if r["condition"] == cond and int(r["n_agents"]) == size]
            means.append(mean(vals))
        ax.plot(
            sizes,
            means,
            marker="o",
            markersize=8,
            linewidth=3.0,
            color=COLORS[cond],
            markeredgecolor="black",
            markeredgewidth=0.8,
            label=SHORT_LABELS[cond],
        )
    ax.set_title(title, pad=18)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#444444")
    ax.set_xlabel("Number of agents")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bar(rows: list[dict[str, str]], n_agents: int, metric: str, title: str, subtitle: str, ylabel: str, filename: str) -> None:
    a, b = paired_values(rows, n_agents, metric)
    means = [mean(a), mean(b)]
    cis = [ci95(a), ci95(b)]
    x = [0, 1.25]

    plt.rcParams.update(
        {"font.size": 14, "axes.titlesize": 22, "axes.titleweight": "bold", "axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 13}
    )
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.bar(
        x,
        means,
        width=0.66,
        color=[COLORS[COND_WITH], COLORS[COND_WITHOUT]],
        edgecolor="black",
        linewidth=0.9,
        yerr=cis,
        capsize=6,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
        zorder=3,
    )
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04, f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[COND_WITH], LABELS[COND_WITHOUT]])
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=18)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#444444")
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65, zorder=0)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(m + c for m, c in zip(means, cis))
    ax.set_ylim(0, ymax + 0.7)
    add_pvalue_bracket(ax, x[0], x[1], ymax + 0.08, 0.06, f"p = {stats.ttest_rel(a, b).pvalue:.4f}")
    ax.text(0.98, 0.95, pvalue_text(a, b), transform=ax.transAxes, ha="right", va="top", fontsize=11, bbox={"facecolor": "white", "edgecolor": "#CFCFCF", "boxstyle": "round,pad=0.3"})
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta(rows: list[dict[str, str]], n_agents: int, metric: str, title: str, subtitle: str, ylabel: str, filename: str) -> None:
    a, b = paired_values(rows, n_agents, metric)
    deltas = [x - y for x, y in zip(a, b)]
    xs = list(range(1, len(deltas) + 1))
    rng = random.Random(42)
    jittered = [x + rng.uniform(-0.08, 0.08) for x in xs]
    plt.rcParams.update(
        {"font.size": 14, "axes.titlesize": 22, "axes.titleweight": "bold", "axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 13}
    )
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axhline(0, color="#C62828", linestyle="--", linewidth=1.5)
    ax.scatter(jittered, deltas, s=90, color="#1565C0", edgecolors="black", linewidths=0.8, zorder=3)
    mean_delta = mean(deltas)
    ax.axhline(mean_delta, color="black", linewidth=2.0)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#444444")
    ax.text(0.98, 0.95, f"Mean improvement = {mean_delta:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=11, bbox={"facecolor": "white", "edgecolor": "#CFCFCF", "boxstyle": "round,pad=0.3"})
    ax.set_title(title, pad=18)
    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_stats_summary(rows: list[dict[str, str]]) -> None:
    metrics = [
        ("food_delivered", "Food delivered"),
        ("late_deliveries", "Late-episode deliveries"),
        ("post_discovery_deliveries", "Post-discovery deliveries"),
    ]
    out = OUT_DIR / "stats_summary.md"
    lines = [
        "# Pheromone-On vs Pheromone-Off Evaluation Summary",
        "",
        "Comparison: `trained_with_pheromone__eval_with_pheromone` vs `trained_with_pheromone__eval_without_pheromone`",
        "",
    ]
    for n_agents in (6, 30):
        lines.append(f"## {n_agents} agents")
        lines.append("")
        for metric, label in metrics:
            a, b = paired_values(rows, n_agents, metric)
            t = stats.ttest_rel(a, b)
            try:
                w = stats.wilcoxon(a, b, zero_method='wilcox')
                w_line = f"- Wilcoxon p-value: `{w.pvalue:.6f}`"
            except ValueError:
                w_line = "- Wilcoxon p-value: `n/a`"
            lines.extend(
                [
                    f"### {label}",
                    f"- Mean with pheromone: `{mean(a):.2f}`",
                    f"- Mean without pheromone: `{mean(b):.2f}`",
                    f"- Mean difference: `{mean([x-y for x, y in zip(a, b)]):.2f}`",
                    f"- Paired t-test p-value: `{t.pvalue:.6f}`",
                    w_line,
                    "",
                ]
            )
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = load_rows()
    plot_scaling(
        rows,
        "food_delivered",
        "Effect of Removing Pheromone at Evaluation Time",
        "Same trained policy, different access to environmental memory",
        "Mean food delivered",
        "eval_pheromone_food_delivered_by_swarm_size.png",
    )
    plot_scaling(
        rows,
        "late_deliveries",
        "Late-Episode Deliveries With and Without Pheromone",
        "Difference appears mainly in later coordination phases",
        "Mean late-episode deliveries",
        "eval_pheromone_late_deliveries_by_swarm_size.png",
    )
    plot_bar(
        rows,
        6,
        "food_delivered",
        "6-Agent Food Delivered Comparison",
        "No statistically significant difference in total deliveries",
        "Mean food delivered",
        "eval_pheromone_food_delivered_6_agents.png",
    )
    plot_bar(
        rows,
        30,
        "food_delivered",
        "30-Agent Food Delivered Comparison",
        "Small but statistically significant gap in total deliveries",
        "Mean food delivered",
        "eval_pheromone_food_delivered_30_agents.png",
    )
    plot_bar(
        rows,
        6,
        "late_deliveries",
        "6-Agent Late-Episode Delivery Comparison",
        "Difference appears after the swarm has had time to organize",
        "Mean late-episode deliveries",
        "eval_pheromone_late_deliveries_6_agents.png",
    )
    plot_bar(
        rows,
        30,
        "late_deliveries",
        "30-Agent Late-Episode Delivery Comparison",
        "No clear late-phase difference at 30 agents in this comparison",
        "Mean late-episode deliveries",
        "eval_pheromone_late_deliveries_30_agents.png",
    )
    plot_delta(
        rows,
        6,
        "food_delivered",
        "Matched-Trial Improvement at 6 Agents",
        "Each point is with-pheromone minus without-pheromone evaluation",
        "Improvement in food delivered",
        "eval_pheromone_delta_food_delivered_6_agents.png",
    )
    plot_delta(
        rows,
        30,
        "food_delivered",
        "Matched-Trial Improvement at 30 Agents",
        "Each point is with-pheromone minus without-pheromone evaluation",
        "Improvement in food delivered",
        "eval_pheromone_delta_food_delivered_30_agents.png",
    )
    write_stats_summary(rows)


if __name__ == "__main__":
    main()
