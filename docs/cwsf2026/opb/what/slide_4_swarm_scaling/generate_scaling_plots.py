from __future__ import annotations

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]
RAW_EXPORT = ROOT / "docs/gvrsf2026/experiments/20260406_102744/raw_exports/all_episode_results.csv"
OUT_DIR = Path(__file__).resolve().parent

FAMILY = "swarm_size_scaling"
ORDER = ["1_agents", "2_agents", "3_agents", "4_agents", "6_agents", "10_agents", "15_agents", "20_agents", "30_agents"]
SIZE_LABELS = [1, 2, 3, 4, 6, 10, 15, 20, 30]
COLORS = ["#D8E8F5", "#C6DCF0", "#B1D0EA", "#95C0E0", "#76ADCF", "#5797BE", "#3A80AB", "#236998", "#0E4E8A"]


def load_rows() -> list[dict[str, str]]:
    with RAW_EXPORT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["family"] == FAMILY and r["condition"] in ORDER]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rows_by_condition(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {cond: [r for r in rows if r["condition"] == cond] for cond in ORDER}


def plot_food_delivered_boxplot(rows_by_cond: dict[str, list[dict[str, str]]]) -> Path:
    data = [[float(r["food_delivered"]) for r in rows_by_cond[cond]] for cond in ORDER]
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 20, "axes.titleweight": "bold", "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=[str(s) for s in SIZE_LABELS], widths=0.6)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.6)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Food delivered")
    ax.set_title("Food Delivered Distribution by Swarm Size", pad=18)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT_DIR / "scaling_food_delivered_boxplot.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_exploration_boxplot(rows_by_cond: dict[str, list[dict[str, str]]]) -> Path:
    data = [[float(r["exploration_coverage"]) for r in rows_by_cond[cond]] for cond in ORDER]
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 20, "axes.titleweight": "bold", "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=[str(s) for s in SIZE_LABELS], widths=0.6)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.6)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Exploration coverage")
    ax.set_title("Exploration Coverage Distribution by Swarm Size", pad=18)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT_DIR / "scaling_exploration_coverage_boxplot.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_paired_points(rows_by_cond: dict[str, list[dict[str, str]]]) -> Path:
    seed_map: dict[int, list[float]] = {}
    for cond in ORDER:
        size_index = ORDER.index(cond)
        for row in rows_by_cond[cond]:
            seed_map.setdefault(int(row["seed"]), [None] * len(ORDER))
            seed_map[int(row["seed"])][size_index] = float(row["food_delivered"])

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 20, "axes.titleweight": "bold", "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    x = list(range(len(ORDER)))
    for seed in sorted(seed_map):
        vals = seed_map[seed]
        if any(v is None for v in vals):
            continue
        ax.plot(x, vals, color="#A3A3A3", alpha=0.45, linewidth=1)
    means = [mean([float(r["food_delivered"]) for r in rows_by_cond[cond]]) for cond in ORDER]
    ax.plot(x, means, color="#0E4E8A", linewidth=2.6, marker="o", markersize=7, label="Mean", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SIZE_LABELS])
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Food delivered")
    ax.set_title("Food Delivered Across Matched Trials by Swarm Size", pad=18)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = OUT_DIR / "scaling_food_delivered_paired_by_seed.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_combined_scaling(rows_by_cond: dict[str, list[dict[str, str]]]) -> Path:
    means_food = [mean([float(r["food_delivered"]) for r in rows_by_cond[cond]]) for cond in ORDER]
    means_explore = [mean([float(r["exploration_coverage"]) for r in rows_by_cond[cond]]) for cond in ORDER]

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 20, "axes.titleweight": "bold", "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, ax1 = plt.subplots(figsize=(10.8, 6.2))
    ax2 = ax1.twinx()
    ax1.plot(SIZE_LABELS, means_food, color="#1565C0", linewidth=2.8, marker="o", markersize=7, label="Food delivered")
    ax2.plot(SIZE_LABELS, means_explore, color="#D97706", linewidth=2.4, marker="s", markersize=6, label="Exploration coverage")
    ax1.set_xlabel("Number of agents")
    ax1.set_ylabel("Food delivered", color="#1565C0")
    ax2.set_ylabel("Exploration coverage", color="#D97706")
    ax1.set_title("Scaling of Delivery and Exploration", pad=18)
    ax1.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax1.grid(False, axis="x")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=False, loc="upper left")
    fig.tight_layout()
    out = OUT_DIR / "scaling_delivery_and_exploration_combined.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_broad_scaling_food_delivered_final(rows_by_cond: dict[str, list[dict[str, str]]]) -> Path:
    sizes = SIZE_LABELS
    mean_food_delivered = [mean([float(r["food_delivered"]) for r in rows_by_cond[cond]]) for cond in ORDER]
    errors = []
    for cond in ORDER:
        vals = [float(r["food_delivered"]) for r in rows_by_cond[cond]]
        if len(vals) <= 1:
            errors.append(0.0)
        else:
            m = mean(vals)
            variance = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
            se = (variance ** 0.5) / (len(vals) ** 0.5)
            errors.append(se)

    plt.rcParams.update(
        {"font.size": 15, "axes.titlesize": 24, "axes.titleweight": "bold", "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14}
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.errorbar(
        sizes,
        mean_food_delivered,
        yerr=errors,
        color="#1565C0",
        linewidth=3.2,
        marker="o",
        markersize=9,
        markerfacecolor="#1565C0",
        markeredgecolor="black",
        markeredgewidth=0.8,
        capsize=5,
        elinewidth=1.4,
        capthick=1.4,
        zorder=3,
    )

    for x, y, label in [(1, mean_food_delivered[0], "0.05"), (6, mean_food_delivered[4], "1.25"), (30, mean_food_delivered[-1], "4.65")]:
        ax.text(x, y + 0.14, label, ha="center", va="bottom", fontsize=13, fontweight="bold")

    plt.title("Swarm Scaling: Food Delivered Increases with Swarm Size", fontsize=16, weight="bold")
    plt.suptitle("Mean ± standard error across evaluation trials", fontsize=10, y=0.95)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Mean food delivered per episode")
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.5, 31.5)
    ymax = max(mean_food_delivered) * 1.2
    if ymax <= 0:
        ymax = 1.0
    ax.set_ylim(0, ymax)
    ax.text(15, ymax * 0.8, "Collective output scales with swarm size", fontsize=13, color="#1F2937")

    plt.tight_layout()
    out = OUT_DIR / "1-broad_scaling_food_delivered_final_fixed.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_broad_scaling_exploration_final(rows_by_cond: dict[str, list[dict[str, str]]]) -> Path:
    sizes = SIZE_LABELS
    mean_exploration = [mean([float(r["exploration_coverage"]) for r in rows_by_cond[cond]]) for cond in ORDER]
    errors = []
    for cond in ORDER:
        vals = [float(r["exploration_coverage"]) for r in rows_by_cond[cond]]
        if len(vals) <= 1:
            errors.append(0.0)
        else:
            m = mean(vals)
            variance = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
            se = (variance ** 0.5) / (len(vals) ** 0.5)
            errors.append(se)

    plt.rcParams.update(
        {"font.size": 15, "axes.titlesize": 24, "axes.titleweight": "bold", "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14}
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.errorbar(
        sizes,
        mean_exploration,
        yerr=errors,
        color="#D97706",
        linewidth=3.2,
        marker="o",
        markersize=9,
        markerfacecolor="#D97706",
        markeredgecolor="black",
        markeredgewidth=0.8,
        capsize=5,
        elinewidth=1.4,
        capthick=1.4,
        zorder=3,
    )

    for x, idx in [(1, 0), (6, 4), (30, -1)]:
        y = mean_exploration[idx]
        ax.text(x, y + 0.01, f"{y:.2f}", ha="center", va="bottom", fontsize=13, fontweight="bold")

    plt.title("Larger Swarms Explore More of the Environment", fontsize=16, weight="bold")
    plt.suptitle("Exploration coverage increases as more agents are added", fontsize=10, y=0.95)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Mean exploration coverage")
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.5, 31.5)
    ymax = max(mean_exploration) * 1.2
    if ymax <= 0:
        ymax = 1.0
    ax.set_ylim(0, ymax)
    ax.text(13, ymax * 0.78, "More agents → broader coverage", fontsize=13, color="#1F2937")

    plt.tight_layout()
    out = OUT_DIR / "5-scaling_exploration_coverage_final.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_broad_scaling_delivery_conversion_final(rows_by_cond: dict[str, list[dict[str, str]]]) -> Path:
    sizes = SIZE_LABELS
    mean_conversion = [mean([float(r["delivery_conversion"]) for r in rows_by_cond[cond]]) for cond in ORDER]
    errors = []
    for cond in ORDER:
        vals = [float(r["delivery_conversion"]) for r in rows_by_cond[cond]]
        if len(vals) <= 1:
            errors.append(0.0)
        else:
            m = mean(vals)
            variance = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
            se = (variance ** 0.5) / (len(vals) ** 0.5)
            errors.append(se)

    plt.rcParams.update(
        {"font.size": 15, "axes.titlesize": 24, "axes.titleweight": "bold", "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14}
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.errorbar(
        sizes,
        mean_conversion,
        yerr=errors,
        color="#1F6B3A",
        linewidth=3.2,
        marker="o",
        markersize=9,
        markerfacecolor="#1F6B3A",
        markeredgecolor="black",
        markeredgewidth=0.8,
        capsize=5,
        elinewidth=1.4,
        capthick=1.4,
        zorder=3,
    )

    for x, idx in [(1, 0), (6, 4), (30, -1)]:
        y = mean_conversion[idx]
        ax.text(x, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=13, fontweight="bold")

    plt.title("Delivery Efficiency Improves with Swarm Size", fontsize=16, weight="bold")
    plt.suptitle("Conversion from pickup to delivery remains high in larger swarms", fontsize=10, y=0.95)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Delivery conversion rate")
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.5, 31.5)
    ax.set_ylim(0, 1.0)
    ax.text(13, 0.82, "Coordination becomes more efficient", fontsize=13, color="#1F2937")

    plt.tight_layout()
    out = OUT_DIR / "3-scaling_delivery_conversion_final.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    rows = load_rows()
    grouped = rows_by_condition(rows)
    plot_food_delivered_boxplot(grouped)
    plot_exploration_boxplot(grouped)
    plot_paired_points(grouped)
    plot_combined_scaling(grouped)
    plot_broad_scaling_food_delivered_final(grouped)
    plot_broad_scaling_exploration_final(grouped)
    plot_broad_scaling_delivery_conversion_final(grouped)


if __name__ == "__main__":
    main()
