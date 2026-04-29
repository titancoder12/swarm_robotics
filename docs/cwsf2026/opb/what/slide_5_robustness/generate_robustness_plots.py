from __future__ import annotations

from pathlib import Path
import csv
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]
RAW_EXPORT = ROOT / "docs/gvrsf2026/experiments/20260406_102744/raw_exports/all_episode_results.csv"
OUT_DIR = Path(__file__).resolve().parent

FAMILY = "robustness_harder_environments"
ORDER = ["control_final_stage", "sensor_noise", "failed_agents_2", "more_obstacles"]
LABELS = {
    "control_final_stage": "Baseline",
    "sensor_noise": "Sensor noise",
    "failed_agents_2": "Agent loss",
    "more_obstacles": "More obstacles",
}
COLORS = {
    "control_final_stage": "#1565C0",
    "sensor_noise": "#5DA5DA",
    "failed_agents_2": "#2E8B57",
    "more_obstacles": "#D97706",
}


def load_rows() -> list[dict[str, str]]:
    with RAW_EXPORT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["family"] == FAMILY and r["condition"] in ORDER]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rows_by_condition(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {cond: [r for r in rows if r["condition"] == cond] for cond in ORDER}


def plot_boxplot(rows_by_cond: dict[str, list[dict[str, str]]], column: str, ylabel: str, title: str, filename: str) -> Path:
    data = [[float(r[column]) for r in rows_by_cond[cond]] for cond in ORDER]
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 20, "axes.titleweight": "bold", "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=[LABELS[c] for c in ORDER], widths=0.58)
    for patch, cond in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COLORS[cond])
        patch.set_alpha(0.8)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.6)
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


def plot_food_delivered_paired(rows_by_cond: dict[str, list[dict[str, str]]]) -> Path:
    seed_map: dict[int, list[float | None]] = {}
    for idx, cond in enumerate(ORDER):
        for row in rows_by_cond[cond]:
            seed = int(row["seed"])
            seed_map.setdefault(seed, [None] * len(ORDER))
            seed_map[seed][idx] = float(row["food_delivered"])

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 20, "axes.titleweight": "bold", "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, ax = plt.subplots(figsize=(10.6, 6.1))
    x = list(range(len(ORDER)))
    for seed in sorted(seed_map):
        vals = seed_map[seed]
        if any(v is None for v in vals):
            continue
        ax.plot(x, vals, color="#A3A3A3", alpha=0.45, linewidth=1)
    means = [mean([float(r["food_delivered"]) for r in rows_by_cond[cond]]) for cond in ORDER]
    ax.plot(x, means, color="black", linewidth=2.2, marker="o", markersize=6, label="Mean", zorder=4)
    for idx, cond in enumerate(ORDER):
        ax.scatter([x[idx]] * len(rows_by_cond[cond]), [float(r["food_delivered"]) for r in rows_by_cond[cond]], color=COLORS[cond], s=22, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in ORDER])
    ax.set_ylabel("Food delivered")
    ax.set_title("Food Delivered Across Matched Robustness Trials", pad=18)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = OUT_DIR / "robustness_food_delivered_paired_by_seed.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_combined_metrics(rows_by_cond: dict[str, list[dict[str, str]]]) -> Path:
    labels = [LABELS[c] for c in ORDER]
    delivered = [mean([float(r["food_delivered"]) for r in rows_by_cond[cond]]) for cond in ORDER]
    explore = [mean([float(r["exploration_coverage"]) for r in rows_by_cond[cond]]) for cond in ORDER]
    x = list(range(len(ORDER)))
    width = 0.34

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 20, "axes.titleweight": "bold", "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, ax1 = plt.subplots(figsize=(10.8, 6.1))
    ax2 = ax1.twinx()
    ax1.bar([i - width / 2 for i in x], delivered, width=width, color="#1565C0", edgecolor="black", linewidth=0.8, label="Food delivered", zorder=3)
    ax2.bar([i + width / 2 for i in x], explore, width=width, color="#D97706", edgecolor="black", linewidth=0.8, label="Exploration coverage", alpha=0.85, zorder=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Food delivered", color="#1565C0")
    ax2.set_ylabel("Exploration coverage", color="#D97706")
    ax1.set_title("Robustness of Delivery and Exploration", pad=18)
    ax1.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax1.grid(False, axis="x")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#1565C0", edgecolor="black"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#D97706", edgecolor="black"),
    ]
    ax1.legend(handles, ["Food delivered", "Exploration coverage"], frameon=False, loc="upper right")
    fig.tight_layout()
    out = OUT_DIR / "robustness_delivery_and_exploration_combined.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_combined_metrics_board_ready(rows_by_cond: dict[str, list[dict[str, str]]]) -> tuple[Path, Path]:
    labels = [LABELS[c] for c in ORDER]
    delivered = [mean([float(r["food_delivered"]) for r in rows_by_cond[cond]]) for cond in ORDER]
    explore = [mean([float(r["exploration_coverage"]) for r in rows_by_cond[cond]]) for cond in ORDER]
    x = list(range(len(ORDER)))
    width = 0.34

    plt.rcParams.update(
        {"font.size": 14, "axes.titlesize": 22, "axes.titleweight": "bold", "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13}
    )
    fig, ax1 = plt.subplots(figsize=(11, 6.4))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2 = ax1.twinx()

    delivery_bars = ax1.bar(
        [i - width / 2 for i in x],
        delivered,
        width=width,
        color="#1565C0",
        edgecolor="black",
        linewidth=0.9,
        label="Food delivered",
        zorder=3,
    )
    coverage_bars = ax2.bar(
        [i + width / 2 for i in x],
        explore,
        width=width,
        color="#D97706",
        edgecolor="black",
        linewidth=0.9,
        alpha=0.88,
        label="Exploration coverage",
        zorder=2,
    )

    for bar, val in zip(delivery_bars, delivered):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.04,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#1F2937",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Food delivered", color="#1565C0")
    ax2.set_ylabel("Exploration coverage", color="#D97706")
    ax1.tick_params(axis="y", colors="#1565C0")
    ax2.tick_params(axis="y", colors="#D97706")
    ax1.set_title("Robustness: Delivery and Exploration Under Stress", pad=16)
    ax1.text(
        0.5,
        1.02,
        "Noise and agent loss reduce performance modestly; dense obstacles cause the largest drop.",
        transform=ax1.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444",
    )
    ax1.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax1.grid(False, axis="x")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    obstacle_idx = ORDER.index("more_obstacles")
    obstacle_bar = delivery_bars[obstacle_idx]
    obstacle_bar.set_linewidth(2.0)
    obstacle_bar.set_edgecolor("#7C2D12")
    ax1.text(
        obstacle_bar.get_x() + obstacle_bar.get_width() / 2,
        obstacle_bar.get_height() + 0.12,
        "Largest drop",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#7C2D12",
        fontweight="bold",
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#1565C0", edgecolor="black", linewidth=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor="#D97706", edgecolor="black", linewidth=0.8),
    ]
    ax1.legend(
        handles,
        ["Food delivered", "Exploration coverage"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=12,
    )

    fig.subplots_adjust(top=0.82, bottom=0.24, left=0.10, right=0.90)
    png_out = OUT_DIR / "robustness_combined_board_ready_fixed.png"
    pdf_out = OUT_DIR / "robustness_combined_board_ready_fixed.pdf"
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    return png_out, pdf_out


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return (variance ** 0.5) / (len(values) ** 0.5)


def plot_food_delivered_board_ready(rows_by_cond: dict[str, list[dict[str, str]]]) -> tuple[Path, Path]:
    labels = [LABELS[c] for c in ORDER]
    delivered_lists = [[float(r["food_delivered"]) for r in rows_by_cond[cond]] for cond in ORDER]
    delivered = [mean(vals) for vals in delivered_lists]
    delivered_sem = [sem(vals) for vals in delivered_lists]
    x = list(range(len(ORDER)))

    plt.rcParams.update(
        {"font.size": 14, "axes.titlesize": 22, "axes.titleweight": "bold", "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13}
    )
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bar_colors = ["#1565C0", "#5DA5DA", "#2E8B57", "#D97706"]
    bars = ax.bar(
        x,
        delivered,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.9,
        width=0.62,
        yerr=delivered_sem,
        error_kw={"elinewidth": 1.2, "capsize": 5, "capthick": 1.2, "ecolor": "#333333"},
        zorder=3,
    )

    baseline_val = delivered[0]
    ax.axhline(baseline_val, color="#444444", linestyle="--", linewidth=1.5, zorder=2)
    ax.text(
        3.38,
        baseline_val + 0.03,
        "Baseline performance",
        ha="right",
        va="bottom",
        fontsize=11,
        color="#444444",
    )

    for bar, val in zip(bars, delivered):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.045,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#1F2937",
        )

    obstacle_bar = bars[-1]
    ax.text(
        obstacle_bar.get_x() + obstacle_bar.get_width() / 2,
        obstacle_bar.get_height() + 0.22,
        "Largest drop:\nnavigation complexity",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#7C2D12",
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean food delivered per episode")
    ax.set_title("Food Delivered Under Robustness Conditions", pad=16)
    ax.set_ylim(0, max(delivered) * 1.32)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    png_out = OUT_DIR / "robustness_food_delivered_board_ready.png"
    pdf_out = OUT_DIR / "robustness_food_delivered_board_ready.pdf"
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    return png_out, pdf_out


def plot_food_delivered_distribution_board_ready(rows_by_cond: dict[str, list[dict[str, str]]]) -> tuple[Path, Path]:
    labels = [LABELS[c] for c in ORDER]
    data = [[float(r["food_delivered"]) for r in rows_by_cond[cond]] for cond in ORDER]
    means = [mean(vals) for vals in data]

    plt.rcParams.update(
        {"font.size": 14, "axes.titlesize": 22, "axes.titleweight": "bold", "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13}
    )
    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bp = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=labels,
        widths=0.56,
        showfliers=False,
    )
    box_colors = ["#1565C0", "#5DA5DA", "#2E8B57", "#D97706"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.8)
    for whisker in bp["whiskers"]:
        whisker.set_color("#444444")
        whisker.set_linewidth(1.1)
    for cap in bp["caps"]:
        cap.set_color("#444444")
        cap.set_linewidth(1.1)

    rng = random.Random(42)
    for i, vals in enumerate(data, start=1):
        jitter_x = [i + rng.uniform(-0.10, 0.10) for _ in vals]
        ax.scatter(jitter_x, vals, s=22, color="#374151", alpha=0.28, linewidths=0, zorder=2)

    mean_handle = ax.scatter(
        list(range(1, len(means) + 1)),
        means,
        marker="D",
        s=70,
        color="#111111",
        edgecolors="white",
        linewidths=0.8,
        zorder=4,
        label="Mean",
    )

    ax.set_title("Trial-to-Trial Robustness Distribution", pad=16)
    ax.set_ylabel("Food delivered per trial")
    ax.text(
        0.5,
        1.01,
        "Boxes show variation across repeated trials.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444",
    )
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=[mean_handle], frameon=False, loc="upper right", fontsize=12)

    fig.tight_layout()
    png_out = OUT_DIR / "robustness_food_delivered_distribution_board_ready.png"
    pdf_out = OUT_DIR / "robustness_food_delivered_distribution_board_ready.pdf"
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    return png_out, pdf_out


def main() -> None:
    rows = load_rows()
    grouped = rows_by_condition(rows)
    plot_boxplot(grouped, "food_delivered", "Food delivered", "Food Delivered Robustness Distribution", "robustness_food_delivered_boxplot.png")
    plot_boxplot(grouped, "delivery_conversion", "Delivery conversion", "Delivery Conversion Under Robustness Conditions", "robustness_delivery_conversion_boxplot.png")
    plot_boxplot(grouped, "exploration_coverage", "Exploration coverage", "Exploration Coverage Under Robustness Conditions", "robustness_exploration_coverage_boxplot.png")
    plot_food_delivered_paired(grouped)
    plot_combined_metrics(grouped)
    plot_combined_metrics_board_ready(grouped)
    plot_food_delivered_board_ready(grouped)
    plot_food_delivered_distribution_board_ready(grouped)


if __name__ == "__main__":
    main()
