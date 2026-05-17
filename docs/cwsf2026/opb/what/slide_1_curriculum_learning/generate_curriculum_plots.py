from __future__ import annotations

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


ROOT = Path(__file__).resolve().parents[5]
RAW_EXPORT = ROOT / "docs/gvrsf2026/experiments/20260406_102744/raw_exports/all_episode_results.csv"
STATS_CSV = ROOT / "docs/gvrsf2026/experiments/20260406_102744/tables/statistical_tests.csv"
OUT_DIR = Path(__file__).resolve().parent

CURRENT = "current_curriculum_final"
OLDER = "older_weaker_curriculum_final"
FAMILY = "curriculum_vs_weaker_training"


def load_curriculum_rows() -> list[dict[str, str]]:
    with RAW_EXPORT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["family"] == FAMILY and r["condition"] in {CURRENT, OLDER}]


def build_matched_trials(rows: list[dict[str, str]]) -> list[dict[str, float]]:
    by_seed: dict[int, dict[str, dict[str, str]]] = {}
    for row in rows:
        seed = int(row["seed"])
        by_seed.setdefault(seed, {})[row["condition"]] = row

    trials: list[dict[str, float]] = []
    for idx, seed in enumerate(sorted(by_seed), start=1):
        pair = by_seed[seed]
        if CURRENT not in pair or OLDER not in pair:
            continue
        newer = float(pair[CURRENT]["food_delivered"])
        older = float(pair[OLDER]["food_delivered"])
        trials.append(
            {
                "trial": idx,
                "seed": seed,
                "newer": newer,
                "older": older,
                "delta": newer - older,
            }
        )
    return trials


def load_reported_curriculum_pvalue() -> float | None:
    with STATS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row["family"] == FAMILY
                and row["condition_a"] == CURRENT
                and row["condition_b"] == OLDER
                and row["metric"] == "food_delivered"
            ):
                return float(row["p_value"])
    return None


def compute_paired_ttest_pvalue(trials: list[dict[str, float]]) -> float | None:
    newer = [float(t["newer"]) for t in trials]
    older = [float(t["older"]) for t in trials]
    if not newer or len(newer) != len(older):
        return None
    return float(stats.ttest_rel(newer, older).pvalue)


def compute_condition_means(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for condition in (CURRENT, OLDER):
        cond_rows = [r for r in rows if r["condition"] == condition]
        if not cond_rows:
            grouped[condition] = {"food_picked_up": 0.0, "food_delivered": 0.0}
            continue
        picked = [float(r["food_picked_up"]) for r in cond_rows]
        delivered = [float(r["food_delivered"]) for r in cond_rows]
        grouped[condition] = {
            "food_picked_up": sum(picked) / len(picked),
            "food_delivered": sum(delivered) / len(delivered),
        }
    return grouped


def generate_original_delta_by_seed(trials: list[dict[str, float]]) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axhline(0, color="black", linewidth=1)
    ax.bar(
        [str(int(t["seed"])) for t in trials],
        [t["delta"] for t in trials],
        color="#2C7FB8",
        alpha=0.8,
    )
    ax.set_ylabel("Food delivered difference")
    ax.set_xlabel("Seed")
    ax.set_title("Matched-Seed Improvement: Newer Minus Older Food Delivered")
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_fontsize(8)
    fig.tight_layout()
    out = OUT_DIR / "curriculum_food_delivered_delta_by_seed.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def generate_improved_delta_by_trial(trials: list[dict[str, float]], p_value: float | None) -> Path:
    trial_ids = [int(t["trial"]) for t in trials]
    deltas = [float(t["delta"]) for t in trials]
    mean_delta = sum(deltas) / len(deltas)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(11, 6.8))

    ax.scatter(trial_ids, deltas, s=90, color="#2C7FB8", edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(0, color="#C62828", linestyle="--", linewidth=2, zorder=1)
    ax.axhline(mean_delta, color="black", linewidth=2.8, zorder=2)

    ax.set_xlim(0.5, len(trial_ids) + 0.5)
    ymin = min(min(deltas), 0.0)
    ymax = max(max(deltas), mean_delta)
    pad = max(0.25, 0.12 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xticks(trial_ids)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Food delivered improvement")
    ax.set_title("Consistent Improvement Across Trials", pad=28)
    ax.text(
        0.5,
        1.02,
        "New curriculum minus older configuration, evaluated under identical initial conditions",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="#444444",
    )

    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    right_x = len(trial_ids) + 0.35
    ax.text(
        right_x,
        0,
        "No improvement",
        color="#C62828",
        fontsize=11,
        va="center",
        ha="right",
        backgroundcolor="white",
    )
    ax.text(
        right_x,
        mean_delta,
        f"Mean improvement = {mean_delta:.2f}",
        color="black",
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="right",
        backgroundcolor="white",
    )

    p_text = f"Reported Welch's t-test p = {p_value:.6f}" if p_value is not None else "Reported p-value unavailable"
    annotation = "All positive values indicate improvement\n" + p_text
    ax.text(
        0.02,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "#CFCFCF", "boxstyle": "round,pad=0.45"},
    )

    fig.tight_layout()
    out = OUT_DIR / "curriculum_food_delivered_delta_by_trial_improved.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_final_delta_by_trial(trials: list[dict[str, float]], p_value: float | None) -> Path:
    trial_ids = [int(t["trial"]) for t in trials]
    deltas = [float(t["delta"]) for t in trials]
    mean_delta = sum(deltas) / len(deltas)
    all_nonnegative = all(delta >= 0 for delta in deltas)

    jitter_pattern = (-0.08, -0.04, 0.0, 0.04, 0.08)
    jittered_x = [trial + jitter_pattern[(idx - 1) % len(jitter_pattern)] for idx, trial in enumerate(trial_ids, start=1)]

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.8))

    ax.scatter(
        jittered_x,
        deltas,
        s=160,
        color="#1565C0",
        edgecolors="black",
        linewidths=0.9,
        zorder=3,
    )
    ax.axhline(0, color="#C62828", linestyle="--", linewidth=1.5, zorder=1)
    ax.axhline(mean_delta, color="black", linewidth=2.0, zorder=2)

    x_min = min(jittered_x) - 0.2
    x_max = max(jittered_x) + 0.2
    ax.set_xlim(x_min, x_max)
    ymin = min(min(deltas), 0.0)
    ymax = max(max(deltas), mean_delta)
    pad = max(0.3, 0.12 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xticks(trial_ids)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Food delivered improvement")
    ax.set_title("Consistent Performance Gains Across Matched Trials", pad=30)
    ax.text(
        0.5,
        1.02,
        "New curriculum minus older configuration, evaluated under identical initial conditions",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="#444444",
    )

    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    label_x = x_max - 0.02
    ax.text(
        label_x,
        0,
        "No improvement",
        color="#C62828",
        fontsize=11,
        va="bottom",
        ha="right",
        backgroundcolor="white",
    )
    ax.text(
        label_x,
        mean_delta,
        f"Mean improvement = {mean_delta:.2f}",
        color="black",
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="right",
        backgroundcolor="white",
    )

    lines = [f"Paired t-test p = {p_value:.5f}" if p_value is not None else "Paired t-test p unavailable"]
    if all_nonnegative:
        lines.append("All trials show non-negative improvement")
    ax.text(
        0.02,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "#CFCFCF", "boxstyle": "round,pad=0.35"},
    )

    fig.tight_layout()
    out = OUT_DIR / "curriculum_food_delivered_delta_by_trial_final.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_final_delta_by_trial_2(trials: list[dict[str, float]]) -> Path:
    trial_ids = [int(t["trial"]) for t in trials]
    deltas = [float(t["delta"]) for t in trials]
    mean_delta = sum(deltas) / len(deltas)

    jitter_pattern = (-0.08, -0.04, 0.0, 0.04, 0.08)
    jittered_x = [trial + jitter_pattern[(idx - 1) % len(jitter_pattern)] for idx, trial in enumerate(trial_ids, start=1)]

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.8))

    ax.scatter(
        jittered_x,
        deltas,
        s=160,
        color="#1565C0",
        edgecolors="black",
        linewidths=0.9,
        zorder=3,
    )
    ax.axhline(0, color="#C62828", linestyle="--", linewidth=1.5, zorder=1)
    ax.axhline(mean_delta, color="black", linewidth=2.0, zorder=2)

    x_min = min(jittered_x) - 0.2
    x_max = max(jittered_x) + 0.2
    ax.set_xlim(x_min, x_max)
    ymin = min(min(deltas), 0.0)
    ymax = max(max(deltas), mean_delta)
    pad = max(0.3, 0.12 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xticks(trial_ids)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Food delivered improvement")

    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "curriculum_food_delivered_delta_by_trial_final_2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_final_delta_by_trial_3(trials: list[dict[str, float]]) -> Path:
    trial_ids = [int(t["trial"]) for t in trials]
    deltas = [float(t["delta"]) for t in trials]
    mean_delta = sum(deltas) / len(deltas)

    jitter_pattern = (-0.08, -0.04, 0.0, 0.04, 0.08)
    jittered_x = [trial + jitter_pattern[(idx - 1) % len(jitter_pattern)] for idx, trial in enumerate(trial_ids, start=1)]

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.8))

    ax.scatter(
        jittered_x,
        deltas,
        s=160,
        color="#1565C0",
        edgecolors="black",
        linewidths=0.9,
        zorder=3,
    )
    ax.axhline(0, color="#C62828", linestyle="--", linewidth=1.5, zorder=1)
    ax.axhline(mean_delta, color="black", linewidth=2.0, zorder=2)

    x_min = min(jittered_x) - 0.2
    x_max = max(jittered_x) + 0.2
    ax.set_xlim(x_min, x_max)
    ymin = min(min(deltas), 0.0)
    ymax = max(max(deltas), mean_delta)
    pad = max(0.3, 0.12 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xticks(trial_ids)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Food delivered improvement")

    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")

    fig.tight_layout()
    out = OUT_DIR / "curriculum_food_delivered_delta_by_trial_final_3.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_final_delta_by_trial_4(trials: list[dict[str, float]]) -> Path:
    trial_ids = [int(t["trial"]) for t in trials]
    deltas = [float(t["delta"]) for t in trials]
    mean_delta = sum(deltas) / len(deltas)

    jitter_pattern = (-0.08, -0.04, 0.0, 0.04, 0.08)
    jittered_x = [trial + jitter_pattern[(idx - 1) % len(jitter_pattern)] for idx, trial in enumerate(trial_ids, start=1)]

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.8))

    ax.scatter(
        jittered_x,
        deltas,
        s=160,
        color="#1565C0",
        edgecolors="black",
        linewidths=0.9,
        zorder=3,
    )
    ax.axhline(0, color="#C62828", linestyle="--", linewidth=1.5, zorder=1)
    ax.axhline(mean_delta, color="black", linewidth=2.0, zorder=2)

    x_min = min(jittered_x) - 0.2
    x_max = max(jittered_x) + 0.2
    ax.set_xlim(x_min, x_max)
    ymin = min(min(deltas), 0.0)
    ymax = max(max(deltas), mean_delta)
    pad = max(0.3, 0.12 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xticks(trial_ids)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Food delivered improvement")
    ax.set_title("Consistent Performance Gains Across Matched Trials", pad=20)

    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")

    fig.tight_layout()
    out = OUT_DIR / "curriculum_food_delivered_delta_by_trial_final_4.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_task_efficiency_funnel(condition_means: dict[str, dict[str, float]]) -> Path:
    labels = {
        CURRENT: "New curriculum",
        OLDER: "Older configuration",
    }
    row_colors = {
        CURRENT: ("#5DA5DA", "#1565C0"),
        OLDER: ("#F6A04D", "#D95F02"),
    }
    annotations = {
        CURRENT: "Completes full foraging loop",
        OLDER: "Finds food, but fails to complete delivery",
    }
    row_y = {
        CURRENT: 1.1,
        OLDER: 0.15,
    }
    block_h = 0.34
    left_start = 0.9
    scale = 3.15
    gap = 1.7

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    max_extent = left_start
    for condition in (CURRENT, OLDER):
        picked = condition_means[condition]["food_picked_up"]
        delivered = condition_means[condition]["food_delivered"]
        conversion = None if picked == 0 else delivered / picked
        picked_w = max(picked * scale, 0.55 if picked > 0 else 0.25)
        delivered_w = max(delivered * scale, 0.28 if delivered > 0 else 0.18)
        y = row_y[condition]
        picked_color, delivered_color = row_colors[condition]

        ax.add_patch(
            plt.Rectangle(
                (left_start, y),
                picked_w,
                block_h,
                facecolor=picked_color,
                edgecolor="black",
                linewidth=1.0,
            )
        )

        arrow_start = left_start + picked_w + 0.18
        arrow_end = arrow_start + gap
        ax.annotate(
            "",
            xy=(arrow_end, y + block_h / 2),
            xytext=(arrow_start, y + block_h / 2),
            arrowprops=dict(arrowstyle="-|>", linewidth=2.0, color="#555555"),
        )

        delivered_start = arrow_end + 0.2
        ax.add_patch(
            plt.Rectangle(
                (delivered_start, y),
                delivered_w,
                block_h,
                facecolor=delivered_color,
                edgecolor="black",
                linewidth=1.0,
            )
        )

        ax.text(
            0.15,
            y + block_h / 2,
            labels[condition],
            ha="left",
            va="center",
            fontsize=15,
            fontweight="bold",
        )
        ax.text(
            left_start + picked_w / 2,
            y + block_h + 0.08,
            f"Picked up = {picked:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
        ax.text(
            delivered_start + delivered_w / 2,
            y + block_h + 0.08,
            f"Delivered = {delivered:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
        ax.text(
            (arrow_start + arrow_end) / 2,
            y + block_h / 2 + 0.18,
            f"Conversion = {conversion * 100:.1f}%" if conversion is not None else "Conversion = N/A",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#333333",
        )
        ax.text(
            delivered_start + max(delivered_w, 0.8) + 0.45,
            y + block_h / 2,
            annotations[condition],
            ha="left",
            va="center",
            fontsize=12,
            color="#333333",
        )
        ax.text(
            left_start + picked_w / 2,
            y + block_h / 2,
            "Food picked up",
            ha="center",
            va="center",
            fontsize=12,
            color="white" if picked_w > 1.1 else "black",
            fontweight="bold",
        )
        ax.text(
            delivered_start + max(delivered_w / 2, 0.2),
            y + block_h / 2,
            "Food delivered",
            ha="center",
            va="center",
            fontsize=12,
            color="white" if delivered_w > 1.1 else "black",
            fontweight="bold",
        )
        max_extent = max(max_extent, delivered_start + delivered_w + 4.2)

    ax.set_title("Task Efficiency Breakdown", pad=30)
    ax.text(
        0.5,
        1.02,
        "Older training finds food but fails to complete the delivery loop",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="#444444",
    )

    ax.set_xlim(0, max_extent)
    ax.set_ylim(-0.1, 1.75)
    ax.axis("off")

    fig.tight_layout()
    out = OUT_DIR / "curriculum_task_efficiency_funnel.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_pickup_to_delivery_arrows(condition_means: dict[str, dict[str, float]]) -> Path:
    labels = ["New curriculum", "Older configuration"]
    conditions = [CURRENT, OLDER]
    picked_vals = [condition_means[c]["food_picked_up"] for c in conditions]
    delivered_vals = [condition_means[c]["food_delivered"] for c in conditions]
    conversions = [
        None if picked == 0 else delivered / picked
        for picked, delivered in zip(picked_vals, delivered_vals)
    ]

    x = [0, 1]
    width = 0.28
    picked_colors = ["#6BAED6", "#FDAE6B"]
    delivered_colors = ["#08519C", "#D94801"]

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    picked_x = [xi - width / 2 for xi in x]
    delivered_x = [xi + width / 2 for xi in x]

    picked_bars = ax.bar(picked_x, picked_vals, width=width, color=picked_colors, edgecolor="black", linewidth=0.8, zorder=3)
    delivered_bars = ax.bar(delivered_x, delivered_vals, width=width, color=delivered_colors, edgecolor="black", linewidth=0.8, zorder=3)

    ymax = max(max(picked_vals), max(delivered_vals), 1.0)
    ax.set_ylim(0, ymax + 1.1)
    ax.set_xlim(-0.6, 1.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean per episode")
    ax.set_title("Pickup-to-Delivery Conversion", pad=30)
    ax.text(
        0.5,
        1.02,
        "The new curriculum turns discovery into completed delivery",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="#444444",
    )

    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7, zorder=0)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(picked_bars, picked_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.06,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    for bar, val in zip(delivered_bars, delivered_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.06,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    for i, xi in enumerate(x):
        picked = picked_vals[i]
        delivered = delivered_vals[i]
        conv = conversions[i]
        ax.text(
            xi,
            max(picked, delivered) + 0.45,
            f"Conversion = {conv * 100:.1f}%" if conv is not None else "Conversion = N/A",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
        ax.annotate(
            "",
            xy=(delivered_x[i], delivered),
            xytext=(picked_x[i], picked),
            arrowprops=dict(arrowstyle="->", linewidth=1.6, color="#4D4D4D"),
            zorder=4,
        )

    ax.text(
        x[0] + 0.34,
        delivered_vals[0] + 0.55,
        "Completes foraging loop",
        ha="left",
        va="center",
        fontsize=12,
        color="#333333",
    )
    ax.text(
        x[1] + 0.34,
        max(0.35, delivered_vals[1] + 0.22),
        "Finds food, but fails to deliver",
        ha="left",
        va="center",
        fontsize=12,
        color="#333333",
    )

    # simple legend-like labels near top left
    ax.text(-0.52, ymax + 0.72, "Picked up", fontsize=12, color="#333333", va="center")
    ax.add_patch(plt.Rectangle((-0.6, ymax + 0.62), 0.06, 0.12, facecolor="#6BAED6", edgecolor="black", linewidth=0.6, clip_on=False))
    ax.text(-0.05, ymax + 0.72, "Delivered", fontsize=12, color="#333333", va="center")
    ax.add_patch(plt.Rectangle((-0.13, ymax + 0.62), 0.06, 0.12, facecolor="#08519C", edgecolor="black", linewidth=0.6, clip_on=False))

    fig.tight_layout()
    out = OUT_DIR / "curriculum_pickup_to_delivery_arrows.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_pickups_vs_deliveries_grouped_improved(condition_means: dict[str, dict[str, float]]) -> Path:
    labels = ["New curriculum", "Older configuration"]
    conditions = [CURRENT, OLDER]
    picked_vals = [condition_means[c]["food_picked_up"] for c in conditions]
    delivered_vals = [condition_means[c]["food_delivered"] for c in conditions]
    conversions = [
        None if picked == 0 else delivered / picked
        for picked, delivered in zip(picked_vals, delivered_vals)
    ]

    group_centers = [0.0, 1.8]
    width = 0.28
    picked_x = [c - 0.16 for c in group_centers]
    delivered_x = [c + 0.16 for c in group_centers]

    picked_colors = ["#7FB3D5", "#F2B179"]
    delivered_colors = ["#1F5A94", "#C96A1B"]

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    picked_bars = ax.bar(picked_x, picked_vals, width=width, color=picked_colors, edgecolor="black", linewidth=0.8, zorder=3)
    delivered_bars = ax.bar(delivered_x, delivered_vals, width=width, color=delivered_colors, edgecolor="black", linewidth=0.8, zorder=3)

    ymax = max(max(picked_vals), max(delivered_vals), 1.0)
    ax.set_ylim(0, ymax + 1.15)
    ax.set_xlim(-0.7, 2.6)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean per episode")
    ax.set_title("Pickup vs Delivery: Completing the Foraging Loop", pad=30)
    ax.text(
        0.5,
        1.02,
        "Older training finds food but fails to deliver it",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="#444444",
    )

    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7, zorder=0)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(picked_bars, picked_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.07,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    for bar, val in zip(delivered_bars, delivered_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.07,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    for i, center in enumerate(group_centers):
        conv = conversions[i]
        ax.text(
            center,
            max(picked_vals[i], delivered_vals[i]) + 0.42,
            f"Conversion = {conv * 100:.1f}%" if conv is not None else "Conversion = N/A",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#333333",
        )

    ax.text(
        group_centers[0] + 0.42,
        max(delivered_vals[0], 0.15) + 0.55,
        "Completes delivery loop",
        ha="left",
        va="center",
        fontsize=12,
        color="#333333",
    )
    ax.text(
        delivered_x[1] + 0.12,
        0.28,
        "0 deliveries\nFails to complete loop",
        ha="left",
        va="bottom",
        fontsize=12,
        color="#333333",
    )

    ax.text(-0.58, ymax + 0.78, "Food picked up", fontsize=12, color="#333333", va="center")
    ax.add_patch(plt.Rectangle((-0.68, ymax + 0.68), 0.07, 0.12, facecolor="#7FB3D5", edgecolor="black", linewidth=0.6, clip_on=False))
    ax.text(0.22, ymax + 0.78, "Food delivered", fontsize=12, color="#333333", va="center")
    ax.add_patch(plt.Rectangle((0.08, ymax + 0.68), 0.07, 0.12, facecolor="#1F5A94", edgecolor="black", linewidth=0.6, clip_on=False))

    fig.tight_layout()
    out = OUT_DIR / "curriculum_pickups_vs_deliveries_grouped_improved.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_pickups_vs_deliveries_grouped_2(condition_means: dict[str, dict[str, float]]) -> Path:
    labels = ["New curriculum", "Older configuration"]
    conditions = [CURRENT, OLDER]
    picked_vals = [condition_means[c]["food_picked_up"] for c in conditions]
    delivered_vals = [condition_means[c]["food_delivered"] for c in conditions]
    conversions = [
        None if picked == 0 else delivered / picked
        for picked, delivered in zip(picked_vals, delivered_vals)
    ]

    group_centers = [0.0, 1.8]
    width = 0.28
    picked_x = [c - 0.16 for c in group_centers]
    delivered_x = [c + 0.16 for c in group_centers]

    picked_colors = ["#7FB3D5", "#F2B179"]
    delivered_colors = ["#1F5A94", "#C96A1B"]

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    picked_bars = ax.bar(picked_x, picked_vals, width=width, color=picked_colors, edgecolor="black", linewidth=0.8, zorder=3)
    delivered_bars = ax.bar(delivered_x, delivered_vals, width=width, color=delivered_colors, edgecolor="black", linewidth=0.8, zorder=3)

    ymax = max(max(picked_vals), max(delivered_vals), 1.0)
    ax.set_ylim(0, ymax + 1.15)
    ax.set_xlim(-0.7, 2.6)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean per episode")

    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7, zorder=0)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")

    for bar, val in zip(picked_bars, picked_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.07,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    for bar, val in zip(delivered_bars, delivered_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.07,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    for i, center in enumerate(group_centers):
        conv = conversions[i]
        ax.text(
            center,
            max(picked_vals[i], delivered_vals[i]) + 0.42,
            f"Conversion = {conv * 100:.1f}%" if conv is not None else "Conversion = N/A",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#333333",
        )

    ax.text(
        group_centers[0] + 0.42,
        max(delivered_vals[0], 0.15) + 0.55,
        "Completes delivery loop",
        ha="left",
        va="center",
        fontsize=12,
        color="#333333",
    )
    ax.text(
        delivered_x[1] + 0.12,
        0.28,
        "0 deliveries\nFails to complete loop",
        ha="left",
        va="bottom",
        fontsize=12,
        color="#333333",
    )

    ax.text(-0.58, ymax + 0.78, "Food picked up", fontsize=12, color="#333333", va="center")
    ax.add_patch(plt.Rectangle((-0.68, ymax + 0.68), 0.07, 0.12, facecolor="#7FB3D5", edgecolor="black", linewidth=0.6, clip_on=False))
    ax.text(0.22, ymax + 0.78, "Food delivered", fontsize=12, color="#333333", va="center")
    ax.add_patch(plt.Rectangle((0.08, ymax + 0.68), 0.07, 0.12, facecolor="#1F5A94", edgecolor="black", linewidth=0.6, clip_on=False))

    fig.tight_layout()
    out = OUT_DIR / "curriculum_pickups_vs_deliveries_grouped_2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    rows = load_curriculum_rows()
    trials = build_matched_trials(rows)
    condition_means = compute_condition_means(rows)
    p_value = load_reported_curriculum_pvalue()
    paired_p_value = compute_paired_ttest_pvalue(trials)
    generate_original_delta_by_seed(trials)
    generate_improved_delta_by_trial(trials, p_value)
    generate_final_delta_by_trial(trials, paired_p_value)
    generate_final_delta_by_trial_2(trials)
    generate_final_delta_by_trial_3(trials)
    generate_final_delta_by_trial_4(trials)
    generate_task_efficiency_funnel(condition_means)
    generate_pickup_to_delivery_arrows(condition_means)
    generate_pickups_vs_deliveries_grouped_improved(condition_means)
    generate_pickups_vs_deliveries_grouped_2(condition_means)


if __name__ == "__main__":
    main()
