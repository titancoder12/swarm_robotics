from __future__ import annotations

from pathlib import Path
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def main() -> None:
    rows = load_curriculum_rows()
    trials = build_matched_trials(rows)
    p_value = load_reported_curriculum_pvalue()
    generate_original_delta_by_seed(trials)
    generate_improved_delta_by_trial(trials, p_value)


if __name__ == "__main__":
    main()
