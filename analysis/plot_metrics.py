from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.experiment_utils import plot_training_metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_dir or os.path.dirname(args.csv_path)
    os.makedirs(out_dir, exist_ok=True)
    plot_training_metrics(args.csv_path, out_dir)


def plot_mean_std_curve(curves, x_key: str, y_key: str, out_path: str, xlabel: str, ylabel: str, title: str, label: str):
    """Plot a mean curve with +/- one standard deviation shading."""
    import matplotlib.pyplot as plt

    if not curves:
        return
    shared_x = sorted(set.intersection(*[set(int(row[x_key]) for row in curve) for curve in curves]))
    if not shared_x:
        return
    ys = []
    for curve in curves:
        lookup = {int(row[x_key]): float(row[y_key]) for row in curve}
        ys.append([lookup[x] for x in shared_x])
    arr = np.array(ys, dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(shared_x, mean, label=label, linewidth=2)
    ax.fill_between(shared_x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_bar_with_error(rows, label_key: str, mean_key: str, std_key: str, out_path: str, ylabel: str, title: str):
    """Plot bar charts with error bars for aggregated experiment metrics."""
    import matplotlib.pyplot as plt

    if not rows:
        return
    labels = [row[label_key] for row in rows]
    means = [float(row[mean_key]) for row in rows]
    stds = [float(row[std_key]) for row in rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
