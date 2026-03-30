from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np


def _configure_matplotlib():
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cwsf2026_mplconfig")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cwsf2026_xdg_cache")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=str, required=True)
    parser.add_argument("--exploration-data-dir", type=str, required=True)
    parser.add_argument("--exploration-png-dir", type=str, required=True)
    parser.add_argument("--exploration-pdf-dir", type=str, required=True)
    parser.add_argument("--graph-png-dir", type=str, required=True)
    parser.add_argument("--graph-pdf-dir", type=str, required=True)
    parser.add_argument("--filename", type=str, required=True)
    return parser.parse_args(argv)


def _load_summary_rows(path: str) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save_comparison_plot(plt, summary_rows: list[dict], metric_key: str, ylabel: str, title: str, filename_root: str, out_png_dir: str, out_pdf_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row["comparison_label"])].append(row)
    for label, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row["number_of_agents"]))
        xs = [int(row["number_of_agents"]) for row in rows]
        ys = [float(row[f"mean_{metric_key}"]) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(out_png_dir, exist_ok=True)
    os.makedirs(out_pdf_dir, exist_ok=True)
    fig.savefig(os.path.join(out_png_dir, f"{filename_root}.png"))
    fig.savefig(os.path.join(out_pdf_dir, f"{filename_root}.pdf"))
    plt.close(fig)


def _plot_exploration_snapshot(plt, npz_path: str, png_dir: str, pdf_dir: str) -> None:
    payload = np.load(npz_path, allow_pickle=False)
    grid = payload["grid"].astype(np.float32)
    nest_position = payload["nest_position"].astype(np.float32)
    coverage_cell_size = float(payload["coverage_cell_size"][0])
    nest_enabled = bool(int(payload["nest_enabled"][0]))
    title = str(payload["title"][0])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.imshow(grid, cmap="magma", origin="lower", interpolation="nearest")
    if nest_enabled and grid.size > 0:
        ax.scatter(
            [nest_position[0] / max(coverage_cell_size, 1.0)],
            [nest_position[1] / max(coverage_cell_size, 1.0)],
            c="cyan",
            s=70,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            label="Nest",
        )
        ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_xlabel("Coverage Grid X")
    ax.set_ylabel("Coverage Grid Y")
    fig.tight_layout()

    stem = os.path.splitext(os.path.basename(npz_path))[0]
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    fig.savefig(os.path.join(png_dir, f"{stem}.png"))
    fig.savefig(os.path.join(pdf_dir, f"{stem}.pdf"))
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    plt = _configure_matplotlib()
    summary_rows = _load_summary_rows(args.summary_csv)

    if summary_rows:
        _save_comparison_plot(plt, summary_rows, "targets_collected", "Mean Targets Collected", "Agents vs Targets Collected in Time", f"{args.filename}_agents_vs_targets_collected", args.graph_png_dir, args.graph_pdf_dir)
        _save_comparison_plot(plt, summary_rows, "coverage_efficiency", "Mean Coverage Efficiency", "Coverage Efficiency vs Agents", f"{args.filename}_coverage_efficiency_vs_agents", args.graph_png_dir, args.graph_pdf_dir)
        _save_comparison_plot(plt, summary_rows, "efficiency", "Mean Efficiency", "Efficiency vs Agents", f"{args.filename}_efficiency_vs_agents", args.graph_png_dir, args.graph_pdf_dir)
        _save_comparison_plot(plt, summary_rows, "time_to_first_discovery", "Mean Time to First Discovery", "Time to First Discovery vs Agents", f"{args.filename}_time_to_first_discovery_vs_agents", args.graph_png_dir, args.graph_pdf_dir)
        _save_comparison_plot(plt, summary_rows, "stuck_event_count", "Mean Stuck Events", "Stuck Events vs Agents", f"{args.filename}_stuck_events_vs_agents", args.graph_png_dir, args.graph_pdf_dir)
        _save_comparison_plot(plt, summary_rows, "successful_escape_count", "Mean Successful Escapes", "Successful Escapes vs Agents", f"{args.filename}_successful_escapes_vs_agents", args.graph_png_dir, args.graph_pdf_dir)

    if os.path.isdir(args.exploration_data_dir):
        for name in sorted(os.listdir(args.exploration_data_dir)):
            if not name.endswith(".npz"):
                continue
            _plot_exploration_snapshot(
                plt,
                os.path.join(args.exploration_data_dir, name),
                args.exploration_png_dir,
                args.exploration_pdf_dir,
            )


if __name__ == "__main__":
    main()
