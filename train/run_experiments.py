from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.plot_metrics import plot_bar_with_error, plot_mean_std_curve
from experiments.benchmark_configs import get_experiment_cases
from train.evaluate import parse_args as parse_eval_args, run as run_eval
from train.experiment_utils import aggregate_rows, load_csv_rows
from train.independent_dqn_pytorch import parse_args as parse_train_args, train as run_train


METRIC_KEYS = [
    "mean_episode_reward",
    "food_retrieved",
    "exploration_coverage",
    "pheromone_usage",
    "episode_length",
    "swarm_efficiency",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=[
            "all",
            "swarm_scaling",
            "stigmergy_ablation",
            "baseline_comparison",
            "robot_failure_test",
            "noise_robustness",
        ],
        default="all",
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--analysis-dir", type=str, default="analysis")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def _write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _case_train_args(args, experiment_name: str, case_name: str, seed: int, extra_args: list[str]):
    save_dir = os.path.join(args.runs_dir, experiment_name, case_name, f"seed_{seed}", "checkpoints")
    output_dir = os.path.join(args.runs_dir, experiment_name, case_name)
    cli = [
        "--headless",
        "--total-steps", str(args.total_steps),
        "--seed", str(seed),
        "--save-dir", save_dir,
        "--save-every", str(args.save_every),
        "--output-dir", output_dir,
        "--experiment-name", f"seed_{seed}",
        "--eval-every", str(args.eval_every),
        "--eval-episodes", str(args.eval_episodes),
    ]
    if args.cuda:
        cli.append("--cuda")
    if args.no_plots:
        cli.append("--no-plots")
    cli.extend(extra_args)
    return cli, save_dir


def _case_eval_args(args, save_dir: str, seed: int, case_results_dir: str, extra_args: list[str]):
    cli = [
        "--checkpoint-dir", save_dir,
        "--seed", str(seed),
        "--output-dir", case_results_dir,
        "--episodes", str(args.eval_episodes),
    ]
    cli.extend(extra_args)
    return cli


def _aggregate_eval_curves(run_dirs: list[str], metric_key: str):
    curves = []
    for run_dir in run_dirs:
        path = os.path.join(run_dir, "eval_metrics.csv")
        if os.path.exists(path):
            curves.append(load_csv_rows(path))
    return curves


def run_experiments(args):
    registry = get_experiment_cases(args.experiment)
    for experiment_name, cases in registry.items():
        trial_rows = []
        aggregate_rows_out = []
        analysis_out_dir = os.path.join(args.analysis_dir, experiment_name)
        results_out_dir = os.path.join(args.results_dir, experiment_name)
        os.makedirs(analysis_out_dir, exist_ok=True)
        os.makedirs(results_out_dir, exist_ok=True)

        for case in cases:
            case_name = case["case_name"]
            run_dirs = []
            case_trial_rows = []

            for trial in range(args.trials):
                seed = args.seed + trial
                train_cli, save_dir = _case_train_args(args, experiment_name, case_name, seed, case["train_args"])
                train_args = parse_train_args(train_cli)
                run_dir = run_train(train_args)
                run_dirs.append(run_dir)

                case_results_dir = os.path.join(results_out_dir, case_name, f"seed_{seed}")
                eval_cli = _case_eval_args(args, save_dir, seed, case_results_dir, case["train_args"])
                eval_args = parse_eval_args(eval_cli)
                run_eval(eval_args)

                summary_path = os.path.join(case_results_dir, "eval_summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary_payload = json.load(f)
                    metric_block = summary_payload.get("metrics", {})
                    final_row = {
                        "mean_episode_reward": metric_block.get("mean_reward", 0.0),
                        "food_retrieved": metric_block.get("mean_food_retrieved", 0.0),
                        "exploration_coverage": metric_block.get("mean_exploration_coverage", 0.0),
                        "pheromone_usage": metric_block.get("mean_pheromone_usage", 0.0),
                        "episode_length": metric_block.get("mean_episode_length", 0.0),
                        "swarm_efficiency": metric_block.get("mean_food_retrieved", 0.0)
                        / max(metric_block.get("mean_episode_length", 1.0), 1.0),
                    }
                else:
                    final_row = {key: 0.0 for key in METRIC_KEYS}
                final_row["experiment"] = experiment_name
                final_row["case_name"] = case_name
                final_row["seed"] = seed
                trial_rows.append(final_row)
                case_trial_rows.append(final_row)

            numeric_case_rows = [
                {key: float(row[key]) for key in METRIC_KEYS}
                for row in case_trial_rows
            ]
            summary = aggregate_rows(numeric_case_rows, METRIC_KEYS)
            summary["experiment"] = experiment_name
            summary["case_name"] = case_name
            aggregate_rows_out.append(summary)

            if not args.no_plots:
                eval_curves = _aggregate_eval_curves(run_dirs, "mean_episode_reward")
                if eval_curves:
                    plot_mean_std_curve(
                        eval_curves,
                        x_key="global_step",
                        y_key="mean_episode_reward",
                        out_path=os.path.join(analysis_out_dir, f"{case_name}_reward_curve.png"),
                        xlabel="Training Step",
                        ylabel="Mean Eval Reward",
                        title=f"{experiment_name}: {case_name}",
                        label=case_name,
                    )

        _write_csv(
            os.path.join(results_out_dir, "trial_metrics.csv"),
            ["experiment", "case_name", "seed"] + METRIC_KEYS,
            trial_rows,
        )
        aggregate_fieldnames = ["experiment", "case_name"]
        for key in METRIC_KEYS:
            aggregate_fieldnames.extend([f"{key}_mean", f"{key}_std"])
        _write_csv(
            os.path.join(results_out_dir, "aggregate_metrics.csv"),
            aggregate_fieldnames,
            aggregate_rows_out,
        )

        if not args.no_plots:
            for metric in ["food_retrieved", "exploration_coverage", "swarm_efficiency", "mean_episode_reward"]:
                plot_bar_with_error(
                    aggregate_rows_out,
                    label_key="case_name",
                    mean_key=f"{metric}_mean",
                    std_key=f"{metric}_std",
                    out_path=os.path.join(analysis_out_dir, f"{metric}_summary.png"),
                    ylabel=metric.replace("_", " ").title(),
                    title=f"{experiment_name}: {metric.replace('_', ' ').title()}",
                )


if __name__ == "__main__":
    run_experiments(parse_args())
