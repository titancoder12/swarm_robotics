from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from analysis.plot_metrics import (
    plot_bar_with_error,
    plot_grouped_errorbar,
    plot_mean_std_curve,
)
from experiments.benchmark_configs import get_experiment_cases
from train.evaluate import parse_args as parse_eval_args, run as run_eval
from train.experiment_utils import aggregate_rows, load_csv_rows, write_json
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
            "collective_intelligence_scaling",
            "rl_algorithm_comparison",
        ],
        default="all",
    )
    parser.add_argument("--trials", type=int, default=None)
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


def _case_eval_args(args, save_dir: str, seed: int, case_results_dir: str, extra_args: list[str], skip_training: bool = False):
    cli = [
        "--seed", str(seed),
        "--output-dir", case_results_dir,
        "--episodes", str(args.eval_episodes),
    ]
    if not skip_training:
        cli.extend(["--checkpoint-dir", save_dir])
    cli.extend(extra_args)
    return cli


def _aggregate_eval_curves(run_dirs: list[str], metric_key: str):
    curves = []
    for run_dir in run_dirs:
        path = os.path.join(run_dir, "eval_metrics.csv")
        if os.path.exists(path):
            curves.append(load_csv_rows(path))
    return curves


def _metadata_fields(case: dict) -> dict:
    fields = {}
    for key in ("n_agents", "condition", "failed_agents", "noise_std", "algorithm"):
        if key in case:
            fields[key] = case[key]
    return fields


def _efficiency_per_robot(food_retrieved: float, n_agents: float | int | None) -> float:
    if not n_agents:
        return 0.0
    return float(food_retrieved) / float(n_agents)


def _convergence_speed(run_dir: str) -> float:
    path = os.path.join(run_dir, "episode_metrics.csv")
    if not os.path.exists(path):
        return 0.0
    rows = load_csv_rows(path)
    if not rows:
        return 0.0
    rewards = np.array([float(row["mean_episode_reward"]) for row in rows], dtype=np.float32)
    if rewards.size == 0:
        return 0.0
    if rewards.size < 5:
        return float(len(rows))
    window = min(5, rewards.size)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(rewards, kernel, mode="valid")
    target = float(np.mean(smoothed[-window:]))
    tol = max(1.0, abs(target) * 0.05)
    for i in range(smoothed.size):
        if np.all(np.abs(smoothed[i:] - target) <= tol):
            return float(i + window)
    return float(len(rows))


def _time_to_first_food(eval_dir: str, fallback_episodes: int) -> float:
    path = os.path.join(eval_dir, "eval_metrics.csv")
    if not os.path.exists(path):
        return float(fallback_episodes + 1)
    rows = load_csv_rows(path)
    for row in rows:
        if float(row.get("food_retrieved", 0.0)) > 0.0:
            return float(row.get("episode", 0))
    return float(len(rows) + 1 if rows else fallback_episodes + 1)


def _skip_train_case(args, experiment_name: str, case_name: str, seed: int, case_metadata: dict):
    run_dir = os.path.join(args.runs_dir, experiment_name, case_name, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    write_json(
        os.path.join(run_dir, "run_config.json"),
        {
            "experiment": experiment_name,
            "case_name": case_name,
            "seed": seed,
            "algorithm": case_metadata.get("algorithm", "rule_based"),
            "policy_kind": "rule_based",
            "obs_dim": 23,
            "trained": False,
        },
    )
    return run_dir


def _series_label(row: dict) -> str:
    algorithm = str(row.get("algorithm", "unknown")).replace("_", " ")
    condition = str(row.get("condition", "")).replace("_", " ").strip()
    if condition:
        return f"{algorithm} | {condition}"
    return algorithm


def _algorithm_bar_rows(rows: list[dict], metric_key: str) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row.get("algorithm", "unknown")), []).append(float(row[metric_key]))
    out = []
    for algorithm, values in sorted(grouped.items()):
        arr = np.array(values, dtype=np.float32)
        out.append(
            {
                "label": algorithm,
                "mean": float(arr.mean()) if arr.size else 0.0,
                "std": float(arr.std()) if arr.size else 0.0,
            }
        )
    return out


def run_experiments(args):
    registry = get_experiment_cases(args.experiment)
    for experiment_name, experiment_spec in registry.items():
        cases = experiment_spec["cases"]
        trial_count = args.trials if args.trials is not None else int(experiment_spec.get("default_trials", 3))
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
            case_metadata = _metadata_fields(case)
            eval_case_args = case.get("eval_args", case["train_args"])
            skip_training = bool(case.get("skip_training", False))

            for trial in range(trial_count):
                seed = args.seed + trial
                train_cli, save_dir = _case_train_args(args, experiment_name, case_name, seed, case["train_args"])
                if skip_training:
                    run_dir = _skip_train_case(args, experiment_name, case_name, seed, case_metadata)
                else:
                    train_args = parse_train_args(train_cli)
                    run_dir = run_train(train_args)
                run_dirs.append(run_dir)

                case_results_dir = os.path.join(results_out_dir, case_name, f"seed_{seed}")
                eval_cli = _case_eval_args(args, save_dir, seed, case_results_dir, eval_case_args, skip_training=skip_training)
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
                final_row.update(case_metadata)
                final_row["efficiency_per_robot"] = _efficiency_per_robot(
                    final_row["food_retrieved"], case_metadata.get("n_agents")
                )
                final_row["convergence_speed"] = _convergence_speed(run_dir)
                final_row["time_to_first_food"] = _time_to_first_food(case_results_dir, args.eval_episodes)
                final_row["series_label"] = _series_label(final_row)
                trial_rows.append(final_row)
                case_trial_rows.append(final_row)

            numeric_case_rows = [
                {
                    key: float(row[key])
                    for key in METRIC_KEYS + ["efficiency_per_robot", "convergence_speed", "time_to_first_food"]
                }
                for row in case_trial_rows
            ]
            summary = aggregate_rows(
                numeric_case_rows,
                METRIC_KEYS + ["efficiency_per_robot", "convergence_speed", "time_to_first_food"],
            )
            summary["experiment"] = experiment_name
            summary["case_name"] = case_name
            summary.update(case_metadata)
            summary["series_label"] = _series_label(summary)
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
            ["experiment", "case_name", "seed", "algorithm", "n_agents", "condition", "failed_agents", "noise_std", "series_label"]
            + METRIC_KEYS
            + ["efficiency_per_robot", "convergence_speed", "time_to_first_food"],
            trial_rows,
        )
        aggregate_fieldnames = [
            "experiment",
            "case_name",
            "algorithm",
            "n_agents",
            "condition",
            "failed_agents",
            "noise_std",
            "series_label",
        ]
        for key in METRIC_KEYS + ["efficiency_per_robot", "convergence_speed", "time_to_first_food"]:
            aggregate_fieldnames.extend([f"{key}_mean", f"{key}_std"])
        _write_csv(
            os.path.join(results_out_dir, "aggregate_metrics.csv"),
            aggregate_fieldnames,
            aggregate_rows_out,
        )

        if not args.no_plots:
            if experiment_name == "collective_intelligence_scaling":
                plot_grouped_errorbar(
                    aggregate_rows_out,
                    x_key="n_agents",
                    mean_key="episode_length_mean",
                    std_key="episode_length_std",
                    group_key="condition",
                    out_path=os.path.join(analysis_out_dir, "completion_time_vs_agents.png"),
                    xlabel="Number of Agents",
                    ylabel="Episode Length",
                    title="Collective Intelligence Scaling: Completion Time vs Agents",
                )
                plot_grouped_errorbar(
                    aggregate_rows_out,
                    x_key="n_agents",
                    mean_key="swarm_efficiency_mean",
                    std_key="swarm_efficiency_std",
                    group_key="condition",
                    out_path=os.path.join(analysis_out_dir, "efficiency_vs_agents.png"),
                    xlabel="Number of Agents",
                    ylabel="Swarm Efficiency",
                    title="Collective Intelligence Scaling: Efficiency vs Agents",
                )
                plot_grouped_errorbar(
                    aggregate_rows_out,
                    x_key="n_agents",
                    mean_key="pheromone_usage_mean",
                    std_key="pheromone_usage_std",
                    group_key="condition",
                    out_path=os.path.join(analysis_out_dir, "pheromone_usage_vs_agents.png"),
                    xlabel="Number of Agents",
                    ylabel="Pheromone Usage",
                    title="Collective Intelligence Scaling: Pheromone Usage vs Agents",
                )
                plot_grouped_errorbar(
                    aggregate_rows_out,
                    x_key="n_agents",
                    mean_key="efficiency_per_robot_mean",
                    std_key="efficiency_per_robot_std",
                    group_key="condition",
                    out_path=os.path.join(analysis_out_dir, "efficiency_per_robot_vs_agents.png"),
                    xlabel="Number of Agents",
                    ylabel="Efficiency Per Robot",
                    title="Collective Intelligence Scaling: Efficiency Per Robot vs Agents",
                )
            elif experiment_name == "rl_algorithm_comparison":
                plot_grouped_errorbar(
                    aggregate_rows_out,
                    x_key="n_agents",
                    mean_key="food_retrieved_mean",
                    std_key="food_retrieved_std",
                    group_key="series_label",
                    out_path=os.path.join(analysis_out_dir, "algorithm_food_retrieval.png"),
                    xlabel="Number of Agents",
                    ylabel="Food Retrieved",
                    title="RL Algorithm Comparison: Food Retrieval vs Agents",
                )
                plot_grouped_errorbar(
                    aggregate_rows_out,
                    x_key="n_agents",
                    mean_key="swarm_efficiency_mean",
                    std_key="swarm_efficiency_std",
                    group_key="series_label",
                    out_path=os.path.join(analysis_out_dir, "algorithm_efficiency.png"),
                    xlabel="Number of Agents",
                    ylabel="Swarm Efficiency",
                    title="RL Algorithm Comparison: Efficiency vs Agents",
                )
                plot_grouped_errorbar(
                    aggregate_rows_out,
                    x_key="n_agents",
                    mean_key="convergence_speed_mean",
                    std_key="convergence_speed_std",
                    group_key="series_label",
                    out_path=os.path.join(analysis_out_dir, "algorithm_convergence_speed.png"),
                    xlabel="Number of Agents",
                    ylabel="Convergence Speed (Episodes)",
                    title="RL Algorithm Comparison: Convergence Speed vs Agents",
                )
                plot_bar_with_error(
                    _algorithm_bar_rows(aggregate_rows_out, "swarm_efficiency_mean"),
                    label_key="label",
                    mean_key="mean",
                    std_key="std",
                    out_path=os.path.join(analysis_out_dir, "algorithm_comparison_bar.png"),
                    ylabel="Swarm Efficiency",
                    title="RL Algorithm Comparison: Overall Swarm Efficiency",
                )
            else:
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
