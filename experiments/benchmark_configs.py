from __future__ import annotations


def get_experiment_cases(name: str):
    """Return named experiment sweeps using existing trainer/evaluator flags."""
    registry = {
        "swarm_scaling": [
            {"case_name": "agents_3", "train_args": ["--n-agents", "3"]},
            {"case_name": "agents_6", "train_args": ["--n-agents", "6"]},
            {"case_name": "agents_10", "train_args": ["--n-agents", "10"]},
        ],
        "stigmergy_ablation": [
            {"case_name": "pheromone_on", "train_args": []},
            {"case_name": "pheromone_off", "train_args": ["--pheromone-disabled"]},
        ],
        "baseline_comparison": [
            {"case_name": "independent_dqn", "train_args": []},
            {"case_name": "shared_dqn", "train_args": ["--shared-policy"]},
        ],
        "robot_failure_test": [
            {"case_name": "failed_0", "train_args": ["--failed-agent-count", "0"]},
            {"case_name": "failed_1", "train_args": ["--failed-agent-count", "1"]},
            {"case_name": "failed_2", "train_args": ["--failed-agent-count", "2"]},
        ],
        "noise_robustness": [
            {"case_name": "noise_0p00", "train_args": ["--observation-noise-std", "0.0"]},
            {"case_name": "noise_0p05", "train_args": ["--observation-noise-std", "0.05"]},
            {"case_name": "noise_0p10", "train_args": ["--observation-noise-std", "0.10"]},
        ],
    }
    if name == "all":
        return registry
    if name not in registry:
        raise ValueError(f"Unknown experiment '{name}'.")
    return {name: registry[name]}
