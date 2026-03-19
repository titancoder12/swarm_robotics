from __future__ import annotations


def get_experiment_cases(name: str):
    """Return named experiment sweeps using existing trainer/evaluator flags."""
    registry = {
        "swarm_scaling": {
            "default_trials": 3,
            "cases": [
                {"case_name": "agents_3", "train_args": ["--n-agents", "3"], "n_agents": 3},
                {"case_name": "agents_6", "train_args": ["--n-agents", "6"], "n_agents": 6},
                {"case_name": "agents_10", "train_args": ["--n-agents", "10"], "n_agents": 10},
            ],
        },
        "stigmergy_ablation": {
            "default_trials": 3,
            "cases": [
                {"case_name": "pheromone_on", "train_args": [], "condition": "pheromone_on"},
                {"case_name": "pheromone_off", "train_args": ["--pheromone-disabled"], "condition": "pheromone_off"},
            ],
        },
        "baseline_comparison": {
            "default_trials": 3,
            "cases": [
                {"case_name": "independent_dqn", "train_args": [], "condition": "independent_dqn"},
                {"case_name": "shared_dqn", "train_args": ["--shared-policy"], "condition": "shared_dqn"},
            ],
        },
        "robot_failure_test": {
            "default_trials": 3,
            "cases": [
                {"case_name": "failed_0", "train_args": ["--failed-agent-count", "0"], "failed_agents": 0},
                {"case_name": "failed_1", "train_args": ["--failed-agent-count", "1"], "failed_agents": 1},
                {"case_name": "failed_2", "train_args": ["--failed-agent-count", "2"], "failed_agents": 2},
            ],
        },
        "noise_robustness": {
            "default_trials": 3,
            "cases": [
                {"case_name": "noise_0p00", "train_args": ["--observation-noise-std", "0.0"], "noise_std": 0.0},
                {"case_name": "noise_0p05", "train_args": ["--observation-noise-std", "0.05"], "noise_std": 0.05},
                {"case_name": "noise_0p10", "train_args": ["--observation-noise-std", "0.10"], "noise_std": 0.10},
            ],
        },
        "collective_intelligence_scaling": {
            "default_trials": 20,
            "cases": [
                {"case_name": "agents_1_pheromone_on", "train_args": ["--n-agents", "1"], "n_agents": 1, "condition": "pheromone_on"},
                {"case_name": "agents_2_pheromone_on", "train_args": ["--n-agents", "2"], "n_agents": 2, "condition": "pheromone_on"},
                {"case_name": "agents_3_pheromone_on", "train_args": ["--n-agents", "3"], "n_agents": 3, "condition": "pheromone_on"},
                {"case_name": "agents_5_pheromone_on", "train_args": ["--n-agents", "5"], "n_agents": 5, "condition": "pheromone_on"},
                {"case_name": "agents_10_pheromone_on", "train_args": ["--n-agents", "10"], "n_agents": 10, "condition": "pheromone_on"},
                {"case_name": "agents_1_pheromone_off", "train_args": ["--n-agents", "1", "--pheromone-disabled"], "n_agents": 1, "condition": "pheromone_off"},
                {"case_name": "agents_2_pheromone_off", "train_args": ["--n-agents", "2", "--pheromone-disabled"], "n_agents": 2, "condition": "pheromone_off"},
                {"case_name": "agents_3_pheromone_off", "train_args": ["--n-agents", "3", "--pheromone-disabled"], "n_agents": 3, "condition": "pheromone_off"},
                {"case_name": "agents_5_pheromone_off", "train_args": ["--n-agents", "5", "--pheromone-disabled"], "n_agents": 5, "condition": "pheromone_off"},
                {"case_name": "agents_10_pheromone_off", "train_args": ["--n-agents", "10", "--pheromone-disabled"], "n_agents": 10, "condition": "pheromone_off"},
            ],
        },
        "rl_algorithm_comparison": {
            "default_trials": 20,
            "cases": [
                {"case_name": "dqn_agents_1_pheromone_on", "train_args": ["--n-agents", "1"], "eval_args": ["--n-agents", "1"], "algorithm": "dqn", "n_agents": 1, "condition": "pheromone_on"},
                {"case_name": "dqn_agents_3_pheromone_on", "train_args": ["--n-agents", "3"], "eval_args": ["--n-agents", "3"], "algorithm": "dqn", "n_agents": 3, "condition": "pheromone_on"},
                {"case_name": "dqn_agents_5_pheromone_on", "train_args": ["--n-agents", "5"], "eval_args": ["--n-agents", "5"], "algorithm": "dqn", "n_agents": 5, "condition": "pheromone_on"},
                {"case_name": "dqn_agents_10_pheromone_on", "train_args": ["--n-agents", "10"], "eval_args": ["--n-agents", "10"], "algorithm": "dqn", "n_agents": 10, "condition": "pheromone_on"},
                {"case_name": "dqn_agents_1_pheromone_off", "train_args": ["--n-agents", "1", "--pheromone-disabled"], "eval_args": ["--n-agents", "1", "--pheromone-disabled"], "algorithm": "dqn", "n_agents": 1, "condition": "pheromone_off"},
                {"case_name": "dqn_agents_3_pheromone_off", "train_args": ["--n-agents", "3", "--pheromone-disabled"], "eval_args": ["--n-agents", "3", "--pheromone-disabled"], "algorithm": "dqn", "n_agents": 3, "condition": "pheromone_off"},
                {"case_name": "dqn_agents_5_pheromone_off", "train_args": ["--n-agents", "5", "--pheromone-disabled"], "eval_args": ["--n-agents", "5", "--pheromone-disabled"], "algorithm": "dqn", "n_agents": 5, "condition": "pheromone_off"},
                {"case_name": "dqn_agents_10_pheromone_off", "train_args": ["--n-agents", "10", "--pheromone-disabled"], "eval_args": ["--n-agents", "10", "--pheromone-disabled"], "algorithm": "dqn", "n_agents": 10, "condition": "pheromone_off"},
                {"case_name": "shared_dqn_agents_1_pheromone_on", "train_args": ["--n-agents", "1", "--shared-policy"], "eval_args": ["--n-agents", "1", "--shared-policy"], "algorithm": "shared_dqn", "n_agents": 1, "condition": "pheromone_on"},
                {"case_name": "shared_dqn_agents_3_pheromone_on", "train_args": ["--n-agents", "3", "--shared-policy"], "eval_args": ["--n-agents", "3", "--shared-policy"], "algorithm": "shared_dqn", "n_agents": 3, "condition": "pheromone_on"},
                {"case_name": "shared_dqn_agents_5_pheromone_on", "train_args": ["--n-agents", "5", "--shared-policy"], "eval_args": ["--n-agents", "5", "--shared-policy"], "algorithm": "shared_dqn", "n_agents": 5, "condition": "pheromone_on"},
                {"case_name": "shared_dqn_agents_10_pheromone_on", "train_args": ["--n-agents", "10", "--shared-policy"], "eval_args": ["--n-agents", "10", "--shared-policy"], "algorithm": "shared_dqn", "n_agents": 10, "condition": "pheromone_on"},
                {"case_name": "shared_dqn_agents_1_pheromone_off", "train_args": ["--n-agents", "1", "--shared-policy", "--pheromone-disabled"], "eval_args": ["--n-agents", "1", "--shared-policy", "--pheromone-disabled"], "algorithm": "shared_dqn", "n_agents": 1, "condition": "pheromone_off"},
                {"case_name": "shared_dqn_agents_3_pheromone_off", "train_args": ["--n-agents", "3", "--shared-policy", "--pheromone-disabled"], "eval_args": ["--n-agents", "3", "--shared-policy", "--pheromone-disabled"], "algorithm": "shared_dqn", "n_agents": 3, "condition": "pheromone_off"},
                {"case_name": "shared_dqn_agents_5_pheromone_off", "train_args": ["--n-agents", "5", "--shared-policy", "--pheromone-disabled"], "eval_args": ["--n-agents", "5", "--shared-policy", "--pheromone-disabled"], "algorithm": "shared_dqn", "n_agents": 5, "condition": "pheromone_off"},
                {"case_name": "shared_dqn_agents_10_pheromone_off", "train_args": ["--n-agents", "10", "--shared-policy", "--pheromone-disabled"], "eval_args": ["--n-agents", "10", "--shared-policy", "--pheromone-disabled"], "algorithm": "shared_dqn", "n_agents": 10, "condition": "pheromone_off"},
                {"case_name": "rule_based_agents_1_pheromone_on", "train_args": [], "eval_args": ["--n-agents", "1", "--policy-kind", "rule_based"], "algorithm": "rule_based", "n_agents": 1, "condition": "pheromone_on", "skip_training": True},
                {"case_name": "rule_based_agents_3_pheromone_on", "train_args": [], "eval_args": ["--n-agents", "3", "--policy-kind", "rule_based"], "algorithm": "rule_based", "n_agents": 3, "condition": "pheromone_on", "skip_training": True},
                {"case_name": "rule_based_agents_5_pheromone_on", "train_args": [], "eval_args": ["--n-agents", "5", "--policy-kind", "rule_based"], "algorithm": "rule_based", "n_agents": 5, "condition": "pheromone_on", "skip_training": True},
                {"case_name": "rule_based_agents_10_pheromone_on", "train_args": [], "eval_args": ["--n-agents", "10", "--policy-kind", "rule_based"], "algorithm": "rule_based", "n_agents": 10, "condition": "pheromone_on", "skip_training": True},
                {"case_name": "rule_based_agents_1_pheromone_off", "train_args": [], "eval_args": ["--n-agents", "1", "--policy-kind", "rule_based", "--pheromone-disabled"], "algorithm": "rule_based", "n_agents": 1, "condition": "pheromone_off", "skip_training": True},
                {"case_name": "rule_based_agents_3_pheromone_off", "train_args": [], "eval_args": ["--n-agents", "3", "--policy-kind", "rule_based", "--pheromone-disabled"], "algorithm": "rule_based", "n_agents": 3, "condition": "pheromone_off", "skip_training": True},
                {"case_name": "rule_based_agents_5_pheromone_off", "train_args": [], "eval_args": ["--n-agents", "5", "--policy-kind", "rule_based", "--pheromone-disabled"], "algorithm": "rule_based", "n_agents": 5, "condition": "pheromone_off", "skip_training": True},
                {"case_name": "rule_based_agents_10_pheromone_off", "train_args": [], "eval_args": ["--n-agents", "10", "--policy-kind", "rule_based", "--pheromone-disabled"], "algorithm": "rule_based", "n_agents": 10, "condition": "pheromone_off", "skip_training": True},
            ],
        },
    }
    if name == "all":
        return registry
    if name not in registry:
        raise ValueError(f"Unknown experiment '{name}'.")
    return {name: registry[name]}
