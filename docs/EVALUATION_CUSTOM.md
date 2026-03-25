# Custom Evaluation Walkthrough

This document explains how evaluation works in [train/evaluate.py](../train/evaluate.py).

It is implementation-grounded. The goal is to help a developer understand exactly what the evaluation script does, how it differs from training, how policies are loaded, and what files it writes.

## What This File Does

[train/evaluate.py](../train/evaluate.py) runs headless evaluation episodes and records metrics.

It does not train.

It supports two policy types:

- custom DQN checkpoints
- rule-based policies

It reuses the same environment as training, but runs policies deterministically:

- DQN uses greedy `argmax` action selection
- rule-based uses `.act(obs)`

The script writes:

- `eval_metrics.csv`
- `eval_summary.json`

## High-Level Evaluation Flow

The file follows this sequence:

1. parse CLI arguments
2. build a `SwarmConfig`
3. create a headless `SwarmEnv`
4. reset once to infer `obs_dim`
5. load DQN models or build rule-based policies
6. run `N` evaluation episodes
7. log one metric row per episode
8. write an aggregated summary JSON

## File Structure

The main parts of the file are:

- `parse_args(...)`
- `_load_models(...)`
- `_build_rule_based_policies(...)`
- `run(args)`

## CLI Arguments

`parse_args(...)` is defined at [train/evaluate.py#L21](../train/evaluate.py#L21).

Main arguments:

- `--checkpoint-dir`
- `--shared-policy`
- `--policy-kind`
- `--episodes`
- `--n-agents`
- `--seed`
- `--output-dir`

It also imports environment-related flags through `add_env_config_args(parser)`, so evaluation can be run under different environment settings such as:

- number of targets
- number of obstacles
- pheromone enabled or disabled
- failed-agent count
- observation noise

## Policy Loading

### DQN checkpoints

`_load_models(...)` is defined at [train/evaluate.py#L34](../train/evaluate.py#L34).

It always evaluates on CPU:

```python
device = torch.device("cpu")
```

There are two modes:

#### Shared policy

If `--shared-policy` is passed:

- instantiate one `QNetwork(obs_dim, action_dim)`
- load `shared.pt`
- reuse that same network for all agents

#### Independent policies

If `--shared-policy` is not passed:

- instantiate one `QNetwork(obs_dim, action_dim)` per agent
- load `agent_0.pt`, `agent_1.pt`, and so on

After loading, all networks are set to evaluation mode with:

```python
net.eval()
```

### Rule-based policies

`_build_rule_based_policies(...)` is defined at [train/evaluate.py#L51](../train/evaluate.py#L51).

It constructs one [RuleBasedSwarmPolicy](../models/rule_based_policy.py) per agent:

```python
[RuleBasedSwarmPolicy(cfg, seed=seed + i) for i in range(cfg.n_agents)]
```

This gives each agent its own rule-based policy instance with a deterministic seed offset.

## Main Evaluation Flow

The main function is `run(args)` at [train/evaluate.py#L55](../train/evaluate.py#L55).

### 1. Create output directory

At the start:

```python
os.makedirs(args.output_dir, exist_ok=True)
```

This ensures the evaluation output folder exists.

### 2. Build config and environment

Next:

```python
cfg = make_swarm_config(args)
env = SwarmEnv(cfg, headless=True)
```

Evaluation is always headless.

That means:

- no PyGame window
- no interactive rendering
- faster evaluation

### 3. Reset once to infer observation size

Then:

```python
obs_dict, _ = env.reset(seed=args.seed)
agent_ids = env.possible_agents
obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0)
obs_dim = obs.shape[1]
```

This is important.

The evaluator does not hardcode observation size. It reads the real observation dimension from the environment at runtime.

That makes it consistent with the current environment contract.

### 4. Choose policy type

At [train/evaluate.py#L63](../train/evaluate.py#L63):

- if `args.policy_kind == "dqn"`:
  - load DQN checkpoints
- else:
  - build rule-based policies

So the evaluator is the shared measurement path for both learned and non-learned baselines.

## Per-Episode Logging Setup

At [train/evaluate.py#L71](../train/evaluate.py#L71), the script creates a CSV logger for:

- `episode`
- `seed`
- `mean_episode_reward`
- `food_retrieved`
- `exploration_coverage`
- `pheromone_usage`
- `episode_length`
- `swarm_efficiency`

This logger writes to:

- `eval_metrics.csv`

It also keeps a `summaries` list in memory so it can compute overall means at the end.

## Episode Loop

The main evaluation loop begins at [train/evaluate.py#L87](../train/evaluate.py#L87):

```python
for episode in range(1, args.episodes + 1):
```

For each episode:

- the environment is reset with a deterministic seed
- per-episode metrics are zeroed

The reset seed is:

```python
args.seed + episode - 1
```

This means different episodes use different seeds, but the sequence is deterministic if the base seed is fixed.

## Action Selection

Inside the episode, the main step loop begins at [train/evaluate.py#L96](../train/evaluate.py#L96):

```python
while True:
```

An action is chosen for each agent.

### DQN case

For DQN:

```python
with torch.no_grad():
    obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
    q_vals = nets[i](obs_tensor)
    actions[i] = int(torch.argmax(q_vals, dim=1).item())
```

This is greedy action selection:

- no epsilon-greedy randomness
- no learning
- just choose the action with the largest Q-value

### Rule-based case

For rule-based:

```python
actions[i] = int(policies[i].act(obs[i]))
```

This uses the same observation vector and the same discrete action space, but produces actions from hand-coded logic instead of a learned network.

## Environment Step

At [train/evaluate.py#L107](../train/evaluate.py#L107):

```python
action_dict = {agent: int(actions[i]) for i, agent in enumerate(agent_ids)}
next_obs_dict, rewards_dict, terminations, truncations, info_dict = env.step(action_dict)
```

Then observations and rewards are converted back into arrays:

```python
obs = np.stack([next_obs_dict[agent] for agent in agent_ids], axis=0)
rewards = np.array([rewards_dict[agent] for agent in agent_ids], dtype=np.float32)
info = info_dict[agent_ids[0]]
```

Because this environment returns the same episode-level info for all agents, the evaluator reads the info from the first agent.

## Metric Accumulation

During the episode, the evaluator updates:

- `episode_rewards += rewards`
- `food_retrieved += int(info.get("food_delivered", 0))`
- `exploration_coverage = max(...)`
- `pheromone_usage_values.append(...)`
- `episode_length = ...`

Interpretation:

- `episode_rewards` stores per-agent accumulated reward
- `food_retrieved` is counted from `food_delivered`
- `exploration_coverage` tracks the maximum coverage reached during the episode
- `pheromone_usage_values` stores the step-by-step pheromone usage so the mean can be computed later
- `episode_length` is taken from env info

## Episode Termination

At [train/evaluate.py#L119](../train/evaluate.py#L119):

```python
if any(terminations.values()) or any(truncations.values()):
    break
```

So an evaluation episode ends when the environment either:

- terminates normally
- truncates at its time limit

## Per-Episode Summary Row

After the episode ends, the evaluator computes:

- `mean_reward = float(episode_rewards.mean())`
- `mean_pheromone = float(np.mean(pheromone_usage_values))`
- `efficiency = float(food_retrieved / max(episode_length, 1))`

Then it writes one row to `eval_metrics.csv`.

The row contains:

- `episode`
- `seed`
- `mean_episode_reward`
- `food_retrieved`
- `exploration_coverage`
- `pheromone_usage`
- `episode_length`
- `swarm_efficiency`

That happens at [train/evaluate.py#L125](../train/evaluate.py#L125) through [train/evaluate.py#L136](../train/evaluate.py#L136).

## Final Summary JSON

After all episodes finish, the evaluator writes:

- `eval_summary.json`

at [train/evaluate.py#L141](../train/evaluate.py#L141).

It contains:

- `episodes`
- `obs_dim`
- `policy_kind`
- aggregated mean metrics:
  - `mean_reward`
  - `mean_food_retrieved`
  - `mean_exploration_coverage`
  - `mean_pheromone_usage`
  - `mean_episode_length`

So:

- `eval_metrics.csv` = per-episode rows
- `eval_summary.json` = overall mean summary

## How This Differs From Training

The main differences from [train/independent_dqn_pytorch.py](../train/independent_dqn_pytorch.py) are:

### No learning

There is:

- no replay buffer
- no optimizer
- no target network updates
- no Bellman loss

### Deterministic action selection

DQN evaluation uses:

- greedy `argmax`

instead of:

- epsilon-greedy exploration

### Fixed outputs

Evaluation only measures policy quality and writes metrics.

It does not produce checkpoints.

## Tensor Shapes During Evaluation

Assume:

- `n_agents = 6`
- `obs_dim = 23`
- `action_dim = 9`

### After reset

```python
obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0)
```

Shape:

- `obs`: `(6, 23)`

### For one DQN agent

```python
obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
q_vals = nets[i](obs_tensor)
```

Shapes:

- `obs[i]`: `(23,)`
- `obs_tensor`: `(1, 23)`
- `q_vals`: `(1, 9)`

Then:

```python
torch.argmax(q_vals, dim=1).item()
```

produces one scalar action index in `[0, 8]`.

### Full action batch

After looping over all agents:

- `actions`: `(6,)`

### After step

After:

```python
obs = np.stack([next_obs_dict[agent] for agent in agent_ids], axis=0)
rewards = np.array([rewards_dict[agent] for agent in agent_ids], dtype=np.float32)
```

Shapes:

- `obs`: `(6, 23)`
- `rewards`: `(6,)`

## Typical Commands

Evaluate saved DQN checkpoints:

```bash
python train/evaluate.py --checkpoint-dir checkpoints --episodes 10 --output-dir runs/eval
```

Evaluate a shared-policy checkpoint:

```bash
python train/evaluate.py --checkpoint-dir checkpoints --shared-policy --episodes 10 --output-dir runs/eval_shared
```

Evaluate the rule-based baseline:

```bash
python train/evaluate.py --policy-kind rule_based --n-agents 5 --episodes 10 --output-dir runs/rule_eval
```

Evaluate with pheromones disabled:

```bash
python train/evaluate.py --checkpoint-dir checkpoints --episodes 10 --pheromone-disabled --output-dir runs/eval_no_pheromone
```

## Practical Mental Model

The shortest correct understanding of this file is:

1. build env
2. infer `obs_dim`
3. load a policy
4. run several full episodes greedily
5. log one row per episode
6. write a mean summary JSON

## Suggested Next Reading

After this document, the best related files to read are:

1. [train/independent_dqn_pytorch.py](../train/independent_dqn_pytorch.py)
2. [models/rule_based_policy.py](../models/rule_based_policy.py)
3. [train/run_experiments.py](../train/run_experiments.py)
