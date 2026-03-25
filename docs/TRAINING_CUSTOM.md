# Custom DQN Training Walkthrough

This document explains how the custom training pipeline works in [train/independent_dqn_pytorch.py](../train/independent_dqn_pytorch.py).

It is implementation-grounded. The goal is to help a developer understand exactly how training runs in this repo, how data flows through the system, and what each major block of the file is doing.

## What This File Does

[train/independent_dqn_pytorch.py](../train/independent_dqn_pytorch.py) implements a classic DQN training loop for the swarm environment.

It uses:

- a replay buffer
- epsilon-greedy action selection
- an online Q-network
- a target Q-network
- Bellman targets
- Huber loss

It supports two training modes:

- independent policies: one Q-network per agent
- shared policy: one Q-network shared by all agents

The environment is still multi-agent in both cases. The difference is whether the agents learn separate networks or share one.

## File Structure

The main parts of the file are:

- `DQNConfig`
- `ReplayBuffer`
- helper conversions between PettingZoo dicts and NumPy arrays
- `linear_schedule(...)`
- `_evaluate_policy(...)`
- `train(args)`
- `parse_args(...)`
- `_save_models(...)`

## High-Level Training Flow

The training loop follows this repeated pattern:

1. reset the environment and get initial observations
2. choose one action per agent with epsilon-greedy
3. call `env.step(...)`
4. store transitions in replay
5. sample minibatches from replay
6. compute Bellman targets
7. update the Q-network
8. periodically sync the target network
9. log episode and evaluation metrics
10. save checkpoints

## Hyperparameters

`DQNConfig` is defined at [train/independent_dqn_pytorch.py#L37](../train/independent_dqn_pytorch.py#L37).

Current defaults:

- `gamma = 0.98`
- `batch_size = 64`
- `buffer_size = 50_000`
- `lr = 3e-4`
- `target_update = 200`
- `epsilon_start = 1.0`
- `epsilon_final = 0.05`
- `epsilon_decay_steps = 8000`
- `warmup_steps = 500`

These are the learning hyperparameters, separate from environment configuration.

## Replay Buffer

`ReplayBuffer` is defined at [train/independent_dqn_pytorch.py#L51](../train/independent_dqn_pytorch.py#L51).

Each stored transition contains:

- `obs`
- `action`
- `reward`
- `next_obs`
- `done`

The arrays are preallocated with fixed shape:

- `obs`: `(capacity, obs_dim)`
- `next_obs`: `(capacity, obs_dim)`
- `actions`: `(capacity,)`
- `rewards`: `(capacity,)`
- `dones`: `(capacity,)`

This is a circular FIFO buffer. When it fills up, old transitions are overwritten.

## Environment Setup

Training begins inside `train(args)` at [train/independent_dqn_pytorch.py#L166](../train/independent_dqn_pytorch.py#L166).

The first important steps are:

1. `cfg = make_swarm_config(args)`
2. `dqn_cfg = DQNConfig()`
3. `env = SwarmEnv(cfg, headless=args.headless)`
4. `obs_dict, _ = env.reset(seed=args.seed)`

The file does not hardcode observation size. Instead, it infers it from the real environment:

- `obs = _dict_to_array(obs_dict, agent_ids, dtype=np.float32)`
- `obs_dim = obs.shape[1]`
- `action_dim = cfg.num_actions`

That means the trainer automatically adapts to the current observation contract as long as the environment exposes it consistently.

## Logging Setup

The trainer creates a run directory and several outputs:

- `episode_metrics.csv`
- `eval_metrics.csv`
- `run_config.json`
- `summary.json`

This happens at [train/independent_dqn_pytorch.py#L183](../train/independent_dqn_pytorch.py#L183) through [train/independent_dqn_pytorch.py#L229](../train/independent_dqn_pytorch.py#L229).

Logged episode metrics include:

- `mean_episode_reward`
- `food_discovered`
- `food_retrieved`
- `exploration_coverage`
- `pheromone_usage`
- `episode_length`
- `collisions`
- reward breakdown components
- `swarm_efficiency`

## Network Initialization

At [train/independent_dqn_pytorch.py#L231](../train/independent_dqn_pytorch.py#L231), the trainer builds:

- online Q-networks
- target Q-networks
- optimizers
- replay buffers

### Shared Policy Mode

If `--shared-policy` is passed:

- one `QNetwork(obs_dim, action_dim)` is created
- one target network is created
- one Adam optimizer is created
- one replay buffer is created per agent
- all agents share the same network object

So:

- `q_nets[i]` points to the same model for every `i`
- `target_nets[i]` points to the same target model for every `i`
- `optimizers[i]` points to the same optimizer for every `i`

### Independent Policy Mode

If `--shared-policy` is not passed:

- each agent gets its own Q-network
- each agent gets its own target network
- each agent gets its own optimizer
- each agent gets its own replay buffer

This is the default mode.

## Episode Bookkeeping

Before entering the training loop, the file initializes episode-level accumulators:

- `episode_rewards`
- `episode_food_discovered`
- `episode_food_retrieved`
- `episode_collisions`
- `episode_exploration_coverage`
- `episode_pheromone_usage`
- `episode_length`
- `episode_reward_breakdown`

These are only for logging. They do not affect the actual DQN update.

## Main Training Loop

The main loop starts at [train/independent_dqn_pytorch.py#L277](../train/independent_dqn_pytorch.py#L277):

```python
while global_step < args.total_steps:
```

Each pass through this loop is one environment step.

### Step 1: Compute Epsilon

At [train/independent_dqn_pytorch.py#L280](../train/independent_dqn_pytorch.py#L280):

```python
epsilon = linear_schedule(...)
```

This linearly decreases exploration from `1.0` to `0.05` over `8000` steps.

### Step 2: Choose Actions

At [train/independent_dqn_pytorch.py#L284](../train/independent_dqn_pytorch.py#L284), one action is chosen for each agent.

For each agent:

- with probability `epsilon`, choose a random action
- otherwise:
  - convert the observation to a tensor
  - run the Q-network
  - choose the action with the highest Q-value

This is epsilon-greedy exploration.

### Step 3: Step the Environment

At [train/independent_dqn_pytorch.py#L295](../train/independent_dqn_pytorch.py#L295):

- the action array is converted into a PettingZoo dict
- `env.step(action_dict)` is called
- next observations, rewards, termination flags, truncation flags, and info come back

Then the trainer computes:

- `terminated`
- `truncated`
- `done_flag = float(terminated or truncated)`

That single `done_flag` is what gets stored in replay for all agents for that environment step.

### Step 4: Store Transitions

At [train/independent_dqn_pytorch.py#L304](../train/independent_dqn_pytorch.py#L304), each agent transition is added to replay:

```python
buffers[i].add(obs[i], actions[i], rewards[i], next_obs[i], done_flag)
```

This is the exact tuple:

- `obs[i]`
- `actions[i]`
- `rewards[i]`
- `next_obs[i]`
- `done_flag`

The trainer also updates episode-level stats from the env `info`.

### Step 5: Move Forward in Time

At [train/independent_dqn_pytorch.py#L322](../train/independent_dqn_pytorch.py#L322):

- `obs = next_obs`
- `global_step += 1`

This makes the next observation become the current observation for the next loop iteration.

### Step 6: Learn From Replay

At [train/independent_dqn_pytorch.py#L326](../train/independent_dqn_pytorch.py#L326), learning begins once warmup is over:

- if `global_step > warmup_steps`
- if each buffer has at least `batch_size` samples

Then for each agent:

1. sample a minibatch
2. move batch tensors to the selected device
3. compute predicted `Q(s, a)`
4. compute Bellman target
5. compute Huber loss
6. backpropagate
7. update weights

This is the core DQN learning block.

### Step 7: Sync Target Networks

At [train/independent_dqn_pytorch.py#L345](../train/independent_dqn_pytorch.py#L345):

```python
if global_step % dqn_cfg.target_update == 0:
```

When that condition is true, the trainer copies the online Q-network weights into the target network.

This stabilizes learning because the Bellman target does not change every gradient step.

### Step 8: Periodic Evaluation

At [train/independent_dqn_pytorch.py#L350](../train/independent_dqn_pytorch.py#L350), if evaluation is enabled:

- `_evaluate_policy(...)` runs deterministic episodes
- the current policy is tested without epsilon-greedy randomness
- metrics are written to `eval_metrics.csv`

This evaluation uses a fresh headless environment.

### Step 9: End-of-Episode Logging

At [train/independent_dqn_pytorch.py#L356](../train/independent_dqn_pytorch.py#L356), if the episode ended:

- aggregate episode metrics
- compute `swarm_efficiency`
- write one row to `episode_metrics.csv`
- print a one-line summary
- reset the environment
- reset episode accumulators

### Step 10: Checkpointing

At [train/independent_dqn_pytorch.py#L397](../train/independent_dqn_pytorch.py#L397):

- if `--save-every > 0`, save intermediate checkpoints every `N` steps

At [train/independent_dqn_pytorch.py#L401](../train/independent_dqn_pytorch.py#L401):

- always save final checkpoints at the end of training

`_save_models(...)` at [train/independent_dqn_pytorch.py#L441](../train/independent_dqn_pytorch.py#L441) writes:

- `metadata.json`
- `shared.pt` if shared policy mode
- `agent_0.pt`, `agent_1.pt`, ... otherwise

## Bellman Update Block: Line-by-Line Walkthrough

This is the core learning block from [train/independent_dqn_pytorch.py#L331](../train/independent_dqn_pytorch.py#L331).

### 1. Sample a minibatch

```python
batch = buffers[i].sample(dqn_cfg.batch_size)
batch = [b.to(device) for b in batch]
b_obs, b_actions, b_rewards, b_next_obs, b_dones = batch
```

Meaning:

- sample random past experience from replay
- move all tensors onto CPU or GPU
- unpack into named tensors

These tensors represent:

- `b_obs`: current states
- `b_actions`: chosen actions
- `b_rewards`: rewards
- `b_next_obs`: next states
- `b_dones`: terminal flags

### 2. Compute predicted Q-values for taken actions

```python
q_vals = q_nets[i](b_obs).gather(1, b_actions.unsqueeze(1)).squeeze(1)
```

Breakdown:

- `q_nets[i](b_obs)` computes all action values for every state in the batch
- output shape is `(batch_size, action_dim)`
- `b_actions.unsqueeze(1)` turns `(batch_size,)` into `(batch_size, 1)`
- `.gather(1, ...)` selects the Q-value for the action that was actually taken
- `.squeeze(1)` turns the result back into `(batch_size,)`

So `q_vals` is the network’s current estimate of:

`Q(s, a)`

for the actions stored in replay.

### 3. Compute target-network next-state values

```python
with torch.no_grad():
    max_next = target_nets[i](b_next_obs).max(dim=1)[0]
```

Breakdown:

- run the target network on the next states
- get all action values for each next state
- take the maximum over actions

So `max_next` is:

`max_a' Q_target(s', a')`

for each sample in the batch.

The `torch.no_grad()` block prevents gradients from flowing through the target calculation.

### 4. Build the Bellman target

```python
target = b_rewards + dqn_cfg.gamma * (1.0 - b_dones) * max_next
```

Meaning:

- if the transition is non-terminal:
  - target = immediate reward + discounted best future value
- if the transition is terminal:
  - the future term is removed because `1.0 - b_dones` becomes `0`

So this is:

`target = r + gamma * max_a' Q_target(s', a')`

or just:

`target = r`

when the episode ended.

### 5. Compute the loss

```python
loss = nn.functional.smooth_l1_loss(q_vals, target)
```

This is Huber loss.

Why use it:

- less sensitive than MSE to large errors
- common in DQN implementations

The trainer is teaching the network:

- make predicted `Q(s, a)` closer to the Bellman target

### 6. Apply gradient descent

```python
optimizers[i].zero_grad()
loss.backward()
optimizers[i].step()
```

Meaning:

- clear old gradients
- backpropagate the new loss
- update the Q-network parameters with Adam

That is one gradient update.

## Tensor-Shape Walkthrough For One Training Step

This section gives a concrete shape walkthrough for one training step.

Assume:

- `n_agents = 6`
- `obs_dim = 23`
- `action_dim = 9`
- `batch_size = 64`

### At environment reset

After:

```python
obs_dict, _ = env.reset(seed=args.seed)
obs = _dict_to_array(obs_dict, agent_ids, dtype=np.float32)
```

the shapes are:

- `obs_dict["agent_0"]`: `(23,)`
- `obs`: `(6, 23)`

So:

- one row per agent
- one 23-dim local observation per row

### Action selection for one agent

Inside the epsilon-greedy loop:

```python
obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
q_vals = q_nets[i](obs_tensor)
```

Shapes:

- `obs[i]`: `(23,)`
- `obs_tensor`: `(1, 23)`
- `q_vals`: `(1, 9)`

Then:

```python
actions[i] = int(torch.argmax(q_vals, dim=1).item())
```

gives one scalar action index in `[0, 8]`.

After all agents choose actions:

- `actions`: `(6,)`

### After stepping the environment

After:

```python
next_obs_dict, rewards_dict, terminations, truncations, info = env.step(action_dict)
next_obs = _dict_to_array(next_obs_dict, agent_ids, dtype=np.float32)
rewards = _dict_to_array(rewards_dict, agent_ids, dtype=np.float32)
```

Shapes:

- `next_obs`: `(6, 23)`
- `rewards`: `(6,)`

Then one transition per agent is stored in replay.

### When sampling a minibatch

After:

```python
batch = buffers[i].sample(dqn_cfg.batch_size)
b_obs, b_actions, b_rewards, b_next_obs, b_dones = batch
```

Shapes:

- `b_obs`: `(64, 23)`
- `b_actions`: `(64,)`
- `b_rewards`: `(64,)`
- `b_next_obs`: `(64, 23)`
- `b_dones`: `(64,)`

### Online-network forward pass

After:

```python
q_nets[i](b_obs)
```

shape is:

- `(64, 9)`

because each of the 64 sampled observations gets 9 Q-values.

Then:

```python
b_actions.unsqueeze(1)
```

changes shape from:

- `(64,)`

to:

- `(64, 1)`

Then:

```python
q_nets[i](b_obs).gather(1, b_actions.unsqueeze(1))
```

has shape:

- `(64, 1)`

Then:

```python
.squeeze(1)
```

gives:

- `q_vals`: `(64,)`

Now each sample has one predicted Q-value for the action that was actually taken.

### Target-network next-state pass

After:

```python
target_nets[i](b_next_obs)
```

shape is:

- `(64, 9)`

Then:

```python
.max(dim=1)[0]
```

gives:

- `max_next`: `(64,)`

Now each sample has one scalar: the highest next-state Q-value.

### Bellman target tensor

After:

```python
target = b_rewards + dqn_cfg.gamma * (1.0 - b_dones) * max_next
```

shape is:

- `target`: `(64,)`

So at loss time:

- predicted values `q_vals`: `(64,)`
- Bellman target `target`: `(64,)`

These shapes match exactly.

### Loss and optimization

After:

```python
loss = nn.functional.smooth_l1_loss(q_vals, target)
```

shape is:

- scalar tensor `()`

Then:

```python
loss.backward()
optimizers[i].step()
```

updates the Q-network weights.

## Evaluation Flow

`_evaluate_policy(...)` is defined at [train/independent_dqn_pytorch.py#L106](../train/independent_dqn_pytorch.py#L106).

It:

- builds a fresh headless environment
- runs greedy action selection only
- records mean metrics across several episodes

It does not train. It only measures current policy quality.

## CLI Arguments

`parse_args(...)` is defined at [train/independent_dqn_pytorch.py#L421](../train/independent_dqn_pytorch.py#L421).

Core arguments:

- `--total-steps`
- `--n-agents`
- `--shared-policy`
- `--headless`
- `--cuda`
- `--seed`
- `--save-dir`
- `--save-every`
- `--experiment-name`
- `--output-dir`
- `--eval-every`
- `--eval-episodes`
- `--no-plots`

It also imports environment-related flags through `add_env_config_args(parser)`.

## Practical Mental Model

If you want the shortest correct understanding of the file, it is this:

1. build env
2. infer `obs_dim`
3. build Q-networks and target networks
4. choose actions with epsilon-greedy
5. store transitions in replay
6. sample replay minibatches
7. compute Bellman targets
8. train with Huber loss
9. sync target networks occasionally
10. log, evaluate, and save checkpoints

## Suggested Next Reading

After this document, the best files to read are:

1. [models/q_network.py](../models/q_network.py)
2. [env/swarm_env.py](../env/swarm_env.py)
3. [train/evaluate.py](../train/evaluate.py)

