# DQN Explained in This Repo

This document explains three closely related parts of the custom DQN trainer in this repository:

1. `QNetwork` and what each tensor means
2. The Bellman update as implemented here
3. One full training iteration in the custom trainer

The relevant code lives in `train/independent_dqn_pytorch.py`.

## 1) `QNetwork` and What Each Tensor Means

`QNetwork` is the function approximator for the Q-function. It takes one observation vector and returns one Q-value for each discrete action.

In this repo:

- input shape: `(obs_dim,)` for one agent observation
- batched input shape: `(batch_size, obs_dim)`
- output shape: `(action_dim,)` for one observation
- batched output shape: `(batch_size, action_dim)`

Example mental model:

- if `obs_dim = 19` and `action_dim = 9`
- the network maps `19 -> 128 -> 128 -> 9`

So for one agent at one timestep:

- `obs_tensor` has shape `(1, 19)`
- `q_vals = q_net(obs_tensor)` has shape `(1, 9)`

Each of those 9 numbers means:

- “How good does the network think action `a` is in this observation?”

Important tensors in training:

- `b_obs`: batch of observations, shape `(batch_size, obs_dim)`
- `b_actions`: batch of chosen action ids, shape `(batch_size,)`
- `b_rewards`: batch of rewards, shape `(batch_size,)`
- `b_next_obs`: batch of next observations, shape `(batch_size, obs_dim)`
- `b_dones`: batch of done flags, shape `(batch_size,)`

This line:

```python
q_vals = q_nets[i](b_obs).gather(1, b_actions.unsqueeze(1)).squeeze(1)
```

means:

- compute all Q-values for all actions for each state in the batch
- keep only the Q-value for the action that was actually taken
- result shape becomes `(batch_size,)`

So `q_vals` here is really:

- predicted `Q(s, a)` for the actions the agent actually took

## 2) The Bellman Equation in This Code

The DQN target is built here:

```python
max_next = target_nets[i](b_next_obs).max(dim=1)[0]
target = b_rewards + dqn_cfg.gamma * (1.0 - b_dones) * max_next
```

That is the Bellman target:

- if not done:
  `target = r + gamma * max_a' Q_target(s', a')`
- if done:
  `target = r`

Why:

- `b_rewards` is the immediate reward
- `gamma` discounts future reward
- `max_next` says “from the next state, assume the best next action”
- `(1.0 - b_dones)` removes the future term when the episode ended

Then the code compares:

- current prediction: `Q(s, a)`
- target: `r + gamma * max Q_target(s', a')`

using Huber loss:

```python
loss = nn.functional.smooth_l1_loss(q_vals, target)
```

So the network is learning:

- “make the Q-value for the chosen action closer to the reward plus discounted future value”

The target network exists for stability. Instead of using the same network on both sides of the target, the code uses `target_nets[i]`, which is copied from the online network every `target_update` steps.

## 3) One Full Training Iteration in the Custom Trainer

One outer-loop iteration in `train()` looks like this:

### Step 1: Compute epsilon

```python
epsilon = linear_schedule(...)
```

This sets the exploration rate.

### Step 2: Choose actions for each agent

- with probability `epsilon`, choose random action
- otherwise, run the network and take `argmax`

```python
obs_tensor = torch.tensor(obs[i], ...).unsqueeze(0)
q_vals = q_nets[i](obs_tensor)
actions[i] = int(torch.argmax(q_vals, dim=1).item())
```

### Step 3: Step the environment

```python
action_dict = _array_to_dict(actions, agent_ids)
next_obs_dict, rewards_dict, terminations, truncations, info = env.step(action_dict)
```

Now the world changes and returns:

- next observations
- rewards
- whether the episode ended

### Step 4: Convert dict outputs to arrays

```python
next_obs = _dict_to_array(next_obs_dict, ...)
rewards = _dict_to_array(rewards_dict, ...)
done_flag = float(terminated or truncated)
```

### Step 5: Store transitions in replay

```python
buffers[i].add(obs[i], actions[i], rewards[i], next_obs[i], done_flag)
```

So each replay item is:

- current observation
- chosen action
- reward
- next observation
- done

### Step 6: Advance current state

```python
obs = next_obs
global_step += 1
```

### Step 7: Learn from replay after warmup

If enough steps have passed and the buffer is big enough:

- sample a random batch
- compute `Q(s, a)`
- compute the Bellman target
- take a gradient step

```python
batch = buffers[i].sample(...)
b_obs, b_actions, b_rewards, b_next_obs, b_dones = batch
q_vals = q_nets[i](b_obs).gather(...)
max_next = target_nets[i](b_next_obs).max(dim=1)[0]
target = b_rewards + gamma * (1.0 - b_dones) * max_next
loss.backward()
optimizers[i].step()
```

### Step 8: Periodically sync the target network

```python
if global_step % dqn_cfg.target_update == 0:
    target_nets[i].load_state_dict(q_nets[i].state_dict())
```

### Step 9: Reset the environment if the episode ended

```python
if terminated or truncated:
    obs_dict, _ = env.reset(...)
    obs = _dict_to_array(obs_dict, ...)
```

That whole cycle repeats until `global_step >= total_steps`.

## 4) Short Mental Model

The shortest correct mental model is:

- act with epsilon-greedy
- store experience
- learn from old random experience
- bootstrap with Bellman targets
- stabilize with a target network
