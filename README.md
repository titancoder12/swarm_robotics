# Artificial Collective Intelligence: Carpenter Ant-Inspired Stigmergic Swarm Robotics for Decentralized Systems Using Deep Reinforcement Learning

Individual ants possess limited cognitive capacity. Yet when thousands interact, colonies exhibit coordinated collective intelligence. This intelligence is not centralized in a single agent but emerges from decentralized interactions shaped by evolutionary processes. A key coordination mechanism is stigmergy—indirect communication through environmental modification (e.g., pheromone trails) that agents can sense and exploit.

Inspired by carpenter ants, this research investigates artificial collective intelligence arising from decentralized interactions with the environment, developing a computational framework for swarm robotics that contrasts with prevailing models of intelligence relying on large-scale centralized computing. To enable scalable experimentation, a custom multi-agent reinforcement learning simulation environment was created for decentralized policy learning. A physical swarm robotics platform was then constructed to evaluate the learned behaviours in the real world.

By coupling learned behaviours with shared “digital pheromone” fields, the system demonstrates how collective intelligence emerges from distributed agents, contributing to the emerging field of Physical AI, where intelligent algorithms interact directly with and control physical systems rather than operating only in digital environments. The robotic platform used to evaluate the system provides an accessible testbed for future swarm robotics research and is fully open-source, including algorithms, models, mechanical designs, firmware, component specifications, and assembly documentation. 

Decentralized swarm systems have applications in environments where communication infrastructure is unreliable or centralized control is fragile, such as planetary exploration and disaster response. By leveraging local decision-making and redundancy, stigmergic swarm systems provide resilience, adaptability, and robustness under uncertainty.


# Swarm RL PyGame Environment (Stigmergy)

A multi-agent PyGame environment for swarm RL with pheromone stigmergy, using
the PettingZoo Parallel API. The repo currently contains:

- a custom DQN baseline
- a recurrent GRU MAPPO training path with CTDE
- demo and evaluation utilities
- a robot-facing runtime that preserves decentralized inference

## Screenshots

<p>
  <img src="docs/images/swarm_default.png" width="480" />
  <br />
  <em>Default scene: tank dynamics with pheromone heatmap enabled.</em>
</p>
<p>
  <img src="docs/images/swarm_no_pheromone.png" width="480" />
  <br />
  <em>No pheromone rendering: same environment without the heatmap overlay.</em>
</p>
<p>
  <img src="docs/images/swarm_hover.png" width="480" />
  <br />
  <em>Hovercraft dynamics: agents drift slightly due to inertia/noise.</em>
</p>
<p>
  <img src="docs/images/swarm_dense.png" width="480" />
  <br />
  <em>Dense swarm: more agents, targets, and obstacles.</em>
</p>

Generate the screenshots locally:

```bash
python train/capture_screenshots.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For robot deployment with [firmware/run.py](firmware/run.py), `pyserial` is now included in `requirements.txt` because [firmware/ant.py](firmware/ant.py) depends on it.

## Random rollout (rendered)

```bash
python train/random_rollout.py
```

## Quickstart: Trail Learning (Recommended Path)

The current recommended research path is recurrent MAPPO with the staged
curriculum. The intended behavior is:

- explore to discover a target
- pick it up
- visibly switch into a carrying-food state
- return to the nest
- deposit pheromone on the successful return route
- complete one delivery by reaching the nest while carrying
- let later agents exploit that trail

The current default task settings now make that loop more explicit:

- there are `3` food sources in play
- each source has `4` uses
- a source loses one use on pickup
- when a source is exhausted, it respawns somewhere else if target respawn is enabled
- agents can carry only one food item at a time
- carrying agents render in a distinct green-highlighted color in demo mode

Recommended full training run:

```bash
python train/train.py --backend mappo --headless --curriculum full --n-agents 6 --total-steps 600000 --rollout-steps 128 --update-epochs 4 --minibatch-size 256 --eval-every 10000 --eval-episodes 5 --stage-repeat-limit 1 --reward-pickup 6 --reward-nest-delivery 30 --reward-undelivered-food -10 --folder-name mappo_full_600k
```

Recommended demo checkpoint after training:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_full_600k/best_greedy_eval --max-steps 300
```

What the main arguments mean:

- `--backend mappo`
  - use the recurrent MAPPO trainer instead of the DQN/SB3/RLlib paths

- `--headless`
  - run without opening a PyGame window so long training is faster and more stable

- `--curriculum full`
  - train through the full staged curriculum rather than only the early stages

- `--n-agents 6`
  - target six agents for the later full-swarm curriculum stages

- `--total-steps 600000`
  - total training budget across the whole curriculum
  - this budget is distributed across the curriculum slice you selected, not across omitted stages
  - this is much more serious than a short `30k`–`200k` run because the later stages need real time

- `--rollout-steps 128`
  - collect on-policy rollouts in chunks of 128 steps before PPO-style updates

- `--update-epochs 4`
  - run four optimization passes over each collected rollout batch

- `--minibatch-size 256`
  - minibatch size used during PPO optimization

- `--eval-every 10000`
  - run evaluation every 10,000 training steps

- `--eval-episodes 5`
  - use five episodes for each scheduled evaluation so eval is less noisy

- `--stage-repeat-limit 1`
  - allow one retry when a stage still fails its minimum greedy pickup/delivery target
  - this helps prevent weak early stages from being silently promoted

- `--reward-pickup 6`
  - keep pickup meaningful, but not as important as completed delivery

- `--reward-nest-delivery 30`
  - make successful return-to-nest delivery the strongest core task reward

- `--reward-undelivered-food -10`
  - penalize ending an episode while still carrying food
  - this helps discourage “pick up but never bring it home”

- `--folder-name mappo_full_600k`
  - base name for checkpoints and run outputs

Why this is the recommended starting point:

- the current curriculum spreads learning across many stages
- the early stages now use more responsive `action_repeat_steps = 1`, while later stages keep smoother `action_repeat_steps = 2`
- the early stages now deliberately simplify the task: pheromone is disabled in stage 1, movement penalties are softened, and pickup/delivery cues are stronger so greedy `pickup -> return -> deliver` behavior can form first
- prompts 30, 31, and 32 now suppress exploration reward while carrying in the return-focused stages, add a dedicated guaranteed-homing stage with controlled target/agent placement, and then reintroduce clutter through a bridge stage before the true obstacle-return stage
- the trainer now decays entropy within each stage instead of keeping one fixed exploration pressure forever, so early rollouts can explore while later updates in the same stage become more deterministic
- the early stages keep a slightly stronger exploration bonus, and later stages reduce `reward_new_cell` so delivery and trail reuse compete less with wandering
- the trainer now keeps a fixed padded centralized critic state dimension across the selected curriculum so the critic can carry across stages instead of resetting whenever the stage shape changes
- stage progression is now greedy-eval-aware, with optional repeats when pickup/delivery remain below minimum promotion targets
- the trainer now prints sampled-vs-greedy pickup/delivery gaps and saves `best_greedy_eval/` so demo can use the strongest greedy checkpoint instead of assuming `latest/` is best
- stage summaries now also print pickup-to-delivery conversion, which is the main signal for whether return-to-nest behavior is actually forming
- the final stage is still large and hard: `1400x950`, `18` obstacles, `6` agents
- a shorter run can finish, but often leaves the later full-swarm stages undertrained
- `600k` is not guaranteed to be optimal, but it is a practical strong starting point for the current repo

Short smoke test:

```bash
python train/train.py --backend mappo --headless --curriculum stage1 --n-agents 6 --total-steps 2400 --rollout-steps 64 --update-epochs 2 --minibatch-size 128 --eval-every 0 --no-plots --n-targets 3 --active-targets 3 --food-source-capacity 4 --target-respawn --folder-name mappo_trail_smoke
```

Render the trained MAPPO policy:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_trail_full/best_greedy_eval --max-steps 300
```

In demo mode, agents now switch to a distinct carrying-food color after pickup
and return to the normal agent color after a completed nest delivery.

Headless MAPPO evaluation:

```bash
python analysis/evaluate.py --policy-kind mappo_gru --checkpoint-dir checkpoints/mappo_trail_full/latest --n-agents 6 --episodes 10 --headless --output-dir runs/eval --filename mappo_trail_full_eval --active-targets 3 --food-source-capacity 4
```

Pheromone comparison example:

```bash
python analysis/evaluate_comparison.py --policy-kind mappo_gru --checkpoint-with-pheromone checkpoints/mappo_trail_full/latest --checkpoint-without-pheromone checkpoints/mappo_trail_no_pher/latest --agent-min 1 --agent-max 6 --episodes-per-agent 3 --headless --output-dir experiments/experiment_data/trail_compare
```

## Quickstart: DQN Baseline

Train headless and save checkpoints:

```bash
python train/independent_dqn_pytorch.py --headless --total-steps 10000 --save-dir checkpoints --folder-name demo_run --save-every 2000
```

Render the trained policy:

```bash
python train/demo.py --checkpoint-dir checkpoints/demo_run/full_policy
```

## Train (headless)

```bash
python train/independent_dqn_pytorch.py --headless --total-steps 10000
```

Use a shared policy:

```bash
python train/independent_dqn_pytorch.py --shared-policy --headless
```

Save checkpoints during training:

```bash
python train/independent_dqn_pytorch.py --headless --save-every 2000 --save-dir checkpoints --folder-name demo_run
```

Train with an explicit epsilon schedule:

```bash
python train/independent_dqn_pytorch.py --headless --total-steps 30000 --save-dir checkpoints --folder-name demo_run --epsilon-start 1.0 --epsilon-final 0.05 --epsilon-decay-steps 20000 --warmup-steps 2000
```

If you run that command manually, keep it on one shell line or use `\` line continuations exactly. Entering `>`-prefixed continuation lines or isolated flag lines can make the shell create empty files such as `--epsilon-final`, `--epsilon-decay-steps`, or `--warmup-steps` in the repo root.

## Training Backends (Optional)

Default (custom DQN):
```bash
python train/train.py --backend custom --headless --total-steps 10000
```

Stable-Baselines3 DQN (shared policy):
```bash
python train/train.py --backend sb3 --headless --total-steps 10000 --save-path checkpoints/sb3_dqn.zip
```

RLlib DQN (shared policy):
```bash
python train/train.py --backend rllib --headless --total-steps 10000 --save-dir checkpoints/rllib_dqn
```

If Ray warns about `/tmp` being full or socket path length, point it to a different temp dir:
```bash
python train/train.py --backend rllib --headless --total-steps 10000 --save-dir checkpoints/rllib_dqn --ray-tmpdir /Users/christopherlin/.ray_tmp
```

## Demo (rendered)

```bash
python train/demo.py --checkpoint-dir checkpoints
```

Shared-policy demo:

```bash
python train/demo.py --checkpoint-dir checkpoints --shared-policy
```

SB3 demo:

```bash
python train/demo.py --backend sb3 --sb3-model checkpoints/sb3_dqn.zip
```

RLlib demo:

```bash
python train/demo.py --backend rllib --rllib-checkpoint checkpoints/rllib_dqn
```

MAPPO demo:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_trail_full/latest --n-agents 6 --max-steps 300
```

With a custom Ray temp dir:
```bash
python train/demo.py --backend rllib --rllib-checkpoint checkpoints/rllib_dqn --ray-tmpdir /Users/christopherlin/.ray_tmp
```

To auto-exit after N steps (useful for smoke tests):
```bash
python train/demo.py --backend custom --max-steps 200
```

## Robot Deployment

The current robot runtime lives in [firmware/](firmware).

Main files:

- [firmware/ant.py](firmware/ant.py)
  - serial client and low-level ESP32 command helpers
- [firmware/run.py](firmware/run.py)
  - direct checkpoint inference loop for the physical robot
- [firmware/ant.service](firmware/ant.service)
  - example systemd unit

Typical deployment flow:

```bash
python firmware/run.py --checkpoint-dir checkpoints --shared-policy
```

This runtime loads [models/q_network.py](models/q_network.py), reads scan lines from `ant.py`, normalizes the observation locally, predicts a discrete action, and sends `turn`, `move`, or `stop` commands directly to the robot.

## Command Center

The repo now includes a separate live operator subsystem in [mission_control/](mission_control). It is a PyGame command center for:

- receiving live robot `POS`, `PHER`, and `SENSE` messages
- visualizing robot positions and trails
- maintaining the authoritative digital pheromone field
- returning simulator-compatible `PHER_RESP` samples back to robots

Run it locally:

```bash
python -m mission_control.main --tcp-host 127.0.0.1 --tcp-port 8765
```

Run a local fake robot against it:

```bash
python -m mission_control.fake_robot --robot-id robot_0 --port 8765
```

Documentation:

- [docs/MISSION_CONTROL.md](docs/MISSION_CONTROL.md)
- [docs/MISSION_CONTROL_FIRMWARE_PSEUDOCODE.md](docs/MISSION_CONTROL_FIRMWARE_PSEUDOCODE.md)

## Environment API (PettingZoo Parallel API)

`SwarmEnv` implements the PettingZoo Parallel API.

Methods:
- `reset(seed=None, options=None) -> (obs_dict, info_dict)`
- `step(action_dict) -> (obs_dict, rewards_dict, terminations, truncations, infos)`
- `render(mode="human", fps=60)`
- `close()`

### Actions
Discrete action space with 18 actions:
`{throttle ∈ [-1,0,1]} × {turn ∈ [-1,0,1]} × {deposit ∈ [0,1]}`.
Provide `actions` as a dict keyed by agent id (e.g., `agent_0`) with values in `[0, 17]`.

### Observations
Each agent gets a local observation history vector; `reset`/`step` return a
dict of `agent_id -> obs`.

Current default per-frame features:

- 9 lidar rays
- 2 nearest detectable target features: distance and relative angle
- 2 nest-direction features
- 2 nearest-neighbor features
- 2 heading features: `sin(theta)`, `cos(theta)`
- 1 normalized speed feature
- 1 food-presence flag
- 1 carrying-food flag
- 3 pheromone samples

Current default observation size:

- 23 features per frame
- `observation_history_steps = 3`
- flattened `obs_dim = 69`

### Stigmergy (pheromone)
The environment maintains a pheromone grid with:

- explicit deposit vs no-deposit action choice
- decay
- bounded diffusion
- carrying-food scaling
- optional gating so deposition only happens while carrying food and making return-to-nest progress

The intended training story is trail formation:

- discovery is expensive early
- successful returns write route hints into the environment
- later agents can exploit those hints

## Files
- `env/swarm_env.py` : environment implementation
- `env/config.py` : configuration dataclass
- `train/random_rollout.py` : random policy sanity check
- `train/independent_dqn_pytorch.py` : independent or shared DQN training
- `train/demo.py` : load and render trained checkpoints
- `firmware/run.py` : physical robot policy runtime
- `firmware/ant.py` : direct serial robot interface
- `docs/ARCHITECTURE.md` : detailed functionality and architecture
- `docs/PROJECT_LOG.md` : decisions, notes, and next steps to resume later

## Notes
- Episode ends when all targets are collected or `max_steps` reached.
- Dynamics are pluggable via `dynamics_mode`: `"tank"`, `"hover"`, or `"mixed"`.

## Beginner Walkthrough (PyGame + RL)

If you’re new to PyGame and RL, this section gives a quick mental model and a practical path to running the project.

### What this project does
- Simulates a swarm of agents in a 2D PyGame world.
- Exposes an RL-style API (`reset`, `step`) with multi-agent observations and rewards.
- Adds stigmergy via a pheromone grid that agents can sense and write to.
- Supports a trail-formation objective where agents learn `discover -> return -> deposit -> exploit`.

### The fastest way to see it working
1) Random sanity check (renders a window):
```bash
python train/random_rollout.py
```

2) Train a basic model (headless, no window):
```bash
python train/independent_dqn_pytorch.py --headless --total-steps 10000 --save-dir checkpoints --save-every 2000
```

3) Run the demo with the trained model (renders a window):
```bash
python train/demo.py --checkpoint-dir checkpoints
```

### How the RL loop works (simple view)
Each step:
1) You give actions for each agent.
2) The environment moves agents, handles collisions, and collects targets.
3) You receive rewards + new observations.

### What the agents are learning
- **+8** when an agent reaches a target
- **-0.01** each step (encourages speed)
- **-0.2** for collisions with walls/obstacles

So the learned behavior should be: “find targets quickly without crashing.”

### How PyGame fits in
- PyGame is only used for rendering and window events.
- If you run with `--headless`, no window is opened and PyGame doesn’t render.

### Where to change behavior
- `env/config.py` controls most parameters:
  - number of agents/targets/obstacles
  - rewards
  - lidar rays
  - pheromone settings
  - dynamics mode (`tank`, `hover`, `mixed`)

### Want deeper details?
See `docs/ARCHITECTURE.md` for a full breakdown of modules and data flow.
