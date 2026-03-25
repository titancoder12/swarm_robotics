# Artificial Collective Intelligence: Carpenter Ant-Inspired Stigmergic Swarm Robotics for Decentralized Systems Using Deep Reinforcement Learning

Individual ants possess limited cognitive capacity. Yet when thousands interact, colonies exhibit coordinated collective intelligence. This intelligence is not centralized in a single agent but emerges from decentralized interactions shaped by evolutionary processes. A key coordination mechanism is stigmergy—indirect communication through environmental modification (e.g., pheromone trails) that agents can sense and exploit.

Inspired by carpenter ants, this research investigates artificial collective intelligence arising from decentralized interactions with the environment, developing a computational framework for swarm robotics that contrasts with prevailing models of intelligence relying on large-scale centralized computing. To enable scalable experimentation, a custom multi-agent reinforcement learning simulation environment was created for decentralized policy learning. A physical swarm robotics platform was then constructed to evaluate the learned behaviours in the real world.

By coupling learned behaviours with shared “digital pheromone” fields, the system demonstrates how collective intelligence emerges from distributed agents, contributing to the emerging field of Physical AI, where intelligent algorithms interact directly with and control physical systems rather than operating only in digital environments. The robotic platform used to evaluate the system provides an accessible testbed for future swarm robotics research and is fully open-source, including algorithms, models, mechanical designs, firmware, component specifications, and assembly documentation. 

Decentralized swarm systems have applications in environments where communication infrastructure is unreliable or centralized control is fragile, such as planetary exploration and disaster response. By leveraging local decision-making and redundancy, stigmergic swarm systems provide resilience, adaptability, and robustness under uncertainty.


# Swarm RL PyGame Environment (Stigmergy)

A minimal multi-agent PyGame environment for swarm RL with pheromone stigmergy, using the PettingZoo Parallel API, plus random rollout and DQN training scripts.

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

## Quickstart: Train → Demo

1. Train headless and save checkpoints:
```bash
python train/independent_dqn_pytorch.py --headless --total-steps 10000 --save-dir checkpoints --save-every 2000
```

2. Render the trained policy:
```bash
python train/demo.py --checkpoint-dir checkpoints
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
python train/independent_dqn_pytorch.py --headless --save-every 2000 --save-dir checkpoints
```

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
Discrete action space with 9 actions: `{throttle ∈ [-1,0,1]} × {turn ∈ [-1,0,1]}`.
Provide `actions` as a dict keyed by agent id (e.g., `agent_0`) with values in `[0, 8]`.

### Observations
Each agent gets a local observation vector; `reset`/`step` return a dict of `agent_id -> obs`:
- 9 lidar rays (normalized)
- 2D relative vector to nearest target (agent frame, normalized)
- 2D relative vector to nearest agent (agent frame, normalized)
- heading as `sin(theta), cos(theta)`
- speed (normalized)
- pheromone samples in front of the agent (3 values, normalized)

Default `obs_dim` = 19.

### Stigmergy (pheromone)
The environment maintains a pheromone grid:
- deposit: each agent deposits per step
- decay: `pheromone *= 0.985`
- diffuse: simple neighbor averaging

Toggle pheromone cues in observation with `SwarmConfig.obs_include_pheromone`.

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
- Adds stigmergy via a pheromone grid that agents can sense.

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
