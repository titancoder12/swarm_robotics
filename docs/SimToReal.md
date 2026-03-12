# Sim-to-Real Deployment Architecture

This document describes a minimal-change path for deploying the learned policy from this repository onto a physical robot whose high-level compute runs on a Raspberry Pi 4/5 and whose low-level sensing and actuation are handled by an Arduino.

The core design principle is simple:

- Keep the learned policy boundary unchanged.
- Replace the simulator-facing observation and action plumbing with a real-world adapter.
- Keep safety and low-level control outside the learned policy.

## 1) Goal

The existing codebase already has a clean policy interface:

- Input: a normalized observation vector
- Output: a discrete action index

In simulation:

- The observation vector is built inside `_get_obs()` in `env/swarm_env.py`
- The action index is mapped to `(throttle, turn)` in `_build_action_table()`
- The environment applies those commands through the kinematics drivers

In the real robot:

- A Pi-side observation builder should create the same observation vector from real sensors
- A Pi-side action bridge should map the chosen action into a motor command
- An Arduino should execute low-level motor control and stream sensor data back to the Pi

This approach avoids rewriting the trainer, the model architecture, or the action semantics.

## 2) Recommended Split of Responsibilities

### Raspberry Pi

The Raspberry Pi should act as the robot "brain" and own:

- Sensor fusion across all incoming data sources
- Observation building
- Model inference
- High-level action selection
- Logging
- Runtime health checks
- Safety gating and action clamping

### Arduino

The Arduino should own:

- Real-time sensor polling when timing is critical
- Encoder counting
- Low-level motor control or wheel velocity control
- Hardware watchdog stop behavior
- Failsafe stop when communication from the Pi times out

This split is practical because the Arduino is good at deterministic low-level I/O, while the Pi is better suited for PyTorch inference and coordination logic.

## 3) End-to-End Data Flow

The recommended runtime loop is:

1. Arduino samples sensors and publishes a `SensorPacket`
2. Pi receives the packet through a `SensorBridge`
3. Pi-side `ObservationBuilder` converts raw measurements into the same normalized observation vector expected by the policy
4. Pi-side `PolicyRunner` runs model inference and chooses a discrete action
5. Pi-side `ActionBridge` maps the action into a high-level command and transmits it to the Arduino
6. Arduino executes the command through low-level motor control
7. Safety checks remain active on both Pi and Arduino

In short:

`SensorPacket -> ObservationBuilder -> PolicyRunner -> ActionBridge -> Arduino motor controller`

## 4) Why This Minimizes Changes

This design preserves the existing training and inference boundary:

- `QNetwork` remains unchanged
- Checkpoint format remains unchanged
- Discrete actions remain unchanged
- Existing action semantics from `SwarmEnv._build_action_table()` remain unchanged

Only the runtime adapter changes.

The simulator still exists for training and offline testing. The real robot runtime becomes a new inference path that uses the same trained model.

## 5) Communication Between Pi and Arduino

Start with USB serial. It is the simplest and most robust initial transport for this project.

Recommended protocol:

- One message type for sensor state: `sensor_packet`
- One message type for action commands: `command_packet`

Use a compact line-delimited format first:

- JSON Lines if you want easy debugging
- MessagePack or CBOR later if you need lower overhead

For a first deployment, JSON Lines over serial is good enough and easy to inspect.

### Example `sensor_packet`

```json
{
  "ts_ms": 123456,
  "imu_yaw": 0.12,
  "imu_yaw_rate": 0.04,
  "speed_mps": 0.31,
  "ranges_m": [0.45, 0.62, 0.81, 0.90, 1.10, 0.87, 0.76, 0.54, 0.40],
  "target_vector_body": [0.15, -0.20],
  "neighbor_vector_body": [0.00, 0.00],
  "pheromone_samples": [0.0, 0.0, 0.0],
  "battery_v": 11.9,
  "estop": false
}
```

### Example `command_packet`

```json
{
  "ts_ms": 123460,
  "mode": "run",
  "action_id": 5,
  "throttle": 1.0,
  "turn": 0.0
}
```

## 6) Observation Compatibility Contract

This is the most important sim-to-real rule:

The real robot must feed the model observations that match training expectations.

That means:

- Same feature order
- Same units
- Same normalization
- Same clipping
- Same missing-data behavior

For the default environment, each observation contains:

1. Lidar-like ranges
2. Nearest target vector
3. Nearest neighbor vector
4. Heading represented as `sin(theta), cos(theta)`
5. Speed
6. Pheromone samples

If the real robot cannot reliably provide a given feature, you must either:

- approximate it in a principled way, or
- remove/change it and retrain the policy

Do not silently change feature meaning while keeping the same vector slot.

## 7) Action Compatibility Contract

The model should continue to output the same discrete action space used during training.

That action is converted in two steps:

1. `action_id -> (throttle, turn)`
2. `(throttle, turn) -> low-level actuator command`

The Arduino should not receive raw neural-network internals. It should receive a clean high-level command.

Examples:

- Target wheel velocities
- Throttle and turn requests
- Linear/angular velocity setpoints

This keeps the deployed system aligned with the training assumptions.

## 8) Safety Architecture

Safety must be outside the policy.

### Pi-side safety

- Reject stale sensor packets
- Clamp action outputs to allowed ranges
- Refuse to command motion when `estop` or fault flags are active
- Fall back to stop if observation data is invalid

### Arduino-side safety

- Stop motors if no command is received within a timeout window
- Honor hardware emergency stop
- Enforce PWM/speed/current limits
- Stop on sensor fault conditions if appropriate

The policy is allowed to suggest actions. The safety layer decides whether those actions are allowed to reach the hardware.

## 9) Software Modules Added for Deployment

The deployment skeleton added in this repository introduces the following Pi-side modules:

- `robot/sensor_bridge.py`
  Reads `SensorPacket` objects from a transport
- `robot/observation_builder.py`
  Converts sensor packets into normalized policy observations
- `robot/policy_runner.py`
  Loads the trained checkpoint and performs inference
- `robot/action_bridge.py`
  Maps discrete actions to actuator commands and sends them to the low-level controller
- `robot/runtime.py`
  Runs the real-world control loop

These are deliberately separate from `env/` and `train/` so the training code remains clean.

## 10) Runtime Loop Design

The real-world runtime should run at a fixed frequency, for example:

- 10 Hz for early debugging
- 20-30 Hz if the full observation pipeline and control path are stable

Each loop iteration should:

1. Read the latest sensor packet
2. Build observation
3. Run policy inference
4. Apply safety checks
5. Send command
6. Log packet, observation summary, chosen action, and timing

The runtime does not need PettingZoo. It only needs to preserve the policy boundary.

## 11) What Should Stay Unchanged

To minimize risk, keep the following unchanged at first:

- `QNetwork` architecture
- checkpoint loading format
- discrete action IDs
- action table semantics
- observation size and order

Only after you have a stable first deployment should you consider retraining with different observations or continuous control.

## 12) Practical First Milestone

The first useful milestone is not a fully autonomous robot. It is a safe inference loop that proves the whole path works:

1. Arduino streams sensor packets over serial
2. Pi converts them into an observation vector
3. Pi runs the trained model
4. Pi sends back action commands
5. Arduino accepts commands and prints or simulates the resulting actuator targets

Only after that should you allow the Arduino to drive motors on the floor.

## 13) Suggested Implementation Path

Phase 1:

- Bring up serial communication
- Validate packet schema
- Run the Pi-side observation builder with recorded sensor packets
- Verify observation size, ranges, and normalization

Phase 2:

- Run the policy in closed loop without actuating motors
- Log chosen actions
- Compare action distributions against simulation

Phase 3:

- Enable low-speed motor actuation
- Keep watchdog and emergency stop enabled
- Log timing, packet loss, and override events

Phase 4:

- Tune observation mapping and controller gains
- Add sim noise and delays to reduce the domain gap
- Retrain if necessary

## 14) Relationship to the Existing Repo

This deployment design intentionally leaves the training flow intact:

- Simulation remains in `env/`
- Training remains in `train/`
- Real-world deployment is added through `robot/`

That separation keeps the codebase easier to reason about and reduces the chance that robot-specific logic accidentally pollutes the simulator.
