# Sim-to-Real Deployment Architecture

This repository now uses a direct robot runtime centered on [firmware/run.py](../firmware/run.py).

## 1) Active Deployment Boundary

The learned policy boundary is unchanged:

- input: one normalized observation vector
- output: one discrete action ID

On the physical robot, the active path is:

`serial scan lines -> local observation builder -> QNetwork inference -> direct robot commands`

Concretely:

- [firmware/ant.py](../firmware/ant.py)
  handles serial communication and the low-level command vocabulary
- [firmware/run.py](../firmware/run.py)
  builds the observation, loads the checkpoint, predicts an action, and executes it
- [models/q_network.py](../models/q_network.py)
  defines the network architecture expected by the custom PyTorch checkpoints

The older `pi/` and `robot/` abstraction layers are no longer part of the live deployment path.

## 2) Runtime Flow

Each loop iteration in the active robot runtime does this:

1. Read scan lines from the ESP32 over serial.
2. Bucketize the scan into fixed lidar rays.
3. Convert distances to meters and normalize them against `--lidar-max-range-mm`.
4. Fill the remaining observation slots with the currently supported values.
5. Load the observation into `QNetwork` and take `argmax` over Q-values.
6. Map the chosen action ID into `turn`, `move`, or `stop`.
7. Send the command directly back through `ant.py`.

## 3) Observation Contract

The real robot must still preserve the policy contract used during training:

- same observation length
- same feature order
- same normalization and clipping
- same action semantics

Important current limitation:

- [firmware/run.py](../firmware/run.py) currently derives lidar from real scan data
- several non-lidar channels are still placeholders, such as target, nest, neighbor, speed, and pheromone values

That means the runtime is operational, but full sim-to-real fidelity still depends on future sensor integration or retraining.

## 4) Action Contract

The action space remains the trained discrete 18-action table:

- throttle in `[-1, 0, 1]`
- turn in `[-1, 0, 1]`
- pheromone deposit bit in `[0, 1]`

The physical mapping is implemented directly in [firmware/run.py](../firmware/run.py) using:

- `--turn-step-deg`
- `--move-distance-mm`
- `--reverse-distance-mm`

So the deployed robot keeps the same policy output semantics even though the transport and execution code are now simpler. When the action's deposit bit is enabled, the runtime can also emit the corresponding pheromone-side effect path where supported.

## 5) Safety Expectations

Safety must still sit outside the learned policy.

On the Pi/runtime side:

- reject invalid or stale sensor data
- stop on observation failures
- clamp motion commands to safe ranges
- preserve manual stop and watchdog behavior

On the ESP32 side:

- stop motors if command traffic is lost
- enforce low-level current, PWM, or motion limits
- honor emergency-stop conditions

## 6) Deployment Files

The minimum robot-side set is now:

- [firmware/ant.py](../firmware/ant.py)
- [firmware/run.py](../firmware/run.py)
- [firmware/ant.service](../firmware/ant.service)
- [models/q_network.py](../models/q_network.py)
- [checkpoints/](../checkpoints)

Plus Python dependencies:

- `torch`
- `numpy`
- `pyserial`

## 7) Practical Recommendation

If your goal is to run one physical robot today, treat [firmware/run.py](../firmware/run.py) as the single source of truth for deployment.

If you later need multiple hardware backends or a more generic deployment framework, reintroduce modular layers only after the direct path is stable and calibrated.
