# Sim-to-Real Deployment Architecture

This repository now deploys the physical robot through [firmware/run.py](../firmware/run.py).

The live robot stack no longer uses the older DQN-only deployment description. The current runtime supports the newer recurrent MAPPO checkpoint layout and optional Mission Control integration.

## 1) Active Robot Runtime

The physical robot loop is:

`ESP32 serial scan -> local pose/observation build -> MAPPO actor inference -> motor command -> optional Mission Control exchange`

The key files are:

- [firmware/ant.py](../firmware/ant.py)
  handles the serial link to the ESP32 and the low-level robot command vocabulary
- [firmware/run.py](../firmware/run.py)
  loads the policy, builds observations, runs inference, and drives the robot
- [firmware/bluetooth.py](../firmware/bluetooth.py)
  provides the BLE transport used for Mission Control integration
- [firmware/command_center_client.py](../firmware/command_center_client.py)
  provides the direct TCP Mission Control client
- [firmware/relay_client.py](../firmware/relay_client.py)
  provides the internet relay transport

## 2) Current Policy Loader

The robot runtime now supports two checkpoint families:

- current path: recurrent MAPPO actor checkpoints such as `checkpoints/mappo_g/latest`
- legacy fallback: older DQN-style checkpoints such as `agent_0.pt` or `shared.pt`

For the current robot deployments, the expected checkpoint set is:

- `actor.pt`
- `critic.pt`
- `trainer.pt`
- `metadata.json`

under a directory like [checkpoints/mappo_g/latest](../checkpoints/mappo_g/latest).

`firmware/run.py` auto-detects the current MAPPO layout and loads the actor for greedy deployment.

## 3) Runtime Loop

Each control iteration in [firmware/run.py](../firmware/run.py) does this:

1. Read scan lines from the ESP32 over serial.
2. Bucketize the scan into fixed lidar rays.
3. Dead-reckon the robot pose locally from commanded motion.
4. Send `POS` and `LIDAR` to Mission Control if a transport is enabled.
5. Query Mission Control with `SENSE` to retrieve pheromone samples.
6. Assemble the observation vector.
7. Run greedy policy inference.
8. Execute the chosen movement on the robot.
9. Optionally send `PHER` if the chosen action includes a deposit bit.

If Mission Control is enabled, the transport-specific operator and connection
instructions live in [MISSION_CONTROL.md](./MISSION_CONTROL.md).

## 4) Observation Contract

The physical runtime still has to preserve the learned policy contract:

- same observation length
- same feature order
- same normalization and clipping
- same action semantics

The current checkpoint metadata for the main deployment path indicates:

- algorithm: `recurrent_mappo_gru`
- observation dimension: `69`
- action dimension: `18`

Important practical limitation:

- lidar and Mission Control pheromone channels are live
- several non-lidar/non-pheromone channels are still approximated or placeholder-like compared with full simulation

So the robot runtime is operational, but sim-to-real fidelity is still constrained by the quality of the real observation reconstruction.

## 5) Action Contract

The action space remains the learned discrete 18-action table:

- throttle in `[-1, 0, 1]`
- turn in `[-1, 0, 1]`
- pheromone deposit bit in `[0, 1]`

The physical mapping is applied in [firmware/run.py](../firmware/run.py) using parameters such as:

- `--turn-step-deg`
- `--move-distance-mm`
- `--reverse-distance-mm`

## 6) Mission Control Transports

The robot can talk to Mission Control through three transports:

1. Internet relay
2. direct TCP
3. Bluetooth

The relay and TCP paths remain the most reliable operational choices today.

The Bluetooth path also remains available, but the full operator-side setup and
bring-up instructions now live in [MISSION_CONTROL.md](./MISSION_CONTROL.md).

## 7) Minimum Robot-Side Files

The minimum robot-side set is now:

- [firmware/ant.py](../firmware/ant.py)
- [firmware/run.py](../firmware/run.py)
- [firmware/bluetooth.py](../firmware/bluetooth.py)
- [firmware/ant.service](../firmware/ant.service)
- [checkpoints/](../checkpoints)

Plus Python dependencies such as:

- `torch`
- `numpy`
- `pyserial`
- `bless` for BLE mode

On the Mission Control side, BLE mode additionally requires:

- `bleak`

## 8) Practical Recommendation

If you are deploying one or more real robots today, treat [firmware/run.py](../firmware/run.py) as the robot-side source of truth and [mission_control/main.py](../mission_control/main.py) as the operator-side source of truth.

If you need the most dependable demo transport, prefer relay or TCP. If you want to continue validating the local wireless path without relying on shared IP networking, use the Bluetooth workflow documented in [MISSION_CONTROL.md](./MISSION_CONTROL.md).
