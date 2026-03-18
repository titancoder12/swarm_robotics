# Pi Migration Guide

This document is for the robot developer who currently has working Raspberry Pi code similar to `AntSwarmFirmware/ant.py` and wants to migrate gradually toward running the learned model from this repository.

The goal is explicit:

- keep today's working robot behavior intact
- let the developer clone this repo onto the Raspberry Pi and continue working immediately
- introduce model-based control in small, low-risk steps

## 1) Compatibility Baseline

The file `pi/ants.py` is included as a compatibility-first copy of the current Raspberry Pi control script.

Its purpose is not to be architecturally elegant. Its purpose is to preserve the current workflow so the robot developer loses nothing by switching to this repo as the working tree.

`pi/ants.py` currently does the same basic job as the original script:

- connect to the ESP32 over `/dev/serial0`
- read JSON scan lines
- choose a movement direction with simple handcrafted free-space logic
- send movement commands like `move(...)`, `turn(...)`, `stop(...)`, and `brake(...)`

This gives the robot developer a safe baseline:

- first prove the robot still behaves the same
- only then start introducing shared modules and model inference

## 2) What Should Not Change First

At the beginning of migration, do not change all of these at once:

- the ESP32 serial protocol
- the known-good movement commands
- the service launch path
- the robot's safety behavior
- the control frequency

The first objective is behavioral continuity, not architectural purity.

## 3) Migration Strategy

Use a phased migration.

### Phase 0: Keep the Existing Behavior

Run `pi/ants.py` as-is and confirm:

- serial connection still works
- scan lines still arrive
- the robot still chooses free-space motions
- command/ack timing is still acceptable

Do not introduce the model yet.

### Phase 1: Share the Low-Level Pi Serial Client

Once the compatibility script is working, begin replacing duplicated code with the reusable wrapper in `pi/esp32_robot.py`.

At this phase:

- keep the same rule-based action selection
- keep the same ESP32 command protocol
- keep the same scan parsing assumptions

Only refactor the transport layer so the codebase starts using shared Pi-side helpers without changing robot behavior.

### Phase 2: Build `SensorPacket` from Real Sensor Data

Introduce `pi/esp32_sensor_adapter.py` and `robot.messages.SensorPacket`.

At this phase the robot still does not need to act on the model.

The objective is to prove that the Pi can convert real serial data into the sensor structure expected by the deployment modules.

Important:

- this is where observation correctness starts to matter
- if the model was trained on a 19-dimensional observation, the real sensor pipeline must preserve the same feature order and scaling

Initially, some channels may remain placeholders while the developer brings up the pipeline.

### Phase 3: Build and Log Observations Without Using Them for Control

Introduce:

- `env/config.py`
- `robot/observation_builder.py`

Now the Pi can create the observation vector that the model expects.

At this phase:

- continue using the current handcrafted motion policy for real control
- log the generated observation vectors to disk
- inspect whether the ranges, normalization, and ordering look sane

This is the safest time to validate the model input contract.

### Phase 4: Load the Checkpoint and Run Inference in Shadow Mode

Introduce:

- `models/q_network.py`
- `robot/policy_runner.py`

Now the Pi can:

- load the checkpoint
- create `obs_tensor`
- run the model
- compute the action ID

But still do not let the model drive the robot yet.

Instead:

- compare the model's chosen action with the handcrafted action
- log both side by side
- identify obvious mismatches

This step is critical because it separates "the model runs" from "the model is safe and useful."

### Phase 5: Translate Model Actions Into Existing ESP32 Commands

Introduce:

- `robot.action_bridge.CommandMapper`
- `pi/esp32_action_bridge.py`

Now the model's discrete action can be mapped into the same serial command vocabulary the robot already uses.

At this phase:

- first test at very low speed
- keep hardware stop behavior intact
- keep manual override available
- verify that action-to-motion mapping is physically reasonable

### Phase 6: Switch Control From Rule-Based to Model-Based

Only after the previous steps are stable should the developer replace the handcrafted chooser in `pi/ants.py` with the learned-policy path in `pi/run_policy.py`.

The transition should look like this:

- old control path:
  `scan -> handcrafted free-space angle -> robot.move(...)`

- new control path:
  `scan -> SensorPacket -> ObservationBuilder -> PolicyRunner -> CommandMapper -> ESP32ActionBridge`

This should be a deliberate cutover, not an early experiment.

## 4) What the Robot Developer Needs to Learn

The developer does not need to learn the whole simulator or training stack.

They mainly need to understand five things:

1. `pi/ants.py`
   This preserves today's behavior.

2. `pi/esp32_robot.py`
   This keeps the existing serial command vocabulary.

3. `robot/messages.py`
   This defines the sensor and action data structures shared by the deployment code.

4. `robot/observation_builder.py`
   This converts real sensor data into the model's observation vector.

5. `models/q_network.py` and `robot/policy_runner.py`
   This loads the saved checkpoint and runs inference.

That is enough to begin migration.

## 5) What Will Still Need Real Hardware Work

The migration is not complete just because the code runs.

The following parts still require robot-specific tuning:

- serial timing
- scan angle interpretation
- lidar/range bucketing
- heading estimation
- speed estimation
- target and neighbor feature definitions
- model action to movement mapping
- safety overrides

Those are hardware integration problems, not just software import problems.

## 6) Recommended First Task for the Robot Developer

The first concrete task should be:

1. clone this repo on the Raspberry Pi
2. run `pi/ants.py` or install `pi/ants.service`
3. confirm the robot behaves the same as before
4. only then begin the migration phases above

This keeps the migration low-risk and prevents a tooling refactor from being confused with an RL integration problem.

## 7) Service Compatibility

To mirror the current Raspberry Pi deployment style, this repo also includes `pi/ants.service`.

It is a compatibility-first systemd unit intended to match the existing launch pattern:

- runs as user `pi`
- starts the Python control script at boot
- restarts automatically on failure
- writes logs to the systemd journal

Before using it on a real Raspberry Pi, the developer should verify:

- the repo checkout path matches the `WorkingDirectory`
- the `ExecStart` path matches where this repo is cloned
- `python3` and required packages are installed on the Pi

If the repo is cloned somewhere other than `/home/pi/ant`, the service file should be updated accordingly.
