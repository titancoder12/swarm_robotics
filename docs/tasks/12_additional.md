# Additional Incremental Steps

Step-by-step changes from commit `2c9ec7085858c4712c3cd4673635cb4d3c330a45` to the current revision `130452c`.


## Step 1 — Rename the Command Center Package

```md
# Step — Rename `server/` to `mission_control/`

The repo currently uses a desktop command-center package named `server/`, but I want the naming to be more explicit and domain-specific.

Please rename the subsystem from `server/` to `mission_control/`.

Requirements:

- Rename the package directory itself.
- Update all imports across the repo.
- Update README and docs so they consistently refer to `mission_control`.
- Update task docs and project-structure docs to match.
- Do not change the runtime behavior or protocol semantics in this task.
- Keep the rename mechanically clean and easy to review.

Definition of done:

- the package is now `mission_control/`
- imports resolve correctly
- repo docs use the new name consistently
- no unrelated behavioral refactors are mixed into the rename
```

## Step 2 — Add BLE Support to Mission Control

```md
# Step — Add BLE Peripheral Support to Mission Control

Extend the current `mission_control/` subsystem so the MacBook command center can communicate with robots over BLE in addition to TCP and serial.

Requirements:

- Add an optional BLE backend to the Mission Control communications layer.
- Keep the existing line-oriented ASCII protocol (`POS`, `PHER`, `SENSE`, `PHER_RESP`) unchanged.
- Expose BLE runtime flags from `mission_control/main.py`.
- Keep the world-state and renderer transport-agnostic.
- Document the BLE setup and GATT UUIDs.
- Do not make `firmware/` depend on `mission_control/`.

Implementation guidance:

- Use a Nordic-UART-style BLE service.
- Preserve the newline-delimited text semantics over BLE.
- Make sure malformed or partial lines are handled safely.
- Keep the UI loop non-blocking.

Definition of done:

- Mission Control can optionally start a BLE peripheral
- robots can send protocol lines over BLE
- Mission Control can respond with `PHER_RESP`
- docs explain how BLE mode works
```

## Step 3 — Add a Dedicated Pi-Side BLE Client

```md
# Step — Add a Raspberry Pi BLE Client for Mission Control

The Raspberry Pi runtime in `firmware/` is where the model loop runs, so I want the Pi to communicate directly with `mission_control/` over BLE.

Please implement a dedicated Mission Control BLE client on the Pi side.

Requirements:

- Add a separate BLE helper module under `firmware/`.
- Do not mix the Mission Control BLE logic into `firmware/ant.py`, which should stay focused on the ESP32 serial link.
- Keep the transport line-oriented and ASCII so it matches Mission Control.
- Allow the Pi runtime to:
  - send `POS`
  - send `PHER`
  - send `SENSE`
  - receive and parse `PHER_RESP`
- Fail gracefully if the BLE link drops.
- Document the expected runtime flow.

Definition of done:

- `firmware/run.py` can use a dedicated BLE helper
- the Pi can talk to Mission Control over BLE without changing the protocol
- the code keeps the ESP32 link and Mission Control link conceptually separate
```

## Step 4 — Integrate Mission Control into the Pi Policy Loop

```md
# Step — Integrate Mission Control Queries into `firmware/run.py`

Now that the Pi has a BLE client, integrate Mission Control into the live policy loop in `firmware/run.py`.

Requirements:

- Maintain a simple dead-reckoned pose estimate on the Pi runtime.
- Send `POS` updates to Mission Control from the Pi loop.
- Query Mission Control with `SENSE` before policy inference.
- Parse `PHER_RESP,<id>,<p0>,<p1>,<p2>` and insert those values into the pheromone portion of the observation vector.
- Keep lidar and low-level hardware I/O local to the Pi/ESP32 path.
- Do not change the model architecture in this task.

Important:

- Keep the command-center abstraction at the level of pose/pheromone/model interaction, not at the ESP32 motor-controller layer.
- Degrade gracefully if Mission Control is unavailable.

Definition of done:

- Pi runtime can use Mission Control pheromone samples during inference
- pose is transmitted to Mission Control
- the rest of the observation remains locally assembled
```

## Step 5 — Improve Mission Control UI and Fake Robot Tooling

```md
# Step — Improve Mission Control Telemetry UI and Local Testing

The Mission Control UI is functional but still sparse. I want it to be more useful for live experiments and easier to test locally.

Please:

- add a richer telemetry/status panel
- show per-robot pose information clearly
- add short-lived visual feedback for operator control keys
- improve the fake robot harness so it is more realistic than a trivial fixed path
- keep the UI lightweight and readable

Optional useful additions:

- show lidar values in the panel
- improve fake robot motion and synthetic lidar generation

Definition of done:

- Mission Control is easier to use in a live demo
- fake-robot behavior is good enough to exercise the UI locally
- the UI changes remain cleanly separated from protocol/world-state logic
```

## Step 6 — Add Lidar Telemetry to the Mission Control Protocol

```md
# Step — Add `LIDAR` Telemetry to Mission Control

I want Mission Control to display the robot's current lidar scan for debugging and demonstration.

Please extend the existing Mission Control protocol with a `LIDAR` message.

Requirements:

- Add a `LIDAR` protocol message carrying the current 180-degree 9-bucket scan.
- Have the Pi runtime upload that telemetry.
- Parse and store it in Mission Control.
- Display it in the UI.
- Keep `POS` / `PHER` / `SENSE` / `PHER_RESP` behavior unchanged.

Do not turn this into a new sensing/control architecture. This is telemetry only.

Definition of done:

- Pi sends `LIDAR`
- Mission Control stores it
- the UI renders it in a useful way
```

## Step 7 — Fix Mission Control Pheromone Rendering Alignment

```md
# Step — Fix Heatmap Alignment in Mission Control

The Mission Control pheromone heatmap does not line up correctly with robot paths and world coordinates.

Please audit the renderer and fix the coordinate alignment issue.

Requirements:

- Ensure the pheromone heatmap uses the same world-coordinate convention as robot positions and trails.
- Verify the y-axis handling carefully.
- Keep the fix minimal and well explained.
- Update any docs if the coordinate explanation needs clarification.

Definition of done:

- pheromone deposits render in the correct place relative to robot motion
- world-coordinate rendering is internally consistent
```

## Step 8 — Let the Policy Control Pheromone Deposition

```md
# Step — Expand the Action Space to Include Pheromone Deposit

Right now pheromone deposition is runtime-driven logic. I want the policy interface itself to control whether pheromone is deposited on a step.

Please change the action contract from 9 movement-only actions to 18 actions that combine:

- movement choice
- binary deposit / no-deposit choice

Requirements:

- update the simulator action table
- add a pheromone-deposit cost in config/reward handling
- make simulator deposition conditional on the chosen action
- update rule-based helpers and policy probes to the new action table
- update the live runtime so `PHER` messages are sent only when the selected action requests deposition
- document the new action semantics

Definition of done:

- simulator and live runtime both share the new 18-action contract
- deposit is now policy-controlled rather than purely heuristic
```

## Step 9 — Improve Training Reset Randomness and Exploration Controls

```md
# Step — Fix Repeated Resets and Expose Epsilon Schedule Controls

I want training and demo behavior to be less misleading and easier to tune.

Please:

- fix repeated reset behavior so episode resets do not replay the same seeded world every time
- expose the epsilon schedule in the custom DQN trainer via CLI flags
- keep the implementation deterministic when the base seed is fixed
- document how the new seed progression and epsilon controls work

Definition of done:

- training and demo resets advance through new seeds across episodes
- epsilon schedule can be adjusted from the CLI
- docs explain the new behavior
```

## Step 10 — Add Training Graph Export

```md
# Step — Export Training and Evaluation Graphs to a Top-Level Folder

I want the training workflow to produce easy-to-browse graph artifacts outside the run directory.

Please:

- add a top-level `training_graphs/` folder
- export the existing training plots there per run
- also generate evaluation plots from the evaluation CSV
- record the graph directory in the run summary
- keep the current run-directory outputs intact

Definition of done:

- each training run gets a graph subdirectory under `training_graphs/`
- training and evaluation plots are both generated
- the summary metadata points to the graph output location
```

## Step 11 — Add Observation History to the Policy Contract

```md
# Step — Add a Short Observation History Window

The current policy is purely reactive to a single frame. I want to give it a short temporal window without introducing an RNN.

Please change the observation contract so that each agent receives a small sliding window of recent observation frames concatenated together.

Requirements:

- add a config value such as `observation_history_steps`
- keep one frame's internal feature layout unchanged
- have the environment stack the most recent frames oldest-to-newest
- update the Pi runtime to build the same stacked observation before inference
- update the rule-based baseline so it can still operate correctly
- document the new default observation shape

Definition of done:

- the default observation is no longer a single 23-D frame
- simulator and runtime agree on the stacked-history format
- docs clearly explain the new shape and ordering
```

## Step 12 — Make Pheromone Awareness Radius Explicit

```md
# Step — Make the Pheromone Sensing Radius Explicit

The simulator currently derives forward pheromone sampling from an inline formula. I want that geometry made more explicit and easier to reason about across simulator, runtime, and Mission Control.

Please:

- add an explicit config parameter for pheromone sample spacing
- route the simulator through helper methods for sample distances and the maximum awareness radius
- mirror the same calculation in the Pi runtime
- document the resulting bounded sensing radius in the observation spec

Definition of done:

- the pheromone sensing geometry is explicit in config/code/docs
- simulator and runtime share the same math
- the bounded awareness radius is documented clearly
```

## Summary

Taken together, the steps above describe a plausible path from the `2c9ec70` baseline to the current repo state:

1. rename `server` to `mission_control`
2. add BLE support on the desktop side
3. add a Pi-side BLE client
4. integrate Mission Control into the live policy loop
5. improve Mission Control UI/test harnesses
6. add lidar telemetry
7. fix heatmap alignment
8. expand the action contract to policy-controlled deposition
9. improve reset randomness and exploration tuning
10. add training graph export
11. add observation history
12. make pheromone awareness geometry explicit

That sequence matches the broad evolution visible in the codebase, even though the exact historical wording was not preserved verbatim.
