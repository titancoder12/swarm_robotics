# Step 49: Add Scripted Physical Robot Test Mode

Work in the existing swarm robotics simulation and physical robot runtime codebase.

Task:
Add a scripted test mode to the Raspberry Pi robot runtime so a physical robot can execute simple deterministic movement patterns without relying on a learned policy.

Main goals:
1. Make hardware bring-up and motor calibration easier.
2. Reuse the existing action-table and ESP32 command path.
3. Keep Mission Control telemetry working during scripted tests.

Goals:
- Allow a robot to run patterns such as `straight,left,straight,right`.
- Support repeatable movement segments with configurable step counts.
- Keep the implementation local to the physical runtime unless a small helper is clearly useful.
- Preserve the existing policy, heuristic, hybrid, Mission Control, camera, and serial behavior.

--------------------------------
CONTEXT / BACKGROUND
--------------------------------

Current repo context:
- `firmware/run.py` is the Raspberry Pi-side runtime that reads ESP32 scan data, builds observations, runs policy or heuristic control, sends motor commands, and optionally talks to Mission Control.
- `firmware/ant.py` owns the low-level ESP32 serial command interface.
- `execute_action(...)` in `firmware/run.py` already converts an 18-action discrete action id into ESP32 movement commands.
- `action_id_from_controls(...)` in `firmware/run.py` can map `(throttle, turn, deposit)` triples back into valid action ids.

Important current assumptions:
- The runtime action contract is still the simulator-compatible 18-action table:
  - throttle in `{-1, 0, 1}`
  - turn in `{-1, 0, 1}`
  - deposit in `{False, True}`
- Mission Control should receive normal `POS`, `LIDAR`, optional `TARGET`, `SENSE`, and optional `PHER` traffic during scripted mode when a Mission Control transport is enabled.
- Scripted mode is for physical robot testing, not for training.

--------------------------------
IMPORTANT CONSTRAINTS
--------------------------------

- Do not change the ESP32 serial command grammar in `firmware/ant.py`.
- Do not change the simulator action space.
- Do not send raw policy outputs to Mission Control.
- Do not bypass `execute_action(...)`; scripted mode should exercise the same command path as policy mode.
- Keep the learned policy loading path intact unless avoiding it in scripted mode is explicitly cleaner and safe.
- Keep all new CLI flags backward compatible.
- Do not rewrite the physical runtime architecture.

--------------------------------
PART 1 — Scripted Control Mode
--------------------------------

Problem / Goal:
- The current physical runtime supports `policy`, `heuristic`, and `hybrid` control modes.
- Hardware bring-up needs a deterministic mode that can run a known motion pattern for testing motor direction, turn calibration, dead-reckoned pose, and Mission Control visualization.

Required changes:
- Add a new control mode, preferably `scripted`, to `--control-mode`.
- Add CLI flags for scripted control, such as:
  - `--scripted-pattern`
  - `--scripted-segment-steps`
  - `--scripted-repeat`
  - optional `--scripted-deposit`
- Implement a helper that converts pattern tokens into action ids using `action_id_from_controls(...)`.
- Supported first-pass tokens should include at least:
  - `straight`
  - `left`
  - `right`
  - `reverse`
  - `stop`
- A pattern like `straight,left,straight,right` should run each token for `--scripted-segment-steps` control iterations before advancing to the next token.

Preferred implementation:
- In `firmware/run.py`:
  - keep parsing and action selection simple
  - make scripted control deterministic
  - use existing `cfg.num_actions` and action helpers
- Represent scripted actions as normal action ids so debug output, pose update, deposit handling, and Mission Control behavior remain consistent.

Important:
- `left` and `right` should be defined clearly:
  - either turn-in-place with throttle `0`
  - or forward-turn with throttle `1`
- Choose one behavior conservatively and document it in code comments or help text.
- Prefer turn-in-place for initial hardware calibration unless there is a strong reason to do otherwise.

--------------------------------
PART 2 — Runtime Behavior And Safety
--------------------------------

Problem / Goal:
- Scripted mode should be useful on real robots without surprising operators.

Required changes:
- Keep `--max-steps` working so scripted tests can stop automatically.
- Preserve `--hz`, `--scan-duration`, `--turn-step-deg`, `--move-distance-mm`, and `--reverse-distance-mm`.
- Preserve debug output so each step prints the selected action, throttle, turn, deposit bit, pose estimate, and timing when `--debug` is enabled.
- Ensure the robot connection still closes cleanly on interrupt or error.

Preferred behavior:
- If `--scripted-repeat` is enabled, loop the pattern until `--max-steps` or user interruption.
- If `--scripted-repeat` is disabled, stop or brake after the pattern finishes.
- If an unknown pattern token is provided, fail fast with a clear error message before connecting to the robot.

Also consider:
- A named shortcut such as `--scripted-pattern square` could expand to `straight,left,straight,left,straight,left,straight,left`.
- A named shortcut such as `zigzag` could expand to `straight,left,straight,right`.
- Keep this optional; a comma-separated token pattern is enough for the first implementation.

--------------------------------
PART 3 — Launcher And Documentation
--------------------------------

Required changes:
- Update `firmware/run_bluetooth.sh` to expose a convenient scripted-control launcher option if it fits the existing launcher style.
- Consider updating `firmware/run_relay.sh` similarly if relay-based Mission Control tests are expected.
- Add a short documentation note in the most relevant runtime doc, likely `docs/SimToReal.md` or `docs/MISSION_CONTROL.md`.
- Update `docs/PROJECT_LOG.md` with a concise summary of the change.

Example commands to support:

```bash
python firmware/run.py \
  --control-mode scripted \
  --scripted-pattern straight,left,straight,right \
  --scripted-segment-steps 4 \
  --max-steps 16 \
  --port /dev/ttyUSB0 \
  --debug
```

```bash
bash firmware/run_bluetooth.sh \
  --scripted-control \
  --robot-id robot_0 \
  -- --scripted-pattern straight,left,straight,right --scripted-segment-steps 4 --max-steps 16
```

--------------------------------
PART 4 — Validation Requirements
--------------------------------

After implementation:

1. Run syntax/import checks for modified Python files.

2. Run a non-hardware sanity check if possible by invoking argument parsing or helper functions without opening serial.

3. If hardware is unavailable, document the exact command that should be run on a Raspberry Pi with the robot attached.

4. Verify that scripted mode:
- does not require policy logits or Q-values
- uses the existing action id to ESP32 command path
- still allows Mission Control telemetry when a transport is configured
- exits cleanly after `--max-steps` when provided

--------------------------------
PART 5 — Non-goals
--------------------------------

- Do not implement a full motion-planning language.
- Do not add closed-loop navigation.
- Do not add wheel odometry or localization in this task.
- Do not change training behavior.
- Do not change the simulator environment.
- Do not refactor Mission Control.
