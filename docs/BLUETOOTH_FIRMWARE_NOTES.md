# Robot-Side Bluetooth Design Notes for Command Center Integration

This note reviews the current [firmware/](../firmware) code and describes the cleanest way to add robot-to-MacBook Bluetooth communication so it works with the current [mission_control/](../mission_control) protocol.

No code changes are made here. This is design guidance only.

In this repo, "robot-side Bluetooth" should be read as **Raspberry Pi-side Bluetooth**, not ESP32-side Bluetooth. The policy loop, observation assembly, and command-center interaction all live in [firmware/run.py](../firmware/run.py), so the natural place for the command-center link is the Pi.

## Current firmware structure

The current robot runtime is centered around:

- [firmware/run.py](../firmware/run.py)
  - main control loop
  - reads scan data
  - builds the observation vector
  - runs the policy
  - executes movement commands
- [firmware/ant.py](../firmware/ant.py)
  - serial transport to the ESP32 motion/sensor controller
  - line-oriented command / response flow

Important repo-specific observation:

- `run_policy.py` currently fills the pheromone channels with zeros in `build_observation(...)`.
- The command center protocol is designed to provide those pheromone channels via `PHER_RESP,<id>,<p0>,<p1>,<p2>`.
- The current firmware already uses a line-oriented transport style in `ant.py`, which is a good fit for a second line-oriented transport to the command center.

## Recommended placement of Bluetooth logic

Do **not** put command-center transport logic into [firmware/ant.py](../firmware/ant.py).

Reason:

- `ant.py` is currently the ESP32 hardware link for local motion and sensor data.
- The command center is a separate remote system with a different responsibility.
- Mixing those two transport roles into one class would make the code harder to reason about and harder to recover from partial failures.

The cleaner design is:

- keep `ESP32Robot` for robot-local serial I/O
- add a separate Raspberry Pi-side command-center transport client in `firmware/`
- let `run_policy.py` orchestrate both links

Conceptually:

```text
run_policy.py
  ├─ ESP32Robot           # local serial link to motors / sensors
  └─ CommandCenterClient  # Raspberry Pi Bluetooth link to MacBook
```

## Recommended new robot-side abstraction

The new robot-side transport should be a small class with a narrow interface, for example:

```python
class CommandCenterClient:
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def send_position(self, robot_id: str, x_cm: float, y_cm: float, heading_deg: float) -> None: ...
    def deposit_pheromone(self, robot_id: str, x_cm: float, y_cm: float, amount: float) -> None: ...
    def sense_pheromone(self, robot_id: str, x_cm: float, y_cm: float, heading_deg: float) -> tuple[float, float, float]: ...
```

Internally, it should:

- send newline-delimited ASCII
- buffer until newline when reading
- validate that the reply is `PHER_RESP` for the expected robot ID
- return `(0.0, 0.0, 0.0)` on timeout or malformed reply

That keeps the Bluetooth transport isolated and allows the policy loop to stay simple.

## Where it fits into the current control loop

The current loop in [firmware/run.py](../firmware/run.py) does this:

1. read scan data from `ESP32Robot`
2. build observation
3. run policy
4. execute action

For command-center integration, the clean robot-side flow should become:

1. read scan data from `ESP32Robot`
2. estimate current `x_cm`, `y_cm`, and `heading_deg`
3. send `POS`
4. send `SENSE`
5. receive `PHER_RESP`
6. build observation with returned pheromone samples inserted into the pheromone slots
7. run policy
8. execute action
9. if the robot decides to perform a digital deposit, send `PHER`

This means the Bluetooth interaction belongs at the observation-building boundary, not at the motion-command boundary.

## Most important current firmware gaps

The current firmware does **not** yet provide three things the Bluetooth integration will need:

1. Pose estimation

   `run_policy.py` currently does not compute real `x_cm`, `y_cm`, or `heading_deg`.
   A Bluetooth command-center client is not useful until the robot can provide at least a rough pose estimate.

2. Pheromone injection into the observation

   `build_observation(...)` currently uses zero pheromone values.
   Those slots need to be replaced with the values returned by `PHER_RESP`.

3. Deposit-event trigger

   There is no current robot-side concept of "drop digital pheromone now".
   That behavioral trigger needs to be defined before `PHER` messages are meaningful.

So Bluetooth itself is not the hardest part. The harder repo-specific issue is getting the robot runtime to produce valid pose and deposit events for the command center.

## Recommended Bluetooth transport strategy on the robot

For the Raspberry Pi-side code in `firmware/`, the simplest practical path is:

### Preferred: Bluetooth serial-style link

Use a Bluetooth transport that behaves like a serial stream:

- line-oriented
- blocking `readline()` or equivalent
- newline-delimited ASCII

Why this fits the repo:

- [firmware/ant.py](../firmware/ant.py) already uses this exact transport style
- the command center protocol is already line-oriented
- request/response timing for `SENSE -> PHER_RESP` is easier to manage with stream semantics

This can be implemented either as:

- an OS-level Bluetooth serial device
- or an RFCOMM-like socket wrapped to look like a file-like stream

### Alternative: BLE client

If the hardware requires BLE instead of serial-style Bluetooth:

- create a small BLE client class
- maintain one persistent connection to the MacBook command center
- reassemble incoming notification fragments into complete lines
- expose the same `send_position(...)`, `deposit_pheromone(...)`, and `sense_pheromone(...)` methods

If BLE is used, the class should still present a line-oriented API to the rest of the firmware.

That way `run_policy.py` does not need to care whether the underlying transport is RFCOMM, BLE, or a virtual serial device.

## Failure-handling recommendation

The Bluetooth client should fail independently from the ESP32 serial link.

Recommended behavior:

- if Bluetooth is disconnected, motion and local sensing should not crash immediately
- `sense_pheromone(...)` should return zeros on timeout or disconnect
- `POS` and `PHER` sends should degrade gracefully if the command center is unavailable
- reconnect attempts should be throttled

This matters because the robot must still remain controllable even if the command center link drops.

## Suggested message timing

To fit the current `run_policy.py` loop:

- send `POS` once per control iteration
- send `SENSE` immediately before inference
- wait only a short bounded time for `PHER_RESP`
- send `PHER` only on explicit deposit events, not every frame

This keeps Bluetooth traffic aligned with the existing low-rate control loop (`--hz`, default `2.0`) and avoids unnecessary link load.

## Suggested parsing contract on the robot

The robot-side Bluetooth client should enforce:

- outgoing messages are always terminated with `\n`
- incoming replies are read until `\n`
- only `PHER_RESP,<robot_id>,<p0>,<p1>,<p2>` is accepted as a valid `SENSE` reply
- replies for the wrong robot ID are ignored
- invalid replies fall back to zero pheromone values

This protects the policy loop from malformed or stale transport data.

## Recommended implementation split

If this is implemented later, the clean repo-level split would be:

- [firmware/ant.py](../firmware/ant.py)
  - unchanged role: ESP32 serial transport only
- new firmware transport helper
  - command-center Bluetooth client only
- [firmware/run.py](../firmware/run.py)
  - owns both clients and merges local sensing with remote pheromone sensing

This keeps hardware control and command-center networking separate.

## Bottom-line recommendation

The best robot-side design for this repo is:

- keep `ESP32Robot` unchanged in role
- add a second, separate Raspberry Pi-side Bluetooth client for the command center
- integrate it in `run_policy.py` right before policy inference
- keep the wire protocol line-oriented and identical to the current command-center protocol
- treat Bluetooth as transport only, not as a change to the policy or observation contract

If you implement this later, prefer a serial-style Bluetooth stream first. It matches both the current firmware style and the current command-center protocol most naturally.
