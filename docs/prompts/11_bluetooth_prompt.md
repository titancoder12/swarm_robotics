# Codex Prompt — Add Bluetooth Transport to the Command Center

You are working in the existing repository and must extend the current [mission_control/](/Users/christopherlin/dev/cwsf2026/sim/mission_control/) subsystem so that the physical robots and the MacBook command center can communicate over **Bluetooth**.

## High-level goal

Add Bluetooth as a **new transport backend** for the command center so that:

1. each physical robot can maintain a direct Bluetooth link to the MacBook,
2. robots can send `POS`, `PHER`, and `SENSE` messages over that link,
3. the command center can respond with `PHER_RESP`,
4. and the existing world-state, pheromone, and rendering logic remain unchanged.

This is a transport-layer extension only.

Do **not** redesign the protocol, the pheromone logic, or the RL observation contract.

For this repo, interpret "robot-side" to mean the Raspberry Pi runtime in [firmware/](/Users/christopherlin/dev/cwsf2026/sim/firmware/), not the low-level ESP32 motor/sensor controller. The Pi is where the policy loop runs and where interaction with the model and observation vector happens.

---

## Current repo context

The current command center already has:

* line-based ASCII protocol parsing in [mission_control/comms/protocol.py](/Users/christopherlin/dev/cwsf2026/sim/mission_control/comms/protocol.py)
* TCP and serial transport workers in [mission_control/comms/receiver.py](/Users/christopherlin/dev/cwsf2026/sim/mission_control/comms/receiver.py)
* transport-agnostic dispatch logic in [mission_control/main.py](/Users/christopherlin/dev/cwsf2026/sim/mission_control/main.py)
* simulator-aligned pheromone sampling in [mission_control/core/pheromone_field.py](/Users/christopherlin/dev/cwsf2026/sim/mission_control/core/pheromone_field.py)

The current codebase does **not** implement a Bluetooth-specific backend yet.

Your job is to add one cleanly.

---

## Architectural constraints

1. **Do not change the robot-facing message grammar.**

   Keep the existing newline-delimited ASCII protocol exactly as it is.

2. **Do not change pheromone semantics.**

   `PHER_RESP,<id>,<p0>,<p1>,<p2>` must remain fully compatible with the existing RL observation vector and simulator-aligned sampling logic.

3. **Do not make `firmware/` depend on `mission_control/`.**

   Any robot-side notes should be documentation or pseudocode only unless explicitly requested otherwise.

4. **Keep Bluetooth isolated to the transport layer.**

   The new backend should plug into the same `on_line(connection_label, line) -> list[str]` flow already used by TCP and serial.

5. **Preserve multi-robot direct-link behavior.**

   Each robot must have its own independent connection and request/response path.

---

## Transport design requirement

Bluetooth must be implemented as **just another transport backend** under the command center communications layer.

Target architecture:

```text
Robot <-> Bluetooth link <-> Bluetooth worker <-> handle_line(...) <-> WorldState
```

The Bluetooth worker must:

* own one robot connection
* receive text data from that connection
* buffer partial incoming chunks until newline-delimited messages are complete
* pass each parsed line into the existing command-center line handler
* send any returned response lines back over the same Bluetooth connection

This must behave analogously to the current `TCPClientWorker` and `SerialWorker`.

---

## Preferred implementation strategy

Implement the Bluetooth support in a way that fits macOS and the current repo cleanly.

### Option A — Preferred first if hardware supports it

Treat Bluetooth as a **serial-style transport**:

* pair each robot with the MacBook
* use a Bluetooth serial device if the OS exposes one
* integrate it through the existing serial-style worker model

This is the lowest-risk path because it matches the current line-oriented transport assumptions very closely.

### Option B — If true BLE is required

Implement a dedicated BLE worker:

* one BLE connection per robot
* one background worker/task per robot
* one read buffer per robot
* one write path per robot
* notification / characteristic handling that reassembles complete newline-delimited lines before dispatch

If BLE packets fragment messages, the implementation must handle that correctly.

---

## Protocol requirements

Keep the existing line protocol exactly as-is.

Robot -> command center:

```text
POS,<id>,<x_cm>,<y_cm>,<heading_deg>
PHER,<id>,<x_cm>,<y_cm>,<amount>
SENSE,<id>,<x_cm>,<y_cm>,<heading_deg>
```

Command center -> robot:

```text
PHER_RESP,<id>,<p0>,<p1>,<p2>
```

Requirements:

* messages remain ASCII
* messages remain newline-delimited
* malformed lines are ignored safely and logged
* `SENSE` requests must receive replies on the same Bluetooth connection that sent the request
* robot IDs remain explicit in every message

Do **not** introduce:

* binary framing
* protobuf / JSON transport redesign
* a multiplexed shared Bluetooth bus for multiple robots

---

## Multi-robot requirements

The Bluetooth design must support multiple robots concurrently.

Specifically:

* one independent Bluetooth connection per robot
* no cross-talk between robots
* one worker/thread/task per connection
* connection loss from one robot must not break others
* reconnection logic should be considered
* render loop must remain non-blocking

The design should preserve the same request/response isolation currently provided by the threaded TCP and serial workers.

---

## macOS-focused implementation details

The command center runs on a MacBook.

Document the Bluetooth design with macOS in mind:

* how robots are identified for connection setup
* how the app maps a Bluetooth device to a robot ID
* whether pairing is expected outside the app or in-app
* whether the implementation assumes a serial device path or a BLE library
* how the connection lifecycle is managed
* how reconnects are handled

If a dependency is needed for BLE, keep it isolated to the command-center transport layer and document it clearly.

---

## Suggested code shape

You may adjust naming, but keep the design in this spirit:

```text
mission_control/
  comms/
    protocol.py
    receiver.py
    bluetooth.py          # optional if a separate module is cleaner
```

Possible additions:

* a `BluetoothWorker`
* a Bluetooth connection config block in [mission_control/config.py](/Users/christopherlin/dev/cwsf2026/sim/mission_control/config.py)
* CLI flags in [mission_control/main.py](/Users/christopherlin/dev/cwsf2026/sim/mission_control/main.py) to configure Bluetooth devices or enable Bluetooth mode

But the protocol parser and world-state logic should remain transport-agnostic.

---

## Robot-side contract

Do **not** implement firmware changes unless explicitly asked in a later prompt.

However, document the expected robot-side transport behavior:

1. establish a direct Bluetooth link to the MacBook
2. periodically send `POS`
3. send `PHER` on digital deposit events
4. send `SENSE` immediately before action selection
5. read a single `PHER_RESP` line
6. insert `p0,p1,p2` into the observation vector

Provide robot-side pseudocode that is transport-agnostic except for the read/write calls.

For clarity:

* the Bluetooth client should live on the Raspberry Pi side of the robot runtime
* the ESP32 should remain focused on low-level motion and sensor control unless there is a separate hardware reason to move Bluetooth lower in the stack
* the Raspberry Pi should be the layer that exchanges `POS`, `PHER`, `SENSE`, and `PHER_RESP` with the command center because it owns observation assembly and model inference in this repo

---

## Error-handling requirements

The Bluetooth transport design must handle:

* partial line delivery
* transient disconnects
* malformed messages
* write failures
* duplicate reconnect attempts
* unknown robots that still send valid `SENSE` coordinates

Logging should be good enough to debug live connection issues during a physical experiment.

---

## Performance requirements

The transport design must support:

* multiple robots at roughly 5–20 Hz message rates
* low-latency `SENSE -> PHER_RESP` turnaround
* a non-blocking PyGame render loop

Bluetooth handling must not stall the UI thread.

---

## Documentation requirements

When implementing this prompt, also update the relevant docs to explain:

* which Bluetooth transport approach was chosen
* how to configure robot connections
* any new command-center CLI flags
* the expected robot-side message flow
* any macOS-specific pairing or setup assumptions
* how Bluetooth maps onto the existing command-center transport architecture

Include a short rationale for why the chosen approach was used instead of redesigning the protocol.

---

## Non-goals

Do **not**:

* redesign the Mission Control pheromone service
* change the RL observation format
* add centralized planning
* push command-center logic into `firmware/`
* replace the existing TCP or serial backends unless necessary

This is an additive transport feature.

---

## Deliverables

When this prompt is executed later, the deliverables should include:

1. Bluetooth transport support in the command center
2. documentation for setup and usage
3. any necessary command-center-only dependency updates
4. a concise summary of files added or changed
5. explicit notes about assumptions and limitations

---

## Definition of done

This task is complete when:

* the command center can communicate with one or more robots over Bluetooth,
* robots can send `POS`, `PHER`, and `SENSE` over Bluetooth,
* the command center returns `PHER_RESP` over the same Bluetooth connection,
* the existing world-state and pheromone logic continue to work unchanged,
* and the Bluetooth design is documented clearly enough for physical deployment on a MacBook.

When done, explain:

* which Bluetooth transport model was chosen
* any macOS-specific assumptions
* any dependency tradeoffs
* any remaining operational risks
