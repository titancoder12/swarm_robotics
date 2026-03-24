# Codex Prompt — Build a Pygame Command Center for Physical Swarm Robots

You are working in the existing repository and must **add a new `command_center/` subsystem** for a real-time desktop application that runs on a MacBook and visualizes a physical swarm robotics experiment.

## High-level goal

Build a **Pygame-based command center** that receives robot updates over Bluetooth/serial, tracks robot positions on a 2D grid, records digital pheromone deposition events, and renders:

1. the live location of each robot,
2. a pheromone heat map,
3. and optionally the path each robot has traveled.

This is **not** a centralized planner. The command center is only a:

* communication receiver,
* world-state tracker,
* pheromone manager,
* and live visualization tool.

The robots themselves remain responsible for their own movement and local decision-making.

---

## Important architectural constraints

1. **Do not bloat or disrupt the existing Pi runtime in `firmware/`.**

   * The Raspberry Pi robot workflow must remain isolated.
   * Do not make `firmware/` depend on the command center.
   * Do not introduce imports from `command_center/` into Pi runtime code.

2. **Keep the command center in the same repo, but as a cleanly separated subsystem.**

3. **Use Python + Pygame** for the first version.

   * Reuse existing project conventions where appropriate.
   * Keep the UI lightweight and visually clear.

4. **Separate core logic from rendering.**

   * The pheromone field, robot registry, trail storage, and protocol parsing should not be entangled with the Pygame drawing loop.

5. **Design for future extraction.**

   * The subsystem should be modular enough that it could later be moved into a separate repo if needed.

---

## Functional requirements

The command center should:

### 1. Receive robot position updates

Each robot sends its coordinate relative to the nest.
The command center must ingest messages such as position updates.

### 2. Receive pheromone drop events

When a robot drops digital pheromone, it notifies the command center.
The command center updates its internal pheromone grid accordingly.

### 3. Maintain a world model

Track per-robot state including at minimum:

* robot ID
* current x/y position
* last update timestamp

Optionally support:

* heading
* carrying-food state
* connection status

### 4. Maintain a pheromone field

Represent pheromone as a 2D grid / heatmap.
Support:

* deposit at a reported coordinate
* rendering intensity
* optional decay over time

Make the grid resolution configurable.

### 5. Optionally maintain trails

Keep a bounded history of each robot’s recent positions and optionally render the path.
This should be toggleable.

### 6. Render a real-time display

The Pygame window should show:

* arena/grid
* nest location
* robot markers
* robot IDs
* pheromone heatmap
* optional trails
* a simple telemetry/status panel

### 7. Support basic operator controls

Implement lightweight keyboard controls such as:

* pause/unpause
* clear/reset pheromone
* toggle trails on/off
* toggle pheromone rendering on/off
* quit cleanly

---

## Protocol requirements

Implement a simple, explicit message protocol for robot -> command center communication.

Use a **very simple line-based text protocol** for v1, not something heavy.
For example, messages like:

```text
POS,2,34.5,18.2
PHER,2,34.5,18.2,1.0
```

Where:

* `POS,<robot_id>,<x>,<y>` means robot position update
* `PHER,<robot_id>,<x>,<y>,<amount>` means pheromone deposit

If helpful, support optional extension fields later, but keep v1 simple and robust.

Please define and document:

* exact message grammar
* how malformed lines are handled
* coordinate units
* origin convention (nest as origin)
* grid-cell quantization rules for pheromone deposits

---

## Suggested subsystem structure

Create something along these lines:

```text
command_center/
  __init__.py
  main.py
  config.py
  core/
    __init__.py
    robot_registry.py
    pheromone_field.py
    trail_store.py
    world_state.py
  comms/
    __init__.py
    protocol.py
    receiver.py
  ui/
    __init__.py
    renderer.py
    panels.py
    colors.py
```

You may adjust naming if needed, but preserve the separation of concerns.

---

## Data model guidance

### Robot registry

Maintain a registry keyed by robot ID.
Each robot record should include at least:

* x
* y
* last_seen

### Pheromone field

Use a 2D structure, ideally NumPy-backed if that is already acceptable in this project.
Support:

* configurable width/height
* configurable cell size
* deposit()
* decay_step()
* coordinate_to_cell()
* get_intensity()

### Trail storage

Maintain a bounded recent history per robot.
Do not allow trails to grow unbounded.

---

## Rendering guidance

The display does not need to be flashy, but it should be clear and science-fair friendly.

Recommended rendering approach:

* dark or neutral background
* grid overlay
* nest marker clearly visible
* robots drawn distinctly and labeled
* pheromone shown as semi-transparent heatmap cells
* optional trail lines or dots per robot
* side panel for simple status/telemetry

Avoid overengineering the UI.
This is a live visualizer, not a full desktop application.

---

## Integration constraints

* Do not break existing training, simulation, or firmware code.
* Do not refactor unrelated parts of the repo unless necessary.
* Keep dependencies minimal.
* If new dependencies are needed, isolate them appropriately.
* If the repo already has requirements organization, extend it cleanly.

If appropriate, add a dedicated requirements file or document any command-center-specific dependencies.

---

## Deliverables

Please implement the feature and also provide:

1. **A concise README or markdown doc** for the command center covering:

   * purpose
   * architecture
   * message protocol
   * how to run it
   * keyboard controls
   * configuration options

2. **Clear inline comments** in non-obvious parts of the code.

3. **A small example or test harness** that can simulate a few robot messages without requiring real Bluetooth hardware, so the display can be tested locally.

   * This can be a simple fake message injector, mock receiver, or replay script.

4. **A short summary of what files were added/changed** and why.

---

## Implementation priorities

Prioritize in this order:

1. correctness and clean architecture
2. minimal disruption to existing code
3. fast local testability
4. visually clear rendering
5. extensibility

---

## Non-goals for this prompt

Do **not** build:

* centralized path planning
* cloud services
* a web frontend
* a database-backed system
* a complicated settings GUI
* deep integration into the Pi firmware runtime

Keep the first version focused.

---

## Definition of done

This task is complete when:

* a developer can run the command center locally,
* inject or receive robot position / pheromone messages,
* see robots rendered on a grid,
* see pheromone deposits appear as a heatmap,
* optionally see robot trails,
* and understand the subsystem through the added documentation.

When done, explain any assumptions you made.

---

## Bidirectional pheromone query (UPDATED REQUIREMENT — AUTHORITATIVE SERVER)

The command center is the **authoritative digital pheromone server** for all robots.

### System role clarification

* The MacBook command center maintains the **single source of truth** pheromone field.
* Robots do **not** maintain their own pheromone maps.
* Robots must **query the command center** to obtain pheromone observations.
* The command center must return values that are **fully compatible with the existing RL observation vector**.

---

## Communication topology (REQUIRED)

The system must use:

> **MacBook command center ↔ direct connection to each robot (no relay node)**

Implications:

* Each robot has an independent connection (Bluetooth/serial/Wi-Fi).
* The command center must track connections per robot.
* Responses must be routed back to the correct robot ID.
* The implementation must support multiple concurrent robot connections.

---

## Protocol responsibilities (CRITICAL)

The protocol is **bidirectional** and must be implemented on BOTH sides.

### Transport style (MUST match existing firmware)

* Use **line-oriented ASCII messages** (newline-delimited)
* Match the style used in `firmware/ant.py` (simple serial text protocol)
* Do NOT introduce binary framing or complex encodings

### Command center (this prompt implements)

* Receive: `POS`, `PHER`, `SENSE`
* Send: `PHER_RESP`

### Robot-side client (MUST BE SPECIFIED, NOT IMPLEMENTED HERE)

The prompt must define a minimal robot-side protocol client contract that:

* Sends messages like:

  ```text
  POS,<id>,<x_cm>,<y_cm>,<heading_deg>
  PHER,<id>,<x_cm>,<y_cm>,<amount>
  SENSE,<id>,<x_cm>,<y_cm>,<heading_deg>
  ```

* Receives messages like:

  ```text
  PHER_RESP,<id>,<p0>,<p1>,<p2>
  ```

* Uses a simple read loop compatible with `serial.readline()`

### Robot-side pseudocode requirement

Provide example pseudocode showing:

* how to send `POS` periodically
* how to send `SENSE` before action selection
* how to parse `PHER_RESP` and inject values into the observation vector

### Integration constraint

* Do NOT add dependencies from `firmware/` → `command_center/`
* Protocol must be compatible with existing serial workflow

---

---

## RL observation compatibility (STRICT REQUIREMENT)

The pheromone response MUST match the simulator’s current observation contract exactly.

### Scope clarification

The command center provides **only the pheromone-related components** of the observation vector.

All other observation features (e.g., lidar, neighbor vectors, speed, food signals, carrying state) remain **locally computed on the robot** and are **out of scope** for this subsystem.

### REQUIRED response format

```text
PHER_RESP,<robot_id>,<p0>,<p1>,<p2>
```

Where:

* `p0, p1, p2` are the **three forward-direction pheromone samples along the robot’s heading**

### Sampling rules (STRICT — EXACT MATCH REQUIRED)

The sampling geometry MUST exactly match the simulator implementation in `env/swarm_env.py`.

Specifically:

```text
For i = 0,1,2:
  d_i = (i + 1) * agent_radius * 1.5
```

* Samples are taken along the robot's heading direction at distances `d_0`, `d_1`, `d_2`
* Use the same forward vector computation as the simulator
* Do NOT approximate distances, angles, or offsets
* Any deviation will break compatibility with the trained model

---

* Use the same forward offsets and sampling pattern as the simulator

### Normalization (STRICT — EXACT MATCH REQUIRED)

Normalization MUST match simulator behavior exactly:

```python
if samples.max() > 0:
    samples = samples / (samples.max() + 1e-6)
```

* Normalize using the **local max of the 3 samples**, NOT a global map max
* Include the `+ 1e-6` epsilon exactly as used by the simulator
* Do NOT introduce smoothing, scaling, or alternate normalization
* This behavior is part of the learned observation distribution

---

---

## Pheromone model alignment (REQUIRED)

The command center pheromone system must be **consistent with the simulator**.

At minimum:

* Support deposit
* Support decay

If the simulator includes diffusion or smoothing:

* Either implement a simplified equivalent
* Or explicitly document differences

The goal is:

> The robot should perceive pheromone in the real system in a way that is consistent with training.

---

## Coordinate system (STRICT DEFINITION REQUIRED)

You MUST define and document a precise coordinate model consistent with the simulator and the physical arena.

### World / arena frame

* Origin: nest = (0, 0)
* Units: **centimeters (cm)** for all x/y values exchanged with the command center
* Axes:

  * +x points to the **right** of the arena
  * +y points **forward/up** in the arena (define explicitly and keep consistent)
* Bounds: define arena width/height in cm

### Heading convention (MUST match simulator)

* Simulator uses angle `theta` in **radians**
* 0 radians points along **+x axis**
* Positive rotation is **counterclockwise (CCW)**
* Transport may use degrees (`heading_deg`), but it must follow the **same zero direction and sign convention**, and be converted to radians internally

### Physical vs display coordinates

* Robot coordinates are **physical-world coordinates** estimated by the robot (e.g., via dead reckoning)
* The command center must treat these as the **authoritative spatial frame**
* Rendering must be a **visualization of the same coordinates**, not a separate pixel-based system

### Grid mapping (pheromone)

Define explicitly:

```python
grid_x = int(x_cm / cell_size_cm)
grid_y = int(y_cm / cell_size_cm)
```

* `cell_size_cm` MUST be defined (recommended: 5 cm)
* Clamp indices at boundaries
* All subsystems must use the **same mapping**

### Sim-to-physical conversion contract

The simulator defines pheromone sampling in simulator world units. For live deployment, you MUST define one explicit shared conversion rule.

Use this default contract unless there is a strong reason not to:

> **1 simulator world unit = 1 centimeter in the physical arena**

Implications:

* Robot-reported protocol coordinates in cm map directly to the command-center world frame
* Pheromone sampling distances derived from `agent_radius` must use the same shared unit convention
* Visualization, pheromone deposit placement, and `PHER_RESP` sampling must all use this same mapping

If any other scale is used, it must be documented explicitly and applied consistently everywhere.

### Unit consistency

* Robots may use **millimeters (mm)** internally for motion/sensing
* All protocol messages (`POS`, `PHER`, `SENSE`) must use **centimeters (cm)**

---

## Multi-robot communication handling (IMPLEMENTATION REQUIREMENT)

* Each robot connection must be handled independently (e.g., thread or asyncio task)
* Ensure request/response pairing per robot (no cross-talk between connections)
* Avoid blocking the render loop
* Maintain low-latency responses for `SENSE → PHER_RESP`

---

## Performance requirements

* Support multiple robots querying at ~5–20 Hz
* Non-blocking I/O (threads or asyncio)
* Ensure responses are low latency
* Avoid blocking render loop

---

## Error handling

* Ignore malformed messages safely
* Log protocol errors
* If robot unknown, still respond using provided coordinates
* Handle connection drops per robot

---

## Documentation updates

Update the command center README to include:

* full protocol specification
* robot-side integration guide
* observation compatibility explanation
* coordinate system definition
* example request/response flows

---

## Updated definition of done

This task is complete when:

* command center receives `POS` and `PHER`
* command center maintains pheromone field
* robots send `SENSE` queries
* command center returns `PHER_RESP` with **3 forward samples matching simulator expectations**
* values can be directly inserted into the robot’s 23-D observation vector
* system works with multiple robots over direct connections

When done, explain:

* assumptions made about simulator alignment
* any approximations in sampling or normalization
