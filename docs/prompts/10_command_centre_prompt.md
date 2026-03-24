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

## Bidirectional pheromone query (NEW REQUIREMENT)

Extend the command center to support **robot queries for local pheromone observations** and send compact responses back to robots.

### Goals

* Robots should **not** receive the full global map.
* Robots should receive **local pheromone samples** suitable for inclusion in the RL observation vector.
* Keep responses **compact, low-latency, and robust** for serial/BLE links.

### New protocol messages

#### Query from robot

```text
SENSE,<robot_id>,<x>,<y>,<heading_deg>
```

* `<x>,<y>` are in the same coordinate system as `POS`
* `<heading_deg>` is the robot’s current heading (degrees, 0–360)

#### Response from command center

Provide one of the following response formats (implement at least one; document which is used):

**Option A: directional samples (preferred for RL)**

```text
PHER_RESP,<robot_id>,<left>,<forward>,<right>
```

* Samples are taken at small offsets relative to heading
* Values are normalized (e.g., 0.0–1.0) or documented units

**Option B: local patch (e.g., 3x3 neighborhood)**

```text
PHER_RESP,<robot_id>,v00,v01,v02,v10,v11,v12,v20,v21,v22
```

* Center cell corresponds to the robot’s current cell
* Row-major order; document clearly

### Sampling rules

* Convert `(x, y)` to grid cell using the same quantization as deposits.
* For directional samples, compute offsets based on heading (e.g., forward = +d along heading; left/right = ±90°).
* Clamp or pad at boundaries.
* Optionally apply a small spatial smoothing kernel.

### Performance

* Handle multiple robots querying at ~5–20 Hz each.
* Ensure non-blocking I/O (use threads or `asyncio`).
* Keep per-query computation O(1) or O(k) for small k (e.g., 3–9 cells).

### Error handling

* Ignore or log malformed `SENSE` messages.
* If a robot is unknown, still respond using provided coordinates.
* Include timeouts or drop policies for slow links.

### Integration with existing system

* This must be implemented **without introducing dependencies into `firmware/`**.
* Keep protocol definitions in `command_center/comms/protocol.py` and, if needed, mirror minimal constants in `shared/`.

### Documentation updates

Update the command center README to include:

* the `SENSE` and `PHER_RESP` message formats
* units and normalization
* sampling geometry (diagram or description)
* example request/response pairs

---

## Updated definition of done

In addition to prior criteria, this task is complete when:

* a robot (or simulator) can send `SENSE` messages,
* the command center returns local pheromone observations,
* and those values can be used as part of the robot’s observation vector.
