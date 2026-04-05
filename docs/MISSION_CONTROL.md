# Mission Control

This subsystem adds a live desktop command center for physical swarm experiments. It is separate from [firmware/](../firmware) and does not introduce any `firmware -> mission_control` dependency.

## Purpose

The command center is a real-time operator tool that:

- receives robot position and pheromone messages
- keeps a live world-state model
- maintains the authoritative digital pheromone field
- answers robot `SENSE` queries with simulator-compatible pheromone samples
- visualizes robots, trails, and pheromone heatmap in a PyGame window

It is not a planner and it does not make robot motion decisions.

## Architecture

The subsystem lives under [mission_control/](../mission_control):

- [mission_control/main.py](../mission_control/main.py)
  - app entrypoint, event loop, keyboard controls, receiver wiring
- [mission_control/config.py](../mission_control/config.py)
  - shared configuration derived from current simulator defaults
- [mission_control/core/robot_registry.py](../mission_control/core/robot_registry.py)
  - latest robot state per ID
- [mission_control/core/trail_store.py](../mission_control/core/trail_store.py)
  - bounded trail history per robot
- [mission_control/core/pheromone_field.py](../mission_control/core/pheromone_field.py)
  - pheromone grid, decay, diffusion, and exact forward-sample query logic
- [mission_control/core/world_state.py](../mission_control/core/world_state.py)
  - thread-safe aggregation of robot registry, trails, and pheromone field
- [mission_control/comms/protocol.py](../mission_control/comms/protocol.py)
  - line protocol parser and `PHER_RESP` formatter
- [mission_control/comms/receiver.py](../mission_control/comms/receiver.py)
  - threaded TCP and serial receivers for direct robot connections
- [mission_control/ui/renderer.py](../mission_control/ui/renderer.py)
  - PyGame rendering
- [mission_control/fake_robot.py](../mission_control/fake_robot.py)
  - local harness for test messages without real hardware

## Protocol

Mission Control uses newline-delimited ASCII messages.

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

Message meanings:

- `POS`
  - updates the latest robot pose and trail point
- `PHER`
  - deposits `amount` into the digital pheromone grid at the reported position
- `SENSE`
  - queries the command center for the three forward pheromone samples used by the current RL observation contract
- `PHER_RESP`
  - returns those three samples as normalized values in `[0, 1]`

Malformed lines are ignored and logged as protocol warnings.

## Coordinate System

World / arena frame:

- origin is the nest at `(0, 0)`
- protocol units are centimeters
- `+x` points right
- `+y` points up / forward in the arena
- default world size is derived from [env/config.py](../env/config.py): `900 cm x 600 cm`

Default sim-to-physical conversion:

- `1 simulator world unit = 1 centimeter`

Pheromone grid mapping:

- the command center uses a centered arena frame around the nest
- default `cell_size_cm` is derived from `cfg.pheromone_cell_size`, currently `6 cm`
- conversion uses arena offsets internally so negative robot coordinates are supported while the nest remains at `(0, 0)`

## Observation Compatibility

Mission Control only provides the pheromone part of the 23-D observation vector.

All other observation channels remain robot-local and are outside this subsystem:

- lidar
- target / nest / neighbor vectors
- speed
- food presence
- carrying-food state

`PHER_RESP` is designed to match `_pheromone_samples()` in [env/swarm_env.py](../env/swarm_env.py):

- sample count: 3
- sample distances:
  - `d_i = (i + 1) * agent_radius * 1.5`
- sample direction:
  - along robot heading only
- normalization:

```python
if samples.max() > 0:
    samples = samples / (samples.max() + 1e-6)
```

The current command center also applies simulator-style decay and diffusion so the field is consistent with training semantics.

## Robot-Side Integration Guide

The command center does not modify [firmware/](../firmware), but the intended robot-side client contract is:

```python
send_line(f"POS,{robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}")
send_line(f"SENSE,{robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}")
reply = read_line()  # e.g. PHER_RESP,robot_0,0.000000,0.571429,1.000000
p0, p1, p2 = parse_reply(reply)
observation[20:23] = [p0, p1, p2]
```

Recommended runtime behavior:

- send `POS` periodically for visualization
- send `PHER` when the robot performs a digital deposit event
- send `SENSE` immediately before action selection if the robot wants live pheromone values
- use a per-robot direct link such as serial or a dedicated TCP connection

The transport should remain line-oriented and compatible with `serial.readline()`.

## TCP Transport

TCP is the recommended non-BLE path for connecting the Raspberry Pi runtime to
Mission Control over:

- home Wi-Fi
- an iPhone hotspot
- any shared private local network

The protocol does not change. Only the transport changes.

Pi -> Mission Control:

```text
POS,<id>,<x_cm>,<y_cm>,<heading_deg>
LIDAR,<id>,<r0_mm>,...,<r8_mm>
PHER,<id>,<x_cm>,<y_cm>,<amount>
SENSE,<id>,<x_cm>,<y_cm>,<heading_deg>
```

Mission Control -> Pi:

```text
PHER_RESP,<id>,<p0>,<p1>,<p2>
```

### MacBook TCP Setup

Start Mission Control on a LAN-reachable address:

```bash
python -m mission_control.main --tcp-host 0.0.0.0 --tcp-port 8765 --log-level INFO
```

Notes:

- `--tcp-host 0.0.0.0` lets another device on the network connect to Mission Control
- `--tcp-port 8765` is the default Mission Control TCP port
- `--ble-enable` is not required for the TCP path

### Find The MacBook IP

The Pi needs the MacBook's local IP on the shared network.

One simple command on the MacBook is:

```bash
ipconfig getifaddr en0
```

Typical examples:

- home Wi-Fi: `192.168.x.x`
- iPhone hotspot: often `172.20.10.x`

### Pi TCP Setup

Run the robot runtime with the new TCP flags:

```bash
python firmware/run.py \
  --checkpoint-dir checkpoints \
  --port /dev/ttyUSB0 \
  --cc-tcp-enable \
  --cc-tcp-host MACBOOK_IP \
  --cc-tcp-port 8765 \
  --cc-tcp-timeout 1.0 \
  --debug
```

Replace `MACBOOK_IP` with the MacBook's real IP on the shared network.

Example:

```bash
python firmware/run.py \
  --checkpoint-dir checkpoints \
  --port /dev/ttyUSB0 \
  --cc-tcp-enable \
  --cc-tcp-host 192.168.1.42 \
  --cc-tcp-port 8765 \
  --cc-tcp-timeout 1.0 \
  --debug
```

### Pi TCP Flags

- `--cc-tcp-enable`
  - enable the Mission Control TCP client
- `--cc-tcp-host`
  - hostname or IP address of the MacBook running Mission Control
- `--cc-tcp-port`
  - Mission Control TCP port, default `8765`
- `--cc-tcp-timeout`
  - socket timeout in seconds

If `--cc-tcp-enable` or `--cc-tcp-host` is provided, the Pi runtime uses TCP instead of BLE.

### What Success Looks Like

On the Pi, debug output should include lines like:

```text
[debug] TCP connecting to 192.168.x.x:8765
[debug] TCP connected to 192.168.x.x:8765
[debug] TCP write -> POS,...
[debug] TCP recv <- PHER_RESP,...
```

On the MacBook, Mission Control should:

- show robot count `1`
- render the robot in the world view
- update `x`, `y`, and heading in the side panel

### Network Notes

TCP works well on:

- home Wi-Fi
- iPhone hotspot
- a private router you control

Public or venue Wi-Fi may fail even if both devices have internet access,
because some networks isolate clients from one another.

For demos, a private network you control is the safer choice.

## Internet Relay Transport

If the MacBook and Pi cannot talk directly over the local network, the repo now
also supports a very small HTTP relay path.

This is the intended fallback for:

- hostile venue Wi-Fi with client isolation
- home Wi-Fi that does not allow peer-to-peer traffic
- cases where both devices can reach the internet but not each other directly

The relay path has three pieces:

1. a tiny HTTP relay server
2. a Mac-side bridge that forwards relay traffic into the existing local Mission Control TCP listener
3. the Pi runtime using the relay directly instead of BLE or direct TCP

### Relay Server

Run the relay server on any host both devices can reach, for example a small
cloud VM:

```bash
python -m relay.relay_server --host 0.0.0.0 --port 8080
```

Example relay URL:

```text
http://YOUR_SERVER_IP:8080
```

### MacBook Relay Setup

Start Mission Control locally:

```bash
python -m mission_control.main --tcp-host 127.0.0.1 --tcp-port 8765 --log-level INFO
```

Then start the relay bridge in a second terminal:

```bash
python -m mission_control.relay_bridge \
  --relay-url http://YOUR_SERVER_IP:8080 \
  --session robot_0 \
  --tcp-host 127.0.0.1 \
  --tcp-port 8765 \
  --debug
```

Notes:

- `--session` is the shared queue/session name for one robot link
- the bridge talks to local Mission Control over `127.0.0.1:8765`
- the bridge only forwards lines; the Mission Control app itself stays unchanged

### Combined Mac Launcher

There is also a combined Mac launcher that starts both Mission Control and the
relay bridge together:

```bash
bash scripts/start_mission_control_relay.sh
```

Or via the desktop-friendly wrapper:

```bash
./scripts/start_mission_control_relay.command
```

Default launcher values:

- relay URL: `https://relay.christopherlin.ca`
- relay session: `robot_0`
- local Mission Control TCP host: `127.0.0.1`
- local Mission Control TCP port: `8765`

You can override those with environment variables before launching:

```bash
MISSION_CONTROL_RELAY_URL=https://relay.example.com \
MISSION_CONTROL_RELAY_SESSION=robot_1 \
bash scripts/start_mission_control_relay.sh
```

Launcher behavior:

- `Ctrl+C` stops both Mission Control and the relay bridge
- closing the Mission Control UI window also causes the launcher to stop the relay bridge
- if the relay bridge exits first, the launcher also stops Mission Control

### Pi Relay Setup

On the Pi:

```bash
python firmware/run.py \
  --checkpoint-dir checkpoints \
  --port /dev/ttyUSB0 \
  --cc-relay-url http://YOUR_SERVER_IP:8080 \
  --cc-relay-session robot_0 \
  --cc-relay-timeout 1.0 \
  --debug
```

### Relay Notes

- The relay is intentionally minimal and keeps line-based message queues in memory.
- It is meant for demos and controlled usage, not production networking.
- The robot-side protocol remains the same: `POS`, `LIDAR`, `PHER`, `SENSE`, and `PHER_RESP`.
- The relay path is less elegant than direct local TCP, but it avoids local client-isolation problems.

## BLE Transport

Mission Control can also expose the same protocol over BLE.

The BLE transport is intended to look like a line-oriented UART link to the
robot runtime even though it is implemented as a GATT service underneath.

Current BLE design:

- the desktop side advertises one BLE peripheral from [mission_control/main.py](../mission_control/main.py)
- the Mission Control-side transport implementation lives in [mission_control/comms/receiver.py](../mission_control/comms/receiver.py)
- the Raspberry Pi-side BLE client lives in [firmware/bluetooth.py](../firmware/bluetooth.py)
- the policy/runtime integration lives in [firmware/run.py](../firmware/run.py)

Default BLE UUIDs:

- service UUID
  - `6E400001-B5A3-F393-E0A9-E50E24DCCA9E`
- write characteristic UUID
  - `6E400002-B5A3-F393-E0A9-E50E24DCCA9E`
- notify characteristic UUID
  - `6E400003-B5A3-F393-E0A9-E50E24DCCA9E`

Those values follow the common Nordic-UART-style layout so the rest of the
runtime can keep using newline-delimited text messages.

BLE message flow:

1. the robot connects as a BLE client
2. it writes `POS,...` lines to the write characteristic
3. it writes `SENSE,...` when it wants pheromone samples
4. Mission Control computes the response
5. Mission Control emits `PHER_RESP,...` on the notify characteristic
6. the robot filters responses by `robot_id`

Digital deposits use the same pattern:

- the robot writes `PHER,<id>,<x_cm>,<y_cm>,<amount>`
- Mission Control applies that deposit to the authoritative pheromone field

### Multi-Robot BLE Behavior

BLE support is now multiplexed by `robot_id` rather than by one dedicated BLE
worker per robot.

That means:

- one shared BLE peripheral can service multiple robots
- each robot must use a distinct `robot_id`
- Mission Control derives a logical per-robot label from the protocol line
- replies still include `robot_id`
- each robot-side BLE client must ignore replies for other robots

This is different from the TCP path, where each robot gets its own socket and
worker thread. BLE multi-robot isolation is protocol-level rather than
connection-level.

### BLE Runtime Expectations

For the current firmware path in [firmware/run.py](../firmware/run.py):

- `--cc-ble-enable` turns on the BLE Mission Control link
- the robot sends `POS` once per control iteration
- the robot sends `SENSE` before inference
- the robot can send `PHER` deposits during forward motion
- returned `PHER_RESP` values are inserted into the pheromone observation slots

If BLE is unavailable or times out, the firmware falls back to zero pheromone
samples instead of crashing the motion loop.

## Running

Start the command center:

```bash
python -m mission_control.main --tcp-host 127.0.0.1 --tcp-port 8765
```

Optional flags:

- `--serial-port /dev/tty.usbserial-...`
  - add one or more direct serial robot connections
- `--serial-baudrate 115200`
- `--ble-enable`
  - expose the same line-oriented protocol over a BLE peripheral
- `--ble-device-name CommandCenter`
  - BLE advertised device name for the command center
- `--ble-service-uuid`, `--ble-write-char-uuid`, `--ble-notify-char-uuid`
  - override the Nordic-UART-style GATT UUIDs if your robot client expects different values
- `--render-scale 1.0`
- `--fps 30`
- `--headless`
  - still uses PyGame, but useful with a dummy display in smoke tests
- `--max-seconds 10`
  - auto-exit after N seconds

BLE mode uses the same newline-delimited `POS`, `PHER`, `SENSE`, and `PHER_RESP` messages as the TCP and serial backends; only the transport changes.

## Keyboard Controls

- `Space`
  - pause / unpause pheromone updates
- `C`
  - clear pheromone field
- `T`
  - toggle trails
- `P`
  - toggle pheromone heatmap
- `Q` or `Esc`
  - quit cleanly

## Local Test Harness

Run Mission Control:

```bash
python -m mission_control.main --tcp-port 8765
```

Then run one or more fake robots in other terminals:

```bash
python -m mission_control.fake_robot --robot-id robot_0 --port 8765
python -m mission_control.fake_robot --robot-id robot_1 --port 8765
```

The fake harness sends `POS`, periodic `PHER`, and `SENSE` messages and prints the returned `PHER_RESP` values.

## Assumptions

- the command center is the single source of truth for digital pheromone
- robots keep computing all non-pheromone observation channels locally
- physical pose estimates are already available on the robot side
- the default conversion `1 sim unit = 1 cm` is acceptable unless a deployment chooses a different documented scale
