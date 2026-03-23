# Pi Migration Guide

The earlier `pi/` and `robot/` deployment layers have been removed.

The current migrated robot runtime is:

- [AntSwarmFirmware/ant.py](/Users/christopherlin/dev/cwsf2026/sim/AntSwarmFirmware/ant.py)
- [AntSwarmFirmware/run_policy.py](/Users/christopherlin/dev/cwsf2026/sim/AntSwarmFirmware/run_policy.py)
- [AntSwarmFirmware/ant.service](/Users/christopherlin/dev/cwsf2026/sim/AntSwarmFirmware/ant.service)

## Current Deployment Path

Use [AntSwarmFirmware/run_policy.py](/Users/christopherlin/dev/cwsf2026/sim/AntSwarmFirmware/run_policy.py) as the single learned-policy runtime on the Raspberry Pi.

Its responsibilities are:

- connect to the ESP32 over serial through `ant.py`
- read JSON scan lines
- bucketize and normalize lidar observations locally
- load the saved checkpoint through [models/q_network.py](/Users/christopherlin/dev/cwsf2026/sim/models/q_network.py)
- predict a discrete action
- send `turn`, `move`, and `stop` commands directly to the robot

## Migration Status

The migration is effectively complete in the sense that there is now one direct deployment path instead of a staged `pi/` to `robot/` architecture.

What still requires hardware-specific work:

- observation correctness for any non-lidar features still represented as placeholders
- action-to-motion calibration
- control frequency tuning
- safety behavior and watchdogs
- sensor calibration and out-of-range handling

## Recommended First Task On Hardware

1. Copy the repo, checkpoints, and dependencies to the Pi.
2. Install the runtime requirements with `pip install -r requirements.txt`.
3. Run `python AntSwarmFirmware/run_policy.py --checkpoint-dir checkpoints --shared-policy`.
4. Verify scan quality, command timing, and emergency-stop behavior before longer runs.

## Service Usage

If you want the policy runtime to start at boot, update [AntSwarmFirmware/ant.service](/Users/christopherlin/dev/cwsf2026/sim/AntSwarmFirmware/ant.service) so `ExecStart` points at `run_policy.py` instead of `ant.py`, then install it with systemd on the Pi.
