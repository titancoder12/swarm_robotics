# Pi Migration Guide

The earlier `pi/` and `robot/` deployment layers have been removed.

The current migrated robot runtime is:

- [firmware/ant.py](../firmware/ant.py)
- [firmware/run.py](../firmware/run.py)
- [firmware/ant.service](../firmware/ant.service)

## Current Deployment Path

Use [firmware/run.py](../firmware/run.py) as the single learned-policy runtime on the Raspberry Pi.

Its responsibilities are:

- connect to the ESP32 over serial through `ant.py`
- read JSON scan lines
- bucketize and normalize lidar observations locally
- load the saved checkpoint through [models/q_network.py](../models/q_network.py)
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
3. Run `python firmware/run.py --checkpoint-dir checkpoints --shared-policy`.
4. Verify scan quality, command timing, and emergency-stop behavior before longer runs.

## Service Usage

If you want the policy runtime to start at boot, update [firmware/ant.service](../firmware/ant.service) so `ExecStart` points at [firmware/run.py](../firmware/run.py) instead of `ant.py`, then install it with systemd on the Pi.
