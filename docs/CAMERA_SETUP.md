# Camera Setup

This document records the working Raspberry Pi camera setup for the current
robot runtime.

It is separate from [CAMERA_PROPOSAL.md](./CAMERA_PROPOSAL.md), which discusses
the broader design direction. This file is for practical setup, debugging, and
known-good launch commands. For the exact model-input feature mapping, see
[OBSERVATION_SPEC.md](./OBSERVATION_SPEC.md).

## Purpose

The current robot runtime can optionally use the onboard camera to fill the
target-related observation slots that were previously hardcoded to zeros.

When camera detection succeeds, the runtime now supplies:

- target distance
- target angle
- food-presence flag

When the camera is unavailable or detection fails, the runtime falls back to the
old behavior and keeps running.

That means the camera path is backward compatible with the existing Mission
Control, relay, TCP, and BLE workflows.

## Future YOLO Option

It is possible to add something like YOLO to recognize specific targets.

The right integration is not to feed raw images directly into the current
policy. The right integration is:

1. run a detector on the camera frame
2. select the relevant target detection
3. convert that detection into the same compact target features the runtime
   already uses

That preserves backward compatibility with the current policy and with the
existing Mission Control, BLE, TCP, and relay workflows.

### When YOLO Makes Sense

YOLO or another learned detector is useful when:

- the target is visually complex
- color thresholding is too fragile
- lighting changes a lot
- the background contains similar colors
- the robot needs to distinguish between multiple object types

YOLO is less necessary when:

- the target can be made visually distinctive
- a simple colored marker is acceptable
- Pi compute budget is tight
- the current HSV detector is already good enough for demos

### Current Interface To Preserve

The current camera path does not pass raw images into the policy.

It produces compact target-related features:

- target visible or not
- relative angle to target
- approximate distance to target
- food-presence flag

That is the contract to preserve.

Any YOLO-based detector should therefore output the same semantic fields:

- `found`
- `angle_rad`
- `distance_m`
- `confidence`

If detection fails, the runtime should still fall back safely to:

- HSV detection
- or zeros, depending on the configured mode

### Proposed Architecture

The clean design is to make detector backend a pluggable choice inside
[firmware/camera.py](../firmware/camera.py).

Example structure:

- `HSVTargetDetector`
- `YOLOTargetDetector`

Then add a runtime selector such as:

- `--camera-detector hsv`
- `--camera-detector yolo`

Possible future flags:

- `--camera-yolo-model PATH`
- `--camera-yolo-class target`
- `--camera-yolo-confidence-threshold 0.25`

This keeps the rest of the robot runtime unchanged.

### How YOLO Would Feed The Model

Once YOLO returns a detection box for the target:

1. use the box center to compute horizontal offset from image center
2. convert that offset into `angle_rad`
3. estimate distance from box size and known object size, or from a calibrated
   lookup
4. set `found = True`
5. set `food_presence = 1.0`

So the model still receives the same kind of observation slots it already
expects.

In plain terms:

- YOLO decides what the target is
- the runtime converts that detection into direction and distance
- the policy continues consuming the same compact observation vector as before

### Why Raw Image Input Is Not The Right First Step

The current policy was trained on structured observations, not pixels.

So replacing the observation vector with raw camera frames would require:

- a new observation space
- a new model architecture
- a new training pipeline
- significant retraining

That is a different project.

For the current system, YOLO should be treated as a feature extractor, not as a
replacement for the policy.

### Recommended Runtime Strategy

Recommended behavior:

1. try YOLO if configured
2. if YOLO succeeds, use its detection
3. if YOLO is unavailable, too slow, or finds nothing:
   - optionally fall back to HSV
   - otherwise fall back to zeros

That keeps the robot operational even if the learned detector is unavailable.

### Main Risks

- inference latency on the Pi
- packaging and dependency complexity
- collecting enough images for the real target classes
- detector instability in poor lighting
- distance estimate noise from bounding-box size alone

### Recommended First Version

If YOLO is added, the first version should:

- detect one target class only
- choose the highest-confidence detection
- convert that box into angle and distance
- preserve the current observation mapping
- keep safe fallback behavior

That gives the project a class-aware detector without requiring a new policy
architecture.

## Recommended Fresh Setup For The Next Robot

If you are setting up a new Raspberry Pi robot from scratch, use this order.
This sequence is based on the exact failure modes encountered during the first
working bring-up.

1. clone or update the repo on the Pi
2. verify the Raspberry Pi camera hardware separately with `rpicam-still`
3. create the robot venv with `--system-site-packages`
4. install `opencv-python` and `bless` into that venv
5. install system packages with `apt`:
   - `python3-picamera2`
   - `python3-libcamera`
   - `python3-torch`
6. verify imports for `torch`, `picamera2`, `cv2`, and `bless`
7. run the known-good Bluetooth command with camera enabled
8. only after capture is working, tune HSV bounds for the actual target object

If you follow that order, you avoid the main traps:

- using a venv that cannot see `picamera2` or `python3-torch`
- trying to install PyTorch from `pip` on the Pi
- debugging target detection before camera capture is actually working
- treating random `/dev/video*` nodes as if they were normal webcam indices

## Fastest Recovery Path From The Same Broken State

This is the shortest known path for a second robot that starts in the same
state as the first one before camera bring-up worked:

- repo present on the Pi
- camera physically connected
- old or missing `firmware/ant-env`
- camera not yet working in the runtime

Run these steps in order.

### 1. Verify The Camera Hardware First

```bash
which rpicam-still
rpicam-still -o test.jpg -t 2000
```

If `rpicam-still` fails, stop there. That is a Pi camera stack or hardware
problem, not a firmware/runtime problem.

### 2. Recreate The Venv The Right Way

Do not try to salvage an old venv that was created without
`--system-site-packages`.

```bash
cd ~/Desktop/swarm_robotics
rm -rf firmware/ant-env
/usr/bin/python3 -m venv --system-site-packages firmware/ant-env
source firmware/ant-env/bin/activate
```

### 3. Install The Required System Packages

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera python3-torch v4l-utils
```

If `apt` was interrupted on a previous attempt:

```bash
sudo dpkg --configure -a
sudo apt install -y python3-picamera2 python3-libcamera python3-torch v4l-utils
```

### 4. Install The Required Venv Packages

```bash
pip install opencv-python bless
```

Do not use `pip install torch`.

### 5. Verify The Four Critical Imports

```bash
python -c "import torch; print(torch.__version__)"
python -c "from picamera2 import Picamera2; print('picamera2 ok')"
python -c "import cv2; print(cv2.__version__)"
python -c "from bless import BlessServer; print('bless ok')"
```

All four must succeed before running the robot.

### 6. Run The Known-Good Camera Command

```bash
bash firmware/run_bluetooth.sh \
  --camera-enable \
  --camera-hsv-lower 10,60,60 \
  --camera-hsv-upper 60,255,255 \
  --camera-min-area-px 50 \
  -- --max-steps 5
```

### 7. Confirm These Exact Success Signals

You want to see all of these:

- `BLE peripheral robot_0 started`
- `camera using Picamera2 backend at index 0`
- `camera target found area=...`
- `camera found=True distance_m=... angle_deg=...`

If you get those lines, the camera pipeline is working.

### 8. Only Then Do Longer Runs

```bash
bash firmware/run_bluetooth.sh \
  --camera-enable \
  --camera-hsv-lower 10,60,60 \
  --camera-hsv-upper 60,255,255 \
  --camera-min-area-px 50
```

### Things To Avoid

- do not use `pip install torch`
- do not keep an old venv that cannot see system packages
- do not guess camera index `1`
- do not debug HSV before `camera using Picamera2 backend at index 0` appears

## Current Status

The following is now working on the Pi:

- camera hardware via Raspberry Pi camera stack
- `picamera2`
- `opencv-python`
- robot runtime integration in [firmware/run.py](../firmware/run.py)
- Picamera2 capture backend in [firmware/camera.py](../firmware/camera.py)
- BLE launch path with camera enabled

Confirmed runtime evidence:

- `camera using Picamera2 backend at index 0`
- repeated `camera target found area=... bbox=...`
- repeated `camera found=True distance_m=... angle_deg=...`

## Python Environment

The working approach on the Pi was:

1. create the robot venv with system packages visible
2. install OpenCV into that venv
3. install `bless` into that venv
4. install `python3-picamera2`, `python3-libcamera`, and `python3-torch` with
   `apt`, not `pip`

### Create The Venv

From the repo root on the Pi:

```bash
/usr/bin/python3 -m venv --system-site-packages ~/Desktop/swarm_robotics/firmware/ant-env
source ~/Desktop/swarm_robotics/firmware/ant-env/bin/activate
```

Why `--system-site-packages` matters:

- `picamera2` is installed as a system Python package
- `python3-torch` is installed as a system Python package
- the venv needs visibility into those system packages

If an older `firmware/ant-env` already exists and was created without
`--system-site-packages`, remove it and recreate it instead of trying to patch
around missing imports:

```bash
rm -rf ~/Desktop/swarm_robotics/firmware/ant-env
/usr/bin/python3 -m venv --system-site-packages ~/Desktop/swarm_robotics/firmware/ant-env
source ~/Desktop/swarm_robotics/firmware/ant-env/bin/activate
```

### Install OpenCV

```bash
pip install opencv-python
```

### Install Torch

Do not use `pip install torch` on the Pi for this setup.

That attempted to download a huge wheel and failed with disk-space issues. The
working path is:

```bash
sudo apt update
sudo apt install -y python3-torch
```

Why this matters:

- `pip install torch` attempted to download a very large wheel
- that is the wrong installation path for this Pi setup
- it can fail with `No space left on device`
- the Debian `python3-torch` package is the correct CPU-oriented install path

### Install Picamera2

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera
```

### BLE Dependency

The rebuilt venv also needs `bless` for robot-side BLE advertising:

```bash
pip install bless
```

## Dependency Verification

Inside the active `ant-env`, these checks should pass:

```bash
python -c "import torch; print(torch.__version__)"
python -c "from picamera2 import Picamera2; print('picamera2 ok')"
python -c "import cv2; print(cv2.__version__)"
python -c "from bless import BlessServer; print('bless ok')"
```

Known-good example output:

```text
2.6.0+debian
picamera2 ok
4.13.0
bless ok
```

If `python3 -c "from picamera2 import Picamera2"` works outside the venv but
`python -c "from picamera2 import Picamera2"` fails inside the venv, the venv
was almost certainly created without `--system-site-packages`. Recreate it.

## Camera Stack Verification

Before debugging the robot runtime, verify the Pi camera stack independently.

### Check Camera CLI Tools

```bash
which rpicam-still
```

On the tested system this returned:

```text
/usr/bin/rpicam-still
```

### Test Still Capture

```bash
rpicam-still -o test.jpg -t 2000
```

This succeeded on the tested Pi and confirmed:

- the camera hardware is connected
- the Raspberry Pi camera stack is working

### Enumerate Video Devices

```bash
v4l2-ctl --list-devices
```

If `v4l2-ctl` is missing on a fresh Pi:

```bash
sudo apt update
sudo apt install -y v4l-utils
```

On the tested system, the CSI camera appeared under:

```text
rp1-cfe (platform:1f00128000.csi)
```

The important lesson is that the Pi camera stack exposes many `/dev/video*`
nodes, but random OpenCV index selection is not a reliable way to identify the
correct stream on newer Pi hardware.

In the original bring-up:

- index `1` was wrong and produced `camera unavailable at index 1`
- index `0` was the right logical camera for the Pi path
- but plain OpenCV capture still failed until the runtime switched to
  Picamera2-first capture

## Runtime Camera Backend

The runtime now prefers:

1. `Picamera2`
2. OpenCV / V4L2 fallback
3. graceful fallback to no-camera behavior

The working backend on the tested Pi was:

```text
[debug] camera using Picamera2 backend at index 0
```

This is the preferred and expected path on Raspberry Pi OS.

If you do not see that line and instead see `camera frame read failed`, then
the runtime has fallen back to the older OpenCV capture path and camera capture
is not yet correctly configured for the Pi.

## Working Launch Command

The following command was confirmed to work on the Pi:

```bash
bash firmware/run_bluetooth.sh \
  --camera-enable \
  --camera-hsv-lower 10,60,60 \
  --camera-hsv-upper 60,255,255 \
  --camera-min-area-px 50 \
  -- --max-steps 5
```

This produced:

- successful Picamera2 backend initialization
- repeated target detections
- `camera found=True`

For a longer live run after bring-up succeeds, remove the short step cap:

```bash
bash firmware/run_bluetooth.sh \
  --camera-enable \
  --camera-hsv-lower 10,60,60 \
  --camera-hsv-upper 60,255,255 \
  --camera-min-area-px 50
```

## Working Detector Parameters

These settings worked on the tested Pi / target setup:

- `--camera-hsv-lower 10,60,60`
- `--camera-hsv-upper 60,255,255`
- `--camera-min-area-px 50`

These values are intentionally broad and are suitable for bright yellow /
orange target-like objects.

## What Success Looks Like

When the camera path is working, debug output should include:

```text
[debug] camera using Picamera2 backend at index 0
[debug] camera target found area=...
[debug] camera found=True distance_m=... angle_deg=...
```

The reported values should change as the target moves:

- `angle_deg` changes when the target moves left or right
- `distance_m` changes when the target moves closer or farther

## Common Failure Modes

### `camera disabled: cv2 is not installed`

Meaning:

- OpenCV is missing from the active venv

Fix:

```bash
pip install opencv-python
```

### `bless is not installed; BLE peripheral transport is unavailable`

Meaning:

- the BLE runtime dependency is missing from the active venv

Fix:

```bash
pip install bless
```

### `camera frame read failed`

Meaning:

- the old OpenCV capture path opened, but failed to deliver frames

This was an intermediate failure mode before switching to the Picamera2-first
capture path.

### `camera unavailable at index 1`

Meaning:

- the chosen camera index is wrong

On the tested Pi, `camera-index 1` was not correct.

Fix:

- use index `0`
- do not guess among `/dev/video*` nodes
- prefer Picamera2 over raw OpenCV device capture on Raspberry Pi OS

### `ModuleNotFoundError: No module named 'picamera2'`

Meaning:

- either `python3-picamera2` is not installed
- or the venv cannot see system Python packages

Fix:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera
```

Then, if needed, recreate the venv with `--system-site-packages`.

### `ModuleNotFoundError: No module named 'torch'`

Meaning:

- the recreated venv did not yet have access to Torch

Fix:

- ensure the venv was created with `--system-site-packages`
- install `python3-torch` with `apt`

### `Error: dpkg was interrupted, you must manually run 'sudo dpkg --configure -a'`

Meaning:

- an `apt` installation was interrupted, for example by shutdown or battery
  loss during package installation

Fix:

```bash
sudo dpkg --configure -a
sudo apt install -y python3-torch
```

If dependency repair is still needed afterward:

```bash
sudo apt install -f
```

### `No space left on device` during `pip install torch`

Meaning:

- `pip` is attempting the wrong PyTorch installation path for the Pi

Fix:

- stop the `pip` approach
- use `sudo apt install -y python3-torch`
- recreate the venv with `--system-site-packages` if needed

### Picamera2 import works, but runtime still hangs or capture does not start

Meaning:

- the Python package is present, but the actual camera runtime still needs to
  be verified independently

Fix:

Run:

```bash
rpicam-still -o test.jpg -t 2000
```

If that succeeds, the hardware and Pi camera stack are working, and the
problem is in Python/runtime configuration rather than the camera hardware.

## Backward Compatibility

The camera integration does not make the rest of the robot runtime depend on the
camera.

If camera detection fails:

- the robot runtime still runs
- Mission Control still works
- relay / TCP / BLE still work
- target features fall back to zeros as before

So the camera path is additive, not required.

## Known-Good Success Checklist

Before handing the next robot to someone else, confirm all of these:

- `rpicam-still -o test.jpg -t 2000` succeeds
- `python -c "import torch; print(torch.__version__)"` succeeds in `ant-env`
- `python -c "from picamera2 import Picamera2; print('picamera2 ok')"` succeeds
- `python -c "import cv2; print(cv2.__version__)"` succeeds
- `python -c "from bless import BlessServer; print('bless ok')"` succeeds
- runtime prints `camera using Picamera2 backend at index 0`
- runtime prints repeated `camera target found ...`
- runtime prints repeated `camera found=True ...`

If all of those pass, the next robot should not need camera-stack debugging
before field testing.

## Recommended Next Steps

1. Keep using the tested HSV range first:
   - `10,60,60` to `60,255,255`
2. Test longer runs with Mission Control active
3. Compare behavior with camera off vs camera on
4. If needed later, add a standalone calibration tool for HSV tuning

## Related Files

- [firmware/camera.py](../firmware/camera.py)
- [firmware/run.py](../firmware/run.py)
- [firmware/run_bluetooth.sh](../firmware/run_bluetooth.sh)
- [firmware/run_relay.sh](../firmware/run_relay.sh)
- [MISSION_CONTROL.md](./MISSION_CONTROL.md)
- [CAMERA_PROPOSAL.md](./CAMERA_PROPOSAL.md)
