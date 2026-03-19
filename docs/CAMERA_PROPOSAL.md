# Camera Proposal

This document proposes how to use an onboard camera in the current swarm robotics system without breaking the existing control architecture.

The recommendation is to use the camera first as a perception sensor that produces compact, policy-ready features, rather than treating raw images as direct policy input.

That recommendation follows directly from the current code structure:

- the simulator policy interface is a fixed low-dimensional observation vector
- the robot runtime expects a `SensorPacket`
- the robot-side observation builder converts that packet into the same vector layout the trained policy expects

In other words, the repository is currently designed around:

`sensor sources -> structured packet -> observation vector -> policy -> action command`

not around:

`raw image -> CNN policy -> action`

## 1. Why Use The Camera At All

A camera can add semantic information that lidar-like ranging alone does not provide well.

The most useful near-term camera roles are:

1. Target detection
   Estimate where food or target objects are relative to the robot.

2. Nest detection
   Detect a visual nest marker, beacon, or AprilTag and estimate its relative bearing.

3. Neighbor detection
   Improve awareness of nearby robots when lidar is noisy or ambiguous.

4. Food presence and pickup confirmation
   Detect whether a target is visible nearby and whether the robot appears to have successfully acquired it.

5. Safety
   Detect hazards or objects that may not be handled well by the existing lidar-style sensing.

6. Ego-motion support
   Provide visual odometry or optic-flow cues if wheel odometry or heading estimates are weak.

The camera should improve semantic awareness first. It should not replace the current observation pipeline all at once.

## 2. Recommendation Summary

The best path for this project is:

1. Keep the current policy architecture vector-based.
2. Run a lightweight perception stack on the Raspberry Pi.
3. Convert camera output into compact geometric and binary features.
4. Feed those features into the existing structured observation pipeline.
5. Retrain the policy on the updated observation contract only after the feature contract is stable.

This is the lowest-risk approach because it fits the current environment and deployment code much better than raw-image RL.

## 3. Why Raw Image Input Is Not The Right First Step

Feeding RGB frames directly into the learned controller is not the best first move for this repository.

Reasons:

1. The current policy is built around a small observation vector, not an image encoder.

2. A raw-image policy would require:
   - a different model architecture
   - a different training pipeline
   - a much larger sim-to-real gap
   - significantly more compute on the Raspberry Pi
   - a more complex simulator camera model

3. Debugging would get harder immediately.
   If the policy fails, it becomes harder to tell whether the problem is:
   - camera calibration
   - perception
   - network architecture
   - domain gap
   - policy learning

The current repository already has a clean place to fuse camera-derived features: the structured `SensorPacket` and observation-builder path.

## 4. Recommended Camera Uses On The Physical Robot

### 4.1 Target Detection

The camera is a strong candidate for producing `target_vector_body`.

For example:

- detect a colored food puck
- detect a fiducial on the target
- detect a distinctive blob or marker

Then estimate:

- horizontal bearing relative to the robot
- approximate distance
- confidence score

The first two values map naturally into the existing 2D target vector concept already used by the observation pipeline.

### 4.2 Nest Detection

If the nest is visually marked, the camera can estimate `nest_vector_body`.

Good options:

- AprilTag
- ArUco marker
- colored fiducial
- light beacon

This is particularly useful because the current architecture already includes nest-direction features conceptually, so vision can supply them in a physically realistic way.

### 4.3 Neighbor Detection

The camera can improve `neighbor_vector_body` if nearby robots have:

- visible markers
- distinctive colors
- LED beacons

This is helpful when the robot needs local swarm coordination and lidar alone cannot reliably separate walls from neighboring robots.

### 4.4 Food Presence And Pickup Confirmation

The camera can help provide:

- `food_presence`
- `carrying_food`

Examples:

- `food_presence = 1` when a target is visible in the near forward field
- `carrying_food = 1` when a grasped object or tag is visible in a pickup zone

This is better than leaving those values as fragile heuristics if the task depends on them.

### 4.5 Safety Override Channel

The camera should also be used outside the policy as a safety sensor.

Examples:

- detect a human or hand near the robot
- detect a cliff or stair edge
- detect transparent or visually obvious obstacles that time-of-flight sensing misses
- detect dead-end geometry ahead

This safety channel should not depend on policy learning. It should clamp or override actions when necessary.

### 4.6 Visual Odometry / Motion Confidence

If heading or speed estimates are weak, the camera can contribute:

- optic flow
- visual odometry
- motion-confidence estimates

These can improve the stability of the heading and speed signals used in the policy input.

## 5. Proposed Integration Principle

The camera should be integrated at the perception layer, not at the policy layer.

That means:

1. Camera frames are processed into compact features on the Pi.
2. Those features populate fields in the structured sensor packet.
3. The observation builder converts the packet into the policy vector.
4. The policy remains a standard vector-input controller.

This preserves the current control architecture and keeps the deployment path understandable.

## 6. Concrete Proposed Observation Contract

The current project already uses a compact observation vector built from:

- lidar-style obstacle ranges
- target vector
- nest vector
- neighbor vector
- heading
- speed
- food presence
- carrying-food state
- pheromone samples

The camera proposal should extend this contract conservatively.

### 6.1 Preferred Near-Term Contract

The preferred first contract is to keep the current semantic slots but improve where their values come from.

That means:

- `ranges_m`
  Still supplied primarily by lidar / ToF style ranging.

- `target_vector_body`
  Supplied by camera-based target detection when available.

- `extras["nest_vector_body"]`
  Supplied by camera-based nest-marker detection when available.

- `neighbor_vector_body`
  Supplied by camera or fused camera+lateral sensing when available.

- `extras["food_presence"]`
  Supplied by near-field visual detection logic.

- `extras["carrying_food"]`
  Supplied by pickup confirmation logic.

This is the lowest-risk contract because it keeps the observation shape conceptually similar.

### 6.2 Proposed Extended Contract

After the above is stable, add a small number of explicit vision-confidence features.

Recommended additional fields:

1. `target_visible`
   Binary or continuous confidence that a target is currently visible.

2. `nest_visible`
   Binary or continuous confidence that the nest marker is currently visible.

3. `neighbor_visible`
   Binary or continuous confidence that a neighbor robot is visually confirmed.

4. `free_space_ahead_visual`
   A scalar in `[0, 1]` summarizing how visually open the forward path appears.

5. `vision_motion_confidence`
   A scalar indicating the quality of visual odometry / optic-flow estimation.

These should be treated as additional scalar features, not as replacements for the geometric vectors.

### 6.3 Recommended Final Observation Layout

If the team decides to revise the observation contract, I would recommend this order:

1. lidar obstacle rays
2. target vector body `(x, y)`
3. nest vector body `(x, y)`
4. neighbor vector body `(x, y)`
5. heading `(sin(theta), cos(theta))`
6. speed
7. food presence
8. carrying food
9. pheromone samples
10. target visible confidence
11. nest visible confidence
12. neighbor visible confidence
13. free-space-ahead visual score
14. vision motion confidence

This preserves the current structure while adding a very small set of camera-derived reliability features.

### 6.4 Normalization Rules

To keep the learning problem stable:

- all XY vectors should remain clipped and normalized relative to a fixed distance scale
- confidence values should remain in `[0, 1]`
- binary flags should remain `0` or `1`
- any missing visual detection should default to:
  - zero vector for geometry
  - zero for confidence
- visual features should never silently change meaning between training and deployment

## 7. Physical-Robot Perception Pipeline Proposal

The onboard Pi camera stack should be organized roughly like this:

1. Capture frame
2. Run lightweight perception
3. Produce structured outputs
4. Fuse with lidar / IMU / odometry
5. Populate a sensor packet
6. Build the policy observation
7. Run policy inference
8. Apply safety gating
9. Send command

### 7.1 Perception Outputs

The perception layer should output compact objects like:

- target bearing and distance estimate
- nest bearing and distance estimate
- neighbor bearing and distance estimate
- target/nest/neighbor confidence
- local free-space confidence
- pickup confirmation

### 7.2 Lightweight Vision Methods

For this project, the first deployed perception methods should probably be simple:

- color segmentation
- fiducial detection
- blob detection
- contour filtering
- narrow classical-CV pipelines

That is a better first step than a large object detector unless the environment is visually uncontrolled.

## 8. How To Integrate The Camera Into The Simulation

There are two possible simulation strategies.

### 8.1 Recommended Strategy: Simulate Camera-Derived Features

This is the best strategy for the current codebase.

Instead of rendering real camera images first, simulate what the camera would have reported as compact features.

For each agent, the simulator should model:

- whether the target is in camera field of view
- whether the nest is in field of view
- whether a neighbor is visually visible
- approximate bearing and range noise
- detection confidence
- missed detections
- latency
- occlusion by obstacles

This allows training on realistic camera semantics without a major architecture change.

### 8.2 Alternative Strategy: Simulate Full RGB Images

This means:

- render a first-person camera image for each agent
- train a CNN or hybrid vision+state policy
- simulate blur, lighting variation, exposure, latency, and compression

This is much more expensive and should be considered a later research direction, not the first integration step.

## 9. Concrete Camera Simulation Model

To simulate feature-level camera perception realistically, the simulator should model the following.

### 9.1 Camera Geometry

- camera field of view
- camera mounting offset
- forward-facing or slightly downward orientation
- effective maximum visual range

### 9.2 Visibility Rules

- target visible only if inside FOV
- target blocked if an obstacle occludes line of sight
- nest marker visible only when not occluded
- neighbor visible only if inside view and not hidden

### 9.3 Noise Model

- angular noise on detected bearing
- distance estimation noise
- occasional missed detections
- occasional false positives if desired
- lower confidence at long range

### 9.4 Timing Model

- camera frame rate lower than control rate
- one or more control-step latency
- dropped frames under load

These matter because real camera data is not perfectly synchronous or perfectly stable.

## 10. Concrete Proposed Sensor Packet Extension

The packet should remain structured and low-dimensional.

A proposed extension would include:

- `target_vector_body`
- `neighbor_vector_body`
- `extras["nest_vector_body"]`
- `extras["food_presence"]`
- `extras["carrying_food"]`
- `extras["target_visible_conf"]`
- `extras["nest_visible_conf"]`
- `extras["neighbor_visible_conf"]`
- `extras["free_space_ahead_visual"]`
- `extras["vision_motion_confidence"]`

This keeps the camera integration additive and inspectable.

## 11. Concrete Proposed Training Plan

The training plan should be phased. Do not jump directly to image-based end-to-end learning.

### Phase 0 - No Camera In Policy

Use the camera only for debugging and data logging on the real robot.

Goals:

- collect sample frames
- understand lighting conditions
- test marker detectability
- estimate realistic detection noise and latency

Outputs:

- camera dataset from the real robot
- rough perception error model

### Phase 1 - Camera As Auxiliary Runtime Sensor Only

Use the camera on the real robot, but do not change the trained policy yet.

Goals:

- populate target/nest/neighbor estimates on the Pi
- compare camera-derived estimates with existing heuristics
- use the camera for safety overrides

This phase is mainly about validating perception without introducing a training change.

### Phase 2 - Feature-Level Camera Simulation

Extend the simulator conceptually so it can produce camera-like feature detections instead of perfect oracle-style geometric values.

Training changes:

- inject visibility constraints
- inject detection noise
- inject latency
- inject missed detections

Goal:

- make training conditions closer to what the robot camera will actually provide

### Phase 3 - Retrain With Camera-Derived Feature Contract

After the feature contract is stable, retrain the policy using the revised observation vector.

Training recommendations:

1. Keep the policy vector-based.
2. Add only a few camera-derived scalar confidence features.
3. Preserve the same discrete action space.
4. Randomize visual detection quality during training.
5. Keep sensor dropout and latency in the loop.

Evaluation questions:

- does target acquisition improve?
- does nest return improve?
- does collision rate go down?
- does performance degrade gracefully when camera detections drop out?

### Phase 4 - Sim-To-Real Validation

Test the retrained policy on the physical robot under controlled conditions.

Validation tasks:

- verify camera-derived target vectors are stable
- verify nest detection is consistent
- verify safety override wins over policy when needed
- measure performance with and without the camera enabled

### Phase 5 - Optional Vision-Heavy Research Direction

Only after the feature-based path is mature should the team consider:

- image encoders
- CNN policies
- vision transformers
- self-supervised visual representations

That is a larger research branch, not the recommended next implementation step.

## 12. Recommended Metrics For Camera Integration

To judge whether the camera is actually helping, measure:

1. target detection success rate
2. nest detection success rate
3. false positive rate
4. control-loop latency impact
5. target pickup success
6. food delivery rate
7. collision rate
8. time to first target
9. episode reward
10. behavior under partial visual failure

## 13. Safety Recommendation

The camera should contribute to safety independently of the learned controller.

That means:

- camera hazard detection should be allowed to stop motion directly
- dangerous conditions should not rely on the policy choosing the correct action
- the camera should be treated as both a semantic sensor and a safety sensor

## 14. Final Recommendation

The best next step for this project is not raw-image RL.

The best next step is:

1. use the camera to estimate target, nest, and neighbor geometry
2. use the camera to provide food-presence and pickup-confirmation cues
3. use the camera as a safety override sensor
4. simulate camera-derived features in the environment
5. retrain the vector-based policy on that revised feature contract

This path is the most consistent with the current repository, the most debuggable, and the most likely to transfer successfully from sim to real.
