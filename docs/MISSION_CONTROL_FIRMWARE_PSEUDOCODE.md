# Mission Control Firmware-Side Pseudocode

This note describes the intended robot-side protocol flow for Mission Control without creating a dependency from [firmware/](/Users/christopherlin/dev/cwsf2026/sim/firmware/) to [mission_control/](/Users/christopherlin/dev/cwsf2026/sim/mission_control/).

## Minimal robot-side flow

```python
robot_id = "robot_0"

while True:
    x_cm, y_cm, heading_deg = estimate_pose()
    local_obs = build_local_observation_without_pheromone()

    send_line(f"POS,{robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}")
    send_line(f"SENSE,{robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}")

    reply = readline().strip()
    # Expected: PHER_RESP,<id>,<p0>,<p1>,<p2>
    parts = reply.split(",")
    if len(parts) == 5 and parts[0] == "PHER_RESP" and parts[1] == robot_id:
        p0 = float(parts[2])
        p1 = float(parts[3])
        p2 = float(parts[4])
    else:
        p0 = p1 = p2 = 0.0

    local_obs[20:23] = [p0, p1, p2]
    action = policy(local_obs)
    execute_action(action)

    if should_drop_pheromone():
        send_line(f"PHER,{robot_id},{x_cm:.2f},{y_cm:.2f},1.0")
```

## Notes

- Use newline-delimited ASCII messages.
- A direct per-robot connection is assumed.
- `PHER_RESP` only fills the pheromone channels of the 23-D observation vector.
- Lidar, target/nest/neighbor cues, speed, food-presence, and carrying-food signals remain local.
