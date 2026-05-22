# April 9, 2026 Robot Debug Note

This note summarizes the physical robot work discussed for `robot_2` on April 9, 2026.

## Main Focus

The work that day centered on bringing up the physical Pi-based robot runtime, fixing Bluetooth startup, and then diagnosing why motion commands were not being executed reliably.

## What Was Happening

1. The robot service was being checked with `journalctl` and `systemctl`.
2. The service was temporarily stopped so the runtime could be tested manually.
3. The manual launcher being used was:

```bash
bash run_bluetooth_2.sh
```

4. The robot runtime was starting, but Bluetooth peripheral startup was initially failing with errors like:
   - `Failed to register advertisement`
   - `Rejected (0x0b)`
5. Bluetooth state was checked with:
   - `systemctl status bluetooth`
   - `rfkill list`
   - `hciconfig -a`
   - `dmesg`
6. The key finding was that the Bluetooth adapter was soft-blocked:
   - `Soft blocked: yes`
   - `hci0` was `DOWN`
7. Bluetooth was restored by unblocking the adapter and bringing `hci0` up.
8. After that, BLE started successfully and logs showed:
   - `BLE peripheral robot_2 started`
   - `BLE notify -> ...`

## Next Problem After BLE Was Fixed

Once Bluetooth was working, the main issue shifted to low-level motion control.

Observed symptoms:
- sensor stream could come up
- camera could initialize
- BLE telemetry could work
- but motion commands often failed with:
  - `TIMEOUT`
  - or sometimes `{"type":"err","msg":"UNKNOWN"}`

Examples of affected commands:
- `T-25`
- `M0,100`
- `M180,60`

This suggested that the robot app itself was running, but the controller/firmware side was not reliably acknowledging or executing movement commands.

## Hardware Context Mentioned That Day

Two hardware changes were relevant:

- the motor had been swapped that morning
- later, the microcontroller was also swapped

After the microcontroller swap:
- BLE, camera, and stream readiness could still work
- but command acknowledgement problems remained

That made a completely dead microcontroller less likely and pushed suspicion toward:
- firmware/protocol mismatch
- motor/encoder wiring problems
- downstream motor driver issues
- power instability

## Likely Working Hypothesis From That Session

The debugging direction moved roughly like this:

1. check service state
2. run robot manually
3. fix BLE startup
4. confirm BLE telemetry works
5. investigate serial/motion failures
6. consider hardware causes beyond BLE, especially motor/controller path issues

## Useful Memory Anchor

If you are trying to remember what you were doing on April 9, 2026, the short version is:

- you were debugging `robot_2` on the Raspberry Pi
- you fixed a Bluetooth adapter issue
- then you chased a separate motion-control problem involving serial command failures after recent hardware changes
