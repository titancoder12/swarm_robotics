from __future__ import annotations

import asyncio
from collections import deque

try:
    from bleak import BleakClient, BleakScanner
except ImportError:  # pragma: no cover - depends on runtime environment
    BleakClient = None
    BleakScanner = None


DEFAULT_BLE_WRITE_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
DEFAULT_BLE_NOTIFY_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
# These UUIDs follow the common Nordic UART pattern so the BLE transport can
# still behave like a newline-delimited text stream from the runtime's point of
# view.


class CommandCenterBLEClient:
    """Line-oriented BLE client for the command-center request/response contract."""

    def __init__(
        self,
        address: str,
        device_name: str,
        write_char_uuid: str = DEFAULT_BLE_WRITE_CHAR_UUID,
        notify_char_uuid: str = DEFAULT_BLE_NOTIFY_CHAR_UUID,
        timeout_s: float = 0.5,
        debug: bool = False,
    ) -> None:
        # The rest of the runtime is synchronous, so this helper hides the
        # asyncio/BLE details behind a small blocking API.
        self.address = address
        self.device_name = device_name
        self.write_char_uuid = write_char_uuid
        self.notify_char_uuid = notify_char_uuid
        self.timeout_s = timeout_s
        self.debug = debug
        self._loop = asyncio.new_event_loop()
        self._client = None
        # Notifications can arrive in arbitrary fragments, so we buffer bytes
        # until full newline-delimited protocol lines are reconstructed.
        self._rx_buffer = ""
        self._lines: deque[str] = deque()

    def _notify_callback(self, _sender, data: bytearray) -> None:
        # BLE notification payloads are not guaranteed to align with message
        # boundaries, so treat them like a byte stream and split on '\n'.
        self._rx_buffer += bytes(data).decode("utf-8", errors="ignore")
        while "\n" in self._rx_buffer:
            line, self._rx_buffer = self._rx_buffer.split("\n", 1)
            line = line.strip()
            if line:
                self._lines.append(line)

    async def _resolve_address(self) -> str:
        # Support both explicit addresses and discovery by advertised device
        # name so deployment can choose whichever is more stable on the target
        # platform.
        if self.address:
            if self.debug:
                print(f"[debug] BLE using explicit address {self.address}", flush=True)
            return self.address
        if not self.device_name:
            raise RuntimeError("BLE command-center client requires --cc-ble-address or --cc-ble-device-name.")
        if BleakScanner is None:
            raise RuntimeError("bleak is not installed; BLE device discovery is unavailable.")
        if self.debug:
            print(f"[debug] BLE scanning for device name {self.device_name!r}", flush=True)
        device = await BleakScanner.find_device_by_name(self.device_name, timeout=self.timeout_s)
        if device is None:
            raise RuntimeError(f"BLE device named {self.device_name!r} was not found.")
        if self.debug:
            print(f"[debug] BLE resolved device {self.device_name!r} to {device.address}", flush=True)
        return str(device.address)

    async def _ensure_connected(self) -> None:
        # Keep one persistent BLE connection for the whole policy loop so each
        # control step only pays request/response cost, not a reconnect cost.
        if BleakClient is None:
            raise RuntimeError("bleak is not installed; BLE command-center transport is unavailable.")
        if self._client is not None and self._client.is_connected:
            return
        address = await self._resolve_address()
        if self.debug:
            print(f"[debug] BLE creating client for {address}", flush=True)
        self._client = BleakClient(address, timeout=self.timeout_s)
        if self.debug:
            print(f"[debug] BLE connecting to {address}", flush=True)
        await asyncio.wait_for(self._client.connect(), timeout=self.timeout_s)
        if self.debug:
            print(f"[debug] BLE connected to {address}; starting notifications on {self.notify_char_uuid}", flush=True)
        await asyncio.wait_for(
            self._client.start_notify(self.notify_char_uuid, self._notify_callback),
            timeout=self.timeout_s,
        )
        if self.debug:
            print(f"[debug] BLE connected to command center at {address}", flush=True)

    async def _write_line(self, line: str) -> None:
        # The Mission Control protocol is line-oriented ASCII even over BLE. Appending a
        # newline keeps behavior aligned with the TCP/serial transports.
        await self._ensure_connected()
        payload = (line.strip() + "\n").encode("utf-8")
        if self.debug:
            print(f"[debug] BLE write -> {line.strip()}", flush=True)
        await self._client.write_gatt_char(self.write_char_uuid, payload)
        if self.debug:
            print(f"[debug] BLE write complete -> {line.strip()}", flush=True)

    async def _read_matching_pheromone(self, robot_id: str, timeout_s: float) -> tuple[float, float, float]:
        # Only PHER_RESP for this robot counts as a valid SENSE reply. Anything
        # else is ignored so stale or unrelated traffic does not corrupt the
        # observation.
        deadline = self._loop.time() + timeout_s
        if self.debug:
            print(f"[debug] BLE waiting for PHER_RESP for {robot_id} (timeout={timeout_s:.2f}s)", flush=True)
        while self._loop.time() < deadline:
            while self._lines:
                line = self._lines.popleft()
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 5 or parts[0].upper() != "PHER_RESP":
                    continue
                if parts[1] != robot_id:
                    continue
                try:
                    if self.debug:
                        print(f"[debug] BLE recv <- {line}", flush=True)
                    return float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    continue
            await asyncio.sleep(0.01)
        if self.debug:
            print(f"[debug] BLE PHER_RESP timeout for {robot_id}", flush=True)
        return 0.0, 0.0, 0.0

    async def _send_position_async(self, robot_id: str, x_cm: float, y_cm: float, heading_deg: float) -> None:
        await self._write_line(f"POS,{robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}")

    async def _send_lidar_async(self, robot_id: str, ranges_mm: list[float] | tuple[float, ...]) -> None:
        values = [f"{float(value):.1f}" for value in ranges_mm]
        await self._write_line(",".join(["LIDAR", robot_id, *values]))

    async def _deposit_pheromone_async(self, robot_id: str, x_cm: float, y_cm: float, amount: float) -> None:
        await self._write_line(f"PHER,{robot_id},{x_cm:.2f},{y_cm:.2f},{amount:.3f}")

    async def _sense_pheromone_async(self, robot_id: str, x_cm: float, y_cm: float, heading_deg: float) -> tuple[float, float, float]:
        # Flush any stale queued lines before issuing a new request so the
        # response we return corresponds to the current pose query.
        self._lines.clear()
        await self._write_line(f"SENSE,{robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}")
        return await self._read_matching_pheromone(robot_id, self.timeout_s)

    def send_position(self, robot_id: str, x_cm: float, y_cm: float, heading_deg: float) -> None:
        # Position updates are best-effort telemetry. Failures are swallowed so
        # the robot can keep moving even if the Mission Control link is flaky.
        try:
            self._loop.run_until_complete(self._send_position_async(robot_id, x_cm, y_cm, heading_deg))
        except Exception as exc:  # pragma: no cover - hardware-dependent path
            if self.debug:
                print(f"[debug] BLE POS send failed: {type(exc).__name__}: {exc!r}", flush=True)

    def deposit_pheromone(self, robot_id: str, x_cm: float, y_cm: float, amount: float) -> None:
        # Deposits follow the same fire-and-forget pattern as position updates.
        try:
            self._loop.run_until_complete(self._deposit_pheromone_async(robot_id, x_cm, y_cm, amount))
        except Exception as exc:  # pragma: no cover - hardware-dependent path
            if self.debug:
                print(f"[debug] BLE PHER send failed: {type(exc).__name__}: {exc!r}", flush=True)

    def send_lidar(self, robot_id: str, ranges_mm: list[float] | tuple[float, ...]) -> None:
        # Lidar uploads are best-effort telemetry like position updates.
        try:
            self._loop.run_until_complete(self._send_lidar_async(robot_id, ranges_mm))
        except Exception as exc:  # pragma: no cover - hardware-dependent path
            if self.debug:
                print(f"[debug] BLE LIDAR send failed: {type(exc).__name__}: {exc!r}", flush=True)

    def sense_pheromone(self, robot_id: str, x_cm: float, y_cm: float, heading_deg: float) -> tuple[float, float, float]:
        # SENSE is the only request that feeds directly into inference, so this
        # call returns a concrete 3-sample vector and falls back to zeros on any
        # transport error or timeout.
        try:
            return self._loop.run_until_complete(
                self._sense_pheromone_async(robot_id, x_cm, y_cm, heading_deg)
            )
        except Exception as exc:  # pragma: no cover - hardware-dependent path
            if self.debug:
                print(f"[debug] BLE SENSE failed: {type(exc).__name__}: {exc!r}", flush=True)
            return 0.0, 0.0, 0.0

    def close(self) -> None:
        # Shut down the BLE client explicitly so repeated test runs do not leave
        # the event loop or connection state hanging around.
        if self._client is None:
            self._loop.close()
            return
        try:
            self._loop.run_until_complete(self._client.disconnect())
        except Exception as exc:  # pragma: no cover - hardware-dependent path
            if self.debug:
                print(f"[debug] BLE disconnect failed: {type(exc).__name__}: {exc!r}", flush=True)
        finally:
            self._loop.close()
