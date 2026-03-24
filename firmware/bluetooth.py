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
        self.address = address
        self.device_name = device_name
        self.write_char_uuid = write_char_uuid
        self.notify_char_uuid = notify_char_uuid
        self.timeout_s = timeout_s
        self.debug = debug
        self._loop = asyncio.new_event_loop()
        self._client = None
        self._rx_buffer = ""
        self._lines: deque[str] = deque()

    def _notify_callback(self, _sender, data: bytearray) -> None:
        self._rx_buffer += bytes(data).decode("utf-8", errors="ignore")
        while "\n" in self._rx_buffer:
            line, self._rx_buffer = self._rx_buffer.split("\n", 1)
            line = line.strip()
            if line:
                self._lines.append(line)

    async def _resolve_address(self) -> str:
        if self.address:
            return self.address
        if not self.device_name:
            raise RuntimeError("BLE command-center client requires --cc-ble-address or --cc-ble-device-name.")
        if BleakScanner is None:
            raise RuntimeError("bleak is not installed; BLE device discovery is unavailable.")
        device = await BleakScanner.find_device_by_name(self.device_name, timeout=self.timeout_s)
        if device is None:
            raise RuntimeError(f"BLE device named {self.device_name!r} was not found.")
        return str(device.address)

    async def _ensure_connected(self) -> None:
        if BleakClient is None:
            raise RuntimeError("bleak is not installed; BLE command-center transport is unavailable.")
        if self._client is not None and self._client.is_connected:
            return
        address = await self._resolve_address()
        self._client = BleakClient(address)
        await self._client.connect()
        await self._client.start_notify(self.notify_char_uuid, self._notify_callback)
        if self.debug:
            print(f"[debug] BLE connected to command center at {address}", flush=True)

    async def _write_line(self, line: str) -> None:
        await self._ensure_connected()
        payload = (line.strip() + "\n").encode("utf-8")
        await self._client.write_gatt_char(self.write_char_uuid, payload)

    async def _read_matching_pheromone(self, robot_id: str, timeout_s: float) -> tuple[float, float, float]:
        deadline = self._loop.time() + timeout_s
        while self._loop.time() < deadline:
            while self._lines:
                line = self._lines.popleft()
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 5 or parts[0].upper() != "PHER_RESP":
                    continue
                if parts[1] != robot_id:
                    continue
                try:
                    return float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    continue
            await asyncio.sleep(0.01)
        return 0.0, 0.0, 0.0

    async def _send_position_async(self, robot_id: str, x_cm: float, y_cm: float, heading_deg: float) -> None:
        await self._write_line(f"POS,{robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}")

    async def _deposit_pheromone_async(self, robot_id: str, x_cm: float, y_cm: float, amount: float) -> None:
        await self._write_line(f"PHER,{robot_id},{x_cm:.2f},{y_cm:.2f},{amount:.3f}")

    async def _sense_pheromone_async(self, robot_id: str, x_cm: float, y_cm: float, heading_deg: float) -> tuple[float, float, float]:
        self._lines.clear()
        await self._write_line(f"SENSE,{robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}")
        return await self._read_matching_pheromone(robot_id, self.timeout_s)

    def send_position(self, robot_id: str, x_cm: float, y_cm: float, heading_deg: float) -> None:
        try:
            self._loop.run_until_complete(self._send_position_async(robot_id, x_cm, y_cm, heading_deg))
        except Exception as exc:  # pragma: no cover - hardware-dependent path
            if self.debug:
                print(f"[debug] BLE POS send failed: {exc}", flush=True)

    def deposit_pheromone(self, robot_id: str, x_cm: float, y_cm: float, amount: float) -> None:
        try:
            self._loop.run_until_complete(self._deposit_pheromone_async(robot_id, x_cm, y_cm, amount))
        except Exception as exc:  # pragma: no cover - hardware-dependent path
            if self.debug:
                print(f"[debug] BLE PHER send failed: {exc}", flush=True)

    def sense_pheromone(self, robot_id: str, x_cm: float, y_cm: float, heading_deg: float) -> tuple[float, float, float]:
        try:
            return self._loop.run_until_complete(
                self._sense_pheromone_async(robot_id, x_cm, y_cm, heading_deg)
            )
        except Exception as exc:  # pragma: no cover - hardware-dependent path
            if self.debug:
                print(f"[debug] BLE SENSE failed: {exc}", flush=True)
            return 0.0, 0.0, 0.0

    def close(self) -> None:
        if self._client is None:
            self._loop.close()
            return
        try:
            self._loop.run_until_complete(self._client.disconnect())
        except Exception as exc:  # pragma: no cover - hardware-dependent path
            if self.debug:
                print(f"[debug] BLE disconnect failed: {exc}", flush=True)
        finally:
            self._loop.close()
