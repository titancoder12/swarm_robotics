from __future__ import annotations

import asyncio
from collections import deque
import threading
import time
from queue import Empty, Queue

try:
    from bless import BlessServer
    from bless.backends.characteristic import GATTAttributePermissions, GATTCharacteristicProperties
except ImportError:  # pragma: no cover - depends on runtime environment
    BlessServer = None
    GATTAttributePermissions = None
    GATTCharacteristicProperties = None


DEFAULT_BLE_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
DEFAULT_BLE_WRITE_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
DEFAULT_BLE_NOTIFY_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
# These UUIDs follow the common Nordic UART pattern so the BLE transport can
# still behave like a newline-delimited text stream from the runtime's point of
# view.


class CommandCenterBLEPeripheral:
    """Line-oriented BLE peripheral that exposes the command-center protocol."""

    def __init__(
        self,
        device_name: str,
        service_uuid: str = DEFAULT_BLE_SERVICE_UUID,
        write_char_uuid: str = DEFAULT_BLE_WRITE_CHAR_UUID,
        notify_char_uuid: str = DEFAULT_BLE_NOTIFY_CHAR_UUID,
        timeout_s: float = 0.5,
        debug: bool = False,
    ) -> None:
        self.device_name = device_name
        self.service_uuid = service_uuid
        self.write_char_uuid = write_char_uuid
        self.notify_char_uuid = notify_char_uuid
        self.timeout_s = timeout_s
        self.debug = debug
        self._server = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._outgoing: Queue[str] = Queue()
        self._incoming_lock = threading.Lock()
        self._incoming_lines: deque[str] = deque()
        self._incoming_buffer = ""

    def _ensure_started(self) -> None:
        if self._thread is not None:
            return
        if BlessServer is None or GATTCharacteristicProperties is None or GATTAttributePermissions is None:
            raise RuntimeError("bless is not installed; BLE peripheral transport is unavailable.")
        self._thread = threading.Thread(target=self._thread_main, name=f"ble-peripheral-{self.device_name}", daemon=True)
        self._thread.start()
        if not self._started_event.wait(timeout=max(self.timeout_s, 1.0) + 2.0):
            raise RuntimeError(f"BLE peripheral {self.device_name!r} did not start.")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _serve() -> None:
            self._server = BlessServer(name=self.device_name, loop=loop)
            await self._server.add_new_service(self.service_uuid)
            await self._server.add_new_characteristic(
                self.service_uuid,
                self.write_char_uuid,
                GATTCharacteristicProperties.write | GATTCharacteristicProperties.write_without_response,
                None,
                GATTAttributePermissions.writeable,
            )
            await self._server.add_new_characteristic(
                self.service_uuid,
                self.notify_char_uuid,
                GATTCharacteristicProperties.notify | GATTCharacteristicProperties.read,
                None,
                GATTAttributePermissions.readable,
            )
            self._server.get_characteristic(self.write_char_uuid).write_callback = self._handle_write
            self._server.write_request_func = self._handle_write
            await self._server.start()
            self._started_event.set()
            if self.debug:
                print(f"[debug] BLE peripheral {self.device_name} started", flush=True)
            try:
                while not self._stop_event.is_set():
                    try:
                        line = self._outgoing.get_nowait()
                    except Empty:
                        await asyncio.sleep(0.02)
                        continue
                    self._notify_line(line)
            finally:
                await self._server.stop()

        try:
            loop.run_until_complete(_serve())
        finally:
            loop.close()

    def _handle_write(self, *args) -> None:
        if len(args) == 1:
            value = args[0]
        elif len(args) >= 2:
            value = args[1]
        else:
            return
        if value is None:
            return
        if isinstance(value, bytearray):
            raw = bytes(value)
        elif isinstance(value, bytes):
            raw = value
        else:
            raw = str(value).encode("utf-8")
        self._incoming_buffer += raw.decode("utf-8", errors="ignore")
        while "\n" in self._incoming_buffer:
            line, self._incoming_buffer = self._incoming_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            with self._incoming_lock:
                self._incoming_lines.append(line)
            if self.debug:
                print(f"[debug] BLE recv <- {line}", flush=True)

    def _notify_line(self, line: str) -> None:
        if self._server is None:
            return
        payload = (line.strip() + "\n").encode("utf-8")
        if self.debug:
            print(f"[debug] BLE notify -> {line.strip()}", flush=True)
        characteristic = self._server.get_characteristic(self.notify_char_uuid)
        characteristic.value = bytearray(payload)
        self._server.update_value(self.service_uuid, self.notify_char_uuid)

    def _send_line(self, line: str) -> None:
        self._ensure_started()
        self._outgoing.put(line.strip())

    def _read_matching_pheromone(self, robot_id: str, timeout_s: float) -> tuple[float, float, float]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._incoming_lock:
                while self._incoming_lines:
                    line = self._incoming_lines.popleft()
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 5 or parts[0].upper() != "PHER_RESP":
                        continue
                    if parts[1] != robot_id:
                        continue
                    try:
                        return float(parts[2]), float(parts[3]), float(parts[4])
                    except ValueError:
                        continue
            time.sleep(0.01)
        return 0.0, 0.0, 0.0

    def send_position(self, robot_id: str, x_cm: float, y_cm: float, heading_deg: float) -> None:
        try:
            self._send_line(f"POS,{robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}")
        except Exception as exc:  # pragma: no cover - hardware-dependent path
            if self.debug:
                print(f"[debug] BLE POS send failed: {type(exc).__name__}: {exc!r}", flush=True)

    def send_lidar(self, robot_id: str, ranges_mm: list[float] | tuple[float, ...]) -> None:
        try:
            values = [f"{float(value):.1f}" for value in ranges_mm]
            self._send_line(",".join(["LIDAR", robot_id, *values]))
        except Exception as exc:  # pragma: no cover - hardware-dependent path
            if self.debug:
                print(f"[debug] BLE LIDAR send failed: {type(exc).__name__}: {exc!r}", flush=True)

    def send_target(self, robot_id: str, x_cm: float, y_cm: float, confidence: float) -> None:
        try:
            self._send_line(f"TARGET,{robot_id},{x_cm:.2f},{y_cm:.2f},{confidence:.3f}")
        except Exception as exc:  # pragma: no cover - hardware-dependent path
            if self.debug:
                print(f"[debug] BLE TARGET send failed: {type(exc).__name__}: {exc!r}", flush=True)

    def deposit_pheromone(self, robot_id: str, x_cm: float, y_cm: float, amount: float) -> None:
        try:
            self._send_line(f"PHER,{robot_id},{x_cm:.2f},{y_cm:.2f},{amount:.3f}")
        except Exception as exc:  # pragma: no cover - hardware-dependent path
            if self.debug:
                print(f"[debug] BLE PHER send failed: {type(exc).__name__}: {exc!r}", flush=True)

    def sense_pheromone(self, robot_id: str, x_cm: float, y_cm: float, heading_deg: float) -> tuple[float, float, float]:
        try:
            with self._incoming_lock:
                self._incoming_lines.clear()
            self._send_line(f"SENSE,{robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}")
            return self._read_matching_pheromone(robot_id, self.timeout_s)
        except Exception as exc:  # pragma: no cover - hardware-dependent path
            if self.debug:
                print(f"[debug] BLE SENSE failed: {type(exc).__name__}: {exc!r}", flush=True)
            return 0.0, 0.0, 0.0

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
