from __future__ import annotations

import logging
import socket
import threading
import time
import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Callable

try:
    import serial
except ImportError:  # pragma: no cover - depends on local runtime env
    serial = None

try:
    from bleak import BleakClient, BleakScanner
except ImportError:  # pragma: no cover - depends on local runtime env
    BleakClient = None
    BleakScanner = None

logger = logging.getLogger(__name__)

DEFAULT_BLE_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
DEFAULT_BLE_WRITE_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
DEFAULT_BLE_NOTIFY_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"


ResponseCallback = Callable[[str, str], list[str]]


@dataclass
class ConnectionInfo:
    label: str
    transport: str


class _Worker(threading.Thread):
    daemon = True


class TCPClientWorker(_Worker):
    def __init__(self, conn: socket.socket, address: tuple[str, int], on_line: ResponseCallback):
        super().__init__(name=f"tcp-client-{address[0]}:{address[1]}")
        self.conn = conn
        self.address = address
        self.on_line = on_line
        self.stop_event = threading.Event()

    @property
    def label(self) -> str:
        return f"tcp:{self.address[0]}:{self.address[1]}"

    def run(self) -> None:
        with self.conn:
            self.conn.settimeout(0.2)
            buffer = ""
            while not self.stop_event.is_set():
                try:
                    chunk = self.conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="ignore")
                # TCP is a byte stream, not a message transport, so accumulate
                # partial data until complete newline-delimited messages arrive.
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    for response in self.on_line(self.label, line.strip()):
                        self._send_line(response)

    def _send_line(self, line: str) -> None:
        payload = (line.strip() + "\n").encode("utf-8")
        try:
            self.conn.sendall(payload)
        except OSError:
            self.stop_event.set()

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


class TCPServerWorker(_Worker):
    def __init__(self, host: str, port: int, on_line: ResponseCallback):
        super().__init__(name=f"tcp-mission-control-{host}:{port}")
        self.host = host
        self.port = port
        self.on_line = on_line
        self.stop_event = threading.Event()
        self.children: list[TCPClientWorker] = []
        self.bound_port: int | None = None
        self._sock: socket.socket | None = None

    def run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.listen()
            sock.settimeout(0.2)
            self.bound_port = sock.getsockname()[1]
            self._sock = sock
            logger.info("command-center TCP listener on %s:%s", self.host, self.bound_port)
            while not self.stop_event.is_set():
                try:
                    conn, address = sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                # Each robot gets its own worker so request/response traffic
                # stays isolated per connection and never cross-talks.
                child = TCPClientWorker(conn, address, self.on_line)
                self.children.append(child)
                child.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        for child in list(self.children):
            child.stop()


class SerialWorker(_Worker):
    def __init__(self, port: str, baudrate: int, timeout_s: float, on_line: ResponseCallback):
        super().__init__(name=f"serial-{port}")
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self.on_line = on_line
        self.stop_event = threading.Event()
        self._ser: serial.Serial | None = None

    @property
    def label(self) -> str:
        return f"serial:{self.port}"

    def run(self) -> None:
        if serial is None:
            logger.error("pyserial is not installed; serial port %s cannot be used", self.port)
            return
        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout_s)
        except serial.SerialException as exc:
            logger.error("failed to open serial port %s: %s", self.port, exc)
            return

        with self._ser:
            while not self.stop_event.is_set():
                try:
                    raw = self._ser.readline()
                except serial.SerialException as exc:
                    logger.error("serial read failed on %s: %s", self.port, exc)
                    break
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                # Serial transport already provides line framing, so each line
                # can be parsed immediately and any reply written back directly.
                for response in self.on_line(self.label, line):
                    self._write_line(response)

    def _write_line(self, line: str) -> None:
        if self._ser is None:
            return
        try:
            self._ser.write((line.strip() + "\n").encode("utf-8"))
            self._ser.flush()
        except serial.SerialException as exc:
            logger.error("serial write failed on %s: %s", self.port, exc)
            self.stop_event.set()

    def stop(self) -> None:
        self.stop_event.set()
        if self._ser is not None:
            try:
                self._ser.close()
            except serial.SerialException:
                pass


class BLEClientWorker(_Worker):
    def __init__(
        self,
        address: str,
        device_name: str,
        write_char_uuid: str,
        notify_char_uuid: str,
        timeout_s: float,
        on_line: ResponseCallback,
    ) -> None:
        super().__init__(name=f"ble-client-{address or device_name or 'unknown'}")
        self.address = address
        self.device_name = device_name
        self.write_char_uuid = write_char_uuid
        self.notify_char_uuid = notify_char_uuid
        self.timeout_s = timeout_s
        self.on_line = on_line
        self.stop_event = threading.Event()
        self._rx_buffer = ""
        self._tx_queue: asyncio.Queue[str] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def label(self) -> str:
        return f"ble-client:{self.address or self.device_name or 'unknown'}"

    def _line_label(self, line: str) -> str:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[1]:
            return f"{self.label}:{parts[1]}"
        return self.label

    async def _resolve_address(self) -> str:
        if self.address:
            return self.address
        if not self.device_name:
            raise RuntimeError("BLE client requires an address or device name")
        if BleakScanner is None:
            raise RuntimeError("bleak is not installed; BLE discovery is unavailable")
        device = await BleakScanner.find_device_by_name(self.device_name, timeout=self.timeout_s)
        if device is None:
            raise RuntimeError(f"BLE device named {self.device_name!r} was not found.")
        return str(device.address)

    async def _write_line(self, client: BleakClient, line: str) -> None:
        await client.write_gatt_char(self.write_char_uuid, (line.strip() + "\n").encode("utf-8"))

    def _notify_callback(self, client: BleakClient, _sender, data: bytearray) -> None:
        self._rx_buffer += bytes(data).decode("utf-8", errors="ignore")
        while "\n" in self._rx_buffer:
            line, self._rx_buffer = self._rx_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            responses = self.on_line(self._line_label(line), line)
            if self._tx_queue is not None:
                for response in responses:
                    self._tx_queue.put_nowait(response)

    async def _drain_writes(self, client: BleakClient) -> None:
        assert self._tx_queue is not None
        while not self.stop_event.is_set() and client.is_connected:
            try:
                line = await asyncio.wait_for(self._tx_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            await self._write_line(client, line)

    async def _run_session(self) -> None:
        if BleakClient is None:
            raise RuntimeError("bleak is not installed; BLE client transport is unavailable")
        address = await self._resolve_address()
        async with BleakClient(address, timeout=self.timeout_s) as client:
            self._rx_buffer = ""
            self._tx_queue = asyncio.Queue()
            await client.start_notify(self.notify_char_uuid, lambda sender, data: self._notify_callback(client, sender, data))
            logger.info("command-center BLE client connected to %s", address)
            writer_task = asyncio.create_task(self._drain_writes(client))
            try:
                while not self.stop_event.is_set() and client.is_connected:
                    await asyncio.sleep(0.1)
            finally:
                logger.info("command-center BLE client disconnected from %s", address)
                writer_task.cancel()
                try:
                    await writer_task
                except asyncio.CancelledError:
                    pass
                try:
                    await client.stop_notify(self.notify_char_uuid)
                except Exception:
                    pass

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            while not self.stop_event.is_set():
                try:
                    loop.run_until_complete(self._run_session())
                except Exception as exc:  # pragma: no cover - runtime/hardware dependent
                    logger.error("BLE client failed on %s: %s", self.address or self.device_name, exc)
                    if self.stop_event.wait(1.0):
                        break
                else:
                    if self.stop_event.wait(1.0):
                        break
        finally:
            loop.close()

    def stop(self) -> None:
        self.stop_event.set()


class ReceiverManager:
    """Owns the live line-based connections to robots."""

    def __init__(self, on_line: ResponseCallback) -> None:
        self.on_line = on_line
        self._workers: list[_Worker] = []

    def add_tcp_server(self, host: str, port: int) -> TCPServerWorker:
        worker = TCPServerWorker(host, port, self.on_line)
        self._workers.append(worker)
        return worker

    def add_serial_port(self, port: str, baudrate: int, timeout_s: float) -> SerialWorker:
        worker = SerialWorker(port, baudrate, timeout_s, self.on_line)
        self._workers.append(worker)
        return worker

    def add_ble_client(
        self,
        address: str = "",
        device_name: str = "",
        write_char_uuid: str = DEFAULT_BLE_WRITE_CHAR_UUID,
        notify_char_uuid: str = DEFAULT_BLE_NOTIFY_CHAR_UUID,
        timeout_s: float = 1.0,
    ) -> BLEClientWorker:
        worker = BLEClientWorker(
            address=address,
            device_name=device_name,
            write_char_uuid=write_char_uuid,
            notify_char_uuid=notify_char_uuid,
            timeout_s=timeout_s,
            on_line=self.on_line,
        )
        self._workers.append(worker)
        return worker

    def start(self) -> None:
        for worker in self._workers:
            worker.start()

    def stop(self) -> None:
        for worker in self._workers:
            worker.stop()
        for worker in self._workers:
            worker.join(timeout=1.0)
