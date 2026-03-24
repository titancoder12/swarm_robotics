from __future__ import annotations

import logging
import socket
import threading
from dataclasses import dataclass
from typing import Callable

try:
    import serial
except ImportError:  # pragma: no cover - depends on local runtime env
    serial = None

logger = logging.getLogger(__name__)


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
        super().__init__(name=f"tcp-server-{host}:{port}")
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

    def start(self) -> None:
        for worker in self._workers:
            worker.start()

    def stop(self) -> None:
        for worker in self._workers:
            worker.stop()
        for worker in self._workers:
            worker.join(timeout=1.0)
