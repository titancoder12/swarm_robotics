from __future__ import annotations

import socket
import time
from collections import deque


class CommandCenterTCPClient:
    """Line-oriented TCP client for the Mission Control request/response contract."""

    def __init__(self, host: str, port: int, timeout_s: float = 1.0, debug: bool = False) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.debug = debug
        self._sock: socket.socket | None = None
        self._rx_buffer = ""
        self._lines: deque[str] = deque()

    def _connect(self) -> None:
        if self._sock is not None:
            return
        if self.debug:
            print(f"[debug] TCP connecting to {self.host}:{self.port}", flush=True)
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        self._sock = sock
        if self.debug:
            print(f"[debug] TCP connected to {self.host}:{self.port}", flush=True)

    def _close_socket(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        except OSError:
            pass
        finally:
            self._sock = None

    def _send_line(self, line: str) -> None:
        self._connect()
        payload = (line.strip() + "\n").encode("utf-8")
        if self.debug:
            print(f"[debug] TCP write -> {line.strip()}", flush=True)
        try:
            assert self._sock is not None
            self._sock.sendall(payload)
        except OSError:
            self._close_socket()
            raise

    def _recv_into_lines(self) -> None:
        if self._sock is None:
            raise RuntimeError("TCP socket is not connected.")
        chunk = self._sock.recv(4096)
        if not chunk:
            raise ConnectionError("Mission Control TCP connection closed.")
        self._rx_buffer += chunk.decode("utf-8", errors="ignore")
        while "\n" in self._rx_buffer:
            line, self._rx_buffer = self._rx_buffer.split("\n", 1)
            line = line.strip()
            if line:
                self._lines.append(line)

    def _drain_pending(self) -> None:
        if self._sock is None:
            return
        old_timeout = self._sock.gettimeout()
        try:
            self._sock.settimeout(0.0)
            while True:
                try:
                    self._recv_into_lines()
                except BlockingIOError:
                    break
                except TimeoutError:
                    break
                except OSError:
                    break
        finally:
            self._sock.settimeout(old_timeout)
        self._lines.clear()
        self._rx_buffer = ""

    def _read_matching_pheromone(self, robot_id: str) -> tuple[float, float, float]:
        deadline = time.time() + self.timeout_s
        if self.debug:
            print(f"[debug] TCP waiting for PHER_RESP for {robot_id} (timeout={self.timeout_s:.2f}s)", flush=True)
        while time.time() < deadline:
            while self._lines:
                line = self._lines.popleft()
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 5 or parts[0].upper() != "PHER_RESP":
                    continue
                if parts[1] != robot_id:
                    continue
                if self.debug:
                    print(f"[debug] TCP recv <- {line}", flush=True)
                return float(parts[2]), float(parts[3]), float(parts[4])
            self._recv_into_lines()
        if self.debug:
            print(f"[debug] TCP PHER_RESP timeout for {robot_id}", flush=True)
        return 0.0, 0.0, 0.0

    def send_position(self, robot_id: str, x_cm: float, y_cm: float, heading_deg: float) -> None:
        try:
            self._send_line(f"POS,{robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}")
        except Exception as exc:  # pragma: no cover - hardware/network-dependent path
            if self.debug:
                print(f"[debug] TCP POS send failed: {type(exc).__name__}: {exc!r}", flush=True)

    def send_lidar(self, robot_id: str, ranges_mm: list[float] | tuple[float, ...]) -> None:
        try:
            values = [f"{float(value):.1f}" for value in ranges_mm]
            self._send_line(",".join(["LIDAR", robot_id, *values]))
        except Exception as exc:  # pragma: no cover - hardware/network-dependent path
            if self.debug:
                print(f"[debug] TCP LIDAR send failed: {type(exc).__name__}: {exc!r}", flush=True)

    def send_target(self, robot_id: str, x_cm: float, y_cm: float, confidence: float) -> None:
        try:
            self._send_line(f"TARGET,{robot_id},{x_cm:.2f},{y_cm:.2f},{confidence:.3f}")
        except Exception as exc:  # pragma: no cover - hardware/network-dependent path
            if self.debug:
                print(f"[debug] TCP TARGET send failed: {type(exc).__name__}: {exc!r}", flush=True)

    def send_status(
        self,
        robot_id: str,
        control_mode: str,
        action_id: int,
        throttle: float,
        turn: float,
        deposit: bool,
        camera_found: bool,
        camera_distance_m: float,
        camera_angle_deg: float,
        front_min_mm: float,
        left_min_mm: float,
        right_min_mm: float,
        serial_ok: bool,
        serial_cmd: str,
        serial_reply: str,
    ) -> None:
        try:
            safe_cmd = str(serial_cmd).replace(",", ";")
            safe_reply = str(serial_reply).replace(",", ";")
            self._send_line(
                "STATUS,"
                f"{robot_id},{control_mode},{int(action_id)},{float(throttle):.1f},{float(turn):.1f},{int(bool(deposit))},"
                f"{int(bool(camera_found))},{float(camera_distance_m):.3f},{float(camera_angle_deg):.1f},"
                f"{float(front_min_mm):.1f},{float(left_min_mm):.1f},{float(right_min_mm):.1f},"
                f"{int(bool(serial_ok))},{safe_cmd},{safe_reply}"
            )
        except Exception as exc:  # pragma: no cover - hardware/network-dependent path
            if self.debug:
                print(f"[debug] TCP STATUS send failed: {type(exc).__name__}: {exc!r}", flush=True)

    def deposit_pheromone(self, robot_id: str, x_cm: float, y_cm: float, amount: float) -> None:
        try:
            self._send_line(f"PHER,{robot_id},{x_cm:.2f},{y_cm:.2f},{amount:.3f}")
        except Exception as exc:  # pragma: no cover - hardware/network-dependent path
            if self.debug:
                print(f"[debug] TCP PHER send failed: {type(exc).__name__}: {exc!r}", flush=True)

    def sense_pheromone(self, robot_id: str, x_cm: float, y_cm: float, heading_deg: float) -> tuple[float, float, float]:
        try:
            self._connect()
            self._drain_pending()
            self._send_line(f"SENSE,{robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}")
            return self._read_matching_pheromone(robot_id)
        except Exception as exc:  # pragma: no cover - hardware/network-dependent path
            self._close_socket()
            if self.debug:
                print(f"[debug] TCP SENSE failed: {type(exc).__name__}: {exc!r}", flush=True)
            return 0.0, 0.0, 0.0

    def close(self) -> None:
        self._close_socket()
