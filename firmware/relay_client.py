from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class CommandCenterRelayClient:
    """HTTP polling client for the internet relay request/response contract."""

    def __init__(self, relay_url: str, session: str, timeout_s: float = 1.0, debug: bool = False) -> None:
        self.relay_url = relay_url.rstrip("/")
        self.session = session
        self.timeout_s = float(timeout_s)
        self.debug = debug

    def _send_line(self, line: str) -> None:
        url = f"{self.relay_url}/send?{urlencode({'session': self.session, 'role': 'robot'})}"
        body = line.strip().encode("utf-8")
        request = Request(url, data=body, method="POST", headers={"Content-Type": "text/plain; charset=utf-8"})
        if self.debug:
            print(f"[debug] RELAY write -> {line.strip()}", flush=True)
        with urlopen(request, timeout=self.timeout_s) as response:
            response.read()

    def _recv_line(self, timeout_s: float) -> str | None:
        url = f"{self.relay_url}/recv?{urlencode({'session': self.session, 'role': 'robot', 'timeout': f'{timeout_s:.3f}'})}"
        request = Request(url, method="GET")
        try:
            with urlopen(request, timeout=max(self.timeout_s, timeout_s + 0.5)) as response:
                if response.status == 204:
                    return None
                payload = json.loads(response.read().decode("utf-8"))
                line = payload.get("line")
                if self.debug and line:
                    print(f"[debug] RELAY recv <- {line}", flush=True)
                return line
        except HTTPError as exc:
            if exc.code == 204:
                return None
            raise

    def _drain_pending(self) -> None:
        while True:
            line = self._recv_line(timeout_s=0.01)
            if not line:
                return

    def send_position(self, robot_id: str, x_cm: float, y_cm: float, heading_deg: float) -> None:
        try:
            self._send_line(f"POS,{robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}")
        except Exception as exc:  # pragma: no cover - network-dependent path
            if self.debug:
                print(f"[debug] RELAY POS send failed: {type(exc).__name__}: {exc!r}", flush=True)

    def send_lidar(self, robot_id: str, ranges_mm: list[float] | tuple[float, ...]) -> None:
        try:
            values = [f"{float(value):.1f}" for value in ranges_mm]
            self._send_line(",".join(["LIDAR", robot_id, *values]))
        except Exception as exc:  # pragma: no cover - network-dependent path
            if self.debug:
                print(f"[debug] RELAY LIDAR send failed: {type(exc).__name__}: {exc!r}", flush=True)

    def deposit_pheromone(self, robot_id: str, x_cm: float, y_cm: float, amount: float) -> None:
        try:
            self._send_line(f"PHER,{robot_id},{x_cm:.2f},{y_cm:.2f},{amount:.3f}")
        except Exception as exc:  # pragma: no cover - network-dependent path
            if self.debug:
                print(f"[debug] RELAY PHER send failed: {type(exc).__name__}: {exc!r}", flush=True)

    def sense_pheromone(self, robot_id: str, x_cm: float, y_cm: float, heading_deg: float) -> tuple[float, float, float]:
        try:
            self._drain_pending()
            self._send_line(f"SENSE,{robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}")
            line = self._recv_line(timeout_s=self.timeout_s)
            if not line:
                if self.debug:
                    print(f"[debug] RELAY PHER_RESP timeout for {robot_id}", flush=True)
                return 0.0, 0.0, 0.0
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 5 and parts[0].upper() == "PHER_RESP" and parts[1] == robot_id:
                return float(parts[2]), float(parts[3]), float(parts[4])
            return 0.0, 0.0, 0.0
        except Exception as exc:  # pragma: no cover - network-dependent path
            if self.debug:
                print(f"[debug] RELAY SENSE failed: {type(exc).__name__}: {exc!r}", flush=True)
            return 0.0, 0.0, 0.0

    def close(self) -> None:
        return
