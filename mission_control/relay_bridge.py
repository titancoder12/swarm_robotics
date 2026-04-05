from __future__ import annotations

import argparse
import json
import socket
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROLE = "mission_control"


class RelayBridge:
    def __init__(
        self,
        relay_url: str,
        session: str,
        tcp_host: str,
        tcp_port: int,
        relay_timeout_s: float,
        tcp_timeout_s: float,
        debug: bool = False,
    ) -> None:
        self.relay_url = relay_url.rstrip("/")
        self.session = session
        self.tcp_host = tcp_host
        self.tcp_port = int(tcp_port)
        self.relay_timeout_s = float(relay_timeout_s)
        self.tcp_timeout_s = float(tcp_timeout_s)
        self.debug = debug
        self._sock: socket.socket | None = None
        self._rx_buffer = ""

    def _connect_tcp(self) -> None:
        if self._sock is not None:
            return
        if self.debug:
            print(f"[debug] bridge TCP connecting to {self.tcp_host}:{self.tcp_port}", flush=True)
        sock = socket.create_connection((self.tcp_host, self.tcp_port), timeout=self.tcp_timeout_s)
        sock.settimeout(self.tcp_timeout_s)
        self._sock = sock
        if self.debug:
            print(f"[debug] bridge TCP connected to {self.tcp_host}:{self.tcp_port}", flush=True)

    def _close_tcp(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        except OSError:
            pass
        finally:
            self._sock = None
            self._rx_buffer = ""

    def _relay_recv(self) -> str | None:
        url = f"{self.relay_url}/recv?{urlencode({'session': self.session, 'role': ROLE, 'timeout': f'{self.relay_timeout_s:.3f}'})}"
        request = Request(url, method="GET")
        try:
            with urlopen(request, timeout=max(self.relay_timeout_s, 1.0) + 0.5) as response:
                if response.status == 204:
                    return None
                payload = json.loads(response.read().decode("utf-8"))
                line = payload.get("line")
                if self.debug and line:
                    print(f"[debug] bridge relay recv <- {line}", flush=True)
                return line
        except HTTPError as exc:
            if exc.code == 204:
                return None
            raise

    def _relay_send(self, line: str) -> None:
        url = f"{self.relay_url}/send?{urlencode({'session': self.session, 'role': ROLE})}"
        request = Request(
            url,
            data=line.strip().encode("utf-8"),
            method="POST",
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )
        if self.debug:
            print(f"[debug] bridge relay send -> {line.strip()}", flush=True)
        with urlopen(request, timeout=max(self.relay_timeout_s, 1.0)) as response:
            response.read()

    def _tcp_send(self, line: str) -> None:
        self._connect_tcp()
        payload = (line.strip() + "\n").encode("utf-8")
        if self.debug:
            print(f"[debug] bridge TCP send -> {line.strip()}", flush=True)
        assert self._sock is not None
        self._sock.sendall(payload)

    def _tcp_recv_available(self) -> list[str]:
        if self._sock is None:
            return []
        lines: list[str] = []
        old_timeout = self._sock.gettimeout()
        try:
            self._sock.settimeout(0.05)
            while True:
                try:
                    chunk = self._sock.recv(4096)
                except socket.timeout:
                    break
                if not chunk:
                    raise ConnectionError("Mission Control TCP connection closed.")
                self._rx_buffer += chunk.decode("utf-8", errors="ignore")
                while "\n" in self._rx_buffer:
                    line, self._rx_buffer = self._rx_buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        if self.debug:
                            print(f"[debug] bridge TCP recv <- {line}", flush=True)
                        lines.append(line)
        finally:
            if self._sock is not None:
                self._sock.settimeout(old_timeout)
        return lines

    def run_forever(self) -> None:
        while True:
            line = self._relay_recv()
            if not line:
                continue
            try:
                self._tcp_send(line)
                for response in self._tcp_recv_available():
                    self._relay_send(response)
            except Exception as exc:
                self._close_tcp()
                if self.debug:
                    print(f"[debug] bridge forwarding failed: {type(exc).__name__}: {exc!r}", flush=True)
                time.sleep(0.2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge relay traffic into a local Mission Control TCP listener.")
    parser.add_argument("--relay-url", required=True)
    parser.add_argument("--session", default="robot_0")
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=8765)
    parser.add_argument("--relay-timeout", type=float, default=1.0)
    parser.add_argument("--tcp-timeout", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    RelayBridge(
        relay_url=args.relay_url,
        session=args.session,
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
        relay_timeout_s=args.relay_timeout,
        tcp_timeout_s=args.tcp_timeout,
        debug=args.debug,
    ).run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
