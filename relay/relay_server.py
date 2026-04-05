from __future__ import annotations

import argparse
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


ROLE_ROBOT = "robot"
ROLE_MISSION_CONTROL = "mission_control"
VALID_ROLES = {ROLE_ROBOT, ROLE_MISSION_CONTROL}


@dataclass
class SessionQueues:
    robot_queue: deque[str] = field(default_factory=deque)
    mission_control_queue: deque[str] = field(default_factory=deque)
    condition: threading.Condition = field(default_factory=threading.Condition)

    def queue_for(self, role: str) -> deque[str]:
        if role == ROLE_ROBOT:
            return self.robot_queue
        return self.mission_control_queue


class RelayStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionQueues] = defaultdict(SessionQueues)

    def send(self, session: str, sender_role: str, line: str) -> int:
        target_role = ROLE_MISSION_CONTROL if sender_role == ROLE_ROBOT else ROLE_ROBOT
        queues = self._sessions[session]
        with queues.condition:
            queue = queues.queue_for(target_role)
            queue.append(line)
            queues.condition.notify_all()
            return len(queue)

    def recv(self, session: str, receiver_role: str, timeout_s: float) -> str | None:
        queues = self._sessions[session]
        deadline = time.time() + max(0.0, timeout_s)
        with queues.condition:
            queue = queues.queue_for(receiver_role)
            while not queue:
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    return None
                queues.condition.wait(timeout=remaining)
            return queue.popleft()


STORE = RelayStore()


def _require_param(values: dict[str, list[str]], key: str) -> str:
    value = values.get(key, [""])[0].strip()
    if not value:
        raise ValueError(f"Missing required query parameter: {key}")
    return value


def _require_role(values: dict[str, list[str]]) -> str:
    role = _require_param(values, "role")
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role {role!r}. Expected one of: {sorted(VALID_ROLES)}")
    return role


class RelayHandler(BaseHTTPRequestHandler):
    server_version = "MissionControlRelay/0.1"

    def _json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_text_body(self) -> str:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b""
        return raw.decode("utf-8", errors="ignore").strip()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json_response(HTTPStatus.OK, {"ok": True})
            return
        if parsed.path != "/recv":
            self._json_response(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return

        try:
            query = parse_qs(parsed.query)
            session = _require_param(query, "session")
            role = _require_role(query)
            timeout_s = float(query.get("timeout", ["1.0"])[0])
            line = STORE.recv(session=session, receiver_role=role, timeout_s=timeout_s)
        except ValueError as exc:
            self._json_response(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if line is None:
            self._json_response(HTTPStatus.NO_CONTENT, {"ok": True, "line": None})
            return
        self._json_response(HTTPStatus.OK, {"ok": True, "line": line})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/send":
            self._json_response(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return

        try:
            query = parse_qs(parsed.query)
            session = _require_param(query, "session")
            role = _require_role(query)
            line = self._read_text_body()
            if not line:
                raise ValueError("Request body must contain one protocol line.")
            queue_depth = STORE.send(session=session, sender_role=role, line=line)
        except ValueError as exc:
            self._json_response(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        self._json_response(HTTPStatus.OK, {"ok": True, "queued": queue_depth})

    def log_message(self, format: str, *args) -> None:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal HTTP relay for Mission Control and robot traffic.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), RelayHandler)
    print(f"relay server listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
