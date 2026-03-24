from __future__ import annotations

import argparse
import math
import random
import socket
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fake robot client for local command-center testing.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--robot-id", default="robot_0")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--radius-cm", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    period = 1.0 / max(args.rate_hz, 1e-6)

    with socket.create_connection((args.host, args.port), timeout=3.0) as sock:
        file = sock.makefile("r", encoding="utf-8", newline="\n")
        for step in range(args.steps):
            theta = step * 0.17
            x_cm = math.cos(theta) * args.radius_cm
            y_cm = math.sin(theta) * args.radius_cm
            heading_deg = (math.degrees(theta) + 90.0) % 360.0

            pos_line = f"POS,{args.robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}\n"
            sock.sendall(pos_line.encode("utf-8"))

            if step % 3 == 0:
                amount = 1.0 + 0.25 * random.random()
                pher_line = f"PHER,{args.robot_id},{x_cm:.2f},{y_cm:.2f},{amount:.3f}\n"
                sock.sendall(pher_line.encode("utf-8"))

            sense_line = f"SENSE,{args.robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}\n"
            sock.sendall(sense_line.encode("utf-8"))
            response = file.readline().strip()
            print(f"step={step} pos=({x_cm:.1f},{y_cm:.1f}) heading={heading_deg:.1f} resp={response}")
            time.sleep(period)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

