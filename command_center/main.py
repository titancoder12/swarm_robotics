from __future__ import annotations

import argparse
import logging
import time

import pygame

from command_center.comms.protocol import (
    PheromoneMessage,
    PositionMessage,
    ProtocolError,
    SenseMessage,
    format_pheromone_response,
    parse_line,
)
from command_center.comms.receiver import ReceiverManager
from command_center.config import CommandCenterConfig
from command_center.core.world_state import WorldState
from command_center.ui.renderer import Renderer


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pygame command center for physical swarm robots.")
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=8765)
    parser.add_argument("--serial-port", action="append", default=[])
    parser.add_argument("--serial-baudrate", type=int, default=115200)
    parser.add_argument("--serial-timeout", type=float, default=0.1)
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    cfg = CommandCenterConfig(
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
        serial_ports=tuple(args.serial_port),
        serial_baudrate=args.serial_baudrate,
        serial_timeout_s=args.serial_timeout,
        fps=args.fps,
    )
    world = WorldState(cfg)

    def handle_line(connection_label: str, line: str) -> list[str]:
        try:
            msg = parse_line(line)
        except ProtocolError as exc:
            logger.warning("protocol error from %s: %s (%r)", connection_label, exc, line)
            return []

        if isinstance(msg, PositionMessage):
            world.update_position(msg.robot_id, msg.x_cm, msg.y_cm, msg.heading_deg, connection_label=connection_label)
            return []

        if isinstance(msg, PheromoneMessage):
            world.deposit_pheromone(msg.x_cm, msg.y_cm, msg.amount)
            return []

        if isinstance(msg, SenseMessage):
            samples = world.sample_pheromone(msg.x_cm, msg.y_cm, msg.heading_deg)
            return [format_pheromone_response(msg.robot_id, samples)]

        return []

    manager = ReceiverManager(handle_line)
    tcp_worker = manager.add_tcp_server(cfg.tcp_host, cfg.tcp_port)
    for port in cfg.serial_ports:
        manager.add_serial_port(port, cfg.serial_baudrate, cfg.serial_timeout_s)

    pygame.init()
    if args.headless:
        pygame.display.set_mode((1, 1))
    renderer = Renderer(cfg, render_scale=args.render_scale)
    manager.start()
    start = time.time()

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        world.toggle_paused()
                    elif event.key == pygame.K_c:
                        world.clear_pheromone()
                    elif event.key == pygame.K_t:
                        world.toggle_trails()
                    elif event.key == pygame.K_p:
                        world.toggle_pheromone()

            world.tick()
            renderer.draw(world.snapshot())
            if args.max_seconds > 0 and time.time() - start >= args.max_seconds:
                running = False
    finally:
        manager.stop()
        pygame.quit()

    bound_port = tcp_worker.bound_port if tcp_worker.bound_port is not None else cfg.tcp_port
    logger.info("command center shut down (tcp listener %s:%s)", cfg.tcp_host, bound_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

