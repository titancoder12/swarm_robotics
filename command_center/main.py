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
    # Keep CLI flags close to the runtime wiring below so transport and UI
    # configuration stay easy to trace from the entrypoint.
    parser = argparse.ArgumentParser(description="Pygame command center for physical swarm robots.")
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=8765)
    parser.add_argument("--serial-port", action="append", default=[])
    parser.add_argument("--serial-baudrate", type=int, default=115200)
    parser.add_argument("--serial-timeout", type=float, default=0.1)
    parser.add_argument("--ble-enable", action="store_true")
    parser.add_argument("--ble-device-name", default="CommandCenter")
    parser.add_argument("--ble-service-uuid", default="6E400001-B5A3-F393-E0A9-E50E24DCCA9E")
    parser.add_argument("--ble-write-char-uuid", default="6E400002-B5A3-F393-E0A9-E50E24DCCA9E")
    parser.add_argument("--ble-notify-char-uuid", default="6E400003-B5A3-F393-E0A9-E50E24DCCA9E")
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    # One shared world model backs both the receiver threads and the renderer.
    cfg = CommandCenterConfig(
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
        serial_ports=tuple(args.serial_port),
        serial_baudrate=args.serial_baudrate,
        serial_timeout_s=args.serial_timeout,
        ble_enable=args.ble_enable,
        ble_device_name=args.ble_device_name,
        ble_service_uuid=args.ble_service_uuid,
        ble_write_char_uuid=args.ble_write_char_uuid,
        ble_notify_char_uuid=args.ble_notify_char_uuid,
        fps=args.fps,
    )
    world = WorldState(cfg)

    def handle_line(connection_label: str, line: str) -> list[str]:
        # All transport workers funnel through one parser/dispatcher so the
        # protocol behavior is consistent across TCP and serial connections.
        try:
            msg = parse_line(line)
        except ProtocolError as exc:
            logger.warning("protocol error from %s: %s (%r)", connection_label, exc, line)
            return []

        if isinstance(msg, PositionMessage):
            # Position updates only mutate the tracked robot state; they do not
            # require an immediate wire reply.
            world.update_position(msg.robot_id, msg.x_cm, msg.y_cm, msg.heading_deg, connection_label=connection_label)
            return []

        if isinstance(msg, PheromoneMessage):
            # Deposits update the command center's authoritative digital field.
            world.deposit_pheromone(msg.x_cm, msg.y_cm, msg.amount)
            return []

        if isinstance(msg, SenseMessage):
            # `SENSE` is the request/response path: sample the field and return
            # the simulator-compatible pheromone observation slice.
            samples = world.sample_pheromone(msg.x_cm, msg.y_cm, msg.heading_deg)
            return [format_pheromone_response(msg.robot_id, samples)]

        return []

    manager = ReceiverManager(handle_line)
    tcp_worker = manager.add_tcp_server(cfg.tcp_host, cfg.tcp_port)
    for port in cfg.serial_ports:
        # Each serial port is treated as an independent direct robot link.
        manager.add_serial_port(port, cfg.serial_baudrate, cfg.serial_timeout_s)
    if cfg.ble_enable:
        # BLE exposes the same line-oriented protocol over a Nordic-UART-style
        # GATT service so the Pi-side client can reuse POS / SENSE / PHER.
        manager.add_ble_peripheral(
            device_name=cfg.ble_device_name,
            service_uuid=cfg.ble_service_uuid,
            write_char_uuid=cfg.ble_write_char_uuid,
            notify_char_uuid=cfg.ble_notify_char_uuid,
        )

    pygame.init()
    if args.headless:
        # PyGame still needs a display surface even when the window is not
        # intended for interactive use.
        pygame.display.set_mode((1, 1))
    renderer = Renderer(cfg, render_scale=args.render_scale)
    manager.start()
    start = time.time()

    try:
        running = True
        while running:
            # The UI loop owns operator controls while receiver threads keep
            # feeding world-state updates in the background.
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
        # Always stop transport workers before tearing down PyGame so sockets
        # and serial ports are not left dangling on exit.
        manager.stop()
        pygame.quit()

    bound_port = tcp_worker.bound_port if tcp_worker.bound_port is not None else cfg.tcp_port
    logger.info("command center shut down (tcp listener %s:%s)", cfg.tcp_host, bound_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
