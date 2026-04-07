from __future__ import annotations

import argparse
import logging
import time

import pygame

from mission_control.comms.protocol import (
    LidarMessage,
    PheromoneMessage,
    PositionMessage,
    ProtocolError,
    SenseMessage,
    TargetMessage,
    format_pheromone_response,
    parse_line,
)
from mission_control.comms.receiver import ReceiverManager
from mission_control.config import CommandCenterConfig
from mission_control.core.world_state import WorldState
from mission_control.ui.renderer import Renderer


logger = logging.getLogger(__name__)


def _parse_multi_value_flags(values: list[str]) -> tuple[str, ...]:
    items: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                items.append(part)
    return tuple(items)


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
    parser.add_argument("--ble-address", action="append", default=[])
    parser.add_argument("--ble-device-name", action="append", default=[])
    parser.add_argument("--ble-timeout", type=float, default=1.0)
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
    ble_addresses = _parse_multi_value_flags(args.ble_address)
    ble_device_names = _parse_multi_value_flags(args.ble_device_name)
    if not ble_addresses and not ble_device_names:
        ble_device_names = ("robot_0",)

    # One shared world model backs both the receiver threads and the renderer.
    cfg = CommandCenterConfig(
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
        serial_ports=tuple(args.serial_port),
        serial_baudrate=args.serial_baudrate,
        serial_timeout_s=args.serial_timeout,
        ble_enable=args.ble_enable,
        ble_addresses=ble_addresses,
        ble_device_names=ble_device_names,
        ble_address=ble_addresses[0] if ble_addresses else "",
        ble_device_name=ble_device_names[0] if ble_device_names else "",
        ble_timeout_s=args.ble_timeout,
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

        if isinstance(msg, LidarMessage):
            world.robot_registry.update_lidar(msg.robot_id, msg.ranges_mm)
            return []

        if isinstance(msg, TargetMessage):
            world.update_target_detection(msg.robot_id, msg.x_cm, msg.y_cm, msg.confidence)
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
        # In reversed-role BLE mode the robot advertises the UART-like
        # peripheral and Mission Control connects as the central/client.
        if cfg.ble_addresses:
            for address in cfg.ble_addresses:
                manager.add_ble_client(
                    address=address,
                    device_name="",
                    write_char_uuid=cfg.ble_write_char_uuid,
                    notify_char_uuid=cfg.ble_notify_char_uuid,
                    timeout_s=cfg.ble_timeout_s,
                )
        else:
            for device_name in cfg.ble_device_names:
                manager.add_ble_client(
                    address="",
                    device_name=device_name,
                    write_char_uuid=cfg.ble_write_char_uuid,
                    notify_char_uuid=cfg.ble_notify_char_uuid,
                    timeout_s=cfg.ble_timeout_s,
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
                        world.flash_control("quit")
                        running = False
                    elif event.key == pygame.K_SPACE:
                        world.flash_control("space")
                        world.toggle_paused()
                    elif event.key == pygame.K_c:
                        world.flash_control("c")
                        world.clear_pheromone()
                    elif event.key == pygame.K_t:
                        world.flash_control("t")
                        world.toggle_trails()
                    elif event.key == pygame.K_p:
                        world.flash_control("p")
                        world.toggle_pheromone()
                    elif event.key == pygame.K_y:
                        world.flash_control("y")
                        world.toggle_targets()

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
