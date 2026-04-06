from __future__ import annotations

from dataclasses import dataclass, field

from env.config import SwarmConfig


def _default_swarm_config() -> SwarmConfig:
    return SwarmConfig()


@dataclass
class CommandCenterConfig:
    """Configuration shared across command-center components."""

    swarm_cfg: SwarmConfig = field(default_factory=_default_swarm_config)
    scale_cm_per_world_unit: float = 1.0
    pheromone_decay_period_s: float = 0.25
    trail_max_points: int = 120
    robot_stale_after_s: float = 3.0
    status_panel_width_px: int = 280
    fps: int = 30
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 8765
    serial_ports: tuple[str, ...] = ()
    serial_baudrate: int = 115200
    serial_timeout_s: float = 0.1
    ble_enable: bool = False
    ble_addresses: tuple[str, ...] = ()
    ble_device_names: tuple[str, ...] = ()
    ble_address: str = ""
    ble_device_name: str = "CommandCenter"
    ble_timeout_s: float = 1.0
    ble_service_uuid: str = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
    ble_write_char_uuid: str = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
    ble_notify_char_uuid: str = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

    @property
    def world_width_cm(self) -> float:
        return float(self.swarm_cfg.width) * self.scale_cm_per_world_unit

    @property
    def world_height_cm(self) -> float:
        return float(self.swarm_cfg.height) * self.scale_cm_per_world_unit

    @property
    def half_width_cm(self) -> float:
        return self.world_width_cm * 0.5

    @property
    def half_height_cm(self) -> float:
        return self.world_height_cm * 0.5

    @property
    def cell_size_cm(self) -> float:
        return float(self.swarm_cfg.pheromone_cell_size) * self.scale_cm_per_world_unit

    @property
    def agent_radius_cm(self) -> float:
        return float(self.swarm_cfg.agent_radius) * self.scale_cm_per_world_unit

    @property
    def pheromone_samples(self) -> int:
        return int(self.swarm_cfg.pheromone_samples)

    @property
    def decay_factor(self) -> float:
        return float(self.swarm_cfg.pheromone_decay)

    @property
    def diffuse_rate(self) -> float:
        return float(self.swarm_cfg.pheromone_diffuse_rate)

    @property
    def deposit_amount(self) -> float:
        return float(self.swarm_cfg.pheromone_deposit)

    @property
    def pheromone_awareness_radius_cm(self) -> float:
        return float(self.pheromone_samples) * float(self.agent_radius_cm) * 1.5

    @property
    def pheromone_deposit_radius_cm(self) -> float:
        return self.pheromone_awareness_radius_cm
