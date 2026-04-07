from __future__ import annotations

import math
import time

import numpy as np
import pygame

from mission_control.config import CommandCenterConfig
from mission_control.core.world_state import WorldSnapshot
from mission_control.ui import colors
from mission_control.ui.panels import draw_status_panel


class Renderer:
    """Owns the Pygame window and draws the current world snapshot."""

    def __init__(self, cfg: CommandCenterConfig, render_scale: float = 1.0) -> None:
        self.cfg = cfg
        self.render_scale = render_scale
        self.world_width_px = int(cfg.world_width_cm * render_scale)
        self.world_height_px = int(cfg.world_height_cm * render_scale)
        self.panel_width_px = cfg.status_panel_width_px
        self.screen_size = (self.world_width_px + self.panel_width_px, self.world_height_px)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Mission Control")
        self.robot_font = pygame.font.SysFont("Menlo", 14, bold=True)
        self.clock = pygame.time.Clock()

    def draw(self, snapshot: WorldSnapshot) -> None:
        self.screen.fill(colors.BG)
        world_rect = pygame.Rect(0, 0, self.world_width_px, self.world_height_px)
        panel_rect = pygame.Rect(self.world_width_px, 0, self.panel_width_px, self.world_height_px)

        self._draw_grid(world_rect)
        self._draw_pheromone(world_rect, snapshot.pheromone_grid)
        self._draw_nest(world_rect)
        self._draw_trails(world_rect, snapshot.trails)
        self._draw_robots(world_rect, snapshot)
        if snapshot.telemetry.get("show_targets", True):
            self._draw_targets(world_rect, snapshot)
        draw_status_panel(self.screen, panel_rect, snapshot.telemetry)
        pygame.display.flip()
        self.clock.tick(self.cfg.fps)

    def _draw_grid(self, rect: pygame.Rect) -> None:
        spacing = max(1, int(self.cfg.cell_size_cm * self.render_scale))
        for x in range(rect.left, rect.right, spacing):
            pygame.draw.line(self.screen, colors.GRID, (x, rect.top), (x, rect.bottom), 1)
        for y in range(rect.top, rect.bottom, spacing):
            pygame.draw.line(self.screen, colors.GRID, (rect.left, y), (rect.right, y), 1)

    def _draw_nest(self, rect: pygame.Rect) -> None:
        center = (rect.centerx, rect.centery)
        radius = max(6, int(self.cfg.swarm_cfg.nest_radius * self.render_scale))
        pygame.draw.circle(self.screen, colors.NEST, center, radius, width=2)
        pygame.draw.circle(self.screen, colors.NEST, center, 4)

    def _draw_pheromone(self, rect: pygame.Rect, grid: np.ndarray) -> None:
        if grid.size == 0:
            return
        max_val = float(grid.max())
        if max_val <= 1e-6:
            return
        cell_px = max(1, int(self.cfg.cell_size_cm * self.render_scale))
        heat = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        for gy in range(grid.shape[0]):
            for gx in range(grid.shape[1]):
                val = float(grid[gy, gx])
                if val <= 1e-6:
                    continue
                ratio = min(1.0, val / max_val)
                color = self._lerp(colors.HEAT_COLD, colors.HEAT_HOT, ratio)
                alpha = int(28 + 150 * ratio)
                px = int(gx * cell_px)
                # Pheromone grid rows increase with +y in world coordinates,
                # while the screen uses top-left origin with +y downward. Flip
                # the row index here so heatmap cells line up with robot/trail
                # rendering, which already uses _world_to_screen(...).
                py = int(rect.height - (gy + 1) * cell_px)
                pygame.draw.rect(heat, (*color, alpha), pygame.Rect(px, py, cell_px, cell_px))
        self.screen.blit(heat, rect.topleft)

    def _draw_trails(self, rect: pygame.Rect, trails: dict[str, list[tuple[float, float]]]) -> None:
        for points in trails.values():
            if len(points) < 2:
                continue
            pygame.draw.lines(
                self.screen,
                colors.TRAIL,
                False,
                [self._world_to_screen(rect, x_cm, y_cm) for x_cm, y_cm in points],
                width=2,
            )

    def _draw_robots(self, rect: pygame.Rect, snapshot: WorldSnapshot) -> None:
        stale_ids = set(snapshot.telemetry.get("stale_ids", []))
        for index, (robot_id, state) in enumerate(sorted(snapshot.robots.items())):
            center = self._world_to_screen(rect, state.x_cm, state.y_cm)
            base_color = colors.ROBOT_COLORS[index % len(colors.ROBOT_COLORS)]
            draw_color = colors.STALE if robot_id in stale_ids else base_color
            if state.heading_deg is not None and state.lidar_ranges_mm:
                self._draw_lidar_semicircle(rect, center, state.heading_deg, state.lidar_ranges_mm)
            pygame.draw.circle(self.screen, draw_color, center, max(6, int(self.cfg.agent_radius_cm * 0.6 * self.render_scale)))
            if state.heading_deg is not None:
                theta = math.radians(state.heading_deg)
                tip = (
                    center[0] + int(math.cos(theta) * 7),
                    center[1] - int(math.sin(theta) * 7),
                )
                pygame.draw.line(self.screen, colors.TEXT, center, tip, 2)
            label = self.robot_font.render(robot_id, True, colors.TEXT)
            self.screen.blit(label, (center[0] + 8, center[1] - 8))

    def _draw_targets(self, rect: pygame.Rect, snapshot: WorldSnapshot) -> None:
        for robot_id, state in sorted(snapshot.robots.items()):
            if state.target_x_cm is None or state.target_y_cm is None or state.last_target_seen is None:
                continue
            age_s = max(0.0, time.time() - state.last_target_seen)
            fade = max(0.0, 1.0 - age_s / 3.0)
            if fade <= 0.0:
                continue
            center = self._world_to_screen(rect, state.target_x_cm, state.target_y_cm)
            radius = max(5, int((6.0 + 12.0 * min(1.0, max(0.0, state.target_confidence))) * self.render_scale))
            ring_radius = radius + max(4, int(6 * self.render_scale))
            marker_surface = pygame.Surface((ring_radius * 2 + 4, ring_radius * 2 + 4), pygame.SRCALPHA)
            local_center = (marker_surface.get_width() // 2, marker_surface.get_height() // 2)
            ring_alpha = int(90 * fade)
            fill_alpha = int(180 * fade)
            pygame.draw.circle(marker_surface, (*colors.TARGET_RING, ring_alpha), local_center, ring_radius, width=2)
            pygame.draw.circle(marker_surface, (*colors.TARGET_MARKER, fill_alpha), local_center, radius)
            self.screen.blit(marker_surface, (center[0] - local_center[0], center[1] - local_center[1]))

    def _draw_lidar_semicircle(
        self,
        rect: pygame.Rect,
        center: tuple[int, int],
        heading_deg: float,
        lidar_ranges_mm: tuple[float, ...],
    ) -> None:
        max_range_cm = float(self.cfg.swarm_cfg.lidar_max_range) * self.cfg.scale_cm_per_world_unit
        if max_range_cm <= 0.0:
            return

        arc_points = []
        for idx in range(25):
            frac = idx / 24.0
            sample_heading_deg = heading_deg - 90.0 + frac * 180.0
            sample_heading_rad = math.radians(sample_heading_deg)
            arc_points.append(
                (
                    center[0] + int(math.cos(sample_heading_rad) * max_range_cm * self.render_scale),
                    center[1] - int(math.sin(sample_heading_rad) * max_range_cm * self.render_scale),
                )
            )
        if len(arc_points) >= 2:
            pygame.draw.lines(self.screen, colors.LIDAR_ARC, False, arc_points, 1)
            pygame.draw.line(self.screen, colors.LIDAR_ARC, arc_points[0], arc_points[-1], 1)

        if not lidar_ranges_mm:
            return
        ray_count = max(1, min(len(lidar_ranges_mm), int(self.cfg.swarm_cfg.lidar_rays)))
        for ray_idx, distance_mm in enumerate(lidar_ranges_mm[:ray_count]):
            frac = ray_idx / max(1, ray_count - 1)
            ray_heading_deg = heading_deg - 90.0 + frac * 180.0
            distance_cm = min(max_range_cm, max(0.0, float(distance_mm) / 10.0))
            dot = (
                center[0] + int(math.cos(math.radians(ray_heading_deg)) * distance_cm * self.render_scale),
                center[1] - int(math.sin(math.radians(ray_heading_deg)) * distance_cm * self.render_scale),
            )
            pygame.draw.circle(self.screen, colors.LIDAR_HIT, dot, 3)

    def _world_to_screen(self, rect: pygame.Rect, x_cm: float, y_cm: float) -> tuple[int, int]:
        # World coordinates are centered at the nest, while screen coordinates
        # are top-left-origin with +y pointing downward.
        x = int((x_cm + self.cfg.half_width_cm) * self.render_scale)
        y = int((self.cfg.half_height_cm - y_cm) * self.render_scale)
        return rect.left + x, rect.top + y

    @staticmethod
    def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
        return tuple(int(x + (y - x) * t) for x, y in zip(a, b))
