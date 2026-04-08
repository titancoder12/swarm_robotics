from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np
import pygame

from mission_control.config import CommandCenterConfig
from mission_control.core.world_state import WorldSnapshot
from mission_control.ui import colors
from mission_control.ui.panels import draw_status_panel


logger = logging.getLogger(__name__)


@dataclass
class _Layout:
    world_rect: pygame.Rect
    panel_rect: pygame.Rect
    render_scale: float


class Renderer:
    """Owns the Pygame window and draws the current world snapshot."""

    def __init__(self, cfg: CommandCenterConfig, render_scale: float = 1.0, fullscreen: bool = False) -> None:
        self.cfg = cfg
        self.base_render_scale = render_scale
        self.world_width_px = int(cfg.world_width_cm * self.base_render_scale)
        self.world_height_px = int(cfg.world_height_cm * self.base_render_scale)
        self.panel_width_px = cfg.status_panel_width_px
        self.screen_size = (self.world_width_px + self.panel_width_px, self.world_height_px)
        self.fullscreen = bool(fullscreen)
        self.screen = self._create_display_surface()
        pygame.display.set_caption("Mission Control")
        self.robot_font = pygame.font.SysFont("Menlo", 14, bold=True)
        self.clock = pygame.time.Clock()

    def _create_display_surface(self) -> pygame.Surface:
        if self.fullscreen:
            try:
                return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            except pygame.error as exc:
                logger.warning("fullscreen mode unavailable; falling back to windowed mode: %s", exc)
                self.fullscreen = False
        return pygame.display.set_mode(self.screen_size)

    def toggle_fullscreen(self) -> bool:
        self.fullscreen = not self.fullscreen
        self.screen = self._create_display_surface()
        pygame.display.set_caption("Mission Control")
        return self.fullscreen

    def _compute_layout(self) -> _Layout:
        screen_width, screen_height = self.screen.get_size()
        base_total_width = max(1, self.world_width_px + self.panel_width_px)
        base_world_height = max(1, self.world_height_px)
        scale_factor = min(screen_width / base_total_width, screen_height / base_world_height)
        scale_factor = max(scale_factor, 0.1)

        render_scale = self.base_render_scale * scale_factor
        world_width_px = max(1, int(round(self.cfg.world_width_cm * render_scale)))
        world_height_px = max(1, int(round(self.cfg.world_height_cm * render_scale)))
        panel_width_px = max(220, int(round(self.panel_width_px * scale_factor)))

        total_width = min(screen_width, world_width_px + panel_width_px)
        origin_x = max(0, (screen_width - total_width) // 2)
        origin_y = max(0, (screen_height - world_height_px) // 2)

        world_rect = pygame.Rect(origin_x, origin_y, world_width_px, world_height_px)
        panel_rect = pygame.Rect(origin_x + world_width_px, origin_y, panel_width_px, world_height_px)
        return _Layout(world_rect=world_rect, panel_rect=panel_rect, render_scale=render_scale)

    def draw(self, snapshot: WorldSnapshot) -> None:
        self.screen.fill(colors.BG)
        layout = self._compute_layout()
        world_rect = layout.world_rect
        panel_rect = layout.panel_rect

        self._draw_grid(world_rect, layout.render_scale)
        self._draw_pheromone(world_rect, snapshot.pheromone_grid, layout.render_scale)
        self._draw_nest(world_rect, layout.render_scale)
        self._draw_trails(world_rect, snapshot.trails, layout.render_scale)
        self._draw_robots(world_rect, snapshot, layout.render_scale)
        if snapshot.telemetry.get("show_targets", True):
            self._draw_targets(world_rect, snapshot, layout.render_scale)
        draw_status_panel(self.screen, panel_rect, snapshot.telemetry)
        pygame.display.flip()
        self.clock.tick(self.cfg.fps)

    def _draw_grid(self, rect: pygame.Rect, render_scale: float) -> None:
        spacing = max(1, int(self.cfg.cell_size_cm * render_scale))
        for x in range(rect.left, rect.right, spacing):
            pygame.draw.line(self.screen, colors.GRID, (x, rect.top), (x, rect.bottom), 1)
        for y in range(rect.top, rect.bottom, spacing):
            pygame.draw.line(self.screen, colors.GRID, (rect.left, y), (rect.right, y), 1)

    def _draw_nest(self, rect: pygame.Rect, render_scale: float) -> None:
        center = (rect.centerx, rect.centery)
        radius = max(6, int(self.cfg.swarm_cfg.nest_radius * render_scale))
        pygame.draw.circle(self.screen, colors.NEST, center, radius, width=2)
        pygame.draw.circle(self.screen, colors.NEST, center, 4)

    def _draw_pheromone(self, rect: pygame.Rect, grid: np.ndarray, render_scale: float) -> None:
        if grid.size == 0:
            return
        max_val = float(grid.max())
        if max_val <= 1e-6:
            return
        cell_px = max(1, int(self.cfg.cell_size_cm * render_scale))
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

    def _draw_trails(self, rect: pygame.Rect, trails: dict[str, list[tuple[float, float]]], render_scale: float) -> None:
        for points in trails.values():
            if len(points) < 2:
                continue
            pygame.draw.lines(
                self.screen,
                colors.TRAIL,
                False,
                [self._world_to_screen(rect, x_cm, y_cm, render_scale) for x_cm, y_cm in points],
                width=2,
            )

    def _draw_robots(self, rect: pygame.Rect, snapshot: WorldSnapshot, render_scale: float) -> None:
        stale_ids = set(snapshot.telemetry.get("stale_ids", []))
        for index, (robot_id, state) in enumerate(sorted(snapshot.robots.items())):
            center = self._world_to_screen(rect, state.x_cm, state.y_cm, render_scale)
            base_color = colors.ROBOT_COLORS[index % len(colors.ROBOT_COLORS)]
            draw_color = colors.STALE if robot_id in stale_ids else base_color
            if state.heading_deg is not None and state.lidar_ranges_mm:
                self._draw_lidar_semicircle(rect, center, state.heading_deg, state.lidar_ranges_mm, render_scale)
            pygame.draw.circle(self.screen, draw_color, center, max(6, int(self.cfg.agent_radius_cm * 0.6 * render_scale)))
            if state.heading_deg is not None:
                theta = math.radians(state.heading_deg)
                tip = (
                    center[0] + int(math.cos(theta) * 7),
                    center[1] - int(math.sin(theta) * 7),
                )
                pygame.draw.line(self.screen, colors.TEXT, center, tip, 2)
            label = self.robot_font.render(robot_id, True, colors.TEXT)
            self.screen.blit(label, (center[0] + 8, center[1] - 8))

    def _draw_targets(self, rect: pygame.Rect, snapshot: WorldSnapshot, render_scale: float) -> None:
        for robot_id, state in sorted(snapshot.robots.items()):
            if state.target_x_cm is None or state.target_y_cm is None or state.last_target_seen is None:
                continue
            age_s = max(0.0, time.time() - state.last_target_seen)
            fade = max(0.0, 1.0 - age_s / 3.0)
            if fade <= 0.0:
                continue
            center = self._world_to_screen(rect, state.target_x_cm, state.target_y_cm, render_scale)
            radius = max(5, int((6.0 + 12.0 * min(1.0, max(0.0, state.target_confidence))) * render_scale))
            ring_radius = radius + max(4, int(6 * render_scale))
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
        render_scale: float,
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
                    center[0] + int(math.cos(sample_heading_rad) * max_range_cm * render_scale),
                    center[1] - int(math.sin(sample_heading_rad) * max_range_cm * render_scale),
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
                center[0] + int(math.cos(math.radians(ray_heading_deg)) * distance_cm * render_scale),
                center[1] - int(math.sin(math.radians(ray_heading_deg)) * distance_cm * render_scale),
            )
            pygame.draw.circle(self.screen, colors.LIDAR_HIT, dot, 3)

    def _world_to_screen(self, rect: pygame.Rect, x_cm: float, y_cm: float, render_scale: float) -> tuple[int, int]:
        # World coordinates are centered at the nest, while screen coordinates
        # are top-left-origin with +y pointing downward.
        x = int((x_cm + self.cfg.half_width_cm) * render_scale)
        y = int((self.cfg.half_height_cm - y_cm) * render_scale)
        return rect.left + x, rect.top + y

    @staticmethod
    def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
        return tuple(int(x + (y - x) * t) for x, y in zip(a, b))
