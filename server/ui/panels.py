from __future__ import annotations

import pygame

from server.ui import colors


def draw_status_panel(surface: pygame.Surface, rect: pygame.Rect, telemetry: dict) -> None:
    pygame.draw.rect(surface, colors.PANEL_BG, rect)
    title_font = pygame.font.SysFont("Menlo", 22, bold=True)
    body_font = pygame.font.SysFont("Menlo", 16)

    y = rect.top + 16
    surface.blit(title_font.render("Command Center", True, colors.TEXT), (rect.left + 16, y))
    y += 36

    rows = [
        f"robots: {telemetry.get('robot_count', 0)}",
        f"paused: {telemetry.get('paused', False)}",
        f"show trails: {telemetry.get('show_trails', False)}",
        f"show pheromone: {telemetry.get('show_pheromone', False)}",
        # Show aggregate field numbers so operators can tell whether the map is
        # decaying, saturating, or has effectively gone empty.
        f"pher total: {telemetry.get('pheromone_total', 0.0):.2f}",
        f"pher max: {telemetry.get('pheromone_max', 0.0):.2f}",
        f"stale: {', '.join(telemetry.get('stale_ids', [])) or 'none'}",
    ]

    for row in rows:
        surface.blit(body_font.render(row, True, colors.TEXT), (rect.left + 16, y))
        y += 24

    y += 8
    controls = [
        "Space: pause",
        "T: toggle trails",
        "P: toggle pheromone",
        "C: clear pheromone",
        "Q / Esc: quit",
    ]
    surface.blit(body_font.render("Controls", True, colors.SUBTEXT), (rect.left + 16, y))
    y += 24
    for row in controls:
        surface.blit(body_font.render(row, True, colors.SUBTEXT), (rect.left + 16, y))
        y += 22
