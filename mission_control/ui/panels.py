from __future__ import annotations

import pygame

from mission_control.ui import colors


def draw_status_panel(surface: pygame.Surface, rect: pygame.Rect, telemetry: dict) -> None:
    pygame.draw.rect(surface, colors.PANEL_BG, rect)
    title_font = pygame.font.SysFont("Menlo", 22, bold=True)
    body_font = pygame.font.SysFont("Menlo", 16)
    body_font_bold = pygame.font.SysFont("Menlo", 16, bold=True)
    body_font_flash = pygame.font.SysFont("Menlo", 19, bold=True)

    y = rect.top + 16
    surface.blit(title_font.render("Mission Control", True, colors.TEXT), (rect.left + 16, y))
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
    robot_rows = telemetry.get("robot_rows", [])
    stale_ids = set(telemetry.get("stale_ids", []))
    surface.blit(body_font.render("Robots", True, colors.SUBTEXT), (rect.left + 16, y))
    y += 24
    if not robot_rows:
        surface.blit(body_font.render("none", True, colors.SUBTEXT), (rect.left + 16, y))
        y += 22
    else:
        for row in robot_rows[:8]:
            heading = row.get("heading_deg")
            heading_text = "--" if heading is None else f"{heading:.0f}"
            is_stale = row["robot_id"] in stale_ids
            status_text = "disconnected" if is_stale else "connected"
            row_color = colors.STALE if is_stale else colors.TEXT
            label = (
                f"{row['robot_id']}: "
                f"{status_text} "
                f"x={row['x_cm']:+.1f} "
                f"y={row['y_cm']:+.1f} "
                f"hdg={heading_text} "
                f"age={row['age_s']:.1f}s"
            )
            surface.blit(body_font.render(label, True, row_color), (rect.left + 16, y))
            y += 22
            lidar_ranges_mm = row.get("lidar_ranges_mm", [])
            if lidar_ranges_mm:
                lidar_text = "lidar: " + " ".join(f"{int(round(value)):>3d}" for value in lidar_ranges_mm[:9])
                lidar_color = colors.SUBTEXT if not is_stale else colors.STALE
                surface.blit(body_font.render(lidar_text, True, lidar_color), (rect.left + 24, y))
                y += 20

    y += 8
    flashed_controls = set(telemetry.get("flashed_controls", []))
    controls = [
        ("space", "Space", ": pause"),
        ("t", "T", ": toggle trails"),
        ("p", "P", ": toggle pheromone"),
        ("c", "C", ": clear pheromone"),
        ("quit", "Q / Esc", ": quit"),
    ]
    surface.blit(body_font.render("Controls", True, colors.SUBTEXT), (rect.left + 16, y))
    y += 24
    for control_id, trigger_label, description in controls:
        x = rect.left + 16
        if control_id in flashed_controls:
            trigger_surface = body_font_flash.render(trigger_label, True, colors.TEXT)
            #highlight_rect = trigger_surface.get_rect(topleft=(x - 6, y - 2)).inflate(12, 6)
            #pygame.draw.rect(surface, colors.TRAIL, highlight_rect, border_radius=6)
            surface.blit(trigger_surface, (x, y - 2))
            description_surface = body_font_bold.render(description, True, colors.TEXT)
            surface.blit(description_surface, (x + trigger_surface.get_width() + 8, y))
        else:
            trigger_surface = body_font.render(trigger_label, True, colors.SUBTEXT)
            description_surface = body_font.render(description, True, colors.SUBTEXT)
            surface.blit(trigger_surface, (x, y))
            surface.blit(description_surface, (x + trigger_surface.get_width(), y))
        y += 22
