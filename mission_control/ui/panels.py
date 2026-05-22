from __future__ import annotations

import pygame

from mission_control.ui import colors


ACTION_LABELS: dict[int, str] = {
    0: "reverse-left",
    1: "reverse-left-deposit",
    2: "reverse-straight",
    3: "reverse-straight-deposit",
    4: "reverse-right",
    5: "reverse-right-deposit",
    6: "stop-left",
    7: "stop-left-deposit",
    8: "stop",
    9: "stop-deposit",
    10: "stop-right",
    11: "stop-right-deposit",
    12: "forward-left",
    13: "forward-left-deposit",
    14: "forward-straight",
    15: "forward-straight-deposit",
    16: "forward-right",
    17: "forward-right-deposit",
}


def _infer_obstacle(front_min_mm: float, left_min_mm: float, right_min_mm: float) -> tuple[str, tuple[int, int, int]]:
    valid = [value for value in (front_min_mm, left_min_mm, right_min_mm) if value > 0.0]
    if not valid:
        return "sense: no obstacle data", colors.SUBTEXT
    if 0.0 < front_min_mm < 120.0:
        return f"sense: blocked ahead ({front_min_mm:.0f} mm)", colors.TARGET_MARKER
    if 0.0 < front_min_mm < 300.0:
        return f"sense: close obstacle ahead ({front_min_mm:.0f} mm)", colors.HEAT_HOT
    nearest = min(valid)
    if nearest < 500.0:
        return f"sense: near obstacle ({nearest:.0f} mm)", colors.HEAT_HOT
    return f"sense: path mostly clear ({nearest:.0f} mm)", colors.TRAIL


def _format_action(row: dict) -> str:
    action_id = int(row.get("action_id", -1))
    label = ACTION_LABELS.get(action_id, f"action_{action_id}")
    throttle = float(row.get("throttle", 0.0))
    turn = float(row.get("turn", 0.0))
    deposit = bool(row.get("deposit", False))
    motion: list[str] = []
    motion.append("forward" if throttle > 0 else ("reverse" if throttle < 0 else "stop"))
    motion.append("left" if turn < 0 else ("right" if turn > 0 else "straight"))
    if deposit:
        motion.append("deposit")
    return f"action: {label} ({', '.join(motion)})"


def _draw_wrapped_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int],
    x: int,
    y: int,
    max_width: int,
    line_height: int,
) -> int:
    words = text.split()
    if not words:
        return y + line_height

    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.size(candidate)[0] <= max_width:
            current = candidate
            continue
        surface.blit(font.render(current, True, color), (x, y))
        y += line_height
        current = word

    surface.blit(font.render(current, True, color), (x, y))
    return y + line_height


def draw_status_panel(surface: pygame.Surface, rect: pygame.Rect, telemetry: dict) -> None:
    pygame.draw.rect(surface, colors.PANEL_BG, rect)
    title_font = pygame.font.SysFont("Menlo", 22, bold=True)
    body_font = pygame.font.SysFont("Menlo", 16)
    body_font_bold = pygame.font.SysFont("Menlo", 16, bold=True)
    body_font_flash = pygame.font.SysFont("Menlo", 19, bold=True)
    content_width = rect.width - 32

    y = rect.top + 16
    surface.blit(title_font.render("Mission Control", True, colors.TEXT), (rect.left + 16, y))
    y += 36

    rows = [
        f"robots: {telemetry.get('robot_count', 0)}",
        f"tracked: {telemetry.get('tracked_count', telemetry.get('robot_count', 0))}",
        f"paused: {telemetry.get('paused', False)}",
        f"show trails: {telemetry.get('show_trails', False)}",
        f"show pheromone: {telemetry.get('show_pheromone', False)}",
        f"show targets: {telemetry.get('show_targets', False)}",
        # Show aggregate field numbers so operators can tell whether the map is
        # decaying, saturating, or has effectively gone empty.
        f"pher total: {telemetry.get('pheromone_total', 0.0):.2f}",
        f"pher max: {telemetry.get('pheromone_max', 0.0):.2f}",
        f"stale: {', '.join(telemetry.get('stale_ids', [])) or 'none'}",
    ]

    for row in rows:
        y = _draw_wrapped_text(surface, body_font, row, colors.TEXT, rect.left + 16, y, content_width, 24)

    y += 8
    robot_rows = telemetry.get("robot_rows", [])
    stale_ids = set(telemetry.get("stale_ids", []))
    surface.blit(body_font.render("Robots", True, colors.SUBTEXT), (rect.left + 16, y))
    y += 24
    if not robot_rows:
        y = _draw_wrapped_text(surface, body_font, "none", colors.SUBTEXT, rect.left + 16, y, content_width, 22)
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
            y = _draw_wrapped_text(surface, body_font, label, row_color, rect.left + 16, y, content_width, 22)
            lidar_ranges_mm = row.get("lidar_ranges_mm", [])
            if lidar_ranges_mm:
                lidar_text = "lidar: " + " ".join(f"{int(round(value)):>3d}" for value in lidar_ranges_mm[:9])
                lidar_color = colors.SUBTEXT if not is_stale else colors.STALE
                y = _draw_wrapped_text(surface, body_font, lidar_text, lidar_color, rect.left + 24, y, content_width - 8, 20)
            target_x_cm = row.get("target_x_cm")
            target_y_cm = row.get("target_y_cm")
            target_age_s = row.get("target_age_s")
            if target_x_cm is not None and target_y_cm is not None and target_age_s is not None and target_age_s <= 3.0:
                target_text = (
                    f"target: x={target_x_cm:+.1f} y={target_y_cm:+.1f} "
                    f"conf={row.get('target_confidence', 0.0):.2f} age={target_age_s:.1f}s"
                )
                target_color = colors.TARGET_MARKER if not is_stale else colors.STALE
                y = _draw_wrapped_text(surface, body_font, target_text, target_color, rect.left + 24, y, content_width - 8, 20)
            status_age_s = row.get("status_age_s")
            if status_age_s is not None and status_age_s <= 5.0:
                action_text = _format_action(row)
                y = _draw_wrapped_text(surface, body_font, action_text, row_color, rect.left + 24, y, content_width - 8, 20)

                obstacle_text, obstacle_color = _infer_obstacle(
                    float(row.get("front_min_mm", 0.0)),
                    float(row.get("left_min_mm", 0.0)),
                    float(row.get("right_min_mm", 0.0)),
                )
                if is_stale:
                    obstacle_color = colors.STALE
                y = _draw_wrapped_text(surface, body_font, obstacle_text, obstacle_color, rect.left + 24, y, content_width - 8, 20)

                if row.get("camera_found", False):
                    camera_text = (
                        f"camera: target {float(row.get('camera_distance_m', 0.0)):.2f} m, "
                        f"{float(row.get('camera_angle_deg', 0.0)):+.1f} deg"
                    )
                    camera_color = colors.TRAIL if not is_stale else colors.STALE
                else:
                    camera_text = "camera: no target"
                    camera_color = colors.SUBTEXT if not is_stale else colors.STALE
                y = _draw_wrapped_text(surface, body_font, camera_text, camera_color, rect.left + 24, y, content_width - 8, 20)

                serial_cmd = str(row.get("serial_cmd", "")).replace(";", ",")
                serial_reply = str(row.get("serial_reply", ""))
                serial_text = f"motor: {serial_cmd or '-'} -> {serial_reply or '-'}"
                serial_color = colors.TRAIL if row.get("serial_ok", False) and not is_stale else (colors.HEAT_HOT if not is_stale else colors.STALE)
                y = _draw_wrapped_text(surface, body_font, serial_text, serial_color, rect.left + 24, y, content_width - 8, 20)

    y += 8
    flashed_controls = set(telemetry.get("flashed_controls", []))
    controls = [
        ("space", "Space", ": pause"),
        ("f", "F / F11", ": toggle fullscreen"),
        ("t", "T", ": toggle trails"),
        ("p", "P", ": toggle pheromone"),
        ("y", "Y", ": toggle targets"),
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
