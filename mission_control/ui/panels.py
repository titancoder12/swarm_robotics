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

SCROLL_STEP_PX = 32


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


def _wrap_text(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _append_wrapped_block(
    blocks: list[tuple[pygame.font.Font, str, tuple[int, int, int], int, int]],
    font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int],
    indent_px: int,
    max_width: int,
    line_height: int,
) -> None:
    for line in _wrap_text(font, text, max_width):
        blocks.append((font, line, color, indent_px, line_height))


def _build_status_blocks(
    telemetry: dict,
    body_font: pygame.font.Font,
    content_width: int,
) -> list[tuple[pygame.font.Font, str, tuple[int, int, int], int, int]]:
    blocks: list[tuple[pygame.font.Font, str, tuple[int, int, int], int, int]] = []
    robot_rows = telemetry.get("robot_rows", [])
    stale_ids = set(telemetry.get("stale_ids", []))

    summary_rows = [
        f"robots: {telemetry.get('robot_count', 0)}",
        f"tracked: {telemetry.get('tracked_count', telemetry.get('robot_count', 0))}",
        f"paused: {telemetry.get('paused', False)}",
        f"show trails: {telemetry.get('show_trails', False)}",
        f"show pheromone: {telemetry.get('show_pheromone', False)}",
        f"show targets: {telemetry.get('show_targets', False)}",
        f"pher total: {telemetry.get('pheromone_total', 0.0):.2f}",
        f"pher max: {telemetry.get('pheromone_max', 0.0):.2f}",
        f"stale: {', '.join(telemetry.get('stale_ids', [])) or 'none'}",
    ]
    for row in summary_rows:
        _append_wrapped_block(blocks, body_font, row, colors.TEXT, 0, content_width, 24)

    blocks.append((body_font, "", colors.TEXT, 0, 8))
    blocks.append((body_font, "Robots", colors.SUBTEXT, 0, 24))

    if not robot_rows:
        _append_wrapped_block(blocks, body_font, "none", colors.SUBTEXT, 0, content_width, 22)
        return blocks

    for row in robot_rows[:8]:
        heading = row.get("heading_deg")
        heading_text = "--" if heading is None else f"{heading:.0f}"
        is_stale = row["robot_id"] in stale_ids
        row_color = colors.STALE if is_stale else colors.TEXT
        label = (
            f"{row['robot_id']}: "
            f"{'disconnected' if is_stale else 'connected'} "
            f"x={row['x_cm']:+.1f} "
            f"y={row['y_cm']:+.1f} "
            f"hdg={heading_text} "
            f"age={row['age_s']:.1f}s"
        )
        _append_wrapped_block(blocks, body_font, label, row_color, 0, content_width, 22)

        lidar_ranges_mm = row.get("lidar_ranges_mm", [])
        if lidar_ranges_mm:
            lidar_text = "lidar: " + " ".join(f"{int(round(value)):>3d}" for value in lidar_ranges_mm[:9])
            lidar_color = colors.SUBTEXT if not is_stale else colors.STALE
            _append_wrapped_block(blocks, body_font, lidar_text, lidar_color, 8, content_width - 8, 20)

        target_x_cm = row.get("target_x_cm")
        target_y_cm = row.get("target_y_cm")
        target_age_s = row.get("target_age_s")
        if target_x_cm is not None and target_y_cm is not None and target_age_s is not None and target_age_s <= 3.0:
            target_text = (
                f"target: x={target_x_cm:+.1f} y={target_y_cm:+.1f} "
                f"conf={row.get('target_confidence', 0.0):.2f} age={target_age_s:.1f}s"
            )
            target_color = colors.TARGET_MARKER if not is_stale else colors.STALE
            _append_wrapped_block(blocks, body_font, target_text, target_color, 8, content_width - 8, 20)

        status_age_s = row.get("status_age_s")
        if status_age_s is not None and status_age_s <= 5.0:
            _append_wrapped_block(blocks, body_font, _format_action(row), row_color, 8, content_width - 8, 20)

            obstacle_text, obstacle_color = _infer_obstacle(
                float(row.get("front_min_mm", 0.0)),
                float(row.get("left_min_mm", 0.0)),
                float(row.get("right_min_mm", 0.0)),
            )
            if is_stale:
                obstacle_color = colors.STALE
            _append_wrapped_block(blocks, body_font, obstacle_text, obstacle_color, 8, content_width - 8, 20)

            if row.get("camera_found", False):
                camera_text = (
                    f"camera: target {float(row.get('camera_distance_m', 0.0)):.2f} m, "
                    f"{float(row.get('camera_angle_deg', 0.0)):+.1f} deg"
                )
                camera_color = colors.TRAIL if not is_stale else colors.STALE
            else:
                camera_text = "camera: no target"
                camera_color = colors.SUBTEXT if not is_stale else colors.STALE
            _append_wrapped_block(blocks, body_font, camera_text, camera_color, 8, content_width - 8, 20)

            serial_cmd = str(row.get("serial_cmd", "")).replace(";", ",")
            serial_reply = str(row.get("serial_reply", ""))
            serial_text = f"motor: {serial_cmd or '-'} -> {serial_reply or '-'}"
            serial_color = colors.TRAIL if row.get("serial_ok", False) and not is_stale else (colors.HEAT_HOT if not is_stale else colors.STALE)
            _append_wrapped_block(blocks, body_font, serial_text, serial_color, 8, content_width - 8, 20)

    return blocks


def _draw_scrollbar(surface: pygame.Surface, viewport_rect: pygame.Rect, content_height: int, scroll_px: int) -> None:
    if content_height <= viewport_rect.height:
        return
    track_width = 8
    track_rect = pygame.Rect(viewport_rect.right - track_width, viewport_rect.top, track_width, viewport_rect.height)
    pygame.draw.rect(surface, colors.GRID, track_rect, border_radius=4)

    visible_ratio = viewport_rect.height / max(content_height, 1)
    thumb_height = max(24, int(track_rect.height * visible_ratio))
    max_scroll = max(0, content_height - viewport_rect.height)
    thumb_travel = max(1, track_rect.height - thumb_height)
    thumb_offset = int((scroll_px / max(max_scroll, 1)) * thumb_travel)
    thumb_rect = pygame.Rect(track_rect.left, track_rect.top + thumb_offset, track_width, thumb_height)
    pygame.draw.rect(surface, colors.SUBTEXT, thumb_rect, border_radius=4)


def draw_status_panel(surface: pygame.Surface, rect: pygame.Rect, telemetry: dict) -> None:
    pygame.draw.rect(surface, colors.PANEL_BG, rect)
    title_font = pygame.font.SysFont("Menlo", 22, bold=True)
    body_font = pygame.font.SysFont("Menlo", 16)
    body_font_bold = pygame.font.SysFont("Menlo", 16, bold=True)
    body_font_flash = pygame.font.SysFont("Menlo", 19, bold=True)

    title_x = rect.left + 16
    title_y = rect.top + 16
    content_width = rect.width - 32

    surface.blit(title_font.render("Mission Control", True, colors.TEXT), (title_x, title_y))

    controls = [
        ("space", "Space", ": pause"),
        ("status_scroll", "Wheel / PgUpDn", ": scroll status"),
        ("f", "F / F11", ": toggle fullscreen"),
        ("t", "T", ": toggle trails"),
        ("p", "P", ": toggle pheromone"),
        ("y", "Y", ": toggle targets"),
        ("c", "C", ": clear pheromone"),
        ("quit", "Q / Esc", ": quit"),
    ]
    controls_y = rect.bottom - (24 + len(controls) * 22 + 20)
    viewport_top = title_y + 36
    viewport_bottom = controls_y - 12
    viewport_height = max(0, viewport_bottom - viewport_top)
    viewport_rect = pygame.Rect(rect.left + 16, viewport_top, content_width, viewport_height)

    blocks = _build_status_blocks(telemetry, body_font, content_width - 12)
    content_height = max(1, sum(line_height for _, _, _, _, line_height in blocks))
    content_surface = pygame.Surface((max(1, content_width - 8), content_height), pygame.SRCALPHA)
    y = 0
    for font, line, color, indent_px, line_height in blocks:
        if line:
            content_surface.blit(font.render(line, True, color), (indent_px, y))
        y += line_height

    scroll_px = max(0, int(telemetry.get("status_scroll_px", 0)))
    max_scroll = max(0, content_height - viewport_rect.height)
    scroll_px = min(scroll_px, max_scroll)

    viewport_surface = pygame.Surface((max(1, viewport_rect.width), max(1, viewport_rect.height)))
    viewport_surface.fill(colors.PANEL_BG)
    if viewport_rect.height > 0:
        viewport_surface.blit(content_surface, (0, -scroll_px))
    surface.blit(viewport_surface, viewport_rect.topleft)
    _draw_scrollbar(surface, viewport_rect, content_height, scroll_px)

    flashed_controls = set(telemetry.get("flashed_controls", []))
    surface.blit(body_font.render("Controls", True, colors.SUBTEXT), (rect.left + 16, controls_y))
    y = controls_y + 24
    for control_id, trigger_label, description in controls:
        x = rect.left + 16
        if control_id in flashed_controls:
            trigger_surface = body_font_flash.render(trigger_label, True, colors.TEXT)
            surface.blit(trigger_surface, (x, y - 2))
            description_surface = body_font_bold.render(description, True, colors.TEXT)
            surface.blit(description_surface, (x + trigger_surface.get_width() + 8, y))
        else:
            trigger_surface = body_font.render(trigger_label, True, colors.SUBTEXT)
            description_surface = body_font.render(description, True, colors.SUBTEXT)
            surface.blit(trigger_surface, (x, y))
            surface.blit(description_surface, (x + trigger_surface.get_width(), y))
        y += 22
