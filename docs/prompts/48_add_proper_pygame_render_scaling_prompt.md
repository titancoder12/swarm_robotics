# Prompt 48: Add Proper PyGame Render Scaling

Implement proper PyGame render scaling so large environments fit on screen without changing simulation coordinates.

Requirements:
- Keep the world/simulation resolution unchanged.
- Render the scene into an off-screen world surface at `(width, height)`.
- Use `render_scale` to compute the actual display window size.
- Create the PyGame window at the scaled display size, not the raw world size.
- Scale/blit the world surface into the display window every frame.
- Make screenshots capture what the user actually sees in the scaled window.
- Expose `--render-scale` through the shared env CLI config path.
- Update the relevant docs and project log.
