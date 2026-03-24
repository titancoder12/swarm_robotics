from __future__ import annotations

from collections import defaultdict, deque


class TrailStore:
    """Stores bounded recent paths for each robot."""

    def __init__(self, max_points: int) -> None:
        self.max_points = max(1, int(max_points))
        self._trails: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.max_points)
        )

    def add_point(self, robot_id: str, x_cm: float, y_cm: float) -> None:
        self._trails[robot_id].append((float(x_cm), float(y_cm)))

    def clear(self) -> None:
        self._trails.clear()

    def snapshot(self) -> dict[str, list[tuple[float, float]]]:
        return {robot_id: list(points) for robot_id, points in self._trails.items()}

