from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - depends on runtime environment
    cv2 = None


@dataclass
class CameraDetection:
    found: bool = False
    distance_m: float = 0.0
    angle_rad: float = 0.0
    confidence: float = 0.0


class CameraTargetDetector:
    """Optional OpenCV-based detector for a visually marked target."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        horizontal_fov_deg: float = 62.0,
        target_width_cm: float = 6.0,
        min_area_px: int = 400,
        hsv_lower: tuple[int, int, int] = (20, 120, 120),
        hsv_upper: tuple[int, int, int] = (40, 255, 255),
        debug: bool = False,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.horizontal_fov_deg = horizontal_fov_deg
        self.target_width_cm = target_width_cm
        self.min_area_px = min_area_px
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper
        self.debug = debug
        self._cap = None
        self._warned_unavailable = False
        # Simple pinhole estimate. Good enough for a first feature-level integration.
        self._focal_px = (self.width * 0.5) / max(math.tan(math.radians(self.horizontal_fov_deg) * 0.5), 1e-6)

    def _ensure_open(self) -> bool:
        if cv2 is None:
            if self.debug and not self._warned_unavailable:
                print("[debug] camera disabled: cv2 is not installed", flush=True)
                self._warned_unavailable = True
            return False
        if self._cap is not None and self._cap.isOpened():
            return True
        cap = cv2.VideoCapture(self.camera_index)
        if not cap or not cap.isOpened():
            if self.debug and not self._warned_unavailable:
                print(f"[debug] camera unavailable at index {self.camera_index}", flush=True)
                self._warned_unavailable = True
            if cap is not None:
                cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        return True

    def detect(self) -> CameraDetection:
        if not self._ensure_open():
            return CameraDetection()

        ok, frame = self._cap.read()
        if not ok or frame is None:
            if self.debug:
                print("[debug] camera frame read failed", flush=True)
            return CameraDetection()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(self.hsv_lower, dtype=np.uint8),
            np.array(self.hsv_upper, dtype=np.uint8),
        )
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return CameraDetection()

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < float(self.min_area_px):
            return CameraDetection()

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return CameraDetection()

        center_x = float(x + w * 0.5)
        pixel_offset = center_x - (self.width * 0.5)
        angle_rad = math.atan2(pixel_offset, self._focal_px)
        distance_m = ((self.target_width_cm / 100.0) * self._focal_px) / max(float(w), 1.0)
        confidence = float(np.clip(area / float(self.width * self.height), 0.0, 1.0))

        if self.debug:
            print(
                f"[debug] camera target found area={area:.1f} bbox=({x},{y},{w},{h}) "
                f"distance_m={distance_m:.3f} angle_deg={math.degrees(angle_rad):.1f}",
                flush=True,
            )

        return CameraDetection(
            found=True,
            distance_m=max(distance_m, 0.0),
            angle_rad=angle_rad,
            confidence=confidence,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
