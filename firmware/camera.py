from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - depends on runtime environment
    cv2 = None

try:
    from picamera2 import Picamera2
except ImportError:  # pragma: no cover - depends on runtime environment
    Picamera2 = None


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
        hold_time_s: float = 0.75,
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
        self.hold_time_s = max(0.0, hold_time_s)
        self.debug = debug
        self._cap = None
        self._picam2 = None
        self._warned_unavailable = False
        self._reported_backend = False
        self._next_open_retry_at = 0.0
        self._open_retry_backoff_s = 2.0
        self._last_detection = CameraDetection()
        self._last_detection_at = 0.0
        # Simple pinhole estimate. Good enough for a first feature-level integration.
        self._focal_px = (self.width * 0.5) / max(math.tan(math.radians(self.horizontal_fov_deg) * 0.5), 1e-6)

    def _schedule_open_retry(self) -> None:
        self._next_open_retry_at = time.monotonic() + self._open_retry_backoff_s

    def _remember_detection(self, detection: CameraDetection) -> CameraDetection:
        self._last_detection = detection
        self._last_detection_at = time.monotonic()
        return detection

    def _held_detection(self) -> CameraDetection:
        if not self._last_detection.found or self.hold_time_s <= 0.0:
            return CameraDetection()
        age_s = time.monotonic() - self._last_detection_at
        if age_s > self.hold_time_s:
            return CameraDetection()
        if self.debug:
            print(
                f"[debug] camera holding last target age_s={age_s:.2f} "
                f"distance_m={self._last_detection.distance_m:.3f} "
                f"angle_deg={math.degrees(self._last_detection.angle_rad):.1f}",
                flush=True,
            )
        return self._last_detection

    def _ensure_picamera_open(self) -> bool:
        if Picamera2 is None:
            return False
        if self._picam2 is not None:
            return True
        if time.monotonic() < self._next_open_retry_at:
            return False
        try:
            picam2 = Picamera2(camera_num=self.camera_index)
            config = picam2.create_still_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                buffer_count=2,
            )
            picam2.configure(config)
            picam2.start(show_preview=False)
            time.sleep(0.5)
            self._picam2 = picam2
            if self.debug and not self._reported_backend:
                print(f"[debug] camera using Picamera2 backend at index {self.camera_index}", flush=True)
                self._reported_backend = True
            return True
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            if self.debug:
                print(f"[debug] Picamera2 init failed at index {self.camera_index}: {type(exc).__name__}: {exc!r}", flush=True)
            if self._picam2 is not None:
                try:
                    self._picam2.stop()
                except Exception:
                    pass
            self._picam2 = None
            self._schedule_open_retry()
            return False

    def _ensure_open(self) -> bool:
        if self._ensure_picamera_open():
            return True
        if cv2 is None:
            if self.debug and not self._warned_unavailable:
                if Picamera2 is None:
                    print("[debug] camera disabled: neither picamera2 nor cv2 is installed", flush=True)
                else:
                    print("[debug] camera disabled: cv2 is not installed and Picamera2 could not initialize", flush=True)
                self._warned_unavailable = True
            return False
        if self._cap is not None and self._cap.isOpened():
            return True
        if time.monotonic() < self._next_open_retry_at:
            return False
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not cap or not cap.isOpened():
            if self.debug and not self._warned_unavailable:
                print(f"[debug] camera unavailable at index {self.camera_index}", flush=True)
                self._warned_unavailable = True
            if cap is not None:
                cap.release()
            self._schedule_open_retry()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        if self.debug and not self._reported_backend:
            print(f"[debug] camera using OpenCV/V4L2 backend at index {self.camera_index}", flush=True)
            self._reported_backend = True
        return True

    def detect(self) -> CameraDetection:
        if not self._ensure_open():
            return self._held_detection()
        if cv2 is None:
            if self.debug and not self._warned_unavailable:
                print("[debug] camera disabled: cv2 is required for HSV target detection", flush=True)
                self._warned_unavailable = True
            return self._held_detection()

        frame = None
        if self._picam2 is not None:
            try:
                frame = self._picam2.capture_array("main")
                if frame is not None and len(frame.shape) == 3:
                    if frame.shape[2] == 4 and cv2 is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    elif frame.shape[2] == 3 and cv2 is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as exc:  # pragma: no cover - depends on runtime environment
                if self.debug:
                    print(f"[debug] Picamera2 frame read failed: {type(exc).__name__}: {exc!r}", flush=True)
                try:
                    self._picam2.stop()
                except Exception:
                    pass
                self._picam2 = None
                self._schedule_open_retry()
                frame = None
        elif self._cap is not None:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self.debug:
                    print("[debug] camera frame read failed", flush=True)
                self._cap.release()
                self._cap = None
                self._schedule_open_retry()
                return self._held_detection()

        if frame is None:
            return self._held_detection()

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
            return self._held_detection()

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < float(self.min_area_px):
            return self._held_detection()

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return self._held_detection()

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

        return self._remember_detection(CameraDetection(
            found=True,
            distance_m=max(distance_m, 0.0),
            angle_rad=angle_rad,
            confidence=confidence,
        ))

    def warmup(self, timeout_s: float = 4.0) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            if not self._ensure_open():
                time.sleep(0.1)
                continue
            detection = self.detect()
            if self._picam2 is not None or (self._cap is not None and self._cap.isOpened()):
                return True
            time.sleep(0.1)
        return False

    def close(self) -> None:
        if self._picam2 is not None:
            try:
                self._picam2.stop()
            except Exception:
                pass
            self._picam2 = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
