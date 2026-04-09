from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - depends on runtime environment
    cv2 = None

try:
    from picamera2 import Picamera2
except ImportError:  # pragma: no cover - depends on runtime environment
    Picamera2 = None


def parse_hsv(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("HSV value must look like H,S,V")
    try:
        hsv = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid HSV value: {value}") from exc
    if any(component < 0 or component > 255 for component in hsv):
        raise argparse.ArgumentTypeError("HSV components must be in [0, 255]")
    return hsv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a test frame from the robot camera and optionally run HSV detection.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-horizontal-fov-deg", type=float, default=62.0)
    parser.add_argument("--camera-target-width-cm", type=float, default=6.0)
    parser.add_argument("--camera-min-area-px", type=int, default=50)
    parser.add_argument("--camera-hsv-lower", type=parse_hsv, default=(10, 60, 60))
    parser.add_argument("--camera-hsv-upper", type=parse_hsv, default=(60, 255, 255))
    parser.add_argument("--output", type=Path, default=Path("camera_frame.jpg"))
    parser.add_argument("--annotated-output", type=Path, default=Path("camera_annotated.jpg"))
    parser.add_argument("--mask-output", type=Path, default=Path("camera_mask.png"))
    parser.add_argument("--show", action="store_true", help="Open preview windows with the captured frame and mask.")
    parser.add_argument("--continuous", action="store_true", help="Continuously capture and print detection results until interrupted.")
    parser.add_argument("--fps", type=float, default=2.0, help="Capture rate for --continuous mode.")
    parser.add_argument("--no-save", action="store_true", help="Do not write frame, annotated frame, or mask to disk.")
    return parser


def capture_frame(camera_index: int, width: int, height: int) -> tuple[np.ndarray, str]:
    if Picamera2 is not None:
        picam2 = None
        try:
            picam2 = Picamera2(camera_num=camera_index)
            config = picam2.create_still_configuration(
                main={"size": (width, height), "format": "RGB888"},
                buffer_count=2,
            )
            picam2.configure(config)
            picam2.start(show_preview=False)
            time.sleep(0.5)
            frame = picam2.capture_array("main")
            if frame is None:
                raise RuntimeError("Picamera2 returned no frame")
            if cv2 is None:
                raise RuntimeError("OpenCV is required to process the frame")
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            return frame, "Picamera2"
        finally:
            if picam2 is not None:
                try:
                    picam2.stop()
                except Exception:
                    pass

    if cv2 is None:
        raise RuntimeError("Neither Picamera2 nor OpenCV/V4L2 capture is available in this environment")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    try:
        if not cap or not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("OpenCV/V4L2 frame read failed")
        return frame, "OpenCV/V4L2"
    finally:
        cap.release()


def run_hsv_detection(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
    horizontal_fov_deg: float,
    target_width_cm: float,
    min_area_px: int,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, dict]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for HSV detection output")

    focal_px = (width * 0.5) / max(math.tan(math.radians(horizontal_fov_deg) * 0.5), 1e-6)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(hsv_lower, dtype=np.uint8),
        np.array(hsv_upper, dtype=np.uint8),
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    annotated = frame.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return annotated, mask, {"found": False}

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < float(min_area_px):
        return annotated, mask, {"found": False, "area": area}

    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return annotated, mask, {"found": False, "area": area}

    center_x = float(x + w * 0.5)
    pixel_offset = center_x - (width * 0.5)
    angle_rad = math.atan2(pixel_offset, focal_px)
    distance_m = ((target_width_cm / 100.0) * focal_px) / max(float(w), 1.0)
    confidence = float(np.clip(area / float(width * height), 0.0, 1.0))

    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(annotated, (int(round(center_x)), int(round(y + h * 0.5))), 5, (0, 255, 255), -1)
    label = f"found area={area:.0f} dist={distance_m:.2f}m angle={math.degrees(angle_rad):+.1f}deg"
    cv2.putText(annotated, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    return annotated, mask, {
        "found": True,
        "area": area,
        "bbox": (x, y, w, h),
        "distance_m": distance_m,
        "angle_deg": math.degrees(angle_rad),
        "confidence": confidence,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if cv2 is None:
        print("error: cv2 is not installed in this environment", file=sys.stderr)
        return 1

    capture_period_s = 1.0 / max(args.fps, 0.1)
    iteration = 0
    announced_settings = False

    try:
        while True:
            iteration += 1
            started = time.time()
            try:
                frame, backend = capture_frame(args.camera_index, args.camera_width, args.camera_height)
            except Exception as exc:
                print(f"error: camera capture failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1

            annotated, mask, result = run_hsv_detection(
                frame,
                width=args.camera_width,
                height=args.camera_height,
                horizontal_fov_deg=args.camera_horizontal_fov_deg,
                target_width_cm=args.camera_target_width_cm,
                min_area_px=args.camera_min_area_px,
                hsv_lower=args.camera_hsv_lower,
                hsv_upper=args.camera_hsv_upper,
            )

            if not args.no_save:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.annotated_output.parent.mkdir(parents=True, exist_ok=True)
                args.mask_output.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.output), frame)
                cv2.imwrite(str(args.annotated_output), annotated)
                cv2.imwrite(str(args.mask_output), mask)

            if not announced_settings:
                print(f"backend: {backend}")
                print(f"hsv lower: {args.camera_hsv_lower}")
                print(f"hsv upper: {args.camera_hsv_upper}")
                print(f"min area px: {args.camera_min_area_px}")
                if not args.no_save:
                    print(f"saved raw frame: {args.output}")
                    print(f"saved annotated frame: {args.annotated_output}")
                    print(f"saved mask: {args.mask_output}")
                announced_settings = True

            if args.continuous:
                prefix = f"[frame {iteration:04d}] "
                if result.get("found"):
                    print(
                        prefix
                        + "found=True "
                        + f"area={result['area']:.1f} bbox={result['bbox']} "
                        + f"distance_m={result['distance_m']:.3f} angle_deg={result['angle_deg']:.1f} "
                        + f"confidence={result['confidence']:.3f}",
                        flush=True,
                    )
                else:
                    if "area" in result:
                        print(prefix + f"found=False largest_area={result['area']:.1f}", flush=True)
                    else:
                        print(prefix + "found=False", flush=True)
            else:
                if result.get("found"):
                    print(
                        "detection: "
                        f"found=True area={result['area']:.1f} bbox={result['bbox']} "
                        f"distance_m={result['distance_m']:.3f} angle_deg={result['angle_deg']:.1f} "
                        f"confidence={result['confidence']:.3f}"
                    )
                else:
                    if "area" in result:
                        print(f"detection: found=False largest_area={result['area']:.1f}")
                    else:
                        print("detection: found=False")

            if args.show:
                cv2.imshow("camera_frame", frame)
                cv2.imshow("camera_mask", mask)
                cv2.imshow("camera_annotated", annotated)
                key = cv2.waitKey(1 if args.continuous else 0) & 0xFF
                if key in (27, ord("q")):
                    break

            if not args.continuous:
                break

            elapsed = time.time() - started
            sleep_s = max(0.0, capture_period_s - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\nstopped camera test", flush=True)
    finally:
        if args.show:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
