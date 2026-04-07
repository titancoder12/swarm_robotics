from __future__ import annotations

import serial
import time
import json
from typing import Optional, Dict, Any
#import cv2

class ESP32Robot:
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        timeout: float = 1.0,
        startup_delay: float = 2.0,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.startup_delay = startup_delay
        self.ser: Optional[serial.Serial] = None

    @staticmethod
    def _require_pyserial() -> None:
        if not hasattr(serial, "Serial"):
            raise ImportError(
                "Expected pyserial's 'serial' module, but 'serial.Serial' is unavailable. "
                "Install 'pyserial' in this environment and remove the unrelated 'serial' package if present."
            )

    def connect(self) -> None:
        self._require_pyserial()
        if self.ser is not None and self.ser.is_open:
            return

        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
        )

        # Give ESP32 time in case opening serial resets it
        time.sleep(self.startup_delay)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def reset_buffers(self) -> None:
        ser = self._require_serial()
        ser.reset_input_buffer()
        ser.reset_output_buffer()

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def _require_serial(self) -> serial.Serial:
        self._require_pyserial()
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("Serial port is not connected. Call connect() first.")
        return self.ser

    def send_raw(self, cmd: str) -> None:
        ser = self._require_serial()
        line = cmd.strip() + "\n"
        ser.write(line.encode("utf-8"))
        ser.flush()

    def read_line(self, timeout_override: float | None = None) -> Optional[str]:
        ser = self._require_serial()
        if timeout_override is None:
            raw = ser.readline()
        else:
            original_timeout = ser.timeout
            try:
                ser.timeout = max(0.0, float(timeout_override))
                raw = ser.readline()
            finally:
                ser.timeout = original_timeout
        if not raw:
            return None

        line = raw.decode("utf-8", errors="ignore").strip()
        #print(line)
        return line if line else None

    def wait_for_stream_ready(self, timeout: float = 3.0, line_timeout: float = 0.1) -> bool:
        ser = self._require_serial()
        start = time.time()
        saw_any_bytes = False

        while time.time() - start < timeout:
            if ser.in_waiting:
                saw_any_bytes = True
                line = self.read_line(timeout_override=line_timeout)
                if not line:
                    time.sleep(0.01)
                    continue
                parsed = self.parse_line(line)
                if parsed.get("type") == "scan":
                    return True
            time.sleep(0.01)

        if saw_any_bytes:
            self.reset_buffers()
        return False

    def wait_response(self, timeout: float = 5.0) -> Dict[str, Any]:
        ser = self._require_serial()
        start = time.time()

        while time.time() - start < timeout:
            if ser.in_waiting:
                line = self.read_line()
                if not line:
                    continue

                parsed = self.parse_line(line)

                if parsed.get("type") == "ack":
                    return {"ok": True, "raw": line, "parsed": parsed}

                if parsed.get("type") == "err":
                    return {"ok": False, "raw": line, "parsed": parsed}

                # Ignore scan/debug lines while waiting for ack/err
            time.sleep(0.01)

        return {
            "ok": False,
            "raw": "TIMEOUT",
            "parsed": {"type": "timeout"},
        }
    
    def command(self, cmd: str, timeout: float = 5.0) -> Dict[str, Any]:
        self.send_raw(cmd)
        return self.wait_response(timeout=timeout)

    # ===== ESP32-matching high-level functions =====

    def turn(self, angle: int, timeout: float = 10.0) -> Dict[str, Any]:
        return self.command(f"T{int(angle)}", timeout=timeout)

    def move(self, angle: int, distance_mm: int, timeout: float = 20.0) -> Dict[str, Any]:
        return self.command(f"M{int(angle)},{int(distance_mm)}", timeout=timeout)

    def stop(self, timeout: float = 3.0) -> Dict[str, Any]:
        return self.command("S", timeout=timeout)

    def brake(self, timeout: float = 3.0) -> Dict[str, Any]:
        return self.command("B", timeout=timeout)

    # ===== Sensor / stream helpers =====

    def read_sensor_lines(self, duration: float = 1.0) -> list[Dict[str, Any]]:
        ser = self._require_serial()
        results: list[Dict[str, Any]] = []
        start = time.time()

        while time.time() - start < duration:
            if ser.in_waiting:
                line = self.read_line(timeout_override=0.05)
                if line:
                    results.append(self.parse_line(line))
            time.sleep(0.005)

        return results

    def get_TOF(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        ser = self._require_serial()
        start = time.time()

        while time.time() - start < timeout:

            if ser.in_waiting:
                line = self.read_line(timeout_override=0.05)
                if not line:
                    continue

                data = self.parse_line(line)

                if data.get("type") == "scan":
                    return data

            time.sleep(0.005)

        return None

    @staticmethod
    def parse_line(line: str) -> Dict[str, Any]:
        try:
            data = json.loads(line)

            if isinstance(data, dict) and "type" in data:
                return data

            return {"type": "raw", "raw": line}

        except json.JSONDecodeError:
            return {"type": "raw", "raw": line}
    
    def read_full_sweep(self, timeout: float = 2.0) -> list[Dict[str, Any]]:
        start = time.time()
        points = []
        seen_angles = set()

        while time.time() - start < timeout:
            if self._require_serial().in_waiting:
                line = self.read_line(timeout_override=0.05)
                if not line:
                    continue

                item = self.parse_line(line)
                if item.get("type") != "scan":
                    continue

                angle = item.get("angle")
                if angle is None:
                    continue

                if points and angle in seen_angles:
                    break

                points.append(item)
                seen_angles.add(angle)

            time.sleep(0.005)

        return points

# def main() -> None:
#     robot = ESP32Robot(port="/dev/ttyUSB0", baudrate=115200)

#     try:
#         robot.connect()
#         print("Connected to ESP32 on /dev/serial0")

#         while True:
#             scan_points = robot.read_full_sweep(timeout=2.0)

#             valid_points = []
#             for item in scan_points:
#                 if item.get("type") != "scan":
#                     continue

#                 angle = item.get("angle")
#                 dist = item.get("tof_mm")

#                 if angle is None or dist is None:
#                     continue

#                 valid_points.append((angle, dist))

#             chosen_angle = 0
#             min_clearance_mm = 100
#             best_angle = None

#             for angle, distance in valid_points:
#                 print(f"Angle: {angle}, Distance: {distance} mm")

#                 if distance > min_clearance_mm or distance == -1:
#                     print(f"Free angle: {angle} degrees, Distance: {distance} mm")

#                     if best_angle is None or abs(angle - 90) < abs(best_angle - 90):
#                         best_angle = angle

#             if best_angle is not None:
#                 chosen_angle = best_angle - 90


#             """if valid_points:
#                 front_points = [
#                     (angle, dist)
#                     for angle, dist in valid_points
#                     if abs(angle - 90) <= front_window
#                 ]

#                 front_blocked = any(
#                     dist != -1 and dist <= min_clearance_mm
#                     for _, dist in front_points
#                 )

#                 if front_blocked:
#                     free_angles = [
#                         angle
#                         for angle, dist in valid_points
#                         if dist == -1 or dist > min_clearance_mm
#                     ]

#                     if free_angles:
#                         chosen_angle = min(free_angles, key=lambda a: abs(a - 90))

#             print("Chosen lidar angle:", chosen_angle)"""

#             move_angle = chosen_angle
#             print("Move angle:", move_angle)
#             resp = robot.move(move_angle, 100)
#             print(resp)

#             time.sleep(0.5)

#     finally:
# #         robot.close()


# if __name__ == "__main__":
#     main()
