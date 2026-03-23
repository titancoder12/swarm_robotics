from __future__ import annotations

import argparse
import time

import serial


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Monitor raw serial output from the robot firmware."
    )
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.2)
    parser.add_argument("--startup-delay", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    print(
        f"[serialmonitor] opening port={args.port} baudrate={args.baudrate}",
        flush=True,
    )

    with serial.Serial(
        port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
    ) as ser:
        time.sleep(args.startup_delay)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print("[serialmonitor] monitoring started; press Ctrl+C to stop", flush=True)

        try:
            while True:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue

                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] {line}", flush=True)
        except KeyboardInterrupt:
            print("\n[serialmonitor] stopped", flush=True)


if __name__ == "__main__":
    main()
