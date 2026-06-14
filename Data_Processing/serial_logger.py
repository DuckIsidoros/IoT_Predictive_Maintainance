"""Capture Arduino serial CSV data and save it to a file.

Expected line format from Arduino_Code/Data_Collection/IIR_Collection/IIR_Collection.ino:

timestamp_ms,raw_x,filt_x,raw_y,filt_y,raw_z,filt_z,label

Each valid line is written to the output CSV with the same header.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

import serial
from serial import SerialException


CSV_HEADER = [
    "timestamp_ms",
    "raw_x",
    "accX",  # filtered_X
    "raw_y",
    "accY",  # filtered_Y
    "raw_z",
    "accZ",  # filtered_Z
    "label",
]

BAUD = int(os.getenv("BAUD", "115200"))
PORT = os.getenv("COM_PORT", "COM5")
OUTPUT = os.getenv("OUTPUT", "arduino_serial_data.csv")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read Arduino serial CSV rows and save them to a file."
    )
    parser.add_argument(
        "--port",
        default=PORT,
        help=f"Serial port, for example COM5 or /dev/ttyUSB0. Default: {PORT}",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=BAUD,
        help=f"Serial baud rate. Default: {BAUD}",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT,
        help=f"Output CSV file path. Default: {OUTPUT}",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing CSV file instead of overwriting it.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Output file encoding, default: utf-8",
    )
    return parser.parse_args()


def iter_serial_lines(ser: serial.SerialBase) -> Iterable[str]:
    while True:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode(errors="replace").strip()
        if line:
            yield line


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists()
    mode = "a" if args.append else "w"

    try:
        with serial.Serial(args.port, args.baud, timeout=1) as ser, output_path.open(
            mode, newline="", encoding=args.encoding
        ) as csv_file:
            writer = csv.writer(csv_file)
            if not args.append or not file_exists:
                writer.writerow(CSV_HEADER)

            print(f"Listening on {args.port} at {args.baud} baud. Saving to {output_path}")
            print("Press Ctrl+C to stop.")

            for line in iter_serial_lines(ser):
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != len(CSV_HEADER):
                    print(f"Skipping malformed line: {line}")
                    continue

                writer.writerow(parts)
                csv_file.flush()
                print(line)

    except SerialException as exc:
        print(f"Serial error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("Stopped by user.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())