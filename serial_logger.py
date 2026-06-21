import argparse
import csv
import os
import re
import serial
import sys
from datetime import datetime

# Configuration defaults
DEFAULT_BAUD = 921600
DEFAULT_PORT = "COM5"  # Change to '/dev/ttyUSB0' or similar on Linux/macOS


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="High-speed Serial Logger for Arduino/ESP32 Data Collection."
    )
    parser.add_argument(
        "-p",
        "--port",
        default=DEFAULT_PORT,
        help=f"Serial port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "-b",
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Baud rate (default: {DEFAULT_BAUD})",
    )
    parser.add_argument(
        "-l",
        "--label",
        default="obstruction",
        help="Label for the data collection session (default: imbalanced)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Generate filename based on the provided label
    filename = f"arduino_serial_data_{args.label}.csv"
    headers = ["timestamp_ms", "accX", "accY", "accZ", "label"]

    # Check if file exists to determine if we write headers
    file_exists = os.path.isfile(filename)

    print(f"Connecting to {args.port} at {args.baud} baud...")
    print(f"Logging data to: {filename}")
    print("Press Ctrl+C to stop logging safely.\n")

    try:
        # Initialize serial connection with a short timeout to stay responsive
        ser = serial.Serial(args.port, args.baud, timeout=1.0)
        ser.flushInput()  # Clear old data in buffer
    except serial.SerialException as e:
        print(f"Error opening serial port {args.port}: {e}")
        sys.exit(1)

    # Compile regex to quickly validate standard CSV floats/ints
    # This filters out 'WARN: sample dropped' or 'ERROR: IMU unresponsive' text
    data_pattern = re.compile(
        r"^\d+(\.\d+)?,(-?\d+(\.\d+)?),(-?\d+(\.\d+)?),(-?\d+(\.\d+)?)$"
    )

    try:
        with open(filename, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)

            if not file_exists:
                writer.writerow(headers)
                csv_file.flush()

            sample_count = 0

            while True:
                if ser.in_waiting > 0:
                    try:
                        # Read line and decode
                        line = (
                            ser.readline().decode("utf-8", errors="ignore").strip()
                        )
                    except Exception:
                        continue

                    if not line:
                        continue

                    # If it's the raw header from setup(), skip it
                    if "timestamp_ms" in line:
                        continue

                    # Validate if it's actual sensor data or system warning
                    if data_pattern.match(line):
                        try:
                            # Split the 4 incoming values: timestamp, accX, accY, accZ
                            row_data = line.split(",")

                            # Append the session label
                            row_data.append(args.label)

                            # Write to CSV
                            writer.writerow(row_data)
                            sample_count += 1

                            # Flush buffer periodically so data isn't lost if killed abruptly
                            if sample_count % 500 == 0:
                                csv_file.flush()
                                print(
                                    f"[{datetime.now().strftime('%H:%M:%S')}] Captured {sample_count} samples...",
                                    end="\r",
                                )

                        except ValueError:
                            # Catch-all for faulty split conversions
                            continue
                    else:
                        # Print warnings/errors from your ESP32 directly to console
                        print(f"\n[ESP32 System Msg]: {line}")

    except KeyboardInterrupt:
        print(f"\n\nLogging stopped by user. Total samples saved: {sample_count}")
    finally:
        if "ser" in locals() and ser.is_open:
            ser.close()
            print("Serial port closed safely.")


if __name__ == "__main__":
    main()