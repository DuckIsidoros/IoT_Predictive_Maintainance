"""Rewrite a CSV file so every row label becomes healthy.

This script is independent from the serial logger. It reads an input CSV,
sets the label column to ``healthy`` for every row, and writes the result
to a new CSV file by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "arduino_serial_data.csv"
DEFAULT_OUTPUT = "arduino_serial_data_healthy.csv"
LABEL_VALUE = "healthy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Change every label value in a CSV file to healthy."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input CSV file path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help='Name of the label column to rewrite. Default: "label".',
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file instead of writing a new file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = input_path if args.inplace else Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    df = pd.read_csv(input_path)

    if args.label_column not in df.columns:
        print(f'Label column "{args.label_column}" not found in {input_path}')
        return 1

    df[args.label_column] = LABEL_VALUE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Rewrote label column to '{LABEL_VALUE}' in {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())