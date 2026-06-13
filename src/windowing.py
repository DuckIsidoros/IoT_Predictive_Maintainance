import argparse
import os
from pathlib import Path

import pandas as pd

from config import (
    SAMPLING_RATE_HZ,
    WINDOW_SIZE,
    OVERLAP_RATIO,
    STEP_SIZE,
    CLASS_LABELS,
    MAX_WINDOWS_PER_CLASS,
    get_paths,
)


# =========================================================
# CONSTANTS
# =========================================================

REQUIRED_COLUMNS = ["timestamp_ms", "accX", "accY", "accZ", "label"]

WINDOW_DURATION_SEC = WINDOW_SIZE / SAMPLING_RATE_HZ
STEP_DURATION_SEC = STEP_SIZE / SAMPLING_RATE_HZ


# =========================================================
# ARGUMENT PARSING
# =========================================================

def parse_args():
    """
    Parse command-line arguments for windowing mode selection.
    """

    parser = argparse.ArgumentParser(
        description="Create sliding windows from segmented sensor data."
    )

    parser.add_argument(
        "--mode",
        choices=["mock", "real"],
        default="mock",
        help="Pipeline mode. Use 'mock' for generated test data or 'real' for real sensor data.",
    )

    return parser.parse_args()


# =========================================================
# SAFETY CHECKS
# =========================================================

def validate_config():
    """
    Ensure windowing configuration matches project requirements.
    """

    if SAMPLING_RATE_HZ != 200:
        raise ValueError(
            f"Invalid sampling rate: {SAMPLING_RATE_HZ}, expected 200 Hz"
        )

    if WINDOW_SIZE != 256:
        raise ValueError(
            f"Invalid window size: {WINDOW_SIZE}, expected 256 samples"
        )

    if OVERLAP_RATIO != 0.5:
        raise ValueError(
            f"Invalid overlap ratio: {OVERLAP_RATIO}, expected 0.5"
        )

    if STEP_SIZE != 128:
        raise ValueError(
            f"Invalid stride: {STEP_SIZE}, expected 128 samples"
        )

    if MAX_WINDOWS_PER_CLASS != 500:
        raise ValueError(
            f"Invalid max windows per class: {MAX_WINDOWS_PER_CLASS}, expected 500"
        )


def validate_segment_dataframe(df, file_path):
    """
    Validate one segment dataframe before creating windows.
    """

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {file_path}: {missing_cols}")

    if len(df) == 0:
        raise ValueError(f"Empty segment file: {file_path}")

    if len(df) < WINDOW_SIZE:
        raise ValueError(
            f"Segment file has only {len(df)} samples, expected at least {WINDOW_SIZE}: {file_path}"
        )

    if df["label"].nunique() != 1:
        raise ValueError(f"Mixed labels detected in segment file: {file_path}")

    label = df["label"].iloc[0]

    if label not in CLASS_LABELS:
        raise ValueError(f"Invalid label '{label}' in file: {file_path}")

    return label


# =========================================================
# WINDOW CREATION
# =========================================================

def create_windows(df):
    """
    Create sliding windows from one segment dataframe.

    Returns:
        list of tuples:
        (start_idx, end_idx_exclusive, window_df)
    """

    windows = []
    total_samples = len(df)

    for start_idx in range(0, total_samples - WINDOW_SIZE + 1, STEP_SIZE):
        end_idx = start_idx + WINDOW_SIZE
        window = df.iloc[start_idx:end_idx].copy()
        windows.append((start_idx, end_idx, window))

    return windows


# =========================================================
# SEGMENT PROCESSING
# =========================================================

def process_segment_file(file_path, output_dir, remaining_windows):
    """
    Process one segment file and create sliding windows.

    Args:
        file_path:
            Path to one segment CSV file.
        output_dir:
            Output directory for generated window CSV files.
        remaining_windows:
            Number of windows still allowed for the current class.

    Returns:
        List of report rows.
    """

    df = pd.read_csv(file_path)
    label = validate_segment_dataframe(df, file_path)

    base_name = Path(file_path).stem

    os.makedirs(output_dir, exist_ok=True)

    windows = create_windows(df)
    report_rows = []

    for local_window_id, (start_idx, end_idx, window) in enumerate(windows):
        if len(report_rows) >= remaining_windows:
            break

        output_name = f"{base_name}_window_{local_window_id:03d}.csv"
        output_path = os.path.join(output_dir, output_name)

        window.to_csv(output_path, index=False)

        report_rows.append({
            "source_segment": str(file_path),
            "output_file": str(output_path),
            "label": label,

            "local_window_id": local_window_id,
            "start_idx": start_idx,
            "end_idx_exclusive": end_idx,
            "end_idx_inclusive": end_idx - 1,

            "num_samples": len(window),

            "start_time_ms": window["timestamp_ms"].iloc[0],
            "end_time_ms": window["timestamp_ms"].iloc[-1],

            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "window_size": WINDOW_SIZE,
            "window_duration_sec": WINDOW_DURATION_SEC,
            "overlap_ratio": OVERLAP_RATIO,
            "step_size": STEP_SIZE,
            "step_duration_sec": STEP_DURATION_SEC,

            "status": "KEPT",
            "reason": "valid",
        })

    return report_rows


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_windowing(mode):
    """
    Run the windowing pipeline for either mock or real mode.
    """

    validate_config()

    paths = get_paths(mode)

    segment_root = paths["segments"]
    output_root = paths["windows"]
    report_path = paths["window_report"]

    all_report_rows = []

    print("======================================")
    print("WINDOWING PIPELINE STARTED")
    print("======================================")
    print(f"Mode                  : {mode}")
    print(f"SEGMENT_ROOT          : {segment_root}")
    print(f"OUTPUT_ROOT           : {output_root}")
    print(f"REPORT_PATH           : {report_path}")
    print("--------------------------------------")
    print(f"Sampling Rate         : {SAMPLING_RATE_HZ} Hz")
    print(f"Window Size           : {WINDOW_SIZE} samples")
    print(f"Window Duration       : {WINDOW_DURATION_SEC:.2f} sec")
    print(f"Overlap Ratio         : {OVERLAP_RATIO}")
    print(f"Stride / Step Size    : {STEP_SIZE} samples")
    print(f"Step Duration         : {STEP_DURATION_SEC:.2f} sec")
    print(f"Max Windows per Class : {MAX_WINDOWS_PER_CLASS}")
    print("======================================")

    if not os.path.exists(segment_root):
        print(f"ERROR: segment root does not exist: {segment_root}")
        return

    os.makedirs(output_root, exist_ok=True)

    for label in CLASS_LABELS:
        label_dir = os.path.join(segment_root, label)

        if not os.path.isdir(label_dir):
            print(f"[WARNING] Missing label folder: {label_dir}")
            continue

        output_dir = os.path.join(output_root, label)
        os.makedirs(output_dir, exist_ok=True)

        csv_files = sorted([
            filename for filename in os.listdir(label_dir)
            if filename.lower().endswith(".csv")
        ])

        print("--------------------------------------")
        print(f"Processing label       : {label}")
        print(f"Segment files found    : {len(csv_files)}")

        created_count = 0

        for filename in csv_files:
            if created_count >= MAX_WINDOWS_PER_CLASS:
                break

            file_path = os.path.join(label_dir, filename)
            remaining_windows = MAX_WINDOWS_PER_CLASS - created_count

            try:
                rows = process_segment_file(
                    file_path=file_path,
                    output_dir=output_dir,
                    remaining_windows=remaining_windows,
                )

                all_report_rows.extend(rows)
                created_count += len(rows)

                print(
                    f"[OK] {filename} -> created {len(rows)} windows "
                    f"| total for class = {created_count}"
                )

            except Exception as exc:
                print(f"[FAIL] {filename}: {exc}")

        print(f"Final windows for {label}: {created_count}")

        if created_count < MAX_WINDOWS_PER_CLASS:
            print(
                f"[WARNING] label={label} has only {created_count} windows, "
                f"required={MAX_WINDOWS_PER_CLASS}"
            )

    if len(all_report_rows) == 0:
        print("No windows created. Check segment input.")
        return

    report_df = pd.DataFrame(all_report_rows)

    report_df.insert(0, "global_window_id", range(len(report_df)))

    report_df.to_csv(report_path, index=False)

    print("\n======================================")
    print("WINDOWING COMPLETED")
    print("======================================")
    print(f"Report saved to: {report_path}")
    print("--------------------------------------")
    print("Window count by label:")
    print(report_df.groupby("label").size())
    print("======================================")


def main():
    """
    Entry point for command-line execution.
    """

    args = parse_args()
    run_windowing(args.mode)


if __name__ == "__main__":
    main()