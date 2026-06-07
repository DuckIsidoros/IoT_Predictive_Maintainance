import os
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEGMENT_ROOT = os.path.join(BASE_DIR, "segments")
OUTPUT_ROOT = os.path.join(BASE_DIR, "windows")
REPORT_PATH = os.path.join(BASE_DIR, "window_report.csv")

EXPECTED_FS = 200  # Hz

WINDOW_SIZE = 256  # samples
OVERLAP_RATIO = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))  # 128 samples

WINDOW_DURATION_SEC = WINDOW_SIZE / EXPECTED_FS      # 1.28 sec
STEP_DURATION_SEC = STEP_SIZE / EXPECTED_FS          # 0.64 sec

MAX_WINDOWS_PER_CLASS = 500

REQUIRED_COLUMNS = ["timestamp_ms", "accX", "accY", "accZ", "label"]
VALID_LABELS = ["healthy", "imbalance", "obstruction"]


# =========================================================
# SAFETY CHECKS
# =========================================================

def validate_config():
    """
    Ensure windowing configuration matches project requirements.
    """

    if EXPECTED_FS != 200:
        raise ValueError(f"Invalid sampling rate: {EXPECTED_FS}, expected 200 Hz")

    if WINDOW_SIZE != 256:
        raise ValueError(f"Invalid window size: {WINDOW_SIZE}, expected 256 samples")

    if OVERLAP_RATIO != 0.5:
        raise ValueError(f"Invalid overlap ratio: {OVERLAP_RATIO}, expected 0.5")

    if STEP_SIZE != 128:
        raise ValueError(f"Invalid stride: {STEP_SIZE}, expected 128 samples")

    if MAX_WINDOWS_PER_CLASS != 500:
        raise ValueError(
            f"Invalid max windows per class: {MAX_WINDOWS_PER_CLASS}, expected 500"
        )


# =========================================================
# WINDOW CREATION
# =========================================================

def create_windows(df):
    """
    Create sliding windows from one segment dataframe.

    Input:
        df: segment dataframe

    Output:
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
    Process one segment file and create windows.

    remaining_windows:
        Number of windows still allowed for the current class.
        This is used to enforce 500 windows per class.
    """

    df = pd.read_csv(file_path)

    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {file_path}: {missing_cols}")

    # Check empty file
    if len(df) == 0:
        raise ValueError(f"Empty segment file: {file_path}")

    # Check label consistency
    if df["label"].nunique() != 1:
        raise ValueError(f"Mixed labels detected in segment file: {file_path}")

    label = df["label"].iloc[0]

    # Check valid label
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label '{label}' in file: {file_path}")

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    os.makedirs(output_dir, exist_ok=True)

    windows = create_windows(df)
    report_rows = []

    for local_window_id, (start_idx, end_idx, window) in enumerate(windows):

        # Stop when current class already has enough windows
        if len(report_rows) >= remaining_windows:
            break

        output_name = f"{base_name}_window_{local_window_id:03d}.csv"
        output_path = os.path.join(output_dir, output_name)

        window.to_csv(output_path, index=False)

        report_rows.append({
            "source_segment": file_path,
            "output_file": output_path,
            "label": label,

            "local_window_id": local_window_id,
            "start_idx": start_idx,
            "end_idx_exclusive": end_idx,
            "end_idx_inclusive": end_idx - 1,

            "num_samples": len(window),

            "start_time_ms": window["timestamp_ms"].iloc[0],
            "end_time_ms": window["timestamp_ms"].iloc[-1],

            "expected_fs": EXPECTED_FS,
            "window_size": WINDOW_SIZE,
            "window_duration_sec": WINDOW_DURATION_SEC,
            "overlap_ratio": OVERLAP_RATIO,
            "step_size": STEP_SIZE,
            "step_duration_sec": STEP_DURATION_SEC,

            "status": "KEPT",
            "reason": "valid"
        })

    return report_rows


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():
    validate_config()

    all_report_rows = []

    print("======================================")
    print("WINDOWING PIPELINE STARTED")
    print("======================================")
    print(f"SEGMENT_ROOT          : {SEGMENT_ROOT}")
    print(f"OUTPUT_ROOT           : {OUTPUT_ROOT}")
    print(f"REPORT_PATH           : {REPORT_PATH}")
    print("--------------------------------------")
    print(f"Sampling Rate         : {EXPECTED_FS} Hz")
    print(f"Window Size           : {WINDOW_SIZE} samples")
    print(f"Window Duration       : {WINDOW_DURATION_SEC:.2f} sec")
    print(f"Overlap Ratio         : {OVERLAP_RATIO}")
    print(f"Stride / Step Size    : {STEP_SIZE} samples")
    print(f"Step Duration         : {STEP_DURATION_SEC:.2f} sec")
    print(f"Max Windows per Class : {MAX_WINDOWS_PER_CLASS}")
    print("======================================")

    if not os.path.exists(SEGMENT_ROOT):
        print("ERROR: SEGMENT_ROOT does not exist.")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for label in sorted(os.listdir(SEGMENT_ROOT)):
        label_dir = os.path.join(SEGMENT_ROOT, label)

        if not os.path.isdir(label_dir):
            continue

        if label not in VALID_LABELS:
            print(f"[SKIP] Unknown label folder: {label}")
            continue

        output_dir = os.path.join(OUTPUT_ROOT, label)
        os.makedirs(output_dir, exist_ok=True)

        csv_files = sorted([
            f for f in os.listdir(label_dir)
            if f.endswith(".csv")
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
                    remaining_windows=remaining_windows
                )

                all_report_rows.extend(rows)
                created_count += len(rows)

                print(
                    f"[OK] {filename} -> created {len(rows)} windows "
                    f"| total for class = {created_count}"
                )

            except Exception as e:
                print(f"[FAIL] {filename}: {e}")

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

    # Add global window id
    report_df.insert(0, "global_window_id", range(len(report_df)))

    report_df.to_csv(REPORT_PATH, index=False)

    print("\n======================================")
    print("WINDOWING COMPLETED")
    print("======================================")
    print(f"Report saved to: {REPORT_PATH}")
    print("--------------------------------------")
    print("Window count by label:")
    print(report_df.groupby("label").size())
    print("======================================")


if __name__ == "__main__":
    main()