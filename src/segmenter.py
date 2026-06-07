import os
import pandas as pd

EXPECTED_FS = 200  # Hz
EXPECTED_DT_MS = 1000 / EXPECTED_FS

MIN_DT_MS = 2.5
MAX_DT_MS = 7.5

MIN_SEGMENT_SECONDS = 2
MIN_SEGMENT_SAMPLES = EXPECTED_FS * MIN_SEGMENT_SECONDS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_ROOT = os.path.join(BASE_DIR, "mock_raw_sensor_stream")
OUTPUT_ROOT = os.path.join(BASE_DIR, "segments")
REPORT_PATH = os.path.join(BASE_DIR, "segment_report.csv")

REQUIRED_COLUMNS = ["timestamp_ms", "accX", "accY", "accZ", "label"]


def find_breakpoints(df):
    timestamps = df["timestamp_ms"].values
    breakpoints = []

    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]

        if dt <= 0 or dt < MIN_DT_MS or dt > MAX_DT_MS:
            breakpoints.append(i)

    return breakpoints


def split_segments(df):
    breakpoints = find_breakpoints(df)

    segments = []
    start_idx = 0

    for bp in breakpoints:
        segment = df.iloc[start_idx:bp].copy()
        segments.append(segment)
        start_idx = bp

    last_segment = df.iloc[start_idx:].copy()
    segments.append(last_segment)

    return segments


def process_file(file_path, output_dir):
    df = pd.read_csv(file_path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    label = df["label"].iloc[0]
    segments = split_segments(df)

    report_rows = []
    kept_count = 0

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    for idx, segment in enumerate(segments):
        if len(segment) < MIN_SEGMENT_SAMPLES:
            status = "REJECTED"
            reason = "too_short"
        else:
            status = "KEPT"
            reason = "valid"

            output_name = f"{base_name}_segment_{kept_count:03d}.csv"
            output_path = os.path.join(output_dir, output_name)
            segment.to_csv(output_path, index=False)

            kept_count += 1

        report_rows.append({
            "source_file": file_path,
            "label": label,
            "segment_id": idx,
            "start_time_ms": segment["timestamp_ms"].iloc[0] if len(segment) > 0 else None,
            "end_time_ms": segment["timestamp_ms"].iloc[-1] if len(segment) > 0 else None,
            "num_samples": len(segment),
            "duration_sec": len(segment) / EXPECTED_FS,
            "status": status,
            "reason": reason
        })

    return report_rows


def main():
    all_report_rows = []

    for label in os.listdir(INPUT_ROOT):
        label_dir = os.path.join(INPUT_ROOT, label)

        if not os.path.isdir(label_dir):
            continue

        output_dir = os.path.join(OUTPUT_ROOT, label)

        for filename in os.listdir(label_dir):
            if not filename.endswith(".csv"):
                continue

            file_path = os.path.join(label_dir, filename)
            rows = process_file(file_path, output_dir)
            all_report_rows.extend(rows)

    report_df = pd.DataFrame(all_report_rows)
    report_df.to_csv(REPORT_PATH, index=False)

    print("Segment splitting completed.")
    print(f"Report saved to: {REPORT_PATH}")
    print(report_df.groupby(["label", "status"]).size())


if __name__ == "__main__":
    main()