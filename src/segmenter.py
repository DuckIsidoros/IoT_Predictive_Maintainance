import argparse
import shutil
from pathlib import Path

import pandas as pd

from config import (
    SAMPLING_RATE_HZ,
    EXPECTED_DT_MS,
    CLASS_LABELS,
    LABEL_ALIASES,
    get_paths,
)


# =========================================================
# SEGMENT CONFIG
# =========================================================

MIN_DT_MS = EXPECTED_DT_MS * 0.5
MAX_DT_MS = EXPECTED_DT_MS * 1.5
GAP_THRESHOLD_MS = EXPECTED_DT_MS * 3

MIN_SEGMENT_SECONDS = 2
MIN_SEGMENT_SAMPLES = SAMPLING_RATE_HZ * MIN_SEGMENT_SECONDS

REQUIRED_COLUMNS = ["timestamp_ms", "accX", "accY", "accZ", "label"]

AUTO_CLEAN_OUTPUT = True


# =========================================================
# CLI ARGUMENTS
# =========================================================

def parse_args():
    """
    Parse command-line arguments.

    Usage:
        python src/segmenter.py --mode mock
        python src/segmenter.py --mode real
    """

    parser = argparse.ArgumentParser(
        description="Split raw sensor CSV files into valid continuous segments."
    )

    parser.add_argument(
        "--mode",
        choices=["mock", "real"],
        default="mock",
        help="Choose between mock data or real data.",
    )

    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete old segment output before running.",
    )

    return parser.parse_args()


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def prepare_output_dir(output_root: Path, auto_clean: bool = True) -> None:
    """
    Prepare segment output directory.

    If auto_clean is enabled, old segment files are removed first.
    This prevents stale segment files from mixing with the current run.
    """

    if auto_clean and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)


def infer_label_from_path(file_path: Path) -> str | None:
    """
    Infer label from parent folder or file name.

    Priority:
        1. Parent folder name
        2. File name
    """

    parent_name = file_path.parent.name.lower().strip()

    if parent_name in CLASS_LABELS:
        return parent_name

    file_name = file_path.name.lower().strip()

    matched_labels = [
        label for label in CLASS_LABELS
        if label.lower() in file_name
    ]

    if len(matched_labels) == 1:
        return matched_labels[0]

    return None


def normalize_label(value) -> str:
    """
    Normalize label value to lowercase string.
    """
    normalized = str(value).lower().strip()
    return LABEL_ALIASES.get(normalized, normalized)


def collect_input_files(input_root: Path, mode: str) -> list[Path]:
    """
    Collect CSV files from the input root.

    Mock mode:
        Only collect files inside class folders.
        This prevents old root-level test files from being mixed into the mock pipeline.

    Real mode:
        Collect CSV files recursively.
        This supports both flat accepted folders and class-based accepted folders.
    """

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    if mode == "mock":
        files = []

        for label in CLASS_LABELS:
            label_dir = input_root / label

            if not label_dir.exists():
                print(f"[WARNING] Missing mock label folder: {label_dir}")
                continue

            files.extend(sorted(label_dir.glob("*.csv")))

    else:
        files = sorted(input_root.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found under: {input_root}")

    return files

# =========================================================
# VALIDATION
# =========================================================

def validate_dataframe(df: pd.DataFrame, file_path: Path) -> str:
    """
    Validate one raw dataframe before segment splitting.

    Returns:
        normalized label
    """

    if df.empty:
        raise ValueError(f"Empty input file: {file_path}")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in {file_path}: {missing_cols}")

    label_values = (
        df["label"]
        .dropna()
        .astype(str)
        .str.lower()
        .str.strip()
        .unique()
    )

    if len(label_values) == 0:
        inferred_label = infer_label_from_path(file_path)

        if inferred_label is None:
            raise ValueError(f"Missing label and cannot infer from path: {file_path}")

        df["label"] = inferred_label
        return inferred_label

    if len(label_values) > 1:
        raise ValueError(
            f"Mixed labels detected in {file_path}: {list(label_values)}"
        )

    label = normalize_label(label_values[0])

    if label not in CLASS_LABELS:
        raise ValueError(f"Invalid label '{label}' in file: {file_path}")

    df["label"] = label

    return label


def normalize_numeric_columns(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    """
    Convert timestamp and accelerometer columns to numeric values.
    Invalid values become NaN and are rejected before segmentation.
    """

    numeric_columns = ["timestamp_ms", "accX", "accY", "accZ"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[numeric_columns].isna().any().any():
        raise ValueError(f"NaN or non-numeric values detected in: {file_path}")

    return df


# =========================================================
# SEGMENT SPLITTING
# =========================================================

# 
def find_breakpoints(df: pd.DataFrame, mode: str) -> list[int]:
    timestamps = df["timestamp_ms"].to_numpy()
    breakpoints = []

    for index in range(1, len(timestamps)):
        dt_ms = timestamps[index] - timestamps[index - 1]

        if mode == "real":
            #  Real data allows minor jitter. Split only on hard timestamp issues.
            if dt_ms <= 0 or dt_ms > GAP_THRESHOLD_MS:
                breakpoints.append(index)
        else:
            #  Mock data remains strict.
            if dt_ms <= 0 or dt_ms < MIN_DT_MS or dt_ms > MAX_DT_MS:
                breakpoints.append(index)

    return breakpoints


def split_segments(df: pd.DataFrame, mode: str) -> list[pd.DataFrame]:
    """
    Split one dataframe into continuous timestamp segments.
    """

    breakpoints = find_breakpoints(df, mode)

    segments = []
    start_idx = 0

    for breakpoint_idx in breakpoints:
        segment = df.iloc[start_idx:breakpoint_idx].copy()
        segments.append(segment)
        start_idx = breakpoint_idx

    last_segment = df.iloc[start_idx:].copy()
    segments.append(last_segment)

    return segments


# =========================================================
# FILE PROCESSING
# =========================================================

def process_file(file_path: Path, output_root: Path, mode: str) -> list[dict]:
    """
    Process one raw CSV file and save valid segments.

    Output structure:
        output_root/<label>/<source_name>_segment_000.csv
    """

    df = pd.read_csv(file_path)

    label = validate_dataframe(df, file_path)
    df = normalize_numeric_columns(df, file_path)

    segments = split_segments(df, mode)

    label_output_dir = output_root / label
    label_output_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []
    kept_count = 0

    base_name = file_path.stem

    for segment_id, segment in enumerate(segments):
        if segment.empty:
            status = "REJECTED"
            reason = "empty_segment"
            output_file = None

        elif len(segment) < MIN_SEGMENT_SAMPLES:
            status = "REJECTED"
            reason = "too_short"
            output_file = None

        else:
            status = "KEPT"
            reason = "valid"

            output_name = f"{base_name}_segment_{kept_count:03d}.csv"
            output_path = label_output_dir / output_name

            segment.to_csv(output_path, index=False)

            output_file = str(output_path)
            kept_count += 1

        report_rows.append({
            "source_file": str(file_path),
            "output_file": output_file,
            "label": label,
            "segment_id": segment_id,
            "kept_segment_id": kept_count - 1 if status == "KEPT" else None,
            "start_time_ms": segment["timestamp_ms"].iloc[0] if not segment.empty else None,
            "end_time_ms": segment["timestamp_ms"].iloc[-1] if not segment.empty else None,
            "num_samples": len(segment),
            "duration_sec": len(segment) / SAMPLING_RATE_HZ,
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "expected_dt_ms": EXPECTED_DT_MS,
            "min_dt_ms": MIN_DT_MS,
            "max_dt_ms": MAX_DT_MS,
            "min_segment_samples": MIN_SEGMENT_SAMPLES,
            "status": status,
            "reason": reason,
        })

    return report_rows


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_segmenter(mode: str, auto_clean: bool = True) -> None:
    """
    Run segment splitting pipeline for mock or real mode.
    """

    paths = get_paths(mode)

    input_root = paths["raw"]
    output_root = paths["segments"]
    report_path = paths["segment_report"]

    print("=" * 70)
    print("SEGMENTER PIPELINE STARTED")
    print("=" * 70)
    print(f"Mode                 : {mode}")
    print(f"Input root           : {input_root}")
    print(f"Output root          : {output_root}")
    print(f"Report path          : {report_path}")
    print(f"Sampling rate        : {SAMPLING_RATE_HZ} Hz")
    print(f"Expected dt          : {EXPECTED_DT_MS:.3f} ms")
    print(f"Min dt               : {MIN_DT_MS:.3f} ms")
    print(f"Max dt               : {MAX_DT_MS:.3f} ms")
    print(f"Min segment seconds  : {MIN_SEGMENT_SECONDS}")
    print(f"Min segment samples  : {MIN_SEGMENT_SAMPLES}")
    print(f"Auto clean output    : {auto_clean}")
    print("=" * 70)

    prepare_output_dir(output_root, auto_clean=auto_clean)

    input_files = collect_input_files(input_root, mode)

    all_report_rows = []
    failed_files = []

    for file_path in input_files:
        try:
            rows = process_file(
                file_path=file_path,
                output_root=output_root,
                mode=mode,
            )

            all_report_rows.extend(rows)

            kept_count = sum(1 for row in rows if row["status"] == "KEPT")
            rejected_count = sum(1 for row in rows if row["status"] == "REJECTED")

            print(
                f"[OK] {file_path.name} | kept={kept_count} | rejected={rejected_count}"
            )

        except Exception as exc:
            failed_files.append({
                "source_file": str(file_path),
                "output_file": None,
                "label": None,
                "segment_id": None,
                "kept_segment_id": None,
                "start_time_ms": None,
                "end_time_ms": None,
                "num_samples": None,
                "duration_sec": None,
                "sampling_rate_hz": SAMPLING_RATE_HZ,
                "expected_dt_ms": EXPECTED_DT_MS,
                "min_dt_ms": MIN_DT_MS,
                "max_dt_ms": MAX_DT_MS,
                "min_segment_samples": MIN_SEGMENT_SAMPLES,
                "status": "FAILED",
                "reason": str(exc),
            })

            print(f"[FAIL] {file_path.name}: {exc}")

    report_rows = all_report_rows + failed_files

    if not report_rows:
        print("No report rows generated.")
        return

    report_df = pd.DataFrame(report_rows)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False)

    print("=" * 70)
    print("SEGMENTER PIPELINE COMPLETED")
    print("=" * 70)
    print(f"Report saved to      : {report_path}")
    print(f"Input files          : {len(input_files)}")
    print(f"Failed files         : {len(failed_files)}")
    print(f"Total report rows    : {len(report_df)}")
    print("-" * 70)

    if "label" in report_df.columns and "status" in report_df.columns:
        print("Segment count by label/status:")
        print(report_df.groupby(["label", "status"]).size())

    print("=" * 70)


def main() -> None:
    """
    Entry point for command-line execution.
    """

    args = parse_args()

    run_segmenter(
        mode=args.mode,
        auto_clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()