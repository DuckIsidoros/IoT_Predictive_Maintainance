import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    SAMPLING_RATE_HZ,
    EXPECTED_DT_MS,
    CLASS_LABELS,
    LABEL_ALIASES,
    get_paths,
)


# =========================================================
# VALIDATION CONFIG
# =========================================================

REQUIRED_COLUMNS = ["timestamp_ms", "accX", "accY", "accZ"]

OPTIONAL_LABEL_COLUMN = "label"

MIN_DT_MS = EXPECTED_DT_MS * 0.5
MAX_DT_MS = EXPECTED_DT_MS * 1.5
GAP_THRESHOLD_MS = EXPECTED_DT_MS * 6

MAX_JITTER_RATIO_WARNING = 0.10
MAX_JITTER_RATIO_FAIL = 0.20

MAX_FS_ERROR_RATIO_WARNING = 0.05
MAX_FS_ERROR_RATIO_FAIL = 0.10

MAX_ABS_ACCEL_VALUE = 1000.0


# =========================================================
# ARGUMENT PARSING
# =========================================================

def parse_args():
    """
    Parse command-line arguments for validation mode selection.
    """

    parser = argparse.ArgumentParser(
        description="Validate raw sensor CSV files for mock or real pipeline."
    )

    parser.add_argument(
        "--mode",
        choices=["mock", "real"],
        default="mock",
        help="Pipeline mode. Use 'mock' for generated data or 'real' for incoming real sensor data.",
    )

    parser.add_argument(
        "--move-real-files",
        action="store_true",
        help="Move real files from incoming to accepted/rejected based on validation status.",
    )

    parser.add_argument(
        "--accept-warnings",
        action="store_true",
        help="Allow real files with WARNING status to be moved to accepted. By default, only PASS is accepted.",
    )

    return parser.parse_args()


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def add_reason(result, severity, reason):
    """
    Add a validation reason and update final status.

    Severity:
        PASS does not downgrade status.
        WARNING downgrades PASS to WARNING.
        FAIL overrides all other statuses.
    """

    result["reasons"].append(reason)

    if severity == "FAIL":
        result["status"] = "FAIL"
    elif severity == "WARNING" and result["status"] == "PASS":
        result["status"] = "WARNING"


def infer_label_from_filename(file_path):
    """
    Infer label from filename when label column is not available.

    Example:
        fan01_healthy_session001.csv -> healthy
    """

    filename = Path(file_path).stem.lower()

    matched_labels = [
        label for label in CLASS_LABELS
        if label.lower() in filename
    ]

    if len(matched_labels) == 1:
        return matched_labels[0]

    return None
def normalize_label(label):
    """
    Normalize label using defined aliases.

    This allows more flexible labeling in the CSV files.
    """

    if label is None:
        return None
    normalized = str(label).strip().lower()
    return LABEL_ALIASES.get(normalized, normalized)


def safe_numeric_series(df, column_name):
    """
    Convert one dataframe column to numeric values.
    Invalid values become NaN and will be counted later.
    """

    return pd.to_numeric(df[column_name], errors="coerce")


def has_duplicated_header_rows(df):
    """
    Detect duplicated CSV header rows inside the data body.
    """

    if df.empty:
        return False

    for col in df.columns:
        if df[col].astype(str).str.strip().eq(col).any():
            return True

    return False


# =========================================================
# VALIDATION FUNCTION
# =========================================================

def validate_file(file_path, expected_status=None, mode="mock"):
    """
    Validate one raw sensor CSV file.

    For mock mode:
        The function validates clean and corrupted test files.

    For real mode:
        The function validates incoming real data before it can be accepted.
    """

    file_path = Path(file_path)

    result = {
        "file": str(file_path),
        "filename": file_path.name,
        "mode": mode,
        "expected_status": expected_status,
        "status": "PASS",
        "reasons": [],
        "label_source": None,
        "label": None,
        "total_samples": 0,
        "mean_dt_ms": None,
        "std_dt_ms": None,
        "min_dt_ms": None,
        "max_dt_ms": None,
        "effective_fs": None,
        "fs_error_ratio": None,
        "jitter_ratio": None,
        "num_non_increasing_ts": 0,
        "num_duplicate_timestamps": 0,
        "num_bad_dt": 0,
        "num_large_gaps": 0,
        "estimated_missing_samples": 0,
        "num_nan_rows": 0,
        "num_inf_rows": 0,
        "num_extreme_sensor_rows": 0,
        "decision": None,
    }
    # Check file extension
    if file_path.suffix.lower() != ".csv":
        add_reason(result, "FAIL", "not_csv_file")
        result["decision"] = result["status"]
        return result
    # Check if file is empty or cannot be read
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        add_reason(result, "FAIL", f"cannot_read_file: {exc}")
        result["decision"] = result["status"]
        return result

    result["total_samples"] = len(df)
    # Check if file is empty
    if len(df) == 0:
        add_reason(result, "FAIL", "empty_file")
        result["decision"] = result["status"]
        return result
    # Check for duplicated header rows
    if has_duplicated_header_rows(df):
        add_reason(result, "FAIL", "duplicated_header_rows_detected")
    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        add_reason(result, "FAIL", f"missing_required_columns: {missing_cols}")
        result["decision"] = result["status"]
        return result
    # Process label column if available, otherwise infer from filename
    if OPTIONAL_LABEL_COLUMN in df.columns:
        unique_labels = df[OPTIONAL_LABEL_COLUMN].dropna().astype(str).str.strip().unique()
    # If label column is present but empty, try to infer from filename and issue a warning for mock data or fail for real data
        if len(unique_labels) == 1:
            raw_label = unique_labels[0]
            label = normalize_label(raw_label)
            result["label"] = label
            result["label_source"] = "column"

            if str(raw_label).strip().lower() != label:
                add_reason(result, "WARNING", f"label_normalized_from_{raw_label}->{label}")
        elif len(unique_labels) == 0:#no valid labels in column, try to infer from filename
            label = infer_label_from_filename(file_path)
            result["label"] = label
            result["label_source"] = "filename"
            if mode =="real":
                add_reason(result, "FAIL", "missing_label_column_and_cannot_infer_from_filename")
            else:
                add_reason(result, "WARNING", "label_column_empty_using_filename")
        else:
            add_reason(result, "FAIL", "mixed_labels_in_file")
            result["label"] = ";".join(unique_labels)
            result["label_source"] = "column"
    else:
        label = infer_label_from_filename(file_path)
        result["label"] = label
        result["label_source"] = "filename"

        if mode == "real":
            add_reason(result, "FAIL", "missing_label_column")
        else:
            add_reason(result, "FAIL", "missing_label_column")
    # Check if label is valid
    if result["label"] not in CLASS_LABELS:
        add_reason(result, "FAIL", f"invalid_or_missing_label: {result['label']}")
    # If there are already reasons for failure, we can skip the rest of the checks to save time
    if len(df) < 2:
        add_reason(result, "FAIL", "not_enough_samples")
        result["decision"] = result["status"]
        return result
    # Check numeric columns and count NaN/inf/extreme values
    numeric_columns = ["timestamp_ms", "accX", "accY", "accZ"]
    # Convert to numeric and count non-numeric values as NaN
    for column in numeric_columns:
        df[column] = safe_numeric_series(df, column)
    # Count NaN values in any of the numeric columns
    nan_rows = df[numeric_columns].isna().any(axis=1).sum()
    result["num_nan_rows"] = int(nan_rows)

    if nan_rows > 0:
        add_reason(result, "FAIL", "nan_or_non_numeric_values_detected")

    inf_rows = np.isinf(df[numeric_columns].to_numpy(dtype=float)).any(axis=1).sum()
    result["num_inf_rows"] = int(inf_rows)

    if inf_rows > 0:
        add_reason(result, "FAIL", "inf_values_detected")

    sensor_abs_max = df[["accX", "accY", "accZ"]].abs().max(axis=1)
    extreme_rows = (sensor_abs_max > MAX_ABS_ACCEL_VALUE).sum()
    result["num_extreme_sensor_rows"] = int(extreme_rows)

    if extreme_rows > 0:
        add_reason(result, "WARNING", "extreme_sensor_values_detected")

    timestamp = df["timestamp_ms"]
    dt = timestamp.diff().dropna()

    mean_dt = dt.mean()
    std_dt = dt.std()
    min_dt = dt.min()
    max_dt = dt.max()

    result["mean_dt_ms"] = mean_dt
    result["std_dt_ms"] = std_dt
    result["min_dt_ms"] = min_dt
    result["max_dt_ms"] = max_dt

    non_increasing = (dt <= 0).sum()
    result["num_non_increasing_ts"] = int(non_increasing)

    if non_increasing > 0:
        add_reason(result, "FAIL", "non_increasing_timestamp")

    duplicate_timestamps = timestamp.duplicated().sum()
    result["num_duplicate_timestamps"] = int(duplicate_timestamps)

    if duplicate_timestamps > 0:
        add_reason(result, "FAIL", "duplicate_timestamps")

    bad_dt = ((dt < MIN_DT_MS) | (dt > MAX_DT_MS)).sum()
    result["num_bad_dt"] = int(bad_dt)
    if bad_dt > 0: 
        if mode == "real":
            add_reason(result, "WARNING", "inconsistent_sampling_interval")
        else:
            add_reason(result, "FAIL", "inconsistent_sampling_interval")

    large_gaps = dt[dt > GAP_THRESHOLD_MS]
    result["num_large_gaps"] = int(len(large_gaps))

    if len(large_gaps) > 0:
        add_reason(result, "FAIL", "large_timestamp_gap")

        estimated_missing = ((large_gaps / EXPECTED_DT_MS).round() - 1).sum()
        result["estimated_missing_samples"] = int(max(0, estimated_missing))

    if mean_dt and mean_dt > 0:
        effective_fs = 1000 / mean_dt
        fs_error_ratio = abs(effective_fs - SAMPLING_RATE_HZ) / SAMPLING_RATE_HZ

        result["effective_fs"] = effective_fs
        result["fs_error_ratio"] = fs_error_ratio
    # more lenient thresholds for real data to account for natural variability, while stricter for mock data which should be clean
        if fs_error_ratio > MAX_FS_ERROR_RATIO_FAIL:
            if mode == "real":
                add_reason(result, "WARNING", "effective_sampling_rate_deviation_warning")
            else:
                add_reason(result, "FAIL", "effective_sampling_rate_deviation_fail")
        elif fs_error_ratio > MAX_FS_ERROR_RATIO_WARNING:
            add_reason(result, "WARNING", "effective_sampling_rate_deviation_warning")

    if mean_dt and mean_dt > 0:
        jitter_ratio = std_dt / mean_dt if pd.notna(std_dt) else 0.0
        result["jitter_ratio"] = jitter_ratio
        # more lenient thresholds for real data to account for natural variability, while stricter for mock data which should be clean
        if jitter_ratio > MAX_JITTER_RATIO_FAIL:
            if mode == "real":
                add_reason(result, "WARNING", "high_sampling_jitter_warning")
            else:
                add_reason(result, "FAIL", "high_sampling_jitter_fail")

    if len(result["reasons"]) == 0:
        result["reasons"] = ["ok"]

    result["decision"] = result["status"]

    return result


# =========================================================
# FILE COLLECTION
# =========================================================

def collect_mock_files(paths):
    """
    Collect mock clean and corrupted CSV files.

    Clean mock files are expected to PASS.
    Corrupted mock files are expected to FAIL.
    """

    files = []

    clean_dir = paths["raw"]
    corrupted_dir = paths["corrupted"]

    for label in CLASS_LABELS:
        class_dir = clean_dir / label

        if not class_dir.exists():
            continue

        for file_path in sorted(class_dir.glob("*.csv")):
            files.append((file_path, "PASS"))

    if corrupted_dir.exists():
        for file_path in sorted(corrupted_dir.rglob("*.csv")):
            files.append((file_path, "FAIL"))

    return files


def collect_real_files(paths):
    """
    Collect real incoming CSV files.

    Real files do not have expected status because they are unknown before validation.
    """

    incoming_dir = paths["incoming"]

    files = []

    if not incoming_dir.exists():
        return files

    for file_path in sorted(incoming_dir.iterdir()):
        if file_path.is_file():
            files.append((file_path, None))

    return files


# =========================================================
# REAL FILE ROUTING
# =========================================================

def route_real_file(file_path, status, paths, accept_warnings=False):
    """
    Move real files from incoming to accepted or rejected folder.

    PASS files are accepted.
    WARNING files are rejected unless --accept-warnings is specified.
    """

    file_path = Path(file_path)

    should_accept = status == "PASS" or (status == "WARNING" and accept_warnings)

    if should_accept:
        target_dir = paths["raw"]
    else:
        target_dir = paths["rejected"]

    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / file_path.name

    if target_path.exists():
        target_path = target_dir / f"{file_path.stem}_validated{file_path.suffix}"

    shutil.move(str(file_path), str(target_path))

    return target_path

# =========================================================
# REPORTING
# =========================================================

def save_report(results, report_file):
    """
    Save validation results into a CSV report.
    """

    report_file = Path(report_file)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(results)
    report_df["reasons"] = report_df["reasons"].apply(
        lambda reasons: ";".join(reasons)
    )

    report_df.to_csv(report_file, index=False)

    return report_df


def print_summary(report_df, mode):
    """
    Print validation summary to console.
    """

    print("--------------------------------------")
    print("Validation completed.")
    print(f"Mode: {mode}")
    print("--------------------------------------")
    print(f"Total files : {len(report_df)}")
    print(f"PASS        : {(report_df['status'] == 'PASS').sum()}")
    print(f"WARNING     : {(report_df['status'] == 'WARNING').sum()}")
    print(f"FAIL        : {(report_df['status'] == 'FAIL').sum()}")

    if "expected_status" in report_df.columns and report_df["expected_status"].notna().any():
        mismatch = (
            report_df["expected_status"].notna()
            & (report_df["status"] != report_df["expected_status"])
        ).sum()
        print(f"Mismatch    : {mismatch}")


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_validation(mode, move_real_files=False, accept_warnings=False):
    """
    Run validation pipeline for mock or real mode.
    """

    paths = get_paths(mode)

    if mode == "mock":
        files = collect_mock_files(paths)
        report_file = paths["qc_reports"] / "qc_report.csv"
    else:
        files = collect_real_files(paths)
        report_file = paths["qc_reports"] / "real_validation_report.csv"

    if not files:
        print(f"No files found for mode={mode}.")
        return

    print("======================================")
    print("VALIDATION PIPELINE STARTED")
    print("======================================")
    print(f"Mode             : {mode}")
    print(f"Expected dt      : {EXPECTED_DT_MS:.3f} ms")
    print(f"Expected fs      : {SAMPLING_RATE_HZ} Hz")
    print(f"Report file      : {report_file}")
    print(f"Move real files  : {move_real_files}")
    print(f"Accept warnings   : {accept_warnings}")
    print("======================================")

    results = []

    for file_path, expected_status in files:
        result = validate_file(
            file_path=file_path,
            expected_status=expected_status,
            mode=mode,
        )

        if mode == "real" and move_real_files:
            routed_path = route_real_file(
                file_path=file_path,
                status=result["status"],
                paths=paths,
                accept_warnings=accept_warnings,
            )
            result["routed_to"] = str(routed_path)
        else:
            result["routed_to"] = None

        results.append(result)

        if expected_status:
            check = "OK" if result["status"] == expected_status else "MISMATCH"
            print(
                f"[{check}] {result['status']} | expected={expected_status} | {file_path}"
            )
        else:
            print(f"[{result['status']}] {file_path}")

        if result["status"] in ["WARNING", "FAIL"]:
            print(f"reasons: {result['reasons']}")

    report_df = save_report(results, report_file)

    print(f"Report saved to: {report_file}")
    print_summary(report_df, mode)


def main():
    """
    Entry point for command-line execution.
    """

    args = parse_args()
    run_validation(
        mode=args.mode,
        move_real_files=args.move_real_files,
        accept_warnings=args.accept_warnings,
    )


if __name__ == "__main__":
    main()