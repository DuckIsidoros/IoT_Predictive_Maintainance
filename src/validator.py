import os
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DATA_DIR = os.path.join(BASE_DIR, "mock_raw_sensor_stream")
CORRUPTED_DATA_DIR = os.path.join(BASE_DIR, "corrupted_data")
REPORT_DIR = os.path.join(BASE_DIR, "qc_reports")
REPORT_FILE = os.path.join(REPORT_DIR, "qc_report.csv")

EXPECTED_FS = 200  # Hz
EXPECTED_DT_MS = 1000 / EXPECTED_FS  # 5 ms

MIN_DT_MS = EXPECTED_DT_MS * 0.5      # 2.5 ms
MAX_DT_MS = EXPECTED_DT_MS * 1.5      # 7.5 ms
GAP_THRESHOLD_MS = EXPECTED_DT_MS * 3 # 15 ms

MAX_JITTER_RATIO = 0.10               # 10%
MAX_FS_ERROR_RATIO = 0.05             # 5%

REQUIRED_COLUMNS = ["timestamp_ms", "accX", "accY", "accZ", "label"]


# =========================
# VALIDATION FUNCTION
# =========================

def validate_file(file_path, expected_status=None):
    result = {
        "file": file_path,
        "expected_status": expected_status,
        "status": "PASS",
        "reasons": [],
        "total_samples": 0,
        "mean_dt_ms": None,
        "std_dt_ms": None,
        "min_dt_ms": None,
        "max_dt_ms": None,
        "effective_fs": None,
        "fs_error_ratio": None,
        "jitter_ratio": None,
        "num_non_increasing_ts": 0,
        "num_bad_dt": 0,
        "num_large_gaps": 0,
        "estimated_missing_samples": 0,
        "num_nan_rows": 0,
    }

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        result["status"] = "FAIL"
        result["reasons"].append(f"cannot_read_file: {e}")
        return result

    result["total_samples"] = len(df)

    # 1. Check columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        result["status"] = "FAIL"
        result["reasons"].append(f"missing_columns: {missing_cols}")
        return result

    # 2. Check NaN
    nan_rows = df[REQUIRED_COLUMNS].isna().any(axis=1).sum()
    result["num_nan_rows"] = int(nan_rows)

    if nan_rows > 0:
        result["status"] = "FAIL"
        result["reasons"].append("nan_values_detected")

    # 3. Check minimum samples
    if len(df) < 2:
        result["status"] = "FAIL"
        result["reasons"].append("not_enough_samples")
        return result

    # 4. Compute dt
    timestamp = df["timestamp_ms"].astype(float)
    dt = timestamp.diff().dropna()

    mean_dt = dt.mean()
    std_dt = dt.std()
    min_dt = dt.min()
    max_dt = dt.max()

    result["mean_dt_ms"] = mean_dt
    result["std_dt_ms"] = std_dt
    result["min_dt_ms"] = min_dt
    result["max_dt_ms"] = max_dt

    # 5. Non-increasing timestamp
    non_increasing = (dt <= 0).sum()
    result["num_non_increasing_ts"] = int(non_increasing)

    if non_increasing > 0:
        result["status"] = "FAIL"
        result["reasons"].append("non_increasing_timestamp")

    # 6. Bad dt: sampling interval too small or too large
    bad_dt = ((dt < MIN_DT_MS) | (dt > MAX_DT_MS)).sum()
    result["num_bad_dt"] = int(bad_dt)

    if bad_dt > 0:
        result["status"] = "FAIL"
        result["reasons"].append("inconsistent_sampling_interval")

    # 7. Large gaps / missing chunks
    large_gaps = dt[dt > GAP_THRESHOLD_MS]
    result["num_large_gaps"] = int(len(large_gaps))

    if len(large_gaps) > 0:
        result["status"] = "FAIL"
        result["reasons"].append("large_timestamp_gap")

        estimated_missing = ((large_gaps / EXPECTED_DT_MS).round() - 1).sum()
        result["estimated_missing_samples"] = int(max(0, estimated_missing))

    # 8. Effective sampling rate
    if mean_dt > 0:
        effective_fs = 1000 / mean_dt
        fs_error_ratio = abs(effective_fs - EXPECTED_FS) / EXPECTED_FS

        result["effective_fs"] = effective_fs
        result["fs_error_ratio"] = fs_error_ratio

        if fs_error_ratio > MAX_FS_ERROR_RATIO:
            result["status"] = "FAIL"
            result["reasons"].append("effective_sampling_rate_deviation")

    # 9. Jitter ratio
    if mean_dt > 0:
        jitter_ratio = std_dt / mean_dt
        result["jitter_ratio"] = jitter_ratio

        if jitter_ratio > MAX_JITTER_RATIO:
            result["status"] = "FAIL"
            result["reasons"].append("high_sampling_jitter")

    if len(result["reasons"]) == 0:
        result["reasons"] = ["ok"]

    return result


# =========================
# FILE COLLECTION
# =========================

def collect_csv_files():
    files = []

    # Clean data: expected PASS
    for root, _, filenames in os.walk(CLEAN_DATA_DIR):
        for filename in filenames:
            if filename.endswith(".csv"):
                files.append((os.path.join(root, filename), "PASS"))

    # Corrupted data: expected FAIL
    for root, _, filenames in os.walk(CORRUPTED_DATA_DIR):
        for filename in filenames:
            if filename.endswith(".csv"):
                files.append((os.path.join(root, filename), "FAIL"))

    return files


# =========================
# MAIN
# =========================

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    files = collect_csv_files()

    if not files:
        print("No CSV files found.")
        return

    results = []

    print("Starting validation...")
    print(f"Expected dt: {EXPECTED_DT_MS:.3f} ms")
    print("--------------------------------------")

    for file_path, expected_status in files:
        result = validate_file(file_path, expected_status)
        results.append(result)

        check = "OK" if result["status"] == expected_status else "MISMATCH"

        print(
            f"[{check}] {result['status']} | expected={expected_status} | {file_path}"
        )

        if result["status"] == "FAIL":
            print(f"     reasons: {result['reasons']}")

    report_df = pd.DataFrame(results)
    report_df["reasons"] = report_df["reasons"].apply(lambda x: ";".join(x))

    report_df.to_csv(REPORT_FILE, index=False)

    print("--------------------------------------")
    print(f"Validation completed.")
    print(f"Report saved to: {REPORT_FILE}")

    total = len(report_df)
    passed = (report_df["status"] == "PASS").sum()
    failed = (report_df["status"] == "FAIL").sum()
    mismatch = (report_df["status"] != report_df["expected_status"]).sum()

    print("--------------------------------------")
    print(f"Total files : {total}")
    print(f"PASS        : {passed}")
    print(f"FAIL        : {failed}")
    print(f"Mismatch    : {mismatch}")


if __name__ == "__main__":
    main()