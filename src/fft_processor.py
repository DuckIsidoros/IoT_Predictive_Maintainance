import argparse  
from pathlib import Path  

import numpy as np
import pandas as pd

from config import (  #  Centralize pipeline configuration in config.py.
    SAMPLING_RATE_HZ,
    WINDOW_SIZE,
    EXPECTED_FFT_BINS,
    EXPECTED_FREQ_MAX_HZ,
    NYQUIST_FREQUENCY_HZ,
    FREQ_RESOLUTION_HZ,
    FEATURE_COLUMNS,
    CLASS_LABELS,
    get_paths,
)


# =========================================================
# CONSTANTS
# =========================================================

#  Keep required columns local because they are specific to window CSV schema.
REQUIRED_COLUMNS = ["timestamp_ms", *FEATURE_COLUMNS, "label"]

#  Keep sensor columns local because FFT only uses acceleration signals.
SENSOR_COLUMNS = FEATURE_COLUMNS




# =========================================================
# CLI ARGUMENTS
# =========================================================

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="FFT processor for deployment vibration window data."
    )

    return parser.parse_args()


# =========================================================
# SAFETY CHECKS
# =========================================================

def validate_config():
    """
    Validate core FFT configuration before processing files.

    
    Values now come from config.py instead of local hard-coded constants.
    """

    if SAMPLING_RATE_HZ != 500:
        raise ValueError(
            f"Invalid sampling rate: {SAMPLING_RATE_HZ}, expected 500 Hz"
        )

    if WINDOW_SIZE != 256:
        raise ValueError(
            f"Invalid window size: {WINDOW_SIZE}, expected 640 samples"
        )

    if EXPECTED_FREQ_MAX_HZ != 250:
        raise ValueError(
            f"Invalid Nyquist frequency: {EXPECTED_FREQ_MAX_HZ}, expected 250 Hz"
        )
    if NYQUIST_FREQUENCY_HZ != SAMPLING_RATE_HZ / 2:
        raise ValueError(
            f"Invalid Nyquist frequency: {NYQUIST_FREQUENCY_HZ}, expected {SAMPLING_RATE_HZ / 2} Hz"
        )
    if FREQ_RESOLUTION_HZ != SAMPLING_RATE_HZ / WINDOW_SIZE:
        raise ValueError(
            f"Invalid frequency resolution: {FREQ_RESOLUTION_HZ}, expected {SAMPLING_RATE_HZ / WINDOW_SIZE} Hz/bin"
         )


def validate_mode_paths(paths: dict):
    """
    Validate required paths for FFT processing.


    This makes missing config keys fail early with a clear message.
    """

    required_keys = ["windows", "fft_windows", "fft_report"]

    missing_keys = [key for key in required_keys if key not in paths]

    if missing_keys:
        raise KeyError(
            f"Missing path config keys: {missing_keys}. "
            f"Please update PIPELINE_PATHS in config.py."
        )


# =========================================================
# SIGNAL PREPARATION
# =========================================================

def compute_fft_ready_signal(df: pd.DataFrame) -> np.ndarray:
    """
    Prepare accZ signal for FFT.

    Steps:
    1. Use accZ as vibration signal.
    2. Apply Hann window.
    """

    acc_z = df["accZ_filt"].values.astype(float)

    # KHÔNG remove DC nữa

    hann_window = np.hanning(len(acc_z))
    windowed_signal = acc_z * hann_window

    return windowed_signal


# =========================================================
# FFT COMPUTATION
# =========================================================

def compute_fft(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided FFT using rFFT.

    
    Keeps the original rFFT logic but now receives fs from config.py.
    """

    n = len(signal)

    fft_values = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)

    #  Normalize magnitude by number of samples.
    magnitude = np.abs(fft_values) / n

    return freqs, magnitude


# =========================================================
# WINDOW FILE PROCESSING
# =========================================================

def process_window_file(file_path: Path, output_dir: Path) -> dict:
    """
    Process one window CSV file and export one FFT CSV file.

    
    Uses pathlib Path objects instead of os.path strings.
    """

    df = pd.read_csv(file_path)

    #  Check required columns before processing.
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {file_path}: {missing_cols}")

    #  Check expected window size from centralized config.
    if len(df) != WINDOW_SIZE:
        raise ValueError(
            f"Invalid window size in {file_path}: "
            f"{len(df)} samples, expected {WINDOW_SIZE}"
        )

    #  Each window must belong to exactly one label.
    if df["label"].nunique() != 1:
        raise ValueError(f"Mixed labels detected in {file_path}")

    label = df["label"].iloc[0]

    #  Validate label using centralized class list from config.py.
    if label not in CLASS_LABELS:
        raise ValueError(f"Invalid label '{label}' in file: {file_path}")

    #  Convert sensor columns to numeric and fail fast if invalid.
    for col in SENSOR_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    signal_z = df["accZ_filt"].values.astype(float) 
    hann_window = np.hanning(len(signal_z))
    windowed_signal = signal_z * hann_window

    freqs, magnitude = compute_fft(windowed_signal, SAMPLING_RATE_HZ)

    #  Check FFT output bins using centralized config.
    if len(freqs) != EXPECTED_FFT_BINS:
        raise ValueError(
            f"Invalid FFT bins in {file_path}: "
            f"{len(freqs)} bins, expected {EXPECTED_FFT_BINS}"
        )

    #  Check max frequency against EXPECTED_FREQ_MAX_HZ from config.py.
    if not np.isclose(freqs[-1], EXPECTED_FREQ_MAX_HZ):
        raise ValueError(
            f"Invalid max FFT frequency in {file_path}: "
            f"{freqs[-1]} Hz, expected {EXPECTED_FREQ_MAX_HZ} Hz"
        )

    #  Ensure output label folder exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{file_path.stem}_fft.csv"
    output_path = output_dir / output_name

    fft_df = pd.DataFrame(
        {
            "frequency_hz": freqs,
            "magnitude": magnitude,
            "label": label,
        }
    )

    fft_df.to_csv(output_path, index=False)

    return {
        "source_window": str(file_path),
        "label": label,
        "fft_file": str(output_path),

        "num_samples": len(df),
        "expected_window_size": WINDOW_SIZE,

        "expected_fs": SAMPLING_RATE_HZ,
        "num_fft_bins": len(freqs),
        "expected_fft_bins": EXPECTED_FFT_BINS,

        "freq_resolution_hz": SAMPLING_RATE_HZ / len(df),
        "expected_freq_resolution_hz": FREQ_RESOLUTION_HZ,

        "max_frequency_hz": freqs[-1],
        "nyquist_frequency_hz": NYQUIST_FREQUENCY_HZ,

        "dc_removed": True,
        "window_function": "hann",
        "fft_type": "rfft",
        "signal_type": "accZ",

        "status": "OK",
        "reason": "valid",
    }


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():
    """
    Main FFT processing pipeline.
    """

    parse_args()

    validate_config()

    paths = get_paths()
    validate_mode_paths(paths)

    # Resolve paths for selected mode.
    window_root = Path(paths["windows"])
    output_root = Path(paths["fft_windows"])
    report_path = Path(paths["fft_report"])

    all_report_rows = []

    print("======================================")
    print("FFT PROCESSING STARTED")
    print("======================================")
    print(f"WINDOW_ROOT               : {window_root}")
    print(f"OUTPUT_ROOT               : {output_root}")
    print(f"REPORT_PATH               : {report_path}")
    print("--------------------------------------")
    print(f"Sampling Rate             : {SAMPLING_RATE_HZ} Hz")
    print(f"Expected Window Size      : {WINDOW_SIZE} samples")
    print(f"Expected FFT Bins         : {EXPECTED_FFT_BINS}")
    print(f"Nyquist Frequency         : {NYQUIST_FREQUENCY_HZ} Hz")
    print(f"Frequency Resolution      : {FREQ_RESOLUTION_HZ} Hz/bin")
    print("======================================")

    if not window_root.exists():
        print(f"ERROR: Input window folder does not exist: {window_root}")
        print("Hint: Run windowing first or check deployment paths in config.py.")
        return

    #  Create FFT output root folder if missing.
    output_root.mkdir(parents=True, exist_ok=True)

    # Process only known class label folders in predictable order.
    for label in CLASS_LABELS:
        label_dir = window_root / label

        if not label_dir.exists():
            print("--------------------------------------")
            print(f"[SKIP] Missing label folder: {label_dir}")
            continue

        if not label_dir.is_dir():
            print("--------------------------------------")
            print(f"[SKIP] Not a directory: {label_dir}")
            continue

        output_dir = output_root / label
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(label_dir.glob("*.csv"))

        print("--------------------------------------")
        print(f"Processing label          : {label}")
        print(f"Window files found        : {len(csv_files)}")

        created_count = 0
        failed_count = 0

        for file_path in csv_files:
            try:
                row = process_window_file(file_path, output_dir)
                all_report_rows.append(row)
                created_count += 1

            except Exception as e:
                failed_count += 1

                #  Failed files are still logged into the report for traceability.
                all_report_rows.append(
                    {
                        "source_window": str(file_path),
                        "label": label,
                        "fft_file": None,
                        "num_samples": None,
                        "expected_window_size": WINDOW_SIZE,
                        "expected_fs": SAMPLING_RATE_HZ,
                        "num_fft_bins": None,
                        "expected_fft_bins": EXPECTED_FFT_BINS,
                        "freq_resolution_hz": None,
                        "expected_freq_resolution_hz": FREQ_RESOLUTION_HZ,
                        "max_frequency_hz": None,
                        "nyquist_frequency_hz": NYQUIST_FREQUENCY_HZ,
                        "dc_removed": True,
                        "window_function": "hann",
                        "fft_type": "rfft",
                        "signal_type": "accZ",
                        "status": "FAIL",
                        "reason": str(e),
                    }
                )

                print(f"[FAIL] {file_path.name}: {e}")

        print(f"FFT files created for {label}: {created_count}")
        print(f"FFT files failed  for {label}: {failed_count}")

    if len(all_report_rows) == 0:
        print("No FFT files created. Check input windows.")
        return

    report_df = pd.DataFrame(all_report_rows)

    #  Ensure report parent folder exists before saving.
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_df.to_csv(report_path, index=False)

    print("\n======================================")
    print("FFT PROCESSING COMPLETED")
    print("======================================")
    print(f"Report saved to: {report_path}")
    print("--------------------------------------")
    print("FFT status by label:")
    print(report_df.groupby(["label", "status"]).size())
    print("======================================")


if __name__ == "__main__":
    main()
