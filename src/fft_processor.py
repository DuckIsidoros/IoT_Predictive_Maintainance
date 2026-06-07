import os
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WINDOW_ROOT = os.path.join(BASE_DIR, "windows")
OUTPUT_ROOT = os.path.join(BASE_DIR, "fft_windows")
REPORT_PATH = os.path.join(BASE_DIR, "fft_report.csv")

EXPECTED_FS = 200  # Hz
EXPECTED_WINDOW_SIZE = 256  # samples

EXPECTED_FFT_BINS = EXPECTED_WINDOW_SIZE // 2 + 1  # 129 bins
NYQUIST_FREQUENCY_HZ = EXPECTED_FS / 2             # 100 Hz
FREQ_RESOLUTION_HZ = EXPECTED_FS / EXPECTED_WINDOW_SIZE  # 0.78125 Hz

REQUIRED_COLUMNS = ["timestamp_ms", "accX", "accY", "accZ", "label"]
SENSOR_COLUMNS = ["accX", "accY", "accZ"]
VALID_LABELS = ["healthy", "imbalance", "obstruction"]


# =========================================================
# SAFETY CHECKS
# =========================================================

def validate_config():
    if EXPECTED_FS != 200:
        raise ValueError(f"Invalid sampling rate: {EXPECTED_FS}, expected 200 Hz")

    if EXPECTED_WINDOW_SIZE != 256:
        raise ValueError(
            f"Invalid window size: {EXPECTED_WINDOW_SIZE}, expected 256 samples"
        )

    if EXPECTED_FFT_BINS != 129:
        raise ValueError(
            f"Invalid FFT bins: {EXPECTED_FFT_BINS}, expected 129"
        )

    if NYQUIST_FREQUENCY_HZ != 100:
        raise ValueError(
            f"Invalid Nyquist frequency: {NYQUIST_FREQUENCY_HZ}, expected 100 Hz"
        )


# =========================================================
# SIGNAL PREPARATION
# =========================================================

def compute_fft_features_ready_signal(df):
    """
    Convert 3-axis acceleration data into one FFT-ready signal.

    Steps:
    1. Compute vector magnitude from accX, accY, accZ.
    2. Remove DC offset.
    3. Apply Hann window to reduce spectral leakage.
    """

    acc_x = df["accX"].values
    acc_y = df["accY"].values
    acc_z = df["accZ"].values

    # Vector magnitude
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    # Remove DC offset
    acc_mag = acc_mag - np.mean(acc_mag)

    # Apply Hann window
    hann_window = np.hanning(len(acc_mag))
    windowed_signal = acc_mag * hann_window

    return windowed_signal


# =========================================================
# FFT COMPUTATION
# =========================================================

def compute_fft(signal, fs):
    """
    Compute one-sided FFT using rFFT.
    """

    n = len(signal)

    fft_values = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)

    magnitude = np.abs(fft_values) / n

    return freqs, magnitude


# =========================================================
# WINDOW FILE PROCESSING
# =========================================================

def process_window_file(file_path, output_dir):
    df = pd.read_csv(file_path)

    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {file_path}: {missing_cols}")

    # Check window size
    if len(df) != EXPECTED_WINDOW_SIZE:
        raise ValueError(
            f"Invalid window size in {file_path}: "
            f"{len(df)} samples, expected {EXPECTED_WINDOW_SIZE}"
        )

    # Check label consistency
    if df["label"].nunique() != 1:
        raise ValueError(f"Mixed labels detected in {file_path}")

    label = df["label"].iloc[0]

    # Check valid label
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label '{label}' in file: {file_path}")

    # Convert sensor columns to numeric
    for col in SENSOR_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    signal = compute_fft_features_ready_signal(df)
    freqs, magnitude = compute_fft(signal, EXPECTED_FS)

    # Check FFT output shape
    if len(freqs) != EXPECTED_FFT_BINS:
        raise ValueError(
            f"Invalid FFT bins in {file_path}: "
            f"{len(freqs)} bins, expected {EXPECTED_FFT_BINS}"
        )

    os.makedirs(output_dir, exist_ok=True)

    output_name = f"{base_name}_fft.csv"
    output_path = os.path.join(output_dir, output_name)

    fft_df = pd.DataFrame({
        "frequency_hz": freqs,
        "magnitude": magnitude,
        "label": label
    })

    fft_df.to_csv(output_path, index=False)

    return {
        "source_window": file_path,
        "label": label,
        "fft_file": output_path,

        "num_samples": len(df),
        "expected_window_size": EXPECTED_WINDOW_SIZE,

        "expected_fs": EXPECTED_FS,
        "num_fft_bins": len(freqs),
        "expected_fft_bins": EXPECTED_FFT_BINS,

        "freq_resolution_hz": EXPECTED_FS / len(df),
        "expected_freq_resolution_hz": FREQ_RESOLUTION_HZ,

        "max_frequency_hz": freqs[-1],
        "nyquist_frequency_hz": NYQUIST_FREQUENCY_HZ,

        "dc_removed": True,
        "window_function": "hann",
        "fft_type": "rfft",
        "signal_type": "vector_magnitude",

        "status": "OK",
        "reason": "valid"
    }


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():
    validate_config()

    all_report_rows = []

    print("======================================")
    print("FFT PROCESSING STARTED")
    print("======================================")
    print(f"WINDOW_ROOT              : {WINDOW_ROOT}")
    print(f"OUTPUT_ROOT              : {OUTPUT_ROOT}")
    print(f"REPORT_PATH              : {REPORT_PATH}")
    print("--------------------------------------")
    print(f"Sampling Rate            : {EXPECTED_FS} Hz")
    print(f"Expected Window Size     : {EXPECTED_WINDOW_SIZE} samples")
    print(f"Expected FFT Bins        : {EXPECTED_FFT_BINS}")
    print(f"Nyquist Frequency        : {NYQUIST_FREQUENCY_HZ} Hz")
    print(f"Frequency Resolution     : {FREQ_RESOLUTION_HZ} Hz/bin")
    print("======================================")

    if not os.path.exists(WINDOW_ROOT):
        print("ERROR: WINDOW_ROOT does not exist.")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for label in sorted(os.listdir(WINDOW_ROOT)):
        label_dir = os.path.join(WINDOW_ROOT, label)

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
        print(f"Window files found     : {len(csv_files)}")

        created_count = 0
        failed_count = 0

        for filename in csv_files:
            file_path = os.path.join(label_dir, filename)

            try:
                row = process_window_file(file_path, output_dir)
                all_report_rows.append(row)
                created_count += 1

            except Exception as e:
                failed_count += 1
                all_report_rows.append({
                    "source_window": file_path,
                    "label": label,
                    "fft_file": None,
                    "num_samples": None,
                    "expected_window_size": EXPECTED_WINDOW_SIZE,
                    "expected_fs": EXPECTED_FS,
                    "num_fft_bins": None,
                    "expected_fft_bins": EXPECTED_FFT_BINS,
                    "freq_resolution_hz": None,
                    "expected_freq_resolution_hz": FREQ_RESOLUTION_HZ,
                    "max_frequency_hz": None,
                    "nyquist_frequency_hz": NYQUIST_FREQUENCY_HZ,
                    "dc_removed": True,
                    "window_function": "hann",
                    "fft_type": "rfft",
                    "signal_type": "vector_magnitude",
                    "status": "FAIL",
                    "reason": str(e)
                })

                print(f"[FAIL] {filename}: {e}")

        print(f"FFT files created for {label}: {created_count}")
        print(f"FFT files failed  for {label}: {failed_count}")

    if len(all_report_rows) == 0:
        print("No FFT files created. Check input windows.")
        return

    report_df = pd.DataFrame(all_report_rows)

    report_df.to_csv(REPORT_PATH, index=False)

    print("\n======================================")
    print("FFT PROCESSING COMPLETED")
    print("======================================")
    print(f"Report saved to: {REPORT_PATH}")
    print("--------------------------------------")
    print("FFT status by label:")
    print(report_df.groupby(["label", "status"]).size())
    print("======================================")


if __name__ == "__main__":
    main()