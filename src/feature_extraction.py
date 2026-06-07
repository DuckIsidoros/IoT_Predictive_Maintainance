from pathlib import Path
import shutil
import pandas as pd
import numpy as np


# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FFT_INPUT_DIR = PROJECT_ROOT / "fft_windows"
FEATURE_OUTPUT_DIR = PROJECT_ROOT / "features"

FEATURE_DATASET_PATH = FEATURE_OUTPUT_DIR / "features_dataset.csv"
FEATURE_REPORT_PATH = FEATURE_OUTPUT_DIR / "feature_extraction_report.csv"

EXPECTED_CLASSES = ["healthy", "imbalance", "obstruction"]

EXPECTED_FFT_BINS = 129
EXPECTED_FREQ_MIN_HZ = 0.0
EXPECTED_FREQ_MAX_HZ = 100.0
# Frequency bands in Hz
LOW_BAND = (0.0, 20.0)
MID_BAND = (20.0, 60.0)
HIGH_BAND = (60.0, 100.0)

AUTO_CLEAN_OUTPUT = True


# =========================
# Utility functions
# =========================

def prepare_output_dir(output_dir: Path, auto_clean: bool = True) -> None:
    """
    Prepare output directory.

    If auto_clean=True, remove old feature artifacts to avoid stale outputs.
    """
    if auto_clean and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """
    Find a matching column from a list of possible column names.
    Case-insensitive.
    """
    lower_map = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        f"Cannot find any matching column from candidates={candidates}. "
        f"Available columns={list(df.columns)}"
    )


def infer_label_from_path(file_path: Path) -> str:
    """
    Infer label from parent folder or filename.

    Preferred structure:
        fft_windows/healthy/*.csv
        fft_windows/imbalance/*.csv
        fft_windows/obstruction/*.csv
    """
    parent_name = file_path.parent.name.lower()

    if parent_name in EXPECTED_CLASSES:
        return parent_name

    file_name = file_path.name.lower()

    for label in EXPECTED_CLASSES:
        if label in file_name:
            return label

    raise ValueError(f"Cannot infer label from path: {file_path}")


def safe_weighted_centroid(freq: np.ndarray, mag: np.ndarray) -> float:
    """
    Calculate spectral centroid safely.
    """
    mag_sum = np.sum(mag)

    if mag_sum <= 0:
        return 0.0

    return float(np.sum(freq * mag) / mag_sum)


def safe_spectral_bandwidth(freq: np.ndarray, mag: np.ndarray, centroid: float) -> float:
    """
    Calculate spectral bandwidth safely.
    """
    mag_sum = np.sum(mag)

    if mag_sum <= 0:
        return 0.0

    variance = np.sum(((freq - centroid) ** 2) * mag) / mag_sum
    return float(np.sqrt(variance))


def calculate_band_energy(
    freq: np.ndarray,
    mag: np.ndarray,
    low: float,
    high: float,
    include_high: bool = False
) -> float:
    """
    Calculate energy in a frequency band.

    Energy = sum(magnitude^2)
    """
    if include_high:
        mask = (freq >= low) & (freq <= high)
    else:
        mask = (freq >= low) & (freq < high)

    return float(np.sum(mag[mask] ** 2))


def validate_fft_vector(freq: np.ndarray, mag: np.ndarray, file_path: Path) -> list[str]:
    """
    Return a list of warnings for one FFT file.
    Does not crash the whole pipeline unless the file is unreadable or malformed.
    """
    warnings = []

    if len(freq) != EXPECTED_FFT_BINS:
        warnings.append(f"Invalid FFT bins: expected {EXPECTED_FFT_BINS}, got {len(freq)}")

    if len(freq) != len(mag):
        warnings.append(f"Frequency and magnitude length mismatch: {len(freq)} vs {len(mag)}")

    if len(freq) > 0:
        freq_min = float(np.min(freq))
        freq_max = float(np.max(freq))

        if not np.isclose(freq_min, EXPECTED_FREQ_MIN_HZ, atol=1e-6):
            warnings.append(f"Unexpected freq_min: expected {EXPECTED_FREQ_MIN_HZ}, got {freq_min}")

        if not np.isclose(freq_max, EXPECTED_FREQ_MAX_HZ, atol=1e-6):
            warnings.append(f"Unexpected freq_max: expected {EXPECTED_FREQ_MAX_HZ}, got {freq_max}")

    if np.any(np.isnan(freq)) or np.any(np.isnan(mag)):
        warnings.append("NaN found in frequency or magnitude")

    if np.any(mag < 0):
        warnings.append("Negative magnitude found")

    return warnings


# =========================
# Core feature extraction
# =========================

def extract_features_from_fft_file(file_path: Path) -> tuple[dict, dict]:
    """
    Extract features from one FFT CSV file.

    Returns:
        feature_row: dict
        report_row: dict
    """
    df = pd.read_csv(file_path)

    freq_col = find_column(
        df,
        candidates=[
            "frequency_hz",
            "freq_hz",
            "frequency",
            "freq",
            "hz"
        ]
    )

    mag_col = find_column(
        df,
        candidates=[
            "magnitude",
            "mag",
            "fft_magnitude",
            "amplitude",
            "spectrum_magnitude"
        ]
    )

    freq = pd.to_numeric(df[freq_col], errors="coerce").to_numpy(dtype=float)
    mag = pd.to_numeric(df[mag_col], errors="coerce").to_numpy(dtype=float)

    label = infer_label_from_path(file_path)

    warnings = validate_fft_vector(freq, mag, file_path)

    # Remove rows with NaN for calculation safety.
    valid_mask = ~np.isnan(freq) & ~np.isnan(mag)
    freq = freq[valid_mask]
    mag = mag[valid_mask]

    if len(freq) == 0 or len(mag) == 0:
        raise ValueError(f"No valid FFT data after cleaning NaN: {file_path}")

    dominant_idx = int(np.argmax(mag))
    dominant_frequency = float(freq[dominant_idx])

    max_magnitude = float(np.max(mag))
    mean_magnitude = float(np.mean(mag))
    std_magnitude = float(np.std(mag))
    total_energy = float(np.sum(mag ** 2))

    spectral_centroid = safe_weighted_centroid(freq, mag)
    spectral_bandwidth = safe_spectral_bandwidth(freq, mag, spectral_centroid)

    low_band_energy = calculate_band_energy(
        freq,
        mag,
        low=LOW_BAND[0],
        high=LOW_BAND[1],
        include_high=False
    )

    mid_band_energy = calculate_band_energy(
        freq,
        mag,
        low=MID_BAND[0],
        high=MID_BAND[1],
        include_high=False
    )

    high_band_energy = calculate_band_energy(
        freq,
        mag,
        low=HIGH_BAND[0],
        high=HIGH_BAND[1],
        include_high=True
    )

    feature_row = {
        "source_file": str(file_path.relative_to(PROJECT_ROOT)),
        "dominant_frequency": dominant_frequency,
        "max_magnitude": max_magnitude,
        "mean_magnitude": mean_magnitude,
        "std_magnitude": std_magnitude,
        "total_energy": total_energy,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "low_band_energy": low_band_energy,
        "mid_band_energy": mid_band_energy,
        "high_band_energy": high_band_energy,
        "label": label,
    }

    report_row = {
        "source_file": str(file_path.relative_to(PROJECT_ROOT)),
        "label": label,
        "status": "WARNING" if warnings else "OK",
        "n_bins": len(freq),
        "freq_min": float(np.min(freq)),
        "freq_max": float(np.max(freq)),
        "warnings": " | ".join(warnings),
    }

    return feature_row, report_row


def collect_fft_files(input_dir: Path) -> list[Path]:
    """
    Collect all FFT CSV files from input directory recursively.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"FFT input directory not found: {input_dir}")

    files = sorted(input_dir.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No FFT CSV files found under: {input_dir}")

    return files


def run_feature_extraction() -> None:
    """
    Main feature extraction pipeline.
    """
    prepare_output_dir(FEATURE_OUTPUT_DIR, auto_clean=AUTO_CLEAN_OUTPUT)

    fft_files = collect_fft_files(FFT_INPUT_DIR)

    feature_rows = []
    report_rows = []
    failed_rows = []

    for file_path in fft_files:
        try:
            feature_row, report_row = extract_features_from_fft_file(file_path)
            feature_rows.append(feature_row)
            report_rows.append(report_row)

        except Exception as exc:
            failed_rows.append({
                "source_file": str(file_path.relative_to(PROJECT_ROOT)),
                "label": "unknown",
                "status": "FAILED",
                "n_bins": None,
                "freq_min": None,
                "freq_max": None,
                "warnings": str(exc),
            })

    features_df = pd.DataFrame(feature_rows)
    report_df = pd.DataFrame(report_rows + failed_rows)

    features_df.to_csv(FEATURE_DATASET_PATH, index=False)
    report_df.to_csv(FEATURE_REPORT_PATH, index=False)

    print("=" * 70)
    print("Feature Extraction Completed")
    print("=" * 70)
    print(f"FFT input directory     : {FFT_INPUT_DIR}")
    print(f"Feature output dataset : {FEATURE_DATASET_PATH}")
    print(f"Feature report         : {FEATURE_REPORT_PATH}")
    print(f"Total FFT files found  : {len(fft_files)}")
    print(f"Successful rows        : {len(features_df)}")
    print(f"Failed rows            : {len(failed_rows)}")

    if not features_df.empty:
        print("\nRows per class:")
        print(features_df["label"].value_counts())

        print("\nMissing values per column:")
        print(features_df.isna().sum())

    if failed_rows:
        print("\nWARNING: Some files failed during feature extraction.")
        print("Check feature_extraction_report.csv for details.")

    print("=" * 70)


if __name__ == "__main__":
    run_feature_extraction()