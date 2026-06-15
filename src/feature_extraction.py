from pathlib import Path
import argparse
import shutil

import numpy as np
import pandas as pd

from config import (
    PROJECT_ROOT,
    get_paths,
    CLASS_LABELS,
    EXPECTED_FFT_BINS,
    EXPECTED_FREQ_MIN_HZ,
    EXPECTED_FREQ_MAX_HZ,
)


# =========================
# Feature band configuration
# =========================

LOW_BAND = (0.0, 20.0)
MID_BAND = (20.0, 60.0)
HIGH_BAND = (60.0, 100.0)

AUTO_CLEAN_OUTPUT = True


# =========================
# CLI arguments
# =========================

def parse_args():
    """
    Parse command-line arguments.

    Usage:
        python src/feature_extractor.py --mode mock
        python src/feature_extractor.py --mode real
    """
    parser = argparse.ArgumentParser(
        description="Hybrid feature extraction from window CSV files and FFT CSV files"
    )

    parser.add_argument(
        "--mode",
        choices=["mock", "real"],
        default="mock",
        help="Choose between mock data or real data"
    )

    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete the old feature output folder before running"
    )

    return parser.parse_args()


# =========================
# Utility functions
# =========================

def prepare_output_dir(output_dir: Path, auto_clean: bool = True) -> None:
    """
    Prepare the output directory.

    If auto_clean is enabled, the old feature folder is removed first.
    This prevents stale output files from mixing with the current run.
    """
    if auto_clean and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """
    Find a matching column name from a list of candidates.

    The matching is case-insensitive.
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
    Infer the class label from the parent folder or file name.

    Supported labels:
        healthy
        imbalance
        obstruction
    """
    parent_name = file_path.parent.name.lower().strip()

    if parent_name in CLASS_LABELS:
        return parent_name

    file_name = file_path.name.lower().strip()

    for label in CLASS_LABELS:
        if label in file_name:
            return label

    raise ValueError(f"Cannot infer label from path: {file_path}")


def safe_relative_path(file_path: Path) -> str:
    """
    Return a safe relative path from PROJECT_ROOT.

    If the file is outside PROJECT_ROOT, return the absolute path instead.
    """
    try:
        return str(file_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(file_path.resolve())


def calculate_rms(values: np.ndarray) -> float:
    """
    Calculate root mean square.

    RMS = sqrt(mean(x^2))
    """
    return float(np.sqrt(np.mean(values ** 2)))


def safe_weighted_centroid(freq: np.ndarray, mag: np.ndarray) -> float:
    """
    Calculate spectral centroid safely.

    If the magnitude sum is zero, return 0.0 to avoid division by zero.
    """
    mag_sum = np.sum(mag)

    if mag_sum <= 0:
        return 0.0

    return float(np.sum(freq * mag) / mag_sum)


def safe_spectral_bandwidth(
    freq: np.ndarray,
    mag: np.ndarray,
    centroid: float
) -> float:
    """
    Calculate spectral bandwidth safely.

    If the magnitude sum is zero, return 0.0 to avoid division by zero.
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

    Band energy = sum(magnitude^2)
    """
    if include_high:
        mask = (freq >= low) & (freq <= high)
    else:
        mask = (freq >= low) & (freq < high)

    return float(np.sum(mag[mask] ** 2))


# =========================
# Validation functions
# =========================

def validate_fft_vector(freq: np.ndarray, mag: np.ndarray) -> list[str]:
    """
    Validate one FFT vector and return warning messages.

    This function does not stop the pipeline immediately.
    It collects warnings so they can be written to the extraction report.
    """
    warnings = []

    if len(freq) != EXPECTED_FFT_BINS:
        warnings.append(
            f"Invalid FFT bins: expected {EXPECTED_FFT_BINS}, got {len(freq)}"
        )

    if len(freq) != len(mag):
        warnings.append(
            f"Frequency and magnitude length mismatch: {len(freq)} vs {len(mag)}"
        )

    if len(freq) > 0:
        freq_min = float(np.nanmin(freq))
        freq_max = float(np.nanmax(freq))

        if not np.isclose(freq_min, EXPECTED_FREQ_MIN_HZ, atol=1e-6):
            warnings.append(
                f"Unexpected minimum frequency: expected {EXPECTED_FREQ_MIN_HZ}, got {freq_min}"
            )

        if not np.isclose(freq_max, EXPECTED_FREQ_MAX_HZ, atol=1e-6):
            warnings.append(
                f"Unexpected maximum frequency: expected {EXPECTED_FREQ_MAX_HZ}, got {freq_max}"
            )

    if np.any(np.isnan(freq)) or np.any(np.isnan(mag)):
        warnings.append("NaN found in frequency or magnitude")

    if np.any(np.isinf(freq)) or np.any(np.isinf(mag)):
        warnings.append("Infinite value found in frequency or magnitude")

    if np.any(mag < 0):
        warnings.append("Negative magnitude found")

    return warnings


def validate_window_dataframe(df: pd.DataFrame, window_file_path: Path) -> None:
    """
    Validate one window dataframe before calculating RMS features.
    """
    if df.empty:
        raise ValueError(f"Empty window file: {window_file_path}")

    label_col = find_column(df, ["label", "Label"])

    if df[label_col].nunique() != 1:
        raise ValueError(f"Mixed labels detected in window file: {window_file_path}")

    label = str(df[label_col].iloc[0]).lower().strip()

    if label not in CLASS_LABELS:
        raise ValueError(
            f"Invalid label '{label}' in window file: {window_file_path}"
        )


# =========================
# File mapping
# =========================

def map_fft_file_to_window_file(
    fft_file_path: Path,
    window_input_dir: Path
) -> Path:
    """
    Map an FFT file to its matching window file.

    Example:
        fft_windows/healthy/sample_001_fft.csv
        windows/healthy/sample_001.csv
    """
    label = infer_label_from_path(fft_file_path)

    fft_stem = fft_file_path.stem

    if fft_stem.endswith("_fft"):
        window_stem = fft_stem[:-4]
    else:
        window_stem = fft_stem

    window_file_path = window_input_dir / label / f"{window_stem}.csv"

    if not window_file_path.exists():
        raise FileNotFoundError(
            f"Matching window file not found for FFT file: {fft_file_path}. "
            f"Expected window file: {window_file_path}"
        )

    return window_file_path


# =========================
# Time-domain feature extraction
# =========================

def extract_time_domain_features_from_window_file(
    window_file_path: Path
) -> tuple[dict, list[str]]:
    """
    Extract time-domain features from one window CSV file.

    The main output of this function is:
        RMS_X
        RMS_Y
        RMS_Z
    """
    warnings = []

    df = pd.read_csv(window_file_path)
    validate_window_dataframe(df, window_file_path)

    acc_x_col = find_column(df, ["accX", "accel_x", "x"])
    acc_y_col = find_column(df, ["accY", "accel_y", "y"])
    acc_z_col = find_column(df, ["accZ", "accel_z", "z"])

    acc_x = pd.to_numeric(df[acc_x_col], errors="coerce").to_numpy(dtype=float)
    acc_y = pd.to_numeric(df[acc_y_col], errors="coerce").to_numpy(dtype=float)
    acc_z = pd.to_numeric(df[acc_z_col], errors="coerce").to_numpy(dtype=float)

    if np.any(np.isnan(acc_x)) or np.any(np.isnan(acc_y)) or np.any(np.isnan(acc_z)):
        warnings.append("NaN found in window acceleration values")

    if np.any(np.isinf(acc_x)) or np.any(np.isinf(acc_y)) or np.any(np.isinf(acc_z)):
        warnings.append("Infinite value found in window acceleration values")

    valid_mask = (
        ~np.isnan(acc_x)
        & ~np.isnan(acc_y)
        & ~np.isnan(acc_z)
        & ~np.isinf(acc_x)
        & ~np.isinf(acc_y)
        & ~np.isinf(acc_z)
    )

    acc_x = acc_x[valid_mask]
    acc_y = acc_y[valid_mask]
    acc_z = acc_z[valid_mask]

    if len(acc_x) == 0:
        raise ValueError(
            f"No valid acceleration samples after cleaning: {window_file_path}"
        )

    time_features = {
        "RMS_X": calculate_rms(acc_x),
        "RMS_Y": calculate_rms(acc_y),
        "RMS_Z": calculate_rms(acc_z),
    }

    return time_features, warnings


# =========================
# Frequency-domain feature extraction
# =========================

def extract_features_from_fft_file(file_path: Path) -> tuple[dict, dict]:
    """
    Extract frequency-domain features from one FFT CSV file.

    This function extracts:
        dominant frequency
        magnitude statistics
        total energy
        spectral centroid
        spectral bandwidth
        band power values
    """
    df = pd.read_csv(file_path)

    freq_col = find_column(
        df,
        candidates=[
            "frequency_hz",
            "freq_hz",
            "frequency",
            "freq",
            "hz",
        ]
    )

    mag_col = find_column(
        df,
        candidates=[
            "magnitude",
            "mag",
            "fft_magnitude",
            "amplitude",
            "spectrum_magnitude",
        ]
    )

    freq = pd.to_numeric(df[freq_col], errors="coerce").to_numpy(dtype=float)
    mag = pd.to_numeric(df[mag_col], errors="coerce").to_numpy(dtype=float)

    label = infer_label_from_path(file_path)

    warnings = validate_fft_vector(freq, mag)

    valid_mask = (
        ~np.isnan(freq)
        & ~np.isnan(mag)
        & ~np.isinf(freq)
        & ~np.isinf(mag)
    )

    freq = freq[valid_mask]
    mag = mag[valid_mask]

    if len(freq) == 0 or len(mag) == 0:
        raise ValueError(f"No valid FFT data after cleaning NaN/Inf: {file_path}")

    dominant_idx = int(np.argmax(mag))
    dominant_frequency = float(freq[dominant_idx])

    max_magnitude = float(np.max(mag))
    mean_magnitude = float(np.mean(mag))
    std_magnitude = float(np.std(mag))
    total_energy = float(np.sum(mag ** 2))

    spectral_centroid = safe_weighted_centroid(freq, mag)
    spectral_bandwidth = safe_spectral_bandwidth(freq, mag, spectral_centroid)

    band_power_low = calculate_band_energy(
        freq,
        mag,
        low=LOW_BAND[0],
        high=LOW_BAND[1],
        include_high=False
    )

    band_power_mid = calculate_band_energy(
        freq,
        mag,
        low=MID_BAND[0],
        high=MID_BAND[1],
        include_high=False
    )

    band_power_high = calculate_band_energy(
        freq,
        mag,
        low=HIGH_BAND[0],
        high=HIGH_BAND[1],
        include_high=True
    )

    source_fft_file = safe_relative_path(file_path)

    fft_features = {
        "source_fft_file": source_fft_file,
        "dominant_frequency": dominant_frequency,
        "max_magnitude": max_magnitude,
        "mean_magnitude": mean_magnitude,
        "std_magnitude": std_magnitude,
        "total_energy": total_energy,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "BandPower_Z_Low": band_power_low,
        "BandPower_Z_Mid": band_power_mid,
        "BandPower_Z_High": band_power_high,
        "label": label,
    }

    report_row = {
        "source_fft_file": source_fft_file,
        "label": label,
        "status": "WARNING" if warnings else "OK",
        "n_bins": len(freq),
        "freq_min": float(np.min(freq)),
        "freq_max": float(np.max(freq)),
        "warnings": " | ".join(warnings),
    }

    return fft_features, report_row


# =========================
# Hybrid feature extraction
# =========================

def extract_hybrid_features_from_fft_file(
    fft_file_path: Path,
    window_input_dir: Path
) -> tuple[dict, dict]:
    """
    Extract one complete feature row from one FFT file and its matching window file.

    The final feature row combines:
        time-domain RMS features
        frequency-domain FFT features
    """
    fft_features, report_row = extract_features_from_fft_file(fft_file_path)

    window_file_path = map_fft_file_to_window_file(
        fft_file_path=fft_file_path,
        window_input_dir=window_input_dir
    )

    time_features, time_warnings = extract_time_domain_features_from_window_file(
        window_file_path=window_file_path
    )

    source_window_file = safe_relative_path(window_file_path)

    feature_row = {
        "source_window_file": source_window_file,
        "source_fft_file": fft_features["source_fft_file"],

        "RMS_X": time_features["RMS_X"],
        "RMS_Y": time_features["RMS_Y"],
        "RMS_Z": time_features["RMS_Z"],

        "BandPower_Z_Low": fft_features["BandPower_Z_Low"],
        "BandPower_Z_Mid": fft_features["BandPower_Z_Mid"],
        "BandPower_Z_High": fft_features["BandPower_Z_High"],

        "dominant_frequency": fft_features["dominant_frequency"],
        "max_magnitude": fft_features["max_magnitude"],
        "mean_magnitude": fft_features["mean_magnitude"],
        "std_magnitude": fft_features["std_magnitude"],
        "total_energy": fft_features["total_energy"],
        "spectral_centroid": fft_features["spectral_centroid"],
        "spectral_bandwidth": fft_features["spectral_bandwidth"],

        "label": fft_features["label"],
    }

    if time_warnings:
        existing_warnings = report_row.get("warnings", "")

        combined_warnings = " | ".join(
            warning for warning in [existing_warnings, *time_warnings]
            if warning
        )

        report_row["warnings"] = combined_warnings
        report_row["status"] = "WARNING"

    report_row["source_window_file"] = source_window_file

    return feature_row, report_row


# =========================
# File collection
# =========================

def collect_fft_files(input_dir: Path) -> list[Path]:
    """
    Collect all FFT CSV files from the input directory recursively.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"FFT input directory not found: {input_dir}")

    files = sorted(input_dir.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No FFT CSV files found under: {input_dir}")

    return files


# =========================
# Main pipeline
# =========================

def run_feature_extraction(
    fft_input_dir: Path,
    window_input_dir: Path,
    feature_output_dir: Path,
    auto_clean: bool = True
) -> None:
    """
    Run the complete hybrid feature extraction pipeline.

    Inputs:
        FFT CSV files
        matching window CSV files

    Outputs:
        features_dataset.csv
        feature_extraction_report.csv
    """
    feature_dataset_path = feature_output_dir / "features_dataset.csv"
    feature_report_path = feature_output_dir / "feature_extraction_report.csv"

    prepare_output_dir(feature_output_dir, auto_clean=auto_clean)

    fft_files = collect_fft_files(fft_input_dir)

    feature_rows = []
    report_rows = []
    failed_rows = []

    for fft_file_path in fft_files:
        try:
            feature_row, report_row = extract_hybrid_features_from_fft_file(
                fft_file_path=fft_file_path,
                window_input_dir=window_input_dir
            )

            feature_rows.append(feature_row)
            report_rows.append(report_row)

        except Exception as exc:
            failed_rows.append({
                "source_fft_file": safe_relative_path(fft_file_path),
                "source_window_file": None,
                "label": "unknown",
                "status": "FAILED",
                "n_bins": None,
                "freq_min": None,
                "freq_max": None,
                "warnings": str(exc),
            })

    features_df = pd.DataFrame(feature_rows)
    report_df = pd.DataFrame(report_rows + failed_rows)

    features_df.to_csv(feature_dataset_path, index=False)
    report_df.to_csv(feature_report_path, index=False)

    print("=" * 70)
    print("Feature Extraction Completed")
    print("=" * 70)
    print(f"FFT input directory     : {fft_input_dir}")
    print(f"Window input directory  : {window_input_dir}")
    print(f"Feature output dataset  : {feature_dataset_path}")
    print(f"Feature report          : {feature_report_path}")
    print(f"Total FFT files found   : {len(fft_files)}")
    print(f"Successful rows         : {len(features_df)}")
    print(f"Failed rows             : {len(failed_rows)}")

    if not features_df.empty:
        print("\nRows per class:")
        print(features_df["label"].value_counts())

        print("\nMissing values per column:")
        print(features_df.isna().sum())

        print("\nFeature columns:")
        print(list(features_df.columns))

    if failed_rows:
        print("\nWARNING: Some files failed during feature extraction.")
        print(f"Check report for details: {feature_report_path}")

    print("=" * 70)


def main() -> None:
    """
    Entry point for the feature extractor.
    """
    args = parse_args()

    paths = get_paths(args.mode)

    fft_input_dir = paths["fft_windows"]
    window_input_dir = paths["windows"]
    feature_output_dir = paths["features"]

    auto_clean = not args.no_clean

    print("=" * 70)
    print(f"Feature Extractor - {args.mode.upper()} MODE")
    print("=" * 70)
    print(f"Input FFT folder    : {fft_input_dir}")
    print(f"Input window folder : {window_input_dir}")
    print(f"Output folder       : {feature_output_dir}")
    print(f"Auto clean          : {auto_clean}")
    print("=" * 70)

    run_feature_extraction(
        fft_input_dir=fft_input_dir,
        window_input_dir=window_input_dir,
        feature_output_dir=feature_output_dir,
        auto_clean=auto_clean
    )


if __name__ == "__main__":
    main()