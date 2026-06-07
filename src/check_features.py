from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = PROJECT_ROOT / "features" / "features_dataset.csv"

EXPECTED_ROWS = 1500
EXPECTED_CLASS_COUNTS = {
    "healthy": 500,
    "imbalance": 500,
    "obstruction": 500,
}

FEATURE_COLUMNS = [
    "dominant_frequency",
    "max_magnitude",
    "mean_magnitude",
    "std_magnitude",
    "total_energy",
    "spectral_centroid",
    "spectral_bandwidth",
    "low_band_energy",
    "mid_band_energy",
    "high_band_energy",
]


def main():
    df = pd.read_csv(FEATURE_PATH)

    print("=" * 70)
    print("Feature Dataset Check")
    print("=" * 70)

    print(f"Dataset path: {FEATURE_PATH}")
    print(f"Shape: {df.shape}")

    # 1. Row count
    if len(df) == EXPECTED_ROWS:
        print(f"[PASS] Row count = {EXPECTED_ROWS}")
    else:
        print(f"[FAIL] Expected {EXPECTED_ROWS} rows, got {len(df)}")

    # 2. Class count
    class_counts = df["label"].value_counts().to_dict()
    print("\nClass counts:")
    print(df["label"].value_counts())

    if class_counts == EXPECTED_CLASS_COUNTS:
        print("[PASS] Class distribution is correct")
    else:
        print("[FAIL] Class distribution mismatch")
        print(f"Expected: {EXPECTED_CLASS_COUNTS}")
        print(f"Actual  : {class_counts}")

    # 3. Missing values
    missing = df.isna().sum()
    total_missing = missing.sum()

    print("\nMissing values:")
    print(missing)

    if total_missing == 0:
        print("[PASS] No missing values")
    else:
        print(f"[FAIL] Total missing values: {total_missing}")

    # 4. Numeric feature sanity
    print("\nFeature sanity check:")

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            print(f"[FAIL] Missing feature column: {col}")
            continue

        if not np.isfinite(df[col]).all():
            print(f"[FAIL] Non-finite values found in {col}")
            continue

        if col != "dominant_frequency" and (df[col] < 0).any():
            print(f"[WARN] Negative values found in {col}")
        else:
            print(f"[PASS] {col}")

    # 5. Dominant frequency range
    if "dominant_frequency" in df.columns:
        invalid_freq = df[
            (df["dominant_frequency"] < 0) |
            (df["dominant_frequency"] > 100)
        ]

        if invalid_freq.empty:
            print("[PASS] dominant_frequency within 0–100 Hz")
        else:
            print(f"[FAIL] Invalid dominant_frequency rows: {len(invalid_freq)}")

    # 6. Summary by class
    print("\nMean feature values by class:")
    print(df.groupby("label")[FEATURE_COLUMNS].mean())

    print("=" * 70)


if __name__ == "__main__":
    main()