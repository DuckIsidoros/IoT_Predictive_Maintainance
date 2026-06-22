import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(BASE_DIR, "mock_raw_sensor_stream")
CORRUPTED_DIR = os.path.join(BASE_DIR, "corrupted_data")

os.makedirs(CORRUPTED_DIR, exist_ok=True)

CLASSES = ["healthy", "imbalance", "obstruction"]


def inject_missing_chunk(df):
    corrupted = df.copy()

    start_idx = np.random.randint(500, len(corrupted) - 700)

    corrupted = corrupted.drop(
        corrupted.index[start_idx:start_idx + 200]
    ).reset_index(drop=True)

    return corrupted


def inject_timestamp_jump(df):
    corrupted = df.copy()

    start_idx = np.random.randint(500, len(corrupted) - 500)

    corrupted.loc[start_idx:, "timestamp_ms"] += 200

    return corrupted


def inject_duplicate_timestamp(df):
    corrupted = df.copy()

    start_idx = np.random.randint(500, len(corrupted) - 500)

    duplicated_chunk = corrupted.iloc[start_idx:start_idx + 50].copy()

    corrupted = pd.concat(
        [
            corrupted.iloc[:start_idx + 50],
            duplicated_chunk,
            corrupted.iloc[start_idx + 50:]
        ],
        ignore_index=True
    )

    return corrupted


def inject_reverse_timestamp(df):
    corrupted = df.copy()

    start_idx = np.random.randint(500, len(corrupted) - 500)

    corrupted.loc[
        start_idx:start_idx + 30,
        "timestamp_ms"
    ] = corrupted.loc[
        start_idx:start_idx + 30,
        "timestamp_ms"
    ].values[::-1]

    return corrupted


def inject_high_jitter(df):
    corrupted = df.copy()

    jitter = np.random.normal(0, 8.0, len(corrupted))

    corrupted["timestamp_ms"] = corrupted["timestamp_ms"] + jitter

    return corrupted


FAULT_INJECTORS = {
    "missing_chunk": inject_missing_chunk,
    "timestamp_jump": inject_timestamp_jump,
    "duplicate_timestamp": inject_duplicate_timestamp,
    "reverse_timestamp": inject_reverse_timestamp,
    "high_jitter": inject_high_jitter,
}


def main():
    print("Generating corrupted data...")

    for class_name in CLASSES:
        source_file = os.path.join(
            SOURCE_DIR,
            class_name,
            f"{class_name}_000.csv"
        )

        clean_df = pd.read_csv(source_file)

        for fault_name, fault_func in FAULT_INJECTORS.items():
            corrupted_df = fault_func(clean_df)

            output_file = os.path.join(
                CORRUPTED_DIR,
                f"err_{class_name}_{fault_name}.csv"
            )

            corrupted_df.to_csv(output_file, index=False)

            print(f"[OK] {output_file}")

    print("Done.")


if __name__ == "__main__":
    main()