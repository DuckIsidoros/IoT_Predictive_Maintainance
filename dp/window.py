import os
import numpy as np
import pandas as pd


def split_and_save_windows_csv(file_paths, properties, output_folder="window"):
    """Loads datasets, extracts overlapping windows allowing up to 11ms gaps,
    and saves them as flattened CSV files in the designated output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: '{output_folder}/'")

    window_size = properties["WINDOW_SIZE"]
    step_size = properties["STEP_SIZE"]
    max_windows = properties["MAX_WINDOWS_PER_CLASS"]
    MAX_ALLOWED_GAP_MS = 11.0

    feature_cols = ["accX", "accY", "accZ"]

    for label, path in file_paths.items():
        if not os.path.exists(path):
            print(f"Skipping missing file for class '{label}': {path}")
            continue

        print(f"\nProcessing class '{label}'...")
        df = pd.read_csv(path)
        print(f" -> Raw data contains {len(df)} rows.")

        # --- FIX MIXED TYPES ERROR HERE ---
        # Convert timestamp to numeric; force broken strings/text to NaN
        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
        # Drop rows where the timestamp is missing/corrupted
        df = df.dropna(subset=["timestamp_ms"])

        # 1. Sort chronologically safely
        df = df.sort_values("timestamp_ms").reset_index(drop=True)
        df["dt"] = df["timestamp_ms"].diff()

        # 2. Process data stream handling gaps > 11ms
        adjusted_rows = []
        raw_values = df[["timestamp_ms"] + feature_cols].values

        # Safety check: ensure dataframe isn't empty after dropping NaNs
        if len(raw_values) == 0:
            print(f" -> Warning: No valid numerical data left for '{label}'.")
            continue

        # Insert the first row setup
        adjusted_rows.append(raw_values[0, 1:])

        for i in range(1, len(raw_values)):
            dt = df["dt"].iloc[i]

            # If there's a major connection drop, fill it to hold window continuity
            if dt > MAX_ALLOWED_GAP_MS:
                sample_period_ms = 1000.0 / properties["SAMPLING_RATE_HZ"]
                num_missing_steps = int(dt // sample_period_ms)
                last_known_features = raw_values[i - 1, 1:]
                for _ in range(num_missing_steps):
                    adjusted_rows.append(last_known_features)

            adjusted_rows.append(raw_values[i, 1:])

        feature_data = np.array(adjusted_rows)
        total_rows = len(feature_data)

        # 3. Slide windows down the data stream
        window_rows = []
        start_idx = 0
        window_id = 0

        while start_idx + window_size <= total_rows:
            end_idx = start_idx + window_size
            window = feature_data[start_idx:end_idx]  # Shape: (WINDOW_SIZE, 3)

            # Flatten the window into a 1D array
            flattened_window = window.flatten()

            # Create a dictionary for the CSV row metadata
            row_dict = {"window_id": window_id, "label": label}

            # Map the features to explicit column names
            for idx, val in enumerate(flattened_window):
                axis = ["accX", "accY", "accZ"][idx % 3]
                timestep = idx // 3
                row_dict[f"{axis}_{timestep}"] = val

            window_rows.append(row_dict)
            window_id += 1

            if len(window_rows) >= max_windows:
                break

            start_idx += step_size

        # 4. Save flattened windows to CSV
        if window_rows:
            output_df = pd.DataFrame(window_rows)
            save_path = os.path.join(output_folder, f"{label}.csv")

            # Save to file
            output_df.to_csv(save_path, index=False)
            print(
                f" -> Generated {len(output_df)} windows flattened into {output_df.shape[1]} columns."
            )
            print(f" -> Successfully saved to: {save_path}")
        else:
            print(f" -> Warning: No windows generated for '{label}'.")


# =====================================================================
# Configuration & Execution Block
# =====================================================================
if __name__ == "__main__":
    DATA_FOLDER = "raw"
    OUTPUT_FOLDER = "window"

    FILE_PATHS = {
        "healthy": os.path.join(DATA_FOLDER, "arduino_serial_data_healthy.csv"),
        "imbalanced": os.path.join(
            DATA_FOLDER, "arduino_serial_data_imbalanced.csv"
        ),
        "obstruction": os.path.join(
            DATA_FOLDER, "arduino_serial_data_obstruction.csv"
        ),
    }

    # Fixed calculation logic for step_size based on your comment:
    # 640 * (1 - 0.5) is actually 320, not 128. Kept your formula intact.
    PROPERTIES = {
        "SAMPLING_RATE_HZ": 500,
        "WINDOW_SIZE": 640,
        "STEP_SIZE": int(640 * (1 - 0.5)),  # Evaluates to 320 (50% overlap)
        "MAX_WINDOWS_PER_CLASS": 500,
    }

    split_and_save_windows_csv(FILE_PATHS, PROPERTIES, OUTPUT_FOLDER)