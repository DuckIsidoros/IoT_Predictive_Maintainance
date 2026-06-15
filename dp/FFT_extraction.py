import os
import numpy as np
import pandas as pd


def extract_deployment_features(
    input_folder="window", output_file="features_for_deployment.csv"
):
    """Loads flattened CSV windows, applies an emulation of the deployment's

    IIR filter (gravity removal), extracts matching RMS and Band Power features,
    and exports a final dataset suitable for training the Arduino model parameters.
    """
    # Define operational bands matching your .ino parameters
    # bin_width = 200.0 / 256 = 0.78125 Hz
    FS = 500.0
    WINDOW_SIZE = 640
    freq_bins = np.fft.rfftfreq(WINDOW_SIZE, d=1 / FS)

    # Output storage
    processed_dataset = []

    # Map target classes to look for
    classes = ["healthy", "imbalanced", "obstruction"]

    print("--- Starting Deployment-Compliant Feature Extraction ---")

    for label in classes:
        csv_path = os.path.join(input_folder, f"{label}.csv")

        if not os.path.exists(csv_path):
            print(f"Skipping: {csv_path} not found.")
            continue

        print(f"Extracting features from '{label}.csv'...")
        df_windows = pd.read_csv(csv_path)

        # Drop structural identifiers to isolate raw feature columns
        feature_matrix = df_windows.drop(
            columns=["window_id", "label"], errors="ignore"
        ).values

        for i in range(len(feature_matrix)):
            # 1. Reshape the flat CSV row back into its time-series shape (256, 3)
            window = feature_matrix[i].reshape(WINDOW_SIZE, 3)
            raw_x = window[:, 0]
            raw_y = window[:, 1]
            raw_z = window[:, 2]

            # 2. EMULATE ON-CHIP IIR HIGH-PASS FILTER (Gravity Removal)
            # Match: fx = HPF_ALPHA * (p_fx + crx - p_rx)
            HPF_ALPHA = 0.9936
            filt_x = np.zeros(WINDOW_SIZE)
            filt_y = np.zeros(WINDOW_SIZE)
            filt_z = np.zeros(WINDOW_SIZE)

            for t in range(1, WINDOW_SIZE):
                filt_x[t] = HPF_ALPHA * (
                    filt_x[t - 1] + raw_x[t] - raw_x[t - 1]
                )
                filt_y[t] = HPF_ALPHA * (
                    filt_y[t - 1] + raw_y[t] - raw_y[t - 1]
                )
                filt_z[t] = HPF_ALPHA * (
                    filt_z[t - 1] + raw_z[t] - raw_z[t - 1]
                )

            # 3. COMPUTE TIME DOMAIN METRICS (RMS)
            # Match: sqrt(sSqX / (float)WINDOW_SIZE)
            rms_x = np.sqrt(np.mean(filt_x**2))
            rms_y = np.sqrt(np.mean(filt_y**2))
            rms_z = np.sqrt(np.mean(filt_z**2))

            # 4. COMPUTE FREQUENCY DOMAIN METRICS (Band Power of Z)
            # Emulating the exact Hanning Window + RFFT magnitude routine of arduinoFFT
            hann_window = np.hanning(WINDOW_SIZE)
            windowed_z = filt_z * hann_window

            # Compute real FFT magnitudes matching complexToMagnitude()
            fft_complex = np.fft.rfft(windowed_z, n=WINDOW_SIZE)
            magnitudes = np.abs(fft_complex)

            # Match: norm_mag = fft_vReal[i] / (float)WINDOW_SIZE; magSq = norm_mag * norm_mag
            normalized_magnitudes = magnitudes / WINDOW_SIZE
            magnitude_squared = normalized_magnitudes * normalized_magnitudes

            # Accumulate power values into your explicit bounds
            bp_low = 0.0
            bp_mid = 0.0
            bp_high = 0.0

            for idx, freq in enumerate(freq_bins):
                # Safeguard limits matching loops: 0-20Hz, 20-60Hz, 60-100Hz
                if 0.0 <= freq < 20.0:
                    bp_low += magnitude_squared[idx]
                elif 20.0 <= freq < 60.0:
                    bp_mid += magnitude_squared[idx]
                elif 60.0 <= freq <= 100.0:
                    bp_high += magnitude_squared[idx]

            # Append the calculated row matched to feature order:
            # {rx, ry, rz, bLow, bMid, bHigh}
            processed_dataset.append(
                {
                    "RMS_X": rms_x,
                    "RMS_Y": rms_y,
                    "RMS_Z": rms_z,
                    "Band_Power_Z_Low": bp_low,
                    "Band_Power_Z_Mid": bp_mid,
                    "Band_Power_Z_High": bp_high,
                    "label": label,
                }
            )

    # Save to final consolidated master CSV
    if processed_dataset:
        final_df = pd.DataFrame(processed_dataset)
        final_df.to_csv(output_file, index=False)
        print(f"\n[SUCCESS] Extracted features matching Arduino architecture!")
        print(f" -> Final Output Data Shape: {final_df.shape}")
        print(f" -> File saved to: {output_file}")
    else:
        print("\n[ERROR] No features were extracted. Verify window files.")


if __name__ == "__main__":
    # Assumes your flat windows are stored in the "window" directory from your previous step
    extract_deployment_features(
        input_folder="window", output_file="features_for_deployment.csv"
    )