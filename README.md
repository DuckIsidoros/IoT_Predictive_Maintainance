# IoT Predictive Maintenance with Edge AI for 12V DC Fan

## 1. Project Overview

This project builds a preprocessing pipeline for an IoT Predictive Maintenance system using Edge AI. The target device is a 12V DC fan. The system collects or simulates 3-axis accelerometer data and classifies the fan condition into three classes:

1. `healthy`
2. `imbalance`
3. `obstruction`

At the current stage, the project does not use real MPU6050 sensor data yet. Instead, mock sensor data is used to validate the complete preprocessing pipeline before moving to real sensor collection and ESP32 real-time inference.

The current scope focuses on:

```text
raw data
→ validation
→ segmentation
→ windowing
→ FFT
→ feature extraction
→ ready for baseline ML
```

Model training, model evaluation, ESP32 deployment, and real-time inference are later stages.

---

## 2. Role and Current Scope

Current role: **Preprocessing Engineer**

Responsibilities completed in this stage:

* Generate mock accelerometer data for three fan conditions.
* Generate corrupted data for validator testing.
* Validate raw sensor data quality.
* Segment valid data.
* Apply windowing with fixed window size and overlap.
* Apply FFT on each window.
* Extract frequency-domain features.
* Prepare final feature dataset for baseline machine learning.

The current preprocessing output is ready to be handed off to the ML training stage.

---

## 3. Technical Parameters

| Parameter                  |          Value |
| -------------------------- | -------------: |
| Sampling rate              |         200 Hz |
| Expected sampling interval |           5 ms |
| Window size                |    256 samples |
| Window duration            |   1.28 seconds |
| Overlap                    |            50% |
| Step size / stride         |    128 samples |
| Step duration              |   0.64 seconds |
| FFT bins                   |            129 |
| Nyquist frequency          |         100 Hz |
| Frequency resolution       | 0.78125 Hz/bin |
| Windows per class          |            500 |
| Total windows              |           1500 |

---

## 4. Class Definition

| Class ID | Label         | Description                                          |
| -------: | ------------- | ---------------------------------------------------- |
|        1 | `healthy`     | Normal fan vibration with low noise                  |
|        2 | `imbalance`   | Stronger vibration with harmonic components          |
|        3 | `obstruction` | Abnormal vibration, higher noise, and spike behavior |

---

## 5. Project Structure

```text
project_root/
│
├── .venv/
│   └── Python virtual environment
│
├── mock_raw_sensor_stream/
│   ├── healthy/
│   ├── imbalance/
│   └── obstruction/
│
├── corrupted_data/
│   └── Corrupted files for validator testing
│
├── qc_reports/
│   └── Data quality check reports
│
├── segments/
│   └── Segmented valid data
│
├── windows/
│   └── Windowed sensor data
│
├── fft_windows/
│   └── FFT output files
│
├── features/
|   |__ plots
    |   └── plot features
│   ├── features_dataset.csv
│   └── feature_extraction_report.csv
│
├── src/
│   ├── generate_data.py
│   ├── generate_faults.py
│   ├── validator.py
│   ├── qc_analyzer.py
│   ├── segmenter.py
│   ├── windowing.py
│   ├── fft_processor.py
│   └── feature_extractor.py
│
├── segment_report.csv
├── window_report.csv
├── fft_report.csv
├── pipeline.md
└── README.md
```

---

## 6. Mock Raw Data

Mock raw data is generated for three classes:

* `healthy`
* `imbalance`
* `obstruction`

Each class contains:

| Item              |      Value |
| ----------------- | ---------: |
| Number of files   |         20 |
| Duration per file | 30 seconds |
| Sampling rate     |     200 Hz |
| Samples per file  |       6000 |

Each raw data file contains the following columns:

```text
timestamp_ms
accX
accY
accZ
label
```

Mock signal behavior:

| Class         | Simulated behavior                                              |
| ------------- | --------------------------------------------------------------- |
| `healthy`     | Light vibration, low noise, frequency around 52 Hz              |
| `imbalance`   | Stronger vibration, harmonic components, around 48 Hz and 96 Hz |
| `obstruction` | Abnormal signal, high noise, spikes, around 30 Hz               |

---

## 7. Corrupted Data

Corrupted data is generated to test the validator.

Current corrupted cases:

```text
missing_chunk
timestamp_jump
duplicate_timestamp
reverse_timestamp
high_jitter
```

These cases are used to verify whether the validator can detect:

* Timestamp discontinuity
* Sampling gap
* High jitter
* Duplicate timestamp
* Reverse timestamp

Recommended future corrupted cases:

```text
missing_column
wrong_dtype
nan_values
invalid_label
empty_file
extreme_sensor_value
```

---

## 8. Validator

The validator checks raw sensor files before they are passed to later preprocessing stages.

Current validation checks:

* Required columns
* Missing values / NaN
* Timestamp monotonicity
* Inconsistent sampling interval
* Large timestamp gap
* Effective sampling rate deviation
* High sampling jitter

Recommended future improvements:

* Check numeric dtype for `accX`, `accY`, `accZ`
* Check whether `timestamp_ms` can be converted to numeric
* Check whether `label` is valid
* Check extreme sensor values
* Avoid crashing when files contain invalid dtype

---

## 9. Windowing

Windowing converts time-series data into fixed-size windows.

Current configuration:

```text
EXPECTED_FS = 200
WINDOW_SIZE = 256
OVERLAP_RATIO = 0.5
STEP_SIZE = 128
MAX_WINDOWS_PER_CLASS = 500
```

Current result:

| Class         |  Windows |
| ------------- | -------: |
| `healthy`     |      500 |
| `imbalance`   |      500 |
| `obstruction` |      500 |
| **Total**     | **1500** |

Each window contains:

```text
256 samples
1.28 seconds
50% overlap
128-sample stride
```

Important note:

Old output folders should be cleaned before regenerating windows to avoid mixing outdated files with new files.

---

## 10. FFT Processing

FFT is applied to each window after windowing.

Current FFT configuration:

```text
EXPECTED_FS = 200
EXPECTED_WINDOW_SIZE = 256
EXPECTED_FFT_BINS = 129
NYQUIST_FREQUENCY_HZ = 100
FREQ_RESOLUTION_HZ = 0.78125
```

FFT processing method:

1. Calculate vector magnitude:

```text
sqrt(accX^2 + accY^2 + accZ^2)
```

2. Remove DC offset.
3. Apply Hann window.
4. Apply real FFT using `np.fft.rfft()`.

Current FFT result:

| Class         | FFT files |
| ------------- | --------: |
| `healthy`     |       500 |
| `imbalance`   |       500 |
| `obstruction` |       500 |
| **Total**     |  **1500** |

Each FFT output has:

```text
129 bins
0–100 Hz frequency range
0.78125 Hz/bin resolution
```

---

## 11. Feature Extraction

Feature extraction converts each FFT file into one feature row.

Input:

```text
fft_windows/
```

Output:

```text
features/features_dataset.csv
features/feature_extraction_report.csv
```

Expected final dataset:

| Class         |     Rows |
| ------------- | -------: |
| `healthy`     |      500 |
| `imbalance`   |      500 |
| `obstruction` |      500 |
| **Total**     | **1500** |

Extracted features:

```text
source_file
dominant_frequency
max_magnitude
mean_magnitude
std_magnitude
total_energy
spectral_centroid
spectral_bandwidth
low_band_energy
mid_band_energy
high_band_energy
label
```

Frequency bands:

| Band      | Frequency range |
| --------- | --------------- |
| Low band  | 0–20 Hz         |
| Mid band  | 20–60 Hz        |
| High band | 60–100 Hz       |

The final dataset is ready for baseline machine learning.

---

## 12. Current Pipeline Status

| Stage                     | Status             |
| ------------------------- | ------------------ |
| Mock raw data generation  | DONE               |
| Corrupted data generation | DONE               |
| Validator                 | DONE basic version |
| QC analyzer               | DONE               |
| Segmenter                 | DONE               |
| Windowing                 | DONE               |
| FFT processing            | DONE               |
| Feature extraction        | DONE               |
| Baseline ML               | NEXT STAGE         |
| Real MPU6050 data         | FUTURE             |
| ESP32 real-time inference | FUTURE             |

---

## 13. How to Run the Pipeline

Recommended execution order:

```bash
python src/generate_data.py
python src/generate_faults.py
python src/validator.py
python src/qc_analyzer.py
python src/segmenter.py
python src/windowing.py
python src/fft_processor.py
python src/feature_extractor.py
```

Before rerunning stages that generate output artifacts, clean old outputs if needed:

```text
windows/
fft_windows/
features/
window_report.csv
fft_report.csv
```

This prevents old files from being mixed with newly generated files.

---

## 14. Feature Dataset Quality Check

After running feature extraction, verify the dataset:

```bash
python -c "import pandas as pd; df=pd.read_csv('features/features_dataset.csv'); print(df.shape); print(df['label'].value_counts()); print(df.isna().sum())"
```

Expected result:

```text
1500 rows
healthy       500
imbalance     500
obstruction   500
No missing values
```

Additional checks:

* `dominant_frequency` should be within 0–100 Hz.
* `total_energy` should not be negative.
* `low_band_energy`, `mid_band_energy`, and `high_band_energy` should not be negative.
* `feature_extraction_report.csv` should not contain failed files.

---

## 15. Definition of Done for Preprocessing Stage

The preprocessing stage is considered complete when:

```text
[ ] features/features_dataset.csv exists
[ ] Dataset has 1500 rows
[ ] healthy has 500 rows
[ ] imbalance has 500 rows
[ ] obstruction has 500 rows
[ ] No missing values
[ ] No inf or -inf values
[ ] FFT output has 129 bins per file
[ ] Frequency range is 0–100 Hz
[ ] Feature extraction report has no FAILED files
[ ] Dataset is ready for Baseline ML
```

Current handoff statement:

```text
Preprocessing stage is completed and ready for Baseline ML handoff.
```

---

## 16. Notes for ML Engineer

The ML training stage should use:

```text
Input dataset:
features/features_dataset.csv
```

Recommended feature columns:

```text
dominant_frequency
max_magnitude
mean_magnitude
std_magnitude
total_energy
spectral_centroid
spectral_bandwidth
low_band_energy
mid_band_energy
high_band_energy
```

Target column:

```text
label
```

Do not use this column as a model feature:

```text
source_file
```

Recommended baseline models:

* Random Forest
* Logistic Regression
* SVM
* KNN

Recommended metrics:

* Accuracy
* Confusion matrix
* Precision
* Recall
* F1-score

Important warning:

Random train/test split may cause data leakage because multiple windows can come from the same original raw file. For stricter evaluation, consider splitting by source file instead of splitting randomly by window.

---

## 17. Future Improvements

Recommended next improvements:

1. Add a shared `config.py` or `config.yaml` to avoid hard-coded parameters.
2. Add `run_pipeline.py` or `Makefile` to run the full pipeline automatically.
3. Add automatic cleaning for generated output folders.
4. Add unit tests or sanity checks:

   * Window size must be 256.
   * FFT bins must be 129.
   * Each class must have 500 windows.
   * Final feature dataset must have 1500 rows.
5. Improve validator:

   * Wrong dtype
   * Missing columns
   * Invalid labels
   * NaN values
   * Empty files
   * Sensor outliers
6. Extract FFT features for each axis separately:

   * `accX`
   * `accY`
   * `accZ`

Current FFT uses vector magnitude, which is simple and suitable for baseline, but it may lose directional vibration information.

---

## 18. Summary

This project has completed the preprocessing pipeline for mock IoT accelerometer data. The generated feature dataset contains balanced samples for three fan conditions and is ready for the next stage: baseline machine learning.

Current output:

```text
features/features_dataset.csv
```

Current status:

```text
Preprocessing DONE.
Ready for Baseline ML.
```
