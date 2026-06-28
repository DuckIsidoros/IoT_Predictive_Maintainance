# IoT Predictive Maintenance for 12V DC Fan
Github Link: https://github.com/DuckIsidoros/IoT_Predictive_Maintainance.git
This project implements an end-to-end IoT predictive maintenance system for vibration-based fault diagnosis on a 12V DC fan. It uses an ESP32 with an MPU6050 accelerometer to collect vibration data, processes the signal through a Python preprocessing pipeline, extracts time-domain and frequency-domain features, and deploys lightweight machine learning models for edge inference.

The system supports three fan conditions:

- `healthy`
- `imbalanced`
- `obstruction`

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Core Configuration](#core-configuration)
- [Expected Dataset Format](#expected-dataset-format)
- [Real Data Processing Pipeline](#real-data-processing-pipeline)
- [Extracted Feature Set](#extracted-feature-set)
- [Installation](#installation)
- [How to Run the Real Data Pipeline](#how-to-run-the-real-data-pipeline)
- [How to Collect Real Sensor Data](#how-to-collect-real-sensor-data)
- [Model Training and Experimentation](#model-training-and-experimentation)
- [Edge Deployment](#edge-deployment)
- [Dashboard and Live Monitoring](#dashboard-and-live-monitoring)
- [Generated Outputs](#generated-outputs)
- [Mock Data and Fault Injection](#mock-data-and-fault-injection)
- [Known Notes and Limitations](#known-notes-and-limitations)
- [Team Contributions](#team-contributions)

---

## Project Overview

The goal of this project is to detect abnormal fan operating conditions from vibration signals. Instead of feeding raw sensor streams directly into a model, the system converts raw accelerometer readings into compact numerical features that are more suitable for lightweight machine learning models and embedded deployment.

The repository includes:

- ESP32/Arduino firmware for data collection
- Python preprocessing pipeline for real sensor streams
- Sliding-window segmentation
- FFT-based frequency analysis
- Hybrid feature extraction
- Logistic Regression deployment variant
- Small Dense Neural Network deployment variant
- MQTT + FastAPI backend for live telemetry
- Next.js dashboard for real-time monitoring
- Mock data generation and corrupted-data testing utilities
- Model training and experimentation notebook

---

## System Architecture

### Offline training / preprocessing flow

```text
ESP32 + MPU6050
      |
      v
Raw CSV sensor stream
      |
      v
Validation
      |
      v
Segmentation
      |
      v
Sliding windows
      |
      v
FFT processing
      |
      v
Feature extraction
      |
      v
features_dataset.csv
      |
      v
Model training / analysis / deployment
```

### Live edge inference flow

```text
ESP32 + MPU6050
      |
      v
On-device feature extraction
      |
      v
Edge ML inference
      |
      v
MQTT publish
      |
      v
FastAPI backend
      |
      v
WebSocket stream
      |
      v
Next.js dashboard
```

---

## Repository Structure

```text
IoT_Predictive_Maintainance/
├─ src/                            # Python preprocessing and feature pipeline
├─ Arduino_Code/                   # Data collection and model deployment sketches
├─ LR_IM/                          # Logistic Regression edge inference variant
├─ Small_Dense/                    # Small Dense Neural Network edge inference variant
├─ dashboard/                      # FastAPI WebSocket host and Next.js dashboard
├─ reactdashboard/                 # Static dashboard data snapshot variant
├─ dataset/                        # Collected CSV datasets
├─ real_raw_sensor_stream/         # Incoming / accepted / rejected real raw files
├─ real_segments/                  # Valid continuous stream segments
├─ real_windows/                   # Sliding-window CSV files
├─ real_fft_windows/               # FFT outputs per window
├─ real_features/                  # Final feature dataset and extraction report
├─ real_qc_reports/                # Validation and QC reports
├─ Machine_Learning_Code_IoT.ipynb # Model training / experimentation notebook
├─ serial_logger.py                # High-speed serial logger for ESP32 data collection
├─ requirements.txt                # Python dependencies for backend / runtime components
└─ Proposal (1).pdf                # Project proposal
```

The repository contains both source code and generated artifacts. The `real_*` folders are used by the real-data preprocessing pipeline and may contain intermediate or final outputs from previous runs.

---

## Core Configuration

The main preprocessing configuration is centralized in:

```text
src/config.py
```

Current core settings:

| Parameter | Value |
|---|---:|
| Sampling rate | `500 Hz` |
| Sample period | `2.0 ms` |
| Window size | `512 samples` |
| Overlap ratio | `0.5` |
| Window stride | `256 samples` |
| Nyquist frequency | `250 Hz` |
| FFT type | One-sided `rFFT` |
| Fault classes | `healthy`, `imbalanced`, `obstruction` |

---

## Expected Dataset Format

The real sensor CSV files should contain the following columns:

```text
timestamp_ms, accX_raw, accY_raw, accZ_raw, accZ_filt, label
```

Column meaning:

| Column | Description |
|---|---|
| `timestamp_ms` | Timestamp in milliseconds |
| `accX_raw` | Raw X-axis acceleration |
| `accY_raw` | Raw Y-axis acceleration |
| `accZ_raw` | Raw Z-axis acceleration |
| `accZ_filt` | Filtered Z-axis acceleration used for FFT and Z-frequency features |
| `label` | Class label: `healthy`, `imbalanced`, or `obstruction` |

The Python validator also handles label normalization through configured label aliases where applicable.

---

## Real Data Processing Pipeline

The real-data pipeline is implemented in the `src/` folder.

### 1. Validation

Implemented in:

```text
src/validator.py
```

Main responsibilities:

- Validate required CSV columns
- Detect empty files
- Remove or reject duplicated header rows
- Normalize labels
- Convert numeric columns
- Detect `NaN` / `Inf` values
- Detect duplicate or non-increasing timestamps
- Detect bad sampling intervals
- Detect large timestamp gaps
- Estimate effective sampling rate
- Measure sampling jitter
- Detect extreme sensor values
- Move real input files into `accepted` or `rejected`
- Generate validation reports in `real_qc_reports/`

### 2. Segmentation

Implemented in:

```text
src/segmenter.py
```

Main responsibilities:

- Read validated files from `real_raw_sensor_stream/accepted/`
- Detect timestamp discontinuities
- Split long streams into valid continuous segments
- Reject segments that are too short
- Save valid segments by class into `real_segments/`
- Generate `real_segment_report.csv`

### 3. Sliding Window Generation

Implemented in:

```text
src/windowing.py
```

Main responsibilities:

- Convert each valid segment into fixed-size windows
- Use `512` samples per window
- Use `50%` overlap between consecutive windows
- Check timestamp monotonicity and maximum gap threshold
- Reject files shorter than one full window
- Limit the number of windows per class using `MAX_WINDOWS_PER_CLASS`
- Save window CSV files into `real_windows/`
- Generate `real_window_report.csv`

### 4. FFT Processing

Implemented in:

```text
src/fft_processor.py
```

Main responsibilities:

- Read window CSV files from `real_windows/`
- Use `accZ_filt` as the Z-axis vibration signal for FFT
- Apply a Hann window before FFT
- Compute one-sided `rFFT`
- Convert the spectrum into frequency and magnitude values
- Validate FFT bin count and maximum frequency
- Save FFT CSV files into `real_fft_windows/`
- Generate `real_fft_report.csv`

### 5. Feature Extraction

Implemented in:

```text
src/feature_extraction.py
```

Main responsibilities:

- Combine time-domain features from window files
- Combine frequency-domain features from FFT files
- Extract RMS features from raw acceleration axes
- Extract Z-axis crest factor and kurtosis
- Extract band-power features from the Z-axis FFT spectrum
- Save the final feature dataset to `real_features/features_dataset.csv`
- Save the feature extraction report to `real_features/feature_extraction_report.csv`

### 6. Feature Sanity Check

Implemented in:

```text
src/check_features.py
```

Main responsibilities:

- Load the generated feature dataset
- Check row count and class distribution
- Check missing values
- Check finite numeric values
- Print per-class feature statistics

Note: this utility should be kept synchronized with the latest feature schema if feature names are changed.

---

## Extracted Feature Set

The final feature dataset is designed to be compact enough for lightweight machine learning models while still capturing useful vibration characteristics.

### Time-domain features

| Feature | Source signal | Purpose |
|---|---|---|
| `RMS_X` | `accX_raw` | Measures vibration intensity on the X-axis |
| `RMS_Y` | `accY_raw` | Measures vibration intensity on the Y-axis |
| `RMS_Z` | `accZ_raw` | Measures vibration intensity on the Z-axis |
| `CrestFactor_Z` | Z-axis signal | Captures peak-to-RMS behavior and impulsive vibration |
| `Kurtosis_Z` | Z-axis signal | Captures sharp spikes and non-Gaussian vibration patterns |

### Frequency-domain features

| Feature | Frequency band | Source signal | Purpose |
|---|---:|---|---|
| `Band_Power_Z_Low` | `50-80 Hz` | `accZ_filt` FFT | Captures low-frequency fault-related energy |
| `Band_Power_Z_Mid` | `110-140 Hz` | `accZ_filt` FFT | Captures mid-frequency vibration behavior |
| `Band_Power_Z_High` | `170-210 Hz` | `accZ_filt` FFT | Captures high-frequency fan vibration behavior |

Depending on the exact extractor version, the dataset may also include helper spectral values such as dominant frequency, spectral centroid, spectral bandwidth, or total energy. The final README should always match the actual column names in `real_features/features_dataset.csv`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DuckIsidoros/IoT_Predictive_Maintainance.git
cd IoT_Predictive_Maintainance
```

### 2. Create a Python virtual environment

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The preprocessing scripts also require common scientific Python packages. If they are not already listed in `requirements.txt`, install them manually:

```bash
pip install pandas numpy scipy
```

For serial data collection from ESP32, install:

```bash
pip install pyserial
```

---

## How to Run the Real Data Pipeline

Place raw CSV files into:

```text
real_raw_sensor_stream/incoming/
```

Then run the pipeline in order:

```bash
python src/validator.py --mode real --move-real-files --accept-warnings
python src/segmenter.py
python src/windowing.py
python src/fft_processor.py
python src/feature_extraction.py
python src/check_features.py
```

Expected main output:

```text
real_features/features_dataset.csv
```

This file is the final tabular feature dataset used for training, evaluation, or deployment alignment.

---

## How to Collect Real Sensor Data

The high-speed serial logger is implemented in:

```text
serial_logger.py
```

It connects to the ESP32 over serial and writes sensor packets into CSV files.

Default output schema:

```text
timestamp_ms, accX_raw, accY_raw, accZ_raw, accZ_filt, label
```

Example session files:

```text
arduino_serial_data_healthy.csv
arduino_serial_data_imbalanced.csv
arduino_serial_data_obstruction.csv
```

Before running the logger, verify:

- ESP32 is connected by USB
- Correct COM port is configured
- Baud rate matches the firmware setting
- The class label matches the current physical fan condition

---

## Model Training and Experimentation

Model training and experimentation are mainly handled in:

```text
Machine_Learning_Code_IoT.ipynb
```

The notebook is used for:

- Loading the extracted feature dataset
- Inspecting feature distributions
- Training machine learning models
- Comparing model behavior
- Preparing model parameters for edge deployment

The repository includes deployment variants for:

- Logistic Regression
- Small Dense Neural Network

---

## Edge Deployment

### Data collection firmware

Implemented in:

```text
Arduino_Code/Data_Collection/IIR_Collection/IIR_Collection.ino
```

Main capabilities:

- Read MPU6050 through I2C
- Sample at 500 Hz
- Apply high-pass IIR filtering on the Z-axis
- Print synchronized CSV rows over serial
- Include basic I2C recovery logic

### Logistic Regression deployment

Implemented across variants such as:

```text
Arduino_Code/model/Deployment_Logistic_Regression/Deployment_LR/Deployment_LR.ino
Arduino_Code/model/Quantitative_LR/Quantitative_LR.ino
LR_IM/LR_IM.ino
```

Main capabilities:

- Real-time MPU6050 sampling
- On-device feature extraction
- Logistic Regression inference
- Softmax prediction scores
- MQTT publishing
- Serial debugging for logits, features, latency, memory, and diagnosis

### Small Dense Neural Network deployment

Implemented across variants such as:

```text
Arduino_Code/model/IIR_deployment_SmallDenseNN/IIR_deployment_SmallDenseNN.ino
Small_Dense/Small_Dense.ino
```

Main capabilities:

- Lightweight neural network inference on ESP32
- Same feature family as the Python pipeline
- MQTT publishing of prediction results
- Runtime diagnostics and performance logging
- Embedded model parameters stored in files such as `model_parameters.h` and `inference.h`

---

## Dashboard and Live Monitoring

The dashboard stack is located mainly in:

```text
dashboard/
```

### FastAPI + MQTT host

Implemented in:

```text
dashboard/host/main_websocket.py
```

Main capabilities:

- Subscribe to MQTT topic `vibration/inference`
- Receive JSON inference payloads from ESP32
- Store telemetry in SQLite database `vibration_data.db`
- Broadcast live snapshots to WebSocket clients
- Provide API endpoints such as:
  - `/ws/dashboard`
  - `/api/history`
  - `/api/export/csv`

### Dashboard simulator

Implemented in:

```text
dashboard/sensorsimulate.py
```

Used for dashboard testing without real hardware by generating simulated sensor, FFT, feature, and inference snapshots.

### React / Next.js dashboard

Implemented in:

```text
dashboard/reactdashboard/
```

Main display components:

- Connection status
- Fan state
- Sensor health
- Raw acceleration values
- RMS and band-power features
- FFT spectrum
- AI prediction
- Per-class confidence scores

---

## Generated Outputs

The pipeline generates several intermediate and final artifacts:

| Folder / File | Description |
|---|---|
| `real_qc_reports/` | Validation and quality-control reports |
| `real_segments/` | Valid continuous data segments |
| `real_windows/` | Sliding-window CSV files |
| `real_fft_windows/` | FFT spectrum outputs per window |
| `real_features/features_dataset.csv` | Final extracted feature dataset |
| `real_features/feature_extraction_report.csv` | Feature extraction status report |
| `real_segment_report.csv` | Segmentation report |
| `real_window_report.csv` | Windowing report |
| `real_fft_report.csv` | FFT processing report |

---

## Mock Data and Fault Injection

The repository also includes utilities for testing the pipeline without real hardware.

### Mock data generation

Implemented in:

```text
src/generate_data.py
```

This script generates synthetic vibration signals for:

- `healthy`
- `imbalanced`
- `obstruction`

It simulates noise, harmonics, spikes, phase jitter, and slight RPM drift.

### Corrupted data generation

Implemented in:

```text
src/generate_faults.py
```

This script creates intentionally corrupted CSV files for validator testing, including:

- Missing chunks
- Timestamp jumps
- Duplicate timestamps
- Reverse timestamps
- High jitter

---

## Known Notes and Limitations

- The repository contains multiple deployment variants instead of one single final firmware path.
- Some older scripts or experimental folders may use naming conventions that differ from the current real-data pipeline.
- The `real_*` folders may contain generated artifacts from previous runs.
- `requirements.txt` mainly covers the backend/streaming stack, so preprocessing dependencies should be verified.
- The feature checker should be updated whenever the feature schema changes.
- The README should be kept synchronized with the actual source code and `features_dataset.csv` column names.

---

## Team Contributions

| Member | Main Responsibilities |
|---|---|
| Thien | Preprocessing pipeline research, implementation, debugging, and feature pipeline completion |
| Duc | Feature engineering research, model architecture, model training, noise filtering method, and edge deployment |
| Kien | Dashboard design, frontend/backend development, MQTT pipeline, and live monitoring integration |

### Project Timeline

| Period | Work Summary |
|---|---|
| Week 1 | Physical system design, sensor setup, and initial hardware preparation |
| Week 2-3 | Preprocessing research and development, feature engineering research, model planning, website and MQTT research |
| Week 3 | Preprocessing debugging, first filtering method, first model training, initial edge deployment, frontend/backend implementation |
| Week 4 | Pipeline debugging, model experimentation, website completion, MQTT pipeline development |
| Week 5 | Final preprocessing completion, deployment bug fixing, feature updates, and dashboard updates |

---
## License

No license has been specified yet. Add a license file if this repository will be shared publicly.

