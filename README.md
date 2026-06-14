# IoT Vibration Classification Project

This repository implements an end-to-end vibration fault classification pipeline for a fan/rotating-machine use case.

The project has two main sides:

- Python notebook and scripts for data preparation, feature extraction, training, and validation.
- Embedded Arduino/ESP32 code for real-time acquisition and inference.

## High-Level Flow

1. Collect raw IMU acceleration data.
2. Segment the signal into fixed windows.
3. Convert each window into FFT-ready data.
4. Extract a small feature vector for classification.
5. Train a lightweight model in the notebook.
6. Export model parameters to embedded C++ for deployment.

## Core Signal Settings

- Sampling rate: 200 Hz
- Window size: 256 samples
- Overlap ratio: 0.5
- FFT bins: 129 one-sided bins (`WINDOW_SIZE // 2 + 1`)
- Frequency range: 0 to 100 Hz
- Class labels: `healthy`, `imbalance`, `obstruction`

These constants are centralized in [src/config.py](src/config.py) and should stay aligned across notebook and embedded code.

## Main Files

### Python pipeline

- [Machine_Learning_Code_IoT.ipynb](Machine_Learning_Code_IoT.ipynb) is the main notebook for exploration, feature analysis, model training, and exporting parameters to C++.
- [src/config.py](src/config.py) stores shared constants, class labels, and mock/real path mappings.
- [src/fft_processor.py](src/fft_processor.py) prepares FFT-ready signals, removes DC offset, applies a Hann window, and exports FFT CSV files.
- [src/feature_extraction.py](src/feature_extraction.py) computes higher-level features from FFT/window data.
- [src/segmenter.py](src/segmenter.py), [src/windowing.py](src/windowing.py), [src/generate_data.py](src/generate_data.py), [src/generate_faults.py](src/generate_faults.py), [src/qc_analyzer.py](src/qc_analyzer.py), [src/validator.py](src/validator.py), and [src/check_features.py](src/check_features.py) support the pipeline with segmentation, synthetic data, QC, validation, and sanity checks.

### Embedded deployment

- [Arduino_Code/Data_Collection/IIR_deployment.ino](Arduino_Code/Data_Collection/IIR_deployment.ino) is the real-time ESP32 deployment sketch.
- [Arduino_Code/Data_Collection/model_parameters.h](Arduino_Code/Data_Collection/model_parameters.h) contains embedded model constants.

## Notebook-to-Embedded Contract

The embedded sketch expects a 6-feature input vector:

- `RMS_X`
- `RMS_Y`
- `RMS_Z`
- `BandPower_Low`
- `BandPower_Mid`
- `BandPower_High`

In the sketch, the flow is:

- Read `accX`, `accY`, `accZ` from the IMU.
- Apply an IIR high-pass filter for gravity/DC removal.
- Compute RMS per axis over a 256-sample window.
- Compute bandpower for 0-20 Hz, 20-60 Hz, and 60-100 Hz.
- Normalize the 6 features with `SCALER_MEAN` and `SCALER_STD`.
- Run logistic regression softmax inference.

The notebook appears to follow the same six-feature design and exports scaler and model parameters for C++ embedding.

## Important Implementation Notes

- The FFT processor removes DC offset explicitly before the Hann window step.
- `feature_extraction.py` works on FFT/window outputs and is separate from the embedded inference path.
- The repository has both mock and real data modes; path mappings are selected through `get_paths(mode)` in `src/config.py`.
- If you change window size, sampling rate, frequency bands, or feature order, update both the Python pipeline and the Arduino deployment code together.

## Expected Data Layout

The pipeline uses workspace folders such as:

- `mock_raw_sensor_stream/`, `windows/`, `fft_windows/`, `features/`
- `real_raw_sensor_stream/accepted`, `real_windows/`, `real_fft_windows/`, `real_features/`
- CSV reports such as `segment_report.csv`, `window_report.csv`, and `fft_report.csv`

## Practical Checks

- Confirm window length is always 256 samples.
- Confirm FFT output has 129 bins.
- Confirm feature order stays consistent with the embedded scaler and weights.
- Confirm the training pipeline and the deployment pipeline use the same preprocessing assumptions.

## Quick AI Context

If you only need a short mental model of the repo:

- This is a vibration classification system for three machine states.
- Python prepares windows, FFTs, and features, then trains a compact model.
- ESP32 code performs live filtering, feature extraction, and inference.
- The deployment contract is six features plus standardized scaling.
