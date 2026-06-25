from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ======================================================
# core signal config
# ======================================================
SAMPLING_RATE_HZ = 500
SAMPLE_PERIOD_MS = 2.0
EXPECTED_DT_MS = SAMPLE_PERIOD_MS

FEATURE_COLUMNS = ["accX_raw", "accY_raw", "accZ_raw", "accZ_filt"]
HPF_ALPHA = 0.9936
MAX_ALLOWED_GAP_MS = 5.0

WINDOW_SIZE = 640
OVERLAP_RATIO = 0.5
STEP_SIZE = 320

EXPECTED_FFT_BINS = WINDOW_SIZE // 2 + 1
EXPECTED_FREQ_MIN_HZ = 0.0
EXPECTED_FREQ_MAX_HZ = SAMPLING_RATE_HZ / 2
NYQUIST_FREQUENCY_HZ = SAMPLING_RATE_HZ / 2
FREQ_RESOLUTION_HZ = SAMPLING_RATE_HZ / WINDOW_SIZE

CLASS_LABELS = ["healthy", "imbalanced", "obstruction"]
# Define label aliases for more flexible labeling
LABEL_ALIASES = {
    "healthy": "healthy",
    "normal": "healthy",
    "good": "healthy",

    "imbalance": "imbalanced",
    "imbalanced": "imbalanced",
    "imabalanced": "imbalanced",

    "obstruction": "obstruction",
    "clogged": "obstruction",
    "blocked": "obstruction",
}

MAX_WINDOWS_PER_CLASS = 500

# ======================================================
# deployment file path
# ======================================================
PIPELINE_PATHS = {
    "incoming": PROJECT_ROOT / "real_raw_sensor_stream" / "incoming",
    "accepted": PROJECT_ROOT / "real_raw_sensor_stream" / "accepted",
    "rejected": PROJECT_ROOT / "real_raw_sensor_stream" / "rejected",
    "metadata": PROJECT_ROOT / "real_raw_sensor_stream" / "metadata",
    "qc_reports": PROJECT_ROOT / "real_qc_reports",
    "segments": PROJECT_ROOT / "real_segments",
    "windows": PROJECT_ROOT / "real_windows",
    "fft_windows": PROJECT_ROOT / "real_fft_windows",
    "features": PROJECT_ROOT / "real_features",
    "segment_report": PROJECT_ROOT / "real_segment_report.csv",
    "window_report": PROJECT_ROOT / "real_window_report.csv",
    "fft_report": PROJECT_ROOT / "real_fft_report.csv",
}


def get_paths() -> dict:
    return PIPELINE_PATHS
