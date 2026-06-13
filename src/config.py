from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ======================================================
# core signal config
# ======================================================
SAMPLING_RATE_HZ = 200
EXPECTED_DT_MS = 1000 / SAMPLING_RATE_HZ

WINDOW_SIZE = 256
OVERLAP_RATIO = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))

EXPECTED_FFT_BINS = WINDOW_SIZE // 2 + 1
EXPECTED_FREQ_MIN_HZ = 0.0
EXPECTED_FREQ_MAX_HZ = SAMPLING_RATE_HZ / 2
NYQUIST_FREQUENCY_HZ = SAMPLING_RATE_HZ / 2
FREQ_RESOLUTION_HZ = SAMPLING_RATE_HZ / WINDOW_SIZE

CLASS_LABELS = ["healthy", "imbalance", "obstruction"]

MAX_WINDOWS_PER_CLASS = 500

# ======================================================
# mock file path
# ======================================================
MOCK_PATHS = {
    "raw": PROJECT_ROOT / "mock_raw_sensor_stream",
    "corrupted": PROJECT_ROOT / "corrupted_data",
    "qc_reports": PROJECT_ROOT / "qc_reports",
    "segments": PROJECT_ROOT / "segments",
    "windows": PROJECT_ROOT / "windows",
    "fft_windows": PROJECT_ROOT / "fft_windows",
    "features": PROJECT_ROOT / "features",
    "segment_report": PROJECT_ROOT / "segment_report.csv",
    "window_report": PROJECT_ROOT / "window_report.csv",
    "fft_report": PROJECT_ROOT / "fft_report.csv",
}
# ======================================================
# real file path
# ======================================================
REAL_PATHS = {
    "raw": PROJECT_ROOT / "real_raw_sensor_stream" / "accepted",
    "incoming": PROJECT_ROOT / "real_raw_sensor_stream" / "incoming",
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
#switch between mock and real data by changing this variable
def get_paths(mode: str)-> dict:
    mode = mode.lower().strip()#make sure the input is case-insensitive and has no leading/trailing spaces
    if mode == "mock":
        return MOCK_PATHS
    elif mode == "real":
        return REAL_PATHS
    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'mock' or 'real'.")