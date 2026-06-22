import numpy as np
import pandas as pd
import os

# =========================================================
# SYSTEM CONFIG
# =========================================================

SAMPLING_RATE = 200  # Hz #there will be 6000 samples per 30-secondrecording
DT = 1.0 / SAMPLING_RATE

RECORD_DURATION_SEC = 30
TOTAL_SAMPLES = int(RECORD_DURATION_SEC * SAMPLING_RATE)

FILES_PER_CLASS = 20
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "mock_raw_sensor_stream")

G_UNIT = 9.81

# =========================================================
# FAN PHYSICAL CHARACTERISTICS
# =========================================================

F_HEALTHY = 52.0
F_IMBALANCE = 48.0
F_OBSTRUCTION = 30.0

# =========================================================
# OUTPUT DIRECTORY
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# SIGNAL GENERATORS
# =========================================================

def generate_healthy_signal(t):

    phase = np.random.uniform(0, 2*np.pi)

    accX = (
        0.25 * np.sin(2 * np.pi * F_HEALTHY * t + phase)
        + np.random.normal(0, 0.12, len(t))
    )

    accY = (
        0.25 * np.cos(2 * np.pi * F_HEALTHY * t + phase)
        + np.random.normal(0, 0.12, len(t))
    )

    accZ = (
        G_UNIT
        + 0.05 * np.sin(2 * np.pi * F_HEALTHY * t + phase)
        + np.random.normal(0, 0.08, len(t))
    )

    return accX, accY, accZ


def generate_imbalance_signal(t):

    phase = np.random.uniform(0, 2*np.pi)

    fundamental = np.sin(2 * np.pi * F_IMBALANCE * t + phase)
    harmonic = np.sin(2 * np.pi * 2 * F_IMBALANCE * t + phase)

    accX = (
        2.5 * fundamental
        + 0.7 * harmonic
        + np.random.normal(0, 0.20, len(t))
    )

    accY = (
        2.5 * np.cos(2 * np.pi * F_IMBALANCE * t + phase)
        + 0.7 * np.cos(2 * np.pi * 2 * F_IMBALANCE * t + phase)
        + np.random.normal(0, 0.20, len(t))
    )

    accZ = (
        G_UNIT
        + 0.9 * fundamental
        + np.random.normal(0, 0.20, len(t))
    )

    return accX, accY, accZ


def generate_obstruction_signal(t):

    phase = np.random.uniform(0, 2*np.pi)

    phase_jitter = np.cumsum(
        np.random.normal(0, 0.05, len(t))
    )

    base_signal = np.sin(
        2 * np.pi * F_OBSTRUCTION * t
        + phase
        + phase_jitter
    )

    spikes = np.zeros(len(t))

    spike_indices = np.random.choice(
        len(t),
        size=int(len(t) * 0.01),
        replace=False
    )

    spikes[spike_indices] = np.random.choice(
        [-4.0, 4.0],
        len(spike_indices)
    )

    accX = (
        1.2 * base_signal
        + spikes
        + np.random.normal(0, 0.6, len(t))
    )

    accY = (
        1.2 * base_signal
        + spikes
        + np.random.normal(0, 0.6, len(t))
    )

    accZ = (
        G_UNIT
        + 1.5 * base_signal
        + spikes
        + np.random.normal(0, 0.5, len(t))
    )

    return accX, accY, accZ


# =========================================================
# RECORD GENERATOR
# =========================================================

def generate_record(class_name):

    t = np.arange(TOTAL_SAMPLES) * DT

    # Simulate slight RPM drift
    drift = np.random.normal(0, 0.00005, TOTAL_SAMPLES)
    t = t + np.cumsum(drift)

    if class_name == "healthy":
        x, y, z = generate_healthy_signal(t)

    elif class_name == "imbalance":
        x, y, z = generate_imbalance_signal(t)

    elif class_name == "obstruction":
        x, y, z = generate_obstruction_signal(t)

    else:
        raise ValueError("Unknown class")

    timestamps_ms = (t * 1000).astype(np.float64)

    df = pd.DataFrame({
        "timestamp_ms": timestamps_ms,
        "accX": x,
        "accY": y,
        "accZ": z,
        "label": class_name
    })

    return df


# =========================================================
# DATASET GENERATION
# =========================================================

for class_name in ["healthy", "imbalance", "obstruction"]:

    class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    for i in range(FILES_PER_CLASS):

        df = generate_record(class_name)

        file_name = f"{class_name}_{i:03d}.csv"

        file_path = os.path.join(class_dir, file_name)

        df.to_csv(file_path, index=False)

        print(f"[OK] Generated: {file_path}")

# =========================================================
# SUMMARY
# =========================================================

print("\n======================================")
print("RAW SENSOR STREAM DATASET GENERATED")
print("======================================")
print(f"Sampling Rate : {SAMPLING_RATE} Hz")
print(f"Duration/File : {RECORD_DURATION_SEC} sec")
print(f"Samples/File  : {TOTAL_SAMPLES}")
print(f"Files/Class   : {FILES_PER_CLASS}")
print("======================================")

