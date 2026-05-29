import numpy as np
import pandas as pd
import os

# =========================================================
# SYSTEM CONFIG (Đã đồng bộ cho quạt Sunon & Cấu hình 200Hz)
# =========================================================
SAMPLING_RATE = 200  # Hz (Mỗi mẫu cách nhau 5ms)
DT = 1.0 / SAMPLING_RATE

# Thời gian thu dữ liệu liên tục để thu về ~500 windows sau khi sliding window
# Công thức: Window_Size + (Target_Windows - 1) * Stride
# 256 mẫu + (500 - 1) * 128 mẫu = 64,128 mẫu. Lấy thời gian: 64,128 / 200Hz = 320.64 giây
RECORD_DURATION_SEC = 320.64
TOTAL_SAMPLES = int(RECORD_DURATION_SEC * SAMPLING_RATE)

OUTPUT_DIR = "mock_raw_sensor_stream"
G_UNIT = 9.81  # Gia tốc trọng trường cho trục Z tĩnh

# =========================================================
# FAN PHYSICAL CHARACTERISTICS (3100 RPM = 51.67 Hz)
# =========================================================
F_ROTATION = 51.67  # Tần số quay cơ bản của quạt Sunon 120mm

# =========================================================
# OUTPUT DIRECTORY
# =========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# SIGNAL GENERATORS (Mô phỏng bản chất cơ khí thực tế)
# =========================================================

def generate_healthy_signal(t):
    """Trạng thái chạy bình thường: Rung động rất nhỏ, ổn định."""
    phase = np.random.uniform(0, 2 * np.pi)
    
    # Biên độ rung cực thấp trên trục X, Y
    accX = 0.15 * np.sin(2 * np.pi * F_ROTATION * t + phase) + np.random.normal(0, 0.05, len(t))
    accY = 0.15 * np.cos(2 * np.pi * F_ROTATION * t + phase) + np.random.normal(0, 0.05, len(t))
    
    # Trục Z chịu trọng lực g (~9.81) và rung nhẹ
    accZ = G_UNIT + 0.05 * np.sin(2 * np.pi * F_ROTATION * t + phase) + np.random.normal(0, 0.03, len(t))
    
    return accX, accY, accZ


def generate_imbalance_signal(t):
    """Trạng thái lệch tâm: Lực ly tâm tăng mạnh biên độ ở tần số cơ bản và hài bậc 2."""
    phase = np.random.uniform(0, 2 * np.pi)
    
    # Biên độ tăng vọt ở trục X và Y (mặt phẳng quay của quạt)
    fundamental_X = 2.8 * np.sin(2 * np.pi * F_ROTATION * t + phase)
    harmonic_X = 0.8 * np.sin(2 * np.pi * 2 * F_ROTATION * t + phase)  # Hài bậc 2 (103.34 Hz)
    
    accX = fundamental_X + harmonic_X + np.random.normal(0, 0.15, len(t))
    accY = 2.8 * np.cos(2 * np.pi * F_ROTATION * t + phase) + 0.8 * np.cos(2 * np.pi * 2 * F_ROTATION * t + phase) + np.random.normal(0, 0.15, len(t))
    
    # Trục Z rung mạnh hơn bình thường do mất cân bằng động tác động lên khớp trục
    accZ = G_UNIT + 1.2 * np.sin(2 * np.pi * F_ROTATION * t + phase) + np.random.normal(0, 0.15, len(t))
    
    return accX, accY, accZ


def generate_obstruction_signal(t):
    """Trạng thái cản gió: Xuất hiện hiện tượng rối dòng khí (Vortex), tăng nhiễu nền tần số cao."""
    phase = np.random.uniform(0, 2 * np.pi)
    
    # Tốc độ quạt suy hao nhẹ do bị cản gió cơ học (ví dụ giảm xuống ~48Hz)
    f_blocked = F_ROTATION * 0.93 
    
    # Rung động không đồng đều, bị xê dịch pha liên tục do gió quẩn
    phase_jitter = np.cumsum(np.random.normal(0, 0.08, len(t)))
    base_signal = np.sin(2 * np.pi * f_blocked * t + phase + phase_jitter)
    
    # Sinh ra các cú giật áp suất (spikes) ngẫu nhiên do cản dòng khí khí động học
    spikes = np.zeros(len(t))
    spike_indices = np.random.choice(len(t), size=int(len(t) * 0.02), replace=False)
    spikes[spike_indices] = np.random.choice([-1.5, 1.5], len(spike_indices))
    
    # Biên độ tăng trung bình nhưng nhiễu nền (vòng lặp trắng) tăng rất mạnh
    accX = 0.8 * base_signal + spikes + np.random.normal(0, 0.6, len(t))
    accY = 0.8 * base_signal + spikes + np.random.normal(0, 0.6, len(t))
    accZ = G_UNIT + 1.0 * base_signal + spikes + np.random.normal(0, 0.5, len(t))
    
    return accX, accY, accZ


# =========================================================
# RECORD GENERATOR
# =========================================================
def generate_record(class_name):
    t = np.arange(TOTAL_SAMPLES) * DT

    # Mô phỏng sự trôi nhẹ RPM trong thực tế của động cơ DC
    drift = np.random.normal(0, 0.001, TOTAL_SAMPLES)
    t = t + np.cumsum(drift)

    if class_name == "healthy":
        x, y, z = generate_healthy_signal(t)
    elif class_name == "imbalance":
        x, y, z = generate_imbalance_signal(t)
    elif class_name == "obstruction":
        x, y, z = generate_obstruction_signal(t)
    else:
        raise ValueError("Unknown class")

    # Ép kiểu thời gian chính xác theo mili-giây
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
# DATASET GENERATION (Sinh một file tổng lực cho mỗi Class)
# =========================================================
for class_name in ["healthy", "imbalance", "obstruction"]:
    df = generate_record(class_name)
    
    # Đặt tên file trực quan để bàn giao cho Preprocessing Team
    file_name = f"test_stream_{class_name}.csv"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    
    df.to_csv(file_path, index=False)
    print(f"[OK] Generated Continuous Stream: {file_path}")

# =========================================================
# SUMMARY & VERIFICATION
# =========================================================
print("\n==================================================")
print("RAW SENSOR STREAM FOR STUDENT DEMO GENERATED")
print("==================================================")
print(f"Sampling Rate       : {SAMPLING_RATE} Hz (Interval: {DT*1000}ms)")
print(f"Total Duration/File : {RECORD_DURATION_SEC:.2f} sec (~5.3 minutes)")
print(f"Total Samples/File  : {TOTAL_SAMPLES} rows")
print(f"Expected Windows    : 499 Windows (at Window=256, Stride=128)")
print("==================================================")