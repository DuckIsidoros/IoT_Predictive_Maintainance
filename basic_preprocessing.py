import pandas as pd
import numpy as np
import os

# Cấu hình hệ thống (Phải khớp với file trước)
WINDOW_SIZE = 256
STRIDE = 128
OUTPUT_DIR = "mock_raw_sensor_stream"

def extract_features_from_window(w_x, w_y, w_z):
    """Giả lập bộ phận Preprocessing trích xuất đặc trưng"""
    # 1. Tính RMS cho 3 trục
    rms_x = np.sqrt(np.mean(w_x**2))
    rms_y = np.sqrt(np.mean(w_y**2))
    rms_z = np.sqrt(np.mean(w_z**2))
    
    # 2. Giả lập FFT bằng cách phân tích năng lượng thô qua các bộ lọc tần số
    # (Ở đây dùng biến đổi Fourier nhanh thực tế để chia BandPower)
    fft_x = np.abs(np.fft.rfft(w_x))
    
    # Với fs = 200Hz, N_FFT = 256 -> rfft sinh ra 129 bins, mỗi bin = 200 / 256 = 0.78 Hz
    # Band_Low: 0 - 30Hz  (Bins 0 - 38)
    # Band_Mid: 30 - 65Hz (Bins 38 - 83)
    # Band_High: 65 - 100Hz (Bins 83 - 128)
    band_low = np.sum(fft_x[0:38]) / 100
    band_mid = np.sum(fft_x[38:83]) / 100
    band_high = np.sum(fft_x[83:128]) / 100
    
    return [rms_x, rms_y, rms_z, band_low, band_mid, band_high]

# Mảng chứa toàn bộ dữ liệu đặc trưng sau xử lý
feature_dataset = []

for class_name in ["healthy", "imbalance", "obstruction"]:
    file_path = os.path.join(OUTPUT_DIR, f"raw_stream_{class_name}.csv")
    if not os.path.exists(file_path):
        print(f"[Lỗi] Không tìm thấy file {file_path}. Hãy chạy file sinh mock data trước!")
        continue
        
    # Đọc file raw stream
    df = pd.read_csv(file_path)
    accX = df['accX'].values
    accY = df['accY'].values
    accZ = df['accZ'].values
    
    # Chạy Sliding Window
    num_samples = len(df)
    start_idx = 0
    while (start_idx + WINDOW_SIZE) <= num_samples:
        w_x = accX[start_idx : start_idx + WINDOW_SIZE]
        w_y = accY[start_idx : start_idx + WINDOW_SIZE]
        w_z = accZ[start_idx : start_idx + WINDOW_SIZE]
        
        # Trích xuất đặc trưng
        features = extract_features_from_window(w_x, w_y, w_z)
        features.append(class_name) # Thêm nhãn nhãn vào cuối
        
        feature_dataset.append(features)
        start_idx += STRIDE

# Chuyển thành DataFrame và lưu lại
columns = ["RMS_X", "RMS_Y", "RMS_Z", "BandPower_Low", "BandPower_Mid", "BandPower_High", "Label"]
df_features = pd.DataFrame(feature_dataset, columns=columns)
df_features.to_csv("dataset_features_for_ml.csv", index=False)

print("\n==================================================")
print("[OK] ĐÃ GIẢ LẬP XONG PREPROCESSING!")
print(f"File đầu ra cho ML Team: 'dataset_features_for_ml.csv'")
print(f"Tổng số mẫu đặc trưng thu được: {len(df_features)} dòng (mỗi dòng là 1 window)")
print("==================================================")
print(df_features.groupby('Label').size())