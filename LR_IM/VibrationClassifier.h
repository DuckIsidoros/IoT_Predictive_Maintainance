/*
 * VibrationClassifier.h
 * Lớp thực thi inference mô hình Logistic Regression trên Edge (ESP32).
 * Hỗ trợ bóc tách đặc trưng 8 chiều: [RMS_X, RMS_Y, RMS_Z, Band_Power_Z_Low, _Mid, _High, CrestFactor_Z, Kurtosis_Z]
 */

#ifndef VIBRATION_CLASSIFIER_H
#define VIBRATION_CLASSIFIER_H

#include <Arduino.h>
#include <math.h>
#include "LR_parameters.h"

class VibrationClassifier
{
private:
    // Hàm kích hoạt Sigmoid đưa logit về khoảng xác suất [0.0, 1.0]
    static inline float sigmoid(float x)
    {
        return 1.0f / (1.0f + expf(-x));
    }

public:
    VibrationClassifier() {}

    /**
     * Hàm thực hiện dự đoán nhãn từ mảng đặc trưng thô (8 phần tử)
     * @param raw_features: Mảng input chứa 8 đặc trưng lai vừa trích xuất
     * @param out_probabilities: Mảng output chứa xác suất của 3 class [MODEL_NUM_CLASSES]
     * @return Tên của nhãn có xác suất cao nhất (String)
     */
    String predict(const float *raw_features, float *out_probabilities)
    {
        float scaled_features[MODEL_NUM_FEATURES];

        // 1. RUNTIME DATA NORMALIZATION (Chuẩn hóa dữ liệu đồng bộ với StandardScaler trên Python)
        // Math: x_scaled = (x_raw - MEAN) / STD
        for (int f = 0; f < MODEL_NUM_FEATURES; f++)
        {
            scaled_features[f] = (raw_features[f] - SCALER_MEAN[f]) / SCALER_STD[f];
        }

        // 2. MATRIX MULTIPLICATION & BIAS ADDITION (Tính Dot Product One-vs-Rest)
        int best_class_idx = 0;
        float max_prob = -1.0f;

        for (int c = 0; c < MODEL_NUM_CLASSES; c++)
        {
            // Khởi tạo tổng bằng giá trị Bias của class đó
            float logit = LGR_BIASES[c];

            // Nhân chập hàng chập cột giữa ma trận trọng số và vector đặc trưng
            for (int f = 0; f < MODEL_NUM_FEATURES; f++)
            {
                logit += scaled_features[f] * LGR_WEIGHTS[c][f];
            }

            // Đi qua hàm kích hoạt Sigmoid để lấy xác suất phần trăm
            out_probabilities[c] = sigmoid(logit);

            // Tìm nhãn chiến thắng có xác suất cao nhất (Argmax)
            if (out_probabilities[c] > max_prob)
            {
                max_prob = out_probabilities[c];
                best_class_idx = c;
            }
        }

        // 3. Trả về nhãn String tương ứng từ siêu dữ liệu (Metadata) trong file .h
        return String(MODEL_CLASS_LABELS[best_class_idx]);
    }
};

#endif // VIBRATION_CLASSIFIER_H