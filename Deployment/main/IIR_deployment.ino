#include <Wire.h>
#include <math.h>
#include "model_parameters.h" // Chứa SCALER_MEAN, SCALER_STD, LGR_WEIGHTS, LGR_BIASES, MODEL_CLASS_LABELS

#define I2C_SDA 21
#define I2C_SCL 22
#define IMU_ADDRESS 0x68
#define IMU_REG_PWR_MGMT_1 0x6B
#define IMU_REG_ACCEL_CONFIG 0x1C
#define IMU_REG_ACCEL_XOUT_H 0x3B

const float ACCEL_SCALE = 1.0f / 16384.0f; // Scale hằng số cho dải đo +/-2g

struct AccelData
{
    int16_t x;
    int16_t y;
    int16_t z;
};

// --- BỘ LỌC THÔNG CAO IIR BẬC 1 (KHỬ TRỌNG LỰC ĐỂ LẤY AC RUNG ĐỘNG) ---
const float HPF_ALPHA = 0.9843f; // Tần số cắt ~0.5Hz ở chu kỳ 200Hz
float prev_raw_x = 0.0f, prev_filt_x = 0.0f;
float prev_raw_y = 0.0f, prev_filt_y = 0.0f;
float prev_raw_z = 0.0f, prev_filt_z = 0.0f;

// --- ĐỊNH THỜI PHẦN CỨNG (HARDWARE TIMER INTERRUPT) ---
hw_timer_t *timer = NULL;
volatile bool samplingTriggered = false;

const int WINDOW_SIZE = 200; // Cửa sổ đánh giá rung động 1 giây (200 mẫu)
int sampleCount = 0;
float sumSqX = 0, sumSqY = 0, sumSqZ = 0;

// Các biến toàn cục lưu trạng thái suy luận phục vụ giao diện Dashboard
float class_probabilities[MODEL_NUM_CLASSES] = {0.0f};
int predicted_class_idx = 0;
unsigned long inference_latency_us = 0;

void IRAM_ATTR onTimer()
{
    samplingTriggered = true;
}

bool IMU_Init()
{
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000); // 400kHz Fast-mode tối ưu I2C Bus Timing
    delay(100);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_PWR_MGMT_1);
    Wire.write(0x00); // Kích hoạt cảm biến hoạt động
    if (Wire.endTransmission() != 0)
        return false;
    delay(50);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_CONFIG);
    Wire.write(0x00); // Cấu hình cứng dải đo +/-2g
    if (Wire.endTransmission() != 0)
        return false;
    delay(50);

    return true;
}

bool IMU_ReadAcceleration(AccelData &data)
{
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_XOUT_H);
    if (Wire.endTransmission(false) != 0)
        return false;

    if (Wire.requestFrom(IMU_ADDRESS, 6) < 6)
        return false;

    data.x = (Wire.read() << 8) | Wire.read();
    data.y = (Wire.read() << 8) | Wire.read();
    data.z = (Wire.read() << 8) | Wire.read();
    return true;
}

// --- THUẬT TOÁN SUY LUẬN LOGISTIC REGRESSION TỐI GIẢN TRÊN CHIP ---
void predict_logistic_regression(float rmsX, float rmsY, float rmsZ)
{
    float raw_features[MODEL_NUM_FEATURES] = {rmsX, rmsY, rmsZ};
    float normalized_features[MODEL_NUM_FEATURES];
    float logit_scores[MODEL_NUM_CLASSES] = {0.0f};

    // 1. Chuẩn hóa dữ liệu đầu vào tự động (Khớp hoàn toàn với StandardScaler từ Python)
    for (int i = 0; i < MODEL_NUM_FEATURES; i++)
    {
        normalized_features[i] = (raw_features[i] - SCALER_MEAN[i]) / SCALER_STD[i];
    }

    float max_score = -999999.0f;
    float min_score = 999999.0f;

    // 2. Tính toán hàm tuyến tính độc lập (One-vs-Rest): Z = W * X + b
    for (int c = 0; c < MODEL_NUM_CLASSES; c++)
    {
        float score = LGR_BIASES[c];
        for (int f = 0; f < MODEL_NUM_FEATURES; f++)
        {
            score += normalized_features[f] * LGR_WEIGHTS[c][f];
        }
        logit_scores[c] = score;

        if (score > max_score)
            max_score = score;
        if (score < min_score)
            min_score = score;
    }

    // 3. Thực thi thuật toán chọn Class (Argmax)
    predicted_class_idx = 0;
    float current_max = logit_scores[0];
    for (int c = 1; c < MODEL_NUM_CLASSES; c++)
    {
        if (logit_scores[c] > current_max)
        {
            current_max = logit_scores[c];
            predicted_class_idx = c;
        }
    }

    // 4. Giả lập xác suất (%) tuyến tính hóa phục vụ hiển thị đồ họa Dashboard (Không dùng exp)
    float score_range = max_score - min_score;
    if (score_range < 0.001f)
        score_range = 0.001f; // Chống chia cho 0

    float sum_normalized_logits = 0.0f;
    for (int c = 0; c < MODEL_NUM_CLASSES; c++)
    {
        // Chuyển dải điểm số về vùng dương [0, max]
        class_probabilities[c] = logit_scores[c] - min_score;
        sum_normalized_logits += class_probabilities[c];
    }

    // Ép tổng mảng xác suất về lại mốc 1.0 (100%)
    if (sum_normalized_logits > 0.0f)
    {
        for (int c = 0; c < MODEL_NUM_CLASSES; c++)
        {
            class_probabilities[c] /= sum_normalized_logits;
        }
    }
    else
    {
        // Trường hợp khẩn cấp nếu tất cả điểm bằng nhau
        for (int c = 0; c < MODEL_NUM_CLASSES; c++)
        {
            class_probabilities[c] = 1.0f / (float)MODEL_NUM_CLASSES;
        }
    }
}

// --- MODULE HIỂN THỊ DASHBOARD TRỰC QUAN QUA ANSI TERMINAL ---
void print_dashboard(float rmsX, float rmsY, float rmsZ)
{
    // Mã điều khiển ANSI: Xóa màn hình cũ và đẩy con trỏ về gốc trái trên cùng
    Serial.print("\033[2J");
    Serial.print("\033[H");

    Serial.println(F("====================================================================="));
    Serial.println(F("[1] EDGE DATA ACQUISITION & SIGNAL FILTERING (Cửa sổ 200 mẫu - 1 Giây)"));
    Serial.printf("- RMS Rung Trục X: %.4fG  |  RMS Trục Y: %.4fG  |  RMS Trục Z (Đã khử g): %.4fG\n", rmsX, rmsY, rmsZ);
    Serial.println();

    Serial.println(F("[2] LOGISTIC REGRESSION ON-CHIP SUY LUẬN (One-vs-Rest Architecture)"));

    // Vẽ thanh tiến trình động dựa trên số lớp được xuất tự động từ file .h
    for (int c = 0; c < MODEL_NUM_CLASSES; c++)
    {
        int bar_length = (int)(class_probabilities[c] * 20.0f);
        if (bar_length > 20)
            bar_length = 20;
        if (bar_length < 0)
            bar_length = 0;

        // In tên nhãn gốc từ Python và vẽ thanh bar
        Serial.printf("- Trạng thái %-15s : [", MODEL_CLASS_LABELS[c]);
        for (int i = 0; i < 20; i++)
        {
            Serial.print(i < bar_length ? "|" : ".");
        }
        Serial.printf("] %.1f%%\n", class_probabilities[c] * 100.0f);
    }
    Serial.println();

    // In kết luận chẩn đoán lỗi cơ học
    Serial.printf(" => KẾT LUẬN HỆ THỐNG: [Class %d] -> QUẠT CHẠY Ở TRẠNG THÁI: %s\n",
                  predicted_class_idx, MODEL_CLASS_LABELS[predicted_class_idx]);
    Serial.println();

    Serial.println(F("[3] HARDWARE TIMING & RESOURCES BENCHMARK"));
    Serial.printf("- Thời gian thực thi phép suy luận ML: %d us (Microseconds)\n", inference_latency_us);
    Serial.printf("- Flash Memory Footprint Arrays     : Cực thấp (Hằng số đã map vào DROM/Flash)\n");
    Serial.printf("- Free Heap RAM hệ thống hiện tại   : %d bytes\n", esp_get_free_heap_size());
    Serial.printf("- Minimum Free Heap (Peak)          : %d bytes\n", esp_get_minimum_free_heap_size());
    Serial.println(F("====================================================================="));
}

void setup()
{
    Serial.begin(115200);
    while (!Serial)
        ;

    if (!IMU_Init())
    {
        Serial.println("IMU Initialization Error! System Halted.");
        while (1)
            ;
    }

    // Cấu hình Hardware Timer định thời ngắt chính xác 5ms (200Hz)
    timer = timerBegin(0, 80, true);
    timerAttachInterrupt(timer, &onTimer, true);
    timerAlarmWrite(timer, 5000, true);
    timerAlarmEnable(timer);
}

void loop()
{
    if (samplingTriggered)
    {
        samplingTriggered = false; // Xóa cờ ngắt phần cứng

        AccelData rawData;
        if (IMU_ReadAcceleration(rawData))
        {
            // Đổi counts thô sang đơn vị gia tốc thực (g)
            float curr_raw_x = (float)rawData.x * ACCEL_SCALE;
            float curr_raw_y = (float)rawData.y * ACCEL_SCALE;
            float curr_raw_z = (float)rawData.z * ACCEL_SCALE;

            // THỰC THI BỘ LỌC THÔNG CAO (TRIỆT TIÊU KHỬ HOÀN TOÀN TRỌNG LỰC TĨNH)
            float curr_filt_x = HPF_ALPHA * (prev_filt_x + curr_raw_x - prev_raw_x);
            float curr_filt_y = HPF_ALPHA * (prev_filt_y + curr_raw_y - prev_raw_y);
            float curr_filt_z = HPF_ALPHA * (prev_filt_z + curr_raw_z - prev_raw_z);

            // Lưu vết trạng thái bộ lọc cho chu kỳ sau
            prev_raw_x = curr_raw_x;
            prev_filt_x = curr_filt_x;
            prev_raw_y = curr_raw_y;
            prev_filt_y = curr_filt_y;
            prev_raw_z = curr_raw_z;
            prev_filt_z = curr_filt_z;

            // Tính toán Feature cuốn chiếu trên dữ liệu ĐÃ SẠCH bóng trọng lực
            sumSqX += curr_filt_x * curr_filt_x;
            sumSqY += curr_filt_y * curr_filt_y;
            sumSqZ += curr_filt_z * curr_filt_z;
            sampleCount++;

            // Khi tích lũy đủ 1 giây cửa sổ dữ liệu (200 mẫu)
            if (sampleCount >= WINDOW_SIZE)
            {
                float rmsX = sqrt(sumSqX / (float)WINDOW_SIZE);
                float rmsY = sqrt(sumSqY / (float)WINDOW_SIZE);
                float rmsZ = sqrt(sumSqZ / (float)WINDOW_SIZE);

                // Đo đạc thời gian suy luận (Inference Benchmark)
                unsigned long start_bench = micros();
                predict_logistic_regression(rmsX, rmsY, rmsZ);
                inference_latency_us = micros() - start_bench;

                // Xuất Dashboard đồ họa ra màn hình Serial Monitor
                print_dashboard(rmsX, rmsY, rmsZ);

                // Giải phóng bộ đệm tích lũy chuẩn bị cho chu kỳ 1 giây tiếp theo
                sampleCount = 0;
                sumSqX = 0.0f;
                sumSqY = 0.0f;
                sumSqZ = 0.0f;
            }
        }
    }
}