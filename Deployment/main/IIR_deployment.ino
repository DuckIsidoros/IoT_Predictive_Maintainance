#include <Wire.h>
#include <math.h>
#include <arduinoFFT.h>       // Thư viện FFT chuẩn cho Arduino/ESP32
#include "model_parameters.h" // Đảm bảo MODEL_NUM_FEATURES trong file này đã sửa thành 6

#define I2C_SDA 21
#define I2C_SCL 22
#define IMU_ADDRESS 0x68
#define IMU_REG_PWR_MGMT_1 0x6B
#define IMU_REG_ACCEL_CONFIG 0x1C
#define IMU_REG_ACCEL_XOUT_H 0x3B

const float ACCEL_SCALE = 1.0f / 16384.0f; // Scale cho dải đo +/-2g

struct AccelData
{
    int16_t x;
    int16_t y;
    int16_t z;
};

// --- BỘ LỌC THÔNG CAO IIR BẬC 1 (KHỬ TRỌNG LỰC) ---
const float HPF_ALPHA = 0.9843f; 
float prev_raw_x = 0.0f, prev_filt_x = 0.0f;
float prev_raw_y = 0.0f, prev_filt_y = 0.0f;
float prev_raw_z = 0.0f, prev_filt_z = 0.0f;

// --- ĐỊNH THỜI PHẦN CỨNG & BỘ ĐỆM FFT ---
hw_timer_t *timer = NULL;
volatile bool samplingTriggered = false;

// Thay đổi sang 256 mẫu (Power of 2) để thỏa mãn điều kiện đầu vào của toán tử FFT
const int WINDOW_SIZE = 256; 
int sampleCount = 0;
float sumSqX = 0, sumSqY = 0, sumSqZ = 0;

// Bộ đệm FFT dành riêng cho trục Z (Trục đo rung cơ học chính của quạt DC)
float fft_vReal[WINDOW_SIZE];
float fft_vImag[WINDOW_SIZE];
ArduinoFFT<float> FFT = ArduinoFFT<float>(fft_vReal, fft_vImag, WINDOW_SIZE, 200.0f);

// Biến toàn cục hệ thống
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
    Wire.setClock(400000); 
    delay(100);
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_PWR_MGMT_1);
    Wire.write(0x00); 
    if (Wire.endTransmission() != 0) return false;
    delay(50);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_CONFIG);
    Wire.write(0x00); 
    if (Wire.endTransmission() != 0) return false;
    delay(50);
    return true;
}

bool IMU_ReadAcceleration(AccelData &data)
{
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_XOUT_H);
    if (Wire.endTransmission(false) != 0) return false;
    if (Wire.requestFrom(IMU_ADDRESS, 6) < 6) return false;

    data.x = (Wire.read() << 8) | Wire.read();
    data.y = (Wire.read() << 8) | Wire.read();
    data.z = (Wire.read() << 8) | Wire.read();
    return true;
}

// --- THUẬT TOÁN SUY LUẬN LOGISTIC REGRESSION CHUẨN SOFTMAX ---
void predict_logistic_regression(float rmsX, float rmsY, float rmsZ, float bpLow, float bpMid, float bpHigh)
{
    // Vector đặc trưng đầu vào đã nâng lên 6 chiều, khớp hoàn chỉnh cấu trúc phân tích
    float raw_features[6] = {rmsX, rmsY, rmsZ, bpLow, bpMid, bpHigh};
    float normalized_features[6];
    float logit_scores[MODEL_NUM_CLASSES] = {0.0f};

    // 1. Chuẩn hóa dữ liệu đầu vào (Z-score Scaling)
    for (int f = 0; f < 6; f++)
    {
        normalized_features[f] = (raw_features[f] - SCALER_MEAN[f]) / SCALER_STD[f];
    }

    // 2. Tính toán điểm số Logit độc lập (One-vs-Rest)
    float max_score = -999999.0f;
    for (int c = 0; c < MODEL_NUM_CLASSES; c++)
    {
        float score = LGR_BIASES[c];
        for (int f = 0; f < 6; f++)
        {
            score += normalized_features[f] * LGR_WEIGHTS[c][f];
        }
        logit_scores[c] = score;
        if (score > max_score) {
            max_score = score; // Giữ lại max_score phục vụ kỹ thuật toán học Softmax
        }
    }

    // 3. Thực thi toán tử Argmax chọn lớp có điểm cao nhất
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

    // 4. Thuật toán SOFTMAX ổn định số học (Numerical Stability) chống tràn số FPU
    float sum_exp = 0.0f;
    for (int c = 0; c < MODEL_NUM_CLASSES; c++)
    {
        class_probabilities[c] = exp(logit_scores[c] - max_score); 
        sum_exp += class_probabilities[c];
    }
    
    // Đưa về dải phân phối xác suất chuẩn [0.0 - 1.0]
    if (sum_exp > 0.0f) {
        for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
            class_probabilities[c] /= sum_exp;
        }
    }
}

// --- MODULE HIỂN THỊ DASHBOARD TERMINAL ---
void print_dashboard(float rmsX, float rmsY, float rmsZ, float bpLow, float bpMid, float bpHigh)
{
    Serial.print("\033[2J");
    Serial.print("\033[H");

    Serial.println(F("====================================================================="));
    Serial.println(F("[1] EDGE DATA ACQUISITION & SIGNAL FILTERING (Cửa sổ 256 mẫu)"));
    Serial.printf("- RMS X: %.4fG  |  RMS Y: %.4fG  |  RMS Z (Khử g): %.4fG\n", rmsX, rmsY, rmsZ);
    Serial.printf("- BandPower Low (0-20Hz): %.2f | Mid (20-60Hz): %.2f | High (60-100Hz): %.2f\n", bpLow, bpMid, bpHigh);
    Serial.println();

    Serial.println(F("[2] LOGISTIC REGRESSION ON-CHIP SUY LUẬN (Softmax Matrix)"));
    for (int c = 0; c < MODEL_NUM_CLASSES; c++)
    {
        int bar_length = (int)(class_probabilities[c] * 20.0f);
        if (bar_length > 20) bar_length = 20;
        if (bar_length < 0)  bar_length = 0;
        
        Serial.printf("- Trạng thái %-15s : [", MODEL_CLASS_LABELS[c]);
        for (int i = 0; i < 20; i++) {
            Serial.print(i < bar_length ? "|" : ".");
        }
        Serial.printf("] %.1f%%\n", class_probabilities[c] * 100.0f);
    }
    Serial.println();
    Serial.printf(" => KẾT LUẬN HỆ THỐNG: [Class %d] -> QUẠT CHẠY Ở TRẠNG THÁI: %s\n",
                  predicted_class_idx, MODEL_CLASS_LABELS[predicted_class_idx]);
    Serial.println();

    Serial.println(F("[3] HARDWARE TIMING & RESOURCES BENCHMARK"));
    Serial.printf("- Thời gian thực thi phép suy luận ML: %d us\n", inference_latency_us);
    Serial.printf("- Free Heap RAM hệ thống hiện tại   : %d bytes\n", esp_get_free_heap_size());
    Serial.println(F("====================================================================="));
}

void setup()
{
    Serial.begin(115200);
    while (!Serial);
    if (!IMU_Init())
    {
        Serial.println("IMU Initialization Error! System Halted.");
        while (1);
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
            // Đổi counts sang g
            float curr_raw_x = (float)rawData.x * ACCEL_SCALE;
            float curr_raw_y = (float)rawData.y * ACCEL_SCALE;
            float curr_raw_z = (float)rawData.z * ACCEL_SCALE;

            // Thực thi bộ lọc thông cao khử trọng lực
            float curr_filt_x = HPF_ALPHA * (prev_filt_x + curr_raw_x - prev_raw_x);
            float curr_filt_y = HPF_ALPHA * (prev_filt_y + curr_raw_y - prev_raw_y);
            float curr_filt_z = HPF_ALPHA * (prev_filt_z + curr_raw_z - prev_raw_z);

            // Lưu vết trạng thái bộ lọc
            prev_raw_x = curr_raw_x; prev_filt_x = curr_filt_x;
            prev_raw_y = curr_raw_y; prev_filt_y = curr_filt_y;
            prev_raw_z = curr_raw_z; prev_filt_z = curr_filt_z;

            // 1. Tính toán RMS cuốn chiếu số thực
            sumSqX += curr_filt_x * curr_filt_x;
            sumSqY += curr_filt_y * curr_filt_y;
            sumSqZ += curr_filt_z * curr_filt_z;

            // 2. Ghi dữ liệu trục Z vào Buffer để chuẩn bị tính toán dải tần FFT
            fft_vReal[sampleCount] = curr_filt_z;
            fft_vImag[sampleCount] = 0.0f; // Phần ảo khởi tạo bằng 0
            
            sampleCount++;

            // Khi tích lũy đủ 256 mẫu cơ học (~1.28 giây dữ liệu thực tế ở tần số 200Hz)
            if (sampleCount >= WINDOW_SIZE)
            {
                // Trích xuất RMS cuối cùng của cửa sổ
                float rmsX = sqrt(sumSqX / (float)WINDOW_SIZE);
                float rmsY = sqrt(sumSqY / (float)WINDOW_SIZE);
                float rmsZ = sqrt(sumSqZ / (float)WINDOW_SIZE);

                // 3. THỰC THI MODULE FFT TRÊN CHIP ĐỂ LẤY BIÊN ĐỘ PHỔ
                FFT.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD); // Áp dụng cửa sổ Hamming khử nhiễu rò rỉ phổ
                FFT.compute(FFT_FORWARD);                        // Thực thi phép biến đổi FFT
                FFT.complexToMagnitude();                        // Chuyển kết quả sang mảng biên độ (Lưu đè vào mảng fft_vReal)

                // 4. TRÍCH XUẤT NĂNG LƯỢNG DẢI TẦN (BAND POWER) KHỚP LOGIC PYTHON
                float bpLow = 0.0f, bpMid = 0.0f, bpHigh = 0.0f;
                
                for (int i = 0; i < WINDOW_SIZE / 2; i++)
                {
                    // Tính toán tần số vật lý chính xác của Bin thứ i: Freq = i * Sampling_Rate / N
                    float freq = (float)i * 200.0f / (float)WINDOW_SIZE;
                    float magSq = fft_vReal[i] * fft_vReal[i]; // Năng lượng = biên độ bình phương

                    // Gom năng lượng vào đúng dải ranh giới logic của Python
                    if (freq >= 0.0f && freq < 20.0f)       bpLow += magSq;
                    else if (freq >= 20.0f && freq < 60.0f)  bpMid += magSq;
                    else if (freq >= 60.0f && freq <= 100.0f) bpHigh += magSq;
                }

                // Đo đạc thời gian suy luận ML bao gồm cả cấu trúc mạng
                unsigned long start_bench = micros();
                predict_logistic_regression(rmsX, rmsY, rmsZ, bpLow, bpMid, bpHigh);
                inference_latency_us = micros() - start_bench;

                // Xuất giao diện đồ họa
                print_dashboard(rmsX, rmsY, rmsZ, bpLow, bpMid, bpHigh);

                // Reset toàn bộ trạng thái chuẩn bị cho chu kỳ gom mẫu tiếp theo
                sampleCount = 0;
                sumSqX = 0.0f; sumSqY = 0.0f; sumSqZ = 0.0f;
            }
        }
    }
}
