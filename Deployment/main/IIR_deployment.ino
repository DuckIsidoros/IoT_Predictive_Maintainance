#include <Wire.h>
#include <math.h>
#include <arduinoFFT.h>
#include "model_parameters.h" // Yêu cầu MODEL_NUM_FEATURES = 6 [cite: 1]

// --- HẰNG SỐ PHẦN CỨNG & CẤU HÌNH ---
#define I2C_SDA             21 [cite: 1]
#define I2C_SCL             22 [cite: 1]
#define IMU_ADDRESS         0x68 [cite: 1]
#define IMU_REG_PWR_MGMT_1  0x6B [cite: 1]
#define IMU_REG_ACCEL_CONFIG 0x1C [cite: 1]
#define IMU_REG_ACCEL_XOUT_H 0x3B [cite: 1]

const float ACCEL_SCALE = 1.0f / 16384.0f; // Dải đo +/-2g [cite: 1, 2]
const int WINDOW_SIZE   = 256;             // Cửa sổ xử lý (Bắt buộc là lũy thừa của 2) [cite: 7]
const float SAMPLING_FREQ = 200.0f;         // Tần số lấy mẫu định thời bằng hardware timer [cite: 4]

struct AccelData {
    int16_t x;
    int16_t y;
    int16_t z;
}; [cite: 2]

// --- BIẾN TRẠNG THÁI HỆ THỐNG (TOÀN CỤC) ---
hw_timer_t *timer = NULL; [cite: 6]
volatile bool samplingTriggered = false; [cite: 6]

// Khởi tạo các mảng tĩnh lưu trữ phổ tần số, chống phân mảnh bộ nhớ (Zero Heap Allocation)
static float fft_vReal[WINDOW_SIZE];
static float fft_vImag[WINDOW_SIZE];
ArduinoFFT<float> FFT = ArduinoFFT<float>(fft_vReal, fft_vImag, WINDOW_SIZE, SAMPLING_FREQ);

// Biến toàn cục phục vụ Dashboard hiển thị dữ liệu
float class_probabilities[MODEL_NUM_CLASSES] = {0.0f}; [cite: 9]
int predicted_class_idx = 0; [cite: 10]
unsigned long inference_latency_us = 0; [cite: 10]

// --- PROTOTYPES ---
bool IMU_Init();
bool IMU_ReadAcceleration(AccelData &data);
void apply_iir_filter(const AccelData &raw, float &fx, float &fy, float &fz);
void accumulate_metrics(float fx, float fy, float fz);
void compute_features_and_infer();
void predict_logistic_regression(float rx, float ry, float rz, float bLow, float bMid, float bHigh);
void print_dashboard(float rx, float ry, float rz, float bLow, float bMid, float bHigh);

// --- HARDWARE TIMER INTERRUPT SERVICE ROUTINE (ISR) ---
void IRAM_ATTR onTimer() {
    samplingTriggered = true; [cite: 10, 11]
}

void setup() {
    Serial.begin(115200); [cite: 49]
    while (!Serial); [cite: 49]

    if (!IMU_Init()) { [cite: 49]
        Serial.println(F("CRITICAL ERROR: IMU Hardware Initialization Failed. System Halted.")); [cite: 50]
        while (1); [cite: 50, 51]
    }

    // Định thời ngắt cứng chính xác 5ms (200Hz)
    timer = timerBegin(0, 80, true); [cite: 52]
    timerAttachInterrupt(timer, &onTimer, true); [cite: 52]
    timerAlarmWrite(timer, 5000, true); [cite: 52]
    timerAlarmEnable(timer); [cite: 52]
}

void loop() {
    if (!samplingTriggered) return;
    samplingTriggered = false; // Xóa cờ ngắt phần cứng ngay lập tức để giải phóng chu kỳ [cite: 53]

    AccelData rawData; [cite: 54]
    if (IMU_ReadAcceleration(rawData)) { [cite: 54]
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        
        apply_iir_filter(rawData, fx, fy, fz);
        accumulate_metrics(fx, fy, fz);
        
        // Khi gom đủ một cửa sổ dữ liệu thực tế, tiến hành rút trích đặc trưng tần số và định danh lỗi
        if (fft_vImag[WINDOW_SIZE - 1] >= 1.0f) { // Sử dụng phần ảo làm cờ báo đầy mảng (Tiết kiệm biến đếm)
            compute_features_and_infer();
        }
    }
}

// --- MODULE 1: GIAO TIẾP VÀ KHỞI TẠO PHẦN CỨNG SENSOR ---
bool IMU_Init() {
    Wire.begin(I2C_SDA, I2C_SCL); [cite: 11]
    Wire.setClock(400000); // 400kHz Fast-mode tối ưu I2C Bus [cite: 11]
    delay(100); [cite: 11]

    Wire.beginTransmission(IMU_ADDRESS); [cite: 12]
    Wire.write(IMU_REG_PWR_MGMT_1); [cite: 12]
    Wire.write(0x00); // Kích hoạt nguồn cảm biến [cite: 12]
    if (Wire.endTransmission() != 0) return false; [cite: 12]
    delay(50); [cite: 13]

    Wire.beginTransmission(IMU_ADDRESS); [cite: 13]
    Wire.write(IMU_REG_ACCEL_CONFIG); [cite: 13]
    Wire.write(0x00); // Khóa cứng dải đo gia tốc toàn thang +/-2g [cite: 13]
    if (Wire.endTransmission() != 0) return false; [cite: 13]
    delay(50); [cite: 14]

    return true; [cite: 14]
}

bool IMU_ReadAcceleration(AccelData &data) {
    Wire.beginTransmission(IMU_ADDRESS); [cite: 14]
    Wire.write(IMU_REG_ACCEL_XOUT_H); [cite: 14]
    if (Wire.endTransmission(false) != 0) return false; [cite: 15]
    if (Wire.requestFrom(IMU_ADDRESS, 6) < 6) return false; [cite: 16]

    data.x = (Wire.read() << 8) | Wire.read(); [cite: 16, 17]
    data.y = (Wire.read() << 8) | Wire.read(); [cite: 17]
    data.z = (Wire.read() << 8) | Wire.read(); [cite: 17]
    return true; [cite: 17]
}

// --- MODULE 2: XỬ LÝ TÍN HIỆU SỐ (DSP) ---
void apply_iir_filter(const AccelData &raw, float &fx, float &fy, float &fz) {
    const float HPF_ALPHA = 0.9843f; // Tần số cắt ~0.5Hz ở chu kỳ 200Hz [cite: 3, 4]
    static float p_rx = 0.0f, p_fx = 0.0f; [cite: 4]
    static float p_ry = 0.0f, p_fy = 0.0f; [cite: 5]
    static float p_rz = 0.0f, p_fz = 0.0f; [cite: 5]

    float crx = (float)raw.x * ACCEL_SCALE; [cite: 54, 55]
    float cry = (float)raw.y * ACCEL_SCALE; [cite: 55]
    float crz = (float)raw.z * ACCEL_SCALE; [cite: 56]

    // Thực thi bộ lọc thông cao IIR để bảo vệ hệ thống trước hiện tượng lệch DC (Drift) tĩnh [cite: 56]
    fx = HPF_ALPHA * (p_fx + crx - p_rx); [cite: 57]
    fy = HPF_ALPHA * (p_fy + cry - p_ry); [cite: 57]
    fz = HPF_ALPHA * (p_fz + crz - p_rz); [cite: 58]

    p_rx = crx; p_fx = fx; [cite: 58, 59]
    p_ry = cry; p_fy = fy; [cite: 59]
    p_rz = crz; p_fz = fz; [cite: 59, 60]
}

void accumulate_metrics(float fx, float fy, float fz) {
    static int idx = 0;
    static float sSqX = 0, sSqY = 0, sSqZ = 0;

    sSqX += fx * fx; [cite: 60, 61]
    sSqY += fy * fy; [cite: 61]
    sSqZ += fz * fz; [cite: 61]

    // Tích lũy chuỗi thời gian trục Z trực tiếp vào bộ đệm của FFT [cite: 61]
    fft_vReal[idx] = fz;
    fft_vImag[idx] = 0.0f;
    idx++;

    if (idx >= WINDOW_SIZE) {
        idx = 0;
        // Đánh dấu mảng đã đầy bằng cách mượn phần tử cuối của mảng ảo (Tiết kiệm RAM biến đếm)
        fft_vImag[WINDOW_SIZE - 1] = 999.0f; 
        
        // Lưu giữ tạm thời kết quả tổng bình phương miền thời gian
        fft_vReal[0] = sSqX; 
        fft_vReal[1] = sSqY; 
        fft_vReal[2] = sSqZ;
        
        sSqX = 0.0f; sSqY = 0.0f; sSqZ = 0.0f; [cite: 66, 67]
    }
}

void compute_features_and_infer() {
    // Phục hồi tổng bình phương miền thời gian từ bộ đệm tạm thời
    float sSqX = fft_vReal[0];
    float sSqY = fft_vReal[1];
    float sSqZ = fft_vReal[2];

    float rmsX = sqrt(sSqX / (float)WINDOW_SIZE); [cite: 62, 63]
    float rmsY = sqrt(sSqY / (float)WINDOW_SIZE); [cite: 63]
    float rmsZ = sqrt(sSqZ / (float)WINDOW_SIZE); [cite: 63, 64]

    // Chạy giải thuật FFT miền tần số cho riêng trục Z
    FFT.windowing(FFT_WIN_TYP_HANN, FFT_FORWARD); 
    FFT.compute(FFT_FORWARD);
    FFT.complexToMagnitude();

    float bpLow = 0.0f, bpMid = 0.0f, bpHigh = 0.0f;
    const float bin_width = SAMPLING_FREQ / (float)WINDOW_SIZE;

    // Trích xuất Band Power với bước Chuẩn hóa Biên độ bắt buộc (Chia cho N) để khớp 100% với Python
    for (int i = 0; i < WINDOW_SIZE / 2; i++) {
        float freq = (float)i * bin_width;
        float norm_mag = fft_vReal[i] / (float)WINDOW_SIZE; 
        float magSq = norm_mag * norm_mag;

        if (freq >= 0.0f && freq < 20.0f)        bpLow += magSq;
        else if (freq >= 20.0f && freq < 60.0f)  bpMid += magSq;
        else if (freq >= 60.0f && freq <= 100.0f) bpHigh += magSq;
    }

    // Thực thi đo đạc hiệu năng và suy luận mô hình ML
    unsigned long start_bench = micros(); [cite: 64]
    predict_logistic_regression(rmsX, rmsY, rmsZ, bpLow, bpMid, bpHigh); [cite: 65]
    inference_latency_us = micros() - start_bench; [cite: 65]

    print_dashboard(rmsX, rmsY, rmsZ, bpLow, bpMid, bpHigh); [cite: 65]
}

// --- MODULE 3: SUY LUẬN LOGISTIC REGRESSION (SOFTMAX ỔN ĐỊNH SỐ HỌC) ---
void predict_logistic_regression(float rx, float ry, float rz, float bLow, float bMid, float bHigh) {
    float raw_features[6] = {rx, ry, rz, bLow, bMid, bHigh}; [cite: 18, 19]
    float normalized_features[6]; [cite: 19]
    float logit_scores[MODEL_NUM_CLASSES] = {0.0f}; [cite: 19]

    // 1. Z-score Normalization (Khớp hoàn toàn cấu trúc StandardScaler của Python)
    for (int f = 0; f < 6; f++) {
        normalized_features[f] = (raw_features[f] - SCALER_MEAN[f]) / SCALER_STD[f]; [cite: 19, 20]
    }

    // 2. Nhân ma trận tính điểm số Logit (One-vs-Rest) & tìm Max Score để chống tràn số toán học [cite: 21]
    float max_score = -999999.0f; [cite: 20]
    for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
        float score = LGR_BIASES[c]; [cite: 21, 22]
        for (int f = 0; f < 6; f++) {
            score += normalized_features[f] * LGR_WEIGHTS[c][f]; [cite: 22, 23]
        }
        logit_scores[c] = score; [cite: 23]
        if (score > max_score) max_score = score; [cite: 24]
    }

    // 3. Phép toán Argmax tìm chỉ số lớp có điểm cao nhất
    predicted_class_idx = 0; [cite: 26]
    float current_max = logit_scores[0]; [cite: 27]
    for (int c = 1; c < MODEL_NUM_CLASSES; c++) {
        if (logit_scores[c] > current_max) { [cite: 27]
            current_max = logit_scores[c]; [cite: 27]
            predicted_class_idx = c; [cite: 28]
        }
    }

    // 4. Hàm SOFTMAX phi tuyến chuẩn định dạng toán học (Chống tràn số bằng Max-subtraction)
    float sum_exp = 0.0f;
    for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
        class_probabilities[c] = exp(logit_scores[c] - max_score); 
        sum_exp += class_probabilities[c]; [cite: 32]
    }

    if (sum_exp > 0.0f) {
        for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
            class_probabilities[c] /= sum_exp; [cite: 33]
        }
    } else {
        // Dự phòng biên an toàn nếu có lỗi tính toán nghiêm trọng
        for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
            class_probabilities[c] = 1.0f / (float)MODEL_NUM_CLASSES; [cite: 34]
        }
    }
}

// --- MODULE 4: HÀM HIỂN THỊ DASHBOARD TERMINAL ---
void print_dashboard(float rx, float ry, float rz, float bLow, float bMid, float bHigh) {
    Serial.print("\033[2J"); [cite: 35]
    Serial.print("\033[H"); [cite: 35]

    Serial.println(F("====================================================================="));
    Serial.println(F("[1] EDGE SIGNAL PROCESSING (Cửa sổ phẳng 256 mẫu - Khử Trọng Lực)"));
    Serial.printf("- RMS Trục X: %.4fG  |  RMS Trục Y: %.4fG  |  RMS Trục Z: %.4fG\n", rx, ry, rz); [cite: 36, 37]
    Serial.printf("- BandPower Low: %.4f | Mid: %.4f | High: %.4f\n", bLow, bMid, bHigh);
    Serial.println(); [cite: 37]

    Serial.println(F("[2] CHUẨN SOFTMAX ON-CHIP SUY LUẬN (Logistic Regression)")); [cite: 37]
    for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
        int bar_length = (int)(class_probabilities[c] * 20.0f); [cite: 38]
        if (bar_length > 20) bar_length = 20; [cite: 38, 39]
        if (bar_length < 0)  bar_length = 0; [cite: 39, 40]
        
        Serial.printf("- Trạng thái %-15s : [", MODEL_CLASS_LABELS[c]); [cite: 40, 41]
        for (int i = 0; i < 20; i++) {
            Serial.print(i < bar_length ? "|" : "."); [cite: 41, 42]
        }
        Serial.printf("] %.1f%%\n", class_probabilities[c] * 100.0f); [cite: 42]
    }
    Serial.println(); [cite: 43]
    Serial.printf(" => KẾT LUẬN CHẨN ĐOÁN: QUẠT CHẠY Ở TRẠNG THÁI -> [%s]\n", MODEL_CLASS_LABELS[predicted_class_idx]); [cite: 43, 44]
    Serial.println(); [cite: 44]

    Serial.println(F("[3] RESOURCE BENCHMARK")); [cite: 44]
    Serial.printf("- Trễ tính toán thuật toán AI (Inference Latency): %d us\n", inference_latency_us); [cite: 45]
    Serial.printf("- Free Heap RAM hiện tại của ESP32               : %d bytes\n", esp_get_free_heap_size()); [cite: 47]
    Serial.println(F("=====================================================================")); [cite: 48]
}
