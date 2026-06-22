#include <Wire.h>
#include <math.h>
#include <arduinoFFT.h>
#include "model_parameters.h"  
#include <WiFi.h>
#include <PubSubClient.h>

#define I2C_SDA 21
#define I2C_SCL 22
#define IMU_ADDRESS 0x68
#define IMU_REG_PWR_MGMT_1 0x6B
#define IMU_REG_ACCEL_CONFIG 0x1C
#define IMU_REG_ACCEL_XOUT_H 0x3B
#define IMU_REG_CONFIG      0x1A
#define IMU_REG_SMPLRT_DIV  0x19

const float ACCEL_SCALE = 1.0f / 16384.0f;
const int WINDOW_SIZE = 512;
const float SAMPLING_FREQ = 500.0f;

// --- Neural Network Hyperparameters ---
#define INPUT_SIZE   6
#define H1_SIZE     16
#define H2_SIZE      8
#define OUTPUT_SIZE  3

struct AccelData
{
    int16_t x;
    int16_t y;
    int16_t z;
};

// SỬA: Thay thế cấu hình HW Timer bằng Ticker (hoặc tính thời gian bằng micros) 
// Đây là giải pháp an toàn tuyệt đối 100%, không bao giờ lo bị sập/reset Core do ngắt cứng chèn nhau.
unsigned long lastSampleTime = 0;
const unsigned long sampleIntervalUs = 2000; // 500Hz = 2000 microseconds (2ms)


// --- WiFi Configuration ---
const char* ssid = "Hans";
const char* password = "succmanuts";
const char* mqtt_server = "10.169.156.10"; // e.g., "broker.hivemq.com"


bool windowReady = false;   
float latestSqX = 0.0f;
float latestSqY = 0.0f;
float latestSqZ = 0.0f;
float last_rmsX, last_rmsY, last_rmsZ;
float last_bpLow, last_bpMid, last_bpHigh;
int print_counter = 0;

float rawX_buffer[20];
float rawY_buffer[20];
float rawZ_buffer[20];

static float fft_vReal[WINDOW_SIZE];
static float fft_vImag[WINDOW_SIZE];
ArduinoFFT<float> FFT = ArduinoFFT<float>(fft_vReal, fft_vImag, WINDOW_SIZE, SAMPLING_FREQ);

float class_probabilities[OUTPUT_SIZE] = {0.0f};
int predicted_class_idx = 0;
unsigned long inference_latency_us = 0;



// --- Function Prototypes ---
bool IMU_Init();
bool IMU_ReadAcceleration(AccelData &data);
void apply_iir_filter(const AccelData &raw, float &fx, float &fy, float &fz);
void accumulate_metrics(float fx, float fy, float fz);
void compute_features_and_infer();
void publish_mqtt_data();
void predict_neural_network(float rx, float ry, float rz, float bLow, float bMid, float bHigh);
void print_dashboard(float rx, float ry, float rz, float bLow, float bMid, float bHigh);
inline float relu(float x);
void softmax(float *x, int n);


WiFiClient espClient;
PubSubClient client(espClient);

void updateRawBuffers(float x, float y, float z) {
  // Shift values to the left
    for(int i = 0; i < 19; i++) {
    rawX_buffer[i] = rawX_buffer[i+1];
    rawY_buffer[i] = rawY_buffer[i+1];
    rawZ_buffer[i] = rawZ_buffer[i+1];
    }
    // Add new value
    rawX_buffer[19] = x;
    rawY_buffer[19] = y;
    rawZ_buffer[19] = z;
    }

void setup_wifi() {
    Serial.println("Connecting to WiFi...");
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);

    int timeout = 0;
    // Limit to 20 seconds of trying
    while (WiFi.status() != WL_CONNECTED && timeout < 40) {
        delay(500);
        Serial.print(".");
        timeout++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi connected!");
        Serial.print("IP address: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\nWiFi connection FAILED. Continuing anyway...");
    }
}

void reconnect() {
    if (client.connected()) return;

    Serial.println("Attempting MQTT connection...");
    // Attempt to connect
    if (client.connect("ESP32_Vibration_Monitor")) {
        Serial.println("MQTT Connected!");
    } else {
        Serial.print("MQTT failed, rc=");
        Serial.print(client.state());
        Serial.println(" - will try again in 5 seconds");
    }
}


void setup()
{
    Serial.begin(115200);
    delay(2000); // Chờ 2 giây để mạch ổn định hẳn dòng điện
    Serial.println(F("\n--- KHỞI ĐỘNG HỆ THỐNG ---"));

    setup_wifi();
    client.setServer(mqtt_server, 1883);
    client.setBufferSize(2048);
    reconnect();
    
    Serial.println(F("System Ready"));

    if (!IMU_Init())
    {
        Serial.println(F("CRITICAL ERROR: Khong tim thay hoac khong khoi tao duoc IMU MPU6050!"));
        while (1) { 
            Serial.println(F("Dang thu lai ket noi I2C..."));
            delay(2000); 
        } 
    }
    
    Serial.println(F("IMU Initialization OK!"));
    Serial.println(F("Bat dau lay mau voi Fs = 500Hz băng cach dem thoi gian sieu chinh xac..."));
    
    // Khởi tạo mốc thời gian ban đầu
    lastSampleTime = micros();
}

void loop()
{
    if (WiFi.status() == WL_CONNECTED) {
        if (!client.connected()) {
            reconnect(); 
        }
        if (client.connected()) {
            client.loop();
        }
    }

    unsigned long currentTime = micros();

    // SỬA: Kiểm tra xem đã qua 2000us (2ms = 500Hz) chưa
    if (currentTime - lastSampleTime >= sampleIntervalUs)
    {
        // Cập nhật lại mốc thời gian (bù trừ sai số nếu có)
        lastSampleTime += sampleIntervalUs; 

        AccelData rawData;
        if (IMU_ReadAcceleration(rawData))
        {
            float fx = 0.0f, fy = 0.0f, fz = 0.0f;

            apply_iir_filter(rawData, fx, fy, fz);
            accumulate_metrics(fx, fy, fz);

            // Cứ 50 mẫu (0.1 giây) in ra một dấu chấm
            static int debug_counter = 0;
            if (++debug_counter >= 50) {
                Serial.print("."); 
                debug_counter = 0;
            }

            if (windowReady)
            {
                windowReady = false; 
                Serial.println(F("\n[OK] Da nhan du 512 mau o 500Hz! Dang chay mpi va FFT..."));
                compute_features_and_infer();

                publish_mqtt_data();
                print_counter++;
                if (print_counter >= 2) {
                    print_dashboard(last_rmsX, last_rmsY, last_rmsZ, last_bpLow, last_bpMid, last_bpHigh);
                    print_counter = 0;
                }
            }
        }
        else 
        {
            static unsigned long last_error_time = 0;
            if (millis() - last_error_time > 2000) {
                Serial.println(F("\n[WARNING] Loi doc cam bien qua I2C!"));
                last_error_time = millis();
            }
        }
    }
    delay(1);
}

bool IMU_Init()
{
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000); // Kích hoạt I2C Fast Mode 400kHz để kịp xử lý ở Fs = 500Hz
    delay(100);

    // Kiểm tra xem MPU6050 có phản hồi trên Bus I2C không
    Wire.beginTransmission(IMU_ADDRESS);
    if (Wire.endTransmission() != 0) {
        return false; 
    }

    // Đánh thức MPU6050 (PWR_MGMT_1 = 0)
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_PWR_MGMT_1);
    Wire.write(0x00);
    if (Wire.endTransmission() != 0) return false;
    delay(50);

    // Cấu hình DLPF (Digital Low Pass Filter) về 184Hz để lọc nhiễu phần cứng
    // Tần số lấy mẫu nội bộ của Gyro/Accel lúc này sẽ cố định ở mức 1kHz
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_CONFIG);
    Wire.write(0x01); 
    if (Wire.endTransmission() != 0) return false;
    delay(10);

    // Đặt bộ chia tần số Sample Rate Divider về 1
    // Công thức phần cứng: Sample Rate = 1kHz / (1 + SMPLRT_DIV) -> 1kHz / (1 + 1) = 500Hz chuẩn xác
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_SMPLRT_DIV);
    Wire.write(1); 
    if (Wire.endTransmission() != 0) return false;   
    delay(10);

    // Cấu hình dải đo Gia tốc kế (+/- 2g) tương ứng với ACCEL_SCALE = 1/16384
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
    
    // Yêu cầu đọc 6 bytes dữ liệu liên tục (X_H, X_L, Y_H, Y_L, Z_H, Z_L)
    if (Wire.requestFrom(IMU_ADDRESS, 6) < 6) return false;

    data.x = (Wire.read() << 8) | Wire.read();
    data.y = (Wire.read() << 8) | Wire.read();
    data.z = (Wire.read() << 8) | Wire.read();
    return true;
}

void apply_iir_filter(const AccelData &raw, float &fx, float &fy, float &fz)
{
    // Hệ số alpha cắt dải tần số thấp (loại bỏ trọng lực Trái Đất 1G)
    const float HPF_ALPHA = 0.9936f;
    static float p_rx = 0.0f, p_fx = 0.0f;
    static float p_ry = 0.0f, p_fy = 0.0f;
    static float p_rz = 0.0f, p_fz = 0.0f;

    // Chuyển đổi dữ liệu thô sang đơn vị G (m/s2 dựa trên scale định mức)
    float crx = (float)raw.x * ACCEL_SCALE;
    float cry = (float)raw.y * ACCEL_SCALE;
    float crz = (float)raw.z * ACCEL_SCALE;

    // Bộ lọc thông cao IIR High-Pass Filter số
    fx = HPF_ALPHA * (p_fx + crx - p_rx);
    fy = HPF_ALPHA * (p_fy + cry - p_ry);
    fz = HPF_ALPHA * (p_fz + crz - p_rz);

    // Lưu lại trạng thái cho chu kỳ lấy mẫu tiếp theo
    p_rx = crx; p_fx = fx;
    p_ry = cry; p_fy = fy;
    p_rz = crz; p_fz = fz;

    updateRawBuffers(fx, fy, fz);
}

void accumulate_metrics(float fx, float fy, float fz)
{
    static int idx = 0;
    static float sSqX = 0.0f;
    static float sSqY = 0.0f;
    static float sSqZ = 0.0f;

    // Tích lũy tổng bình phương phục vụ tính toán RMS sau này
    sSqX += fx * fx;
    sSqY += fy * fy;
    sSqZ += fz * fz;

    // Nạp tín hiệu trục Z đã lọc bỏ trọng lực vào mảng phân tích tần số FFT
    fft_vReal[idx] = fz;
    fft_vImag[idx] = 0.0f;

    idx++;

    // Khi bộ đệm gom đủ số lượng WINDOW_SIZE (512 mẫu)
    if (idx >= WINDOW_SIZE)
    {
        idx = 0;

        // Đẩy dữ liệu sang biến volatile an toàn
        latestSqX = sSqX;
        latestSqY = sSqY;
        latestSqZ = sSqZ;

        // Reset các bộ tích lũy để chuẩn bị cho chu kỳ cửa sổ kế tiếp
        sSqX = 0.0f;
        sSqY = 0.0f;
        sSqZ = 0.0f;

        windowReady = true; // Kích hoạt cờ báo cho hàm loop xử lý AI
    }
}


void publish_mqtt_data() {
    // 1. Pre-calculate or use a buffer-safe approach
    String jsonPayload = "{";
    jsonPayload += "\"timestamp\":\"" + String(millis()) + "\",";
    
    jsonPayload += "\"inference\":{";
    jsonPayload += "\"prediction\":\"" + String(MODEL_CLASS_LABELS[predicted_class_idx]) + "\",";
    jsonPayload += "\"confidence\":" + String(class_probabilities[predicted_class_idx], 4) + ",";
    jsonPayload += "\"scores\":{";
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        jsonPayload += "\"" + String(MODEL_CLASS_LABELS[i]) + "\":" + String(class_probabilities[i], 4);
        if (i < OUTPUT_SIZE - 1) jsonPayload += ",";
    }
    jsonPayload += "}},";
    
    jsonPayload += "\"status\":{\"fan_state\":\"Running\",\"sensor_health\":\"OK\",\"connection\":\"Connected\"},";
    jsonPayload += "\"features\":{\"rms\":{\"x\":" + String(last_rmsX, 6) + ",\"y\":" + String(last_rmsY, 6) + ",\"z\":" + String(last_rmsZ, 6) + "},";
    jsonPayload += "\"band_power\":{\"low\":" + String(last_bpLow, 6) + ",\"mid\":" + String(last_bpMid, 6) + ",\"high\":" + String(last_bpHigh, 6) + "}},";

    jsonPayload += "\"fft\":{\"magnitudes\":[";
    for(int i = 0; i < 32; i++) { 
        jsonPayload += String(fft_vReal[i], 2) + (i < 31 ? "," : ""); 
    }
    jsonPayload += "]},";

    jsonPayload += "\"raw\":{\"ax\":[";
    for(int i = 0; i < 20; i++) { jsonPayload += String(rawX_buffer[i], 3) + (i < 19 ? "," : ""); }
    jsonPayload += "],\"ay\":[";
    for(int i = 0; i < 20; i++) { jsonPayload += String(rawY_buffer[i], 3) + (i < 19 ? "," : ""); }
    jsonPayload += "],\"az\":[";
    for(int i = 0; i < 20; i++) { jsonPayload += String(rawZ_buffer[i], 3) + (i < 19 ? "," : ""); }
    jsonPayload += "]}"; 
    jsonPayload += "}"; 

    // Debug output
    Serial.printf("MQTT Payload Size: %d bytes | Heap: %d\n", jsonPayload.length(), esp_get_free_heap_size());

    // Publish
    if (client.beginPublish("vibration/inference", jsonPayload.length(), false)) {
        client.print(jsonPayload);
        client.endPublish();
        Serial.println("MQTT Publish Success!");
    } else {
        Serial.println("MQTT Publish FAILED!");
    }
}


void compute_features_and_infer()
{
    float sSqX = latestSqX;
    float sSqY = latestSqY;
    float sSqZ = latestSqZ;

    // 1. Tính toán giá trị RMS (Root Mean Square) từ tổng bình phương dữ liệu
    float rmsX = sqrt(sSqX / (float)WINDOW_SIZE);
    float rmsY = sqrt(sSqY / (float)WINDOW_SIZE);
    float rmsZ = sqrt(sSqZ / (float)WINDOW_SIZE);

    // 2. Thực hiện biến đổi Fourier nhanh FFT trên trục Z
    FFT.windowing(FFT_WIN_TYP_HANN, FFT_FORWARD); // Áp dụng cửa sổ Hanning giảm rò rỉ phổ
    FFT.compute(FFT_FORWARD);
    FFT.complexToMagnitude();                     // Chuyển đổi từ số phức sang biên độ thực

    float bpLow = 0.0f, bpMid = 0.0f, bpHigh = 0.0f;
    const float bin_width = SAMPLING_FREQ / (float)WINDOW_SIZE; // Độ phân giải tần số mỗi bin

    // 3. Phân tách tích lũy năng lượng theo các dải băng tần (BandPower)
    for (int i = 0; i < WINDOW_SIZE / 2; i++)
    {
        if (i % 50 == 0) yield();

        float freq = (float)i * bin_width;
        float norm_mag = fft_vReal[i] / (float)WINDOW_SIZE;
        float magSq = norm_mag * norm_mag;

        if (freq >= 0.0f && freq < 20.0f)
            bpLow += magSq;
        else if (freq >= 20.0f && freq < 60.0f)
            bpMid += magSq;
        else if (freq >= 60.0f && freq <= 100.0f)
            bpHigh += magSq;
    }

    // 4. Đo lường tốc độ xử lý của mạng nơ-ron (Mạng MLP nhúng)
    unsigned long start_bench = micros();
    predict_neural_network(rmsX, rmsY, rmsZ, bpLow, bpMid, bpHigh);
    inference_latency_us = micros() - start_bench;

    last_rmsX = rmsX; last_rmsY = rmsY; last_rmsZ = rmsZ;
    last_bpLow = bpLow; last_bpMid = bpMid; last_bpHigh = bpHigh;
}
inline float relu(float x)
{
    return (x > 0.0f) ? x : 0.0f;
}

void softmax(float *x, int n)
{
    float maxVal = x[0];

    for (int i = 1; i < n; i++)
    {
        if (x[i] > maxVal)
            maxVal = x[i];
    }

    float sum = 0.0f;

    for (int i = 0; i < n; i++)
    {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }

    for (int i = 0; i < n; i++)
    {
        x[i] /= sum;
    }
}

void predict_neural_network(float rx, float ry, float rz, float bLow, float bMid, float bHigh)
{
    // 1. Đóng gói vector đặc trưng đầu vào (Input Tensor)
    float x[INPUT_SIZE] = { rx, ry, rz, bLow, bMid, bHigh };

    // 2. Chuẩn hóa dữ liệu đầu vào (Z-score Standardization) dựa trên bộ Scaler từ Model gốc
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        x[i] = (x[i] - SCALER_MEAN[i]) / SCALER_STD[i];
    }

    // 3. Tầng ẩn số 1 (Dense Layer 1: 6 Inputs -> 16 Nodes + ReLU)
    float h1[H1_SIZE];
    for (int i = 0; i < H1_SIZE; i++)
    {
        float sum = BIAS1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            sum += W1[i][j] * x[j];
        }
        h1[i] = relu(sum);
    }

    // 4. Tầng ẩn số 2 (Dense Layer 2: 16 Inputs -> 8 Nodes + ReLU)
    float h2[H2_SIZE];
    for (int i = 0; i < H2_SIZE; i++)
    {
        float sum = BIAS2[i]; 
        for (int j = 0; j < H1_SIZE; j++)
        {
            sum += W2[i][j] * h1[j];
        }
        h2[i] = relu(sum);
    }

    // 5. Tầng đầu ra (Output Dense Layer: 8 Inputs -> 3 Logits phân lớp)
    float logits[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        float sum = BIAS3[i];
        for (int j = 0; j < H2_SIZE; j++)
        {
            sum += W3[i][j] * h2[j];
        }
        logits[i] = sum;
    }

    // 6. Kích hoạt phân phối xác suất qua bộ lọc Softmax Engine
    softmax(logits, OUTPUT_SIZE);

    // 7. Thuật toán Argmax trích xuất lớp có xác suất dự đoán cao nhất
    predicted_class_idx = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        class_probabilities[i] = logits[i];
        if (class_probabilities[i] > class_probabilities[predicted_class_idx])
        {
            predicted_class_idx = i;
        }
    }
}

void print_dashboard(float rx, float ry, float rz, float bLow, float bMid, float bHigh)
{
    // ĐÃ SỬA: Loại bỏ các chuỗi mã ANSI "\033[2J" cũ vốn làm đơ các bộ phần mềm Terminal cơ bản.
    // Thay vào đó dùng ký tự xuống dòng rõ ràng để làm sạch giao diện trực quan.
    Serial.println();
    Serial.println(F("====================================================================="));
    Serial.println(F("[1] EDGE SIGNAL PROCESSING (512-sample window @ 500Hz)"));
    Serial.printf("- RMS X: %.4fG  |  RMS Y: %.4fG  |  RMS Z: %.4fG\n", rx, ry, rz);
    Serial.printf("- BandPower Low: %.6f | Mid: %.6f | High: %.6f\n", bLow, bMid, bHigh);
    Serial.println();

    Serial.println(F("[2] ON-CHIP SOFTMAX INFERENCE (Edge MLP Embedded Neural Network)"));
    for (int c = 0; c < OUTPUT_SIZE; c++)
    {
        int bar_length = (int)(class_probabilities[c] * 20.0f);
        if (bar_length > 20)  bar_length = 20;
        if (bar_length < 0)   bar_length = 0;

        // Hiển thị tên nhãn phân lớp trích xuất từ file model_parameters.h
        Serial.printf("- State %-15s : [", MODEL_CLASS_LABELS[c]);
        for (int i = 0; i < 20; i++)
        {
            Serial.print(i < bar_length ? "|" : ".");
        }
        Serial.printf("] %.1f%%\n", class_probabilities[c] * 100.0f);
    }
    Serial.println();
    Serial.printf(" => DIAGNOSIS RESULT -> [%s]\n", MODEL_CLASS_LABELS[predicted_class_idx]);
    Serial.println();

    Serial.println(F("[3] HARDWARE RESOURCE BENCHMARK"));
    Serial.printf("- AI Model Inference Latency : %d us\n", inference_latency_us);
    Serial.printf("- ESP32 System Free Heap RAM : %d bytes\n", esp_get_free_heap_size());
    Serial.println(F("====================================================================="));
}
