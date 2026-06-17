#include <Wire.h>
#include <math.h>
#include <arduinoFFT.h>
#include "model_parameters.h"

// ============================================================
//  CẤU HÌNH HARDWARE
// ============================================================
#define I2C_SDA               21
#define I2C_SCL               22
#define IMU_ADDRESS           0x68
#define IMU_REG_PWR_MGMT_1    0x6B
#define IMU_REG_ACCEL_CONFIG  0x1C
#define IMU_REG_ACCEL_XOUT_H  0x3B
#define IMU_REG_CONFIG        0x1A
#define IMU_REG_SMPLRT_DIV    0x19

// ============================================================
//  THÔNG SỐ LẤY MẪU
// ============================================================
#define WINDOW_SIZE           512
#define SAMPLING_FREQ         500.0f
#define SAMPLE_INTERVAL_US    2000    // 500Hz
#define I2C_TIMEOUT_MS        50
#define MAX_I2C_ERRORS        10

// ============================================================
//  KÍCH THƯỚC MẠNG NEURAL (4 LAYERS)
// ============================================================
#define INPUT_SIZE   6
#define H1_SIZE      32
#define H2_SIZE      16
#define H3_SIZE      8
#define OUTPUT_SIZE  3

// ============================================================
//  BỘ LỌC HPF + LPF CASCADE
// ============================================================
static const float HPF_ALPHA = 0.9936f;
static const float LPF_ALPHA = 0.571f;
static const float ACCEL_SCALE = 1.0f / 16384.0f;

static float prev_raw_x = 0.0f, prev_filt_x = 0.0f, prev_lpf_x = 0.0f;
static float prev_raw_y = 0.0f, prev_filt_y = 0.0f, prev_lpf_y = 0.0f;
static float prev_raw_z = 0.0f, prev_filt_z = 0.0f, prev_lpf_z = 0.0f;

// ============================================================
//  BỘ NHỚ ĐỆM (STATIC - TRÁNH STACK OVERFLOW)
// ============================================================
static float fft_vReal[WINDOW_SIZE];
static float fft_vImag[WINDOW_SIZE];
static ArduinoFFT<float> FFT(fft_vReal, fft_vImag, WINDOW_SIZE, SAMPLING_FREQ);

// Neural Network buffers
static float nn_input[INPUT_SIZE];
static float nn_h1[H1_SIZE];
static float nn_h2[H2_SIZE];
static float nn_h3[H3_SIZE];
static float nn_logits[OUTPUT_SIZE];
static float nn_probs[OUTPUT_SIZE];

// ============================================================
//  TRẠNG THÁI HỆ THỐNG
// ============================================================
struct SystemState {
    unsigned long lastSampleTime;
    unsigned long inferenceTime;
    uint32_t errorCount;
    uint32_t windowIndex;
    uint32_t inferenceCount;
    float sqX, sqY, sqZ;
    float latestSqX, latestSqY, latestSqZ;
    bool windowReady;
    bool isCollecting;
    int predictedClass;
    int debugCounter;
} state = {
    .lastSampleTime = 0,
    .inferenceTime = 0,
    .errorCount = 0,
    .windowIndex = 0,
    .inferenceCount = 0,
    .sqX = 0.0f, .sqY = 0.0f, .sqZ = 0.0f,
    .latestSqX = 0.0f, .latestSqY = 0.0f, .latestSqZ = 0.0f,
    .windowReady = false,
    .isCollecting = true,
    .predictedClass = 0,
    .debugCounter = 0
};

// ============================================================
//  PROTOTYPES
// ============================================================
bool IMU_Init();
bool IMU_ReadAcceleration(int16_t &x, int16_t &y, int16_t &z);
void apply_advanced_filter(int16_t rawX, int16_t rawY, int16_t rawZ, 
                          float &fx, float &fy, float &fz);
void accumulate_metrics(float fx, float fy, float fz);
void compute_features_and_infer();
void predict_neural_network(float rx, float ry, float rz, 
                           float bLow, float bMid, float bHigh);
void print_dashboard(float rx, float ry, float rz, 
                    float bLow, float bMid, float bHigh);
inline float relu(float x);
void softmax(float *x, int n);

// ============================================================
//  SETUP
// ============================================================
void setup()
{
    Serial.begin(115200);
    delay(2000);
    
    Serial.println(F("\n========================================"));
    Serial.println(F("  ESP32 EDGE AI - 4 LAYERS NN"));
    Serial.println(F("========================================"));
    Serial.printf("Free Heap: %d bytes\n", ESP.getFreeHeap());
    Serial.println();
    
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000);
    Wire.setTimeout(I2C_TIMEOUT_MS);
    delay(100);
    
    if (!IMU_Init())
    {
        Serial.println(F("[ERROR] IMU initialization failed!"));
        while (1) {
            delay(1000);
            Serial.print(".");
        }
    }
    
    state.lastSampleTime = micros();
    Serial.println(F("[OK] System ready! Sampling at 500Hz...\n"));
    Serial.println(F("[DEBUG] Waiting for 512 samples..."));
}

// ============================================================
//  MAIN LOOP
// ============================================================
void loop()
{
    unsigned long currentTime = micros();
    
    // DEBUG: In trạng thái mỗi 3 giây
    static unsigned long lastStatusTime = 0;
    if (millis() - lastStatusTime > 3000) {
        Serial.printf("[DEBUG] Window: %d/%d, Ready: %d, Errors: %d, Inf: %d\n", 
                     state.windowIndex, WINDOW_SIZE, 
                     state.windowReady, state.errorCount, state.inferenceCount);
        lastStatusTime = millis();
    }
    
    if (currentTime - state.lastSampleTime >= SAMPLE_INTERVAL_US)
    {
        state.lastSampleTime += SAMPLE_INTERVAL_US;
        
        int16_t ax, ay, az;
        if (IMU_ReadAcceleration(ax, ay, az))
        {
            state.errorCount = 0;
            
            float fx, fy, fz;
            apply_advanced_filter(ax, ay, az, fx, fy, fz);
            accumulate_metrics(fx, fy, fz);
            
            if (++state.debugCounter >= 50) {
                Serial.print(".");
                state.debugCounter = 0;
            }
            
            // Khi có đủ dữ liệu
            if (state.windowReady)
            {
                state.windowReady = false;
                state.isCollecting = false;
                Serial.println(F("\n[OK] Window ready! Running inference..."));
                compute_features_and_infer();
                
                // Reset để bắt đầu thu thập lại
                state.isCollecting = true;
                state.windowIndex = 0;
                state.sqX = 0.0f;
                state.sqY = 0.0f;
                state.sqZ = 0.0f;
                state.latestSqX = 0.0f;
                state.latestSqY = 0.0f;
                state.latestSqZ = 0.0f;
                state.windowReady = false;
                state.inferenceCount++;
                
                Serial.println(F("[DEBUG] Reset for next window...\n"));
            }
        }
        else
        {
            state.errorCount++;
            if (state.errorCount >= MAX_I2C_ERRORS)
            {
                Serial.println(F("\n[WARN] I2C error limit! Resetting bus..."));
                Wire.end();
                delay(10);
                Wire.begin(I2C_SDA, I2C_SCL);
                state.errorCount = 0;
            }
        }
    }
}

// ============================================================
//  IMU FUNCTIONS
// ============================================================
bool IMU_Init()
{
    Wire.beginTransmission(IMU_ADDRESS);
    if (Wire.endTransmission() != 0)
    {
        Serial.println(F("[ERROR] IMU not detected!"));
        return false;
    }
    
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_PWR_MGMT_1);
    Wire.write(0x00);
    if (Wire.endTransmission() != 0) return false;
    delay(50);
    
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_CONFIG);
    Wire.write(0x01);
    if (Wire.endTransmission() != 0) return false;
    delay(10);
    
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_SMPLRT_DIV);
    Wire.write(1);
    if (Wire.endTransmission() != 0) return false;
    delay(10);
    
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_CONFIG);
    Wire.write(0x00);
    if (Wire.endTransmission() != 0) return false;
    delay(50);
    
    return true;
}

bool IMU_ReadAcceleration(int16_t &x, int16_t &y, int16_t &z)
{
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_XOUT_H);
    if (Wire.endTransmission(false) != 0)
        return false;
    
    uint32_t start = millis();
    while (Wire.requestFrom(IMU_ADDRESS, 6) < 6)
    {
        if (millis() - start > I2C_TIMEOUT_MS)
            return false;
        delay(1);
    }
    
    x = (Wire.read() << 8) | Wire.read();
    y = (Wire.read() << 8) | Wire.read();
    z = (Wire.read() << 8) | Wire.read();
    return true;
}

// ============================================================
//  BỘ LỌC HPF + LPF CASCADE
// ============================================================
void apply_advanced_filter(int16_t rawX, int16_t rawY, int16_t rawZ,
                          float &fx, float &fy, float &fz)
{
    float crx = (float)rawX * ACCEL_SCALE;
    float cry = (float)rawY * ACCEL_SCALE;
    float crz = (float)rawZ * ACCEL_SCALE;
    
    float hx = HPF_ALPHA * (prev_filt_x + crx - prev_raw_x);
    float hy = HPF_ALPHA * (prev_filt_y + cry - prev_raw_y);
    float hz = HPF_ALPHA * (prev_filt_z + crz - prev_raw_z);
    
    fx = LPF_ALPHA * hx + (1.0f - LPF_ALPHA) * prev_lpf_x;
    fy = LPF_ALPHA * hy + (1.0f - LPF_ALPHA) * prev_lpf_y;
    fz = LPF_ALPHA * hz + (1.0f - LPF_ALPHA) * prev_lpf_z;
    
    prev_raw_x = crx;
    prev_filt_x = hx;
    prev_lpf_x = fx;
    
    prev_raw_y = cry;
    prev_filt_y = hy;
    prev_lpf_y = fy;
    
    prev_raw_z = crz;
    prev_filt_z = hz;
    prev_lpf_z = fz;
}

// ============================================================
//  TÍCH LŨY CHỈ SỐ
// ============================================================
void accumulate_metrics(float fx, float fy, float fz)
{
    state.sqX += fx * fx;
    state.sqY += fy * fy;
    state.sqZ += fz * fz;
    
    fft_vReal[state.windowIndex] = fz;
    fft_vImag[state.windowIndex] = 0.0f;
    
    state.windowIndex++;
    
    if (state.windowIndex >= WINDOW_SIZE)
    {
        state.latestSqX = state.sqX;
        state.latestSqY = state.sqY;
        state.latestSqZ = state.sqZ;
        state.windowReady = true;
    }
}

// ============================================================
//  TRÍCH XUẤT ĐẶC TRƯNG & DỰ ĐOÁN
// ============================================================
void compute_features_and_infer()
{
    float rmsX = sqrtf(state.latestSqX / (float)WINDOW_SIZE);
    float rmsY = sqrtf(state.latestSqY / (float)WINDOW_SIZE);
    float rmsZ = sqrtf(state.latestSqZ / (float)WINDOW_SIZE);
    
    FFT.windowing(FFT_WIN_TYP_HANN, FFT_FORWARD);
    FFT.compute(FFT_FORWARD);
    FFT.complexToMagnitude();
    
    float bpLow = 0.0f, bpMid = 0.0f, bpHigh = 0.0f;
    const float bin_width = SAMPLING_FREQ / (float)WINDOW_SIZE;
    
    for (int i = 0; i < WINDOW_SIZE / 2; i++)
    {
        float freq = (float)i * bin_width;
        float magSq = (fft_vReal[i] / WINDOW_SIZE) * (fft_vReal[i] / WINDOW_SIZE);
        
        if (freq < 20.0f)
            bpLow += magSq;
        else if (freq < 60.0f)
            bpMid += magSq;
        else if (freq <= 100.0f)
            bpHigh += magSq;
    }
    
    unsigned long start = micros();
    predict_neural_network(rmsX, rmsY, rmsZ, bpLow, bpMid, bpHigh);
    state.inferenceTime = micros() - start;
    
    print_dashboard(rmsX, rmsY, rmsZ, bpLow, bpMid, bpHigh);
}

// ============================================================
//  MẠNG NEURAL - 4 LAYERS
// ============================================================
inline float relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

void softmax(float *x, int n)
{
    float maxVal = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > maxVal) maxVal = x[i];
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }
    
    for (int i = 0; i < n; i++)
        x[i] /= sum;
}

void predict_neural_network(float rx, float ry, float rz, 
                           float bLow, float bMid, float bHigh)
{
    nn_input[0] = (rx - SCALER_MEAN[0]) / SCALER_STD[0];
    nn_input[1] = (ry - SCALER_MEAN[1]) / SCALER_STD[1];
    nn_input[2] = (rz - SCALER_MEAN[2]) / SCALER_STD[2];
    nn_input[3] = (bLow - SCALER_MEAN[3]) / SCALER_STD[3];
    nn_input[4] = (bMid - SCALER_MEAN[4]) / SCALER_STD[4];
    nn_input[5] = (bHigh - SCALER_MEAN[5]) / SCALER_STD[5];
    
    for (int i = 0; i < H1_SIZE; i++)
    {
        float sum = BIAS1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            sum += W1[i][j] * nn_input[j];
        nn_h1[i] = relu(sum);
    }
    
    for (int i = 0; i < H2_SIZE; i++)
    {
        float sum = BIAS2[i];
        for (int j = 0; j < H1_SIZE; j++)
            sum += W2[i][j] * nn_h1[j];
        nn_h2[i] = relu(sum);
    }
    
    for (int i = 0; i < H3_SIZE; i++)
    {
        float sum = BIAS3[i];
        for (int j = 0; j < H2_SIZE; j++)
            sum += W3[i][j] * nn_h2[j];
        nn_h3[i] = relu(sum);
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        float sum = BIAS4[i];
        for (int j = 0; j < H3_SIZE; j++)
            sum += W4[i][j] * nn_h3[j];
        nn_logits[i] = sum;
    }
    
    softmax(nn_logits, OUTPUT_SIZE);
    
    state.predictedClass = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        nn_probs[i] = nn_logits[i];
        if (nn_probs[i] > nn_probs[state.predictedClass])
            state.predictedClass = i;
    }
}

// ============================================================
//  HIỂN THỊ KẾT QUẢ
// ============================================================
void print_dashboard(float rx, float ry, float rz, 
                    float bLow, float bMid, float bHigh)
{
    Serial.println();
    Serial.println(F("============================================================"));
    Serial.println(F("[1] SIGNAL PROCESSING (512 samples @ 500Hz)"));
    Serial.println(F("    Filter: HPF(0.51Hz) + LPF(106Hz) Cascade"));
    Serial.printf("    RMS X: %.4fG  RMS Y: %.4fG  RMS Z: %.4fG\n", rx, ry, rz);
    Serial.printf("    BandPower Low: %.6f  Mid: %.6f  High: %.6f\n", bLow, bMid, bHigh);
    Serial.println();
    
    Serial.println(F("[2] NEURAL NETWORK (4 Layers: 6-32-16-8-3)"));
    for (int c = 0; c < OUTPUT_SIZE; c++)
    {
        int barLen = (int)(nn_probs[c] * 20.0f);
        if (barLen > 20) barLen = 20;
        
        Serial.printf("    %-12s [", MODEL_CLASS_LABELS[c]);
        for (int i = 0; i < 20; i++)
            Serial.print(i < barLen ? "|" : ".");
        Serial.printf("] %5.1f%%\n", nn_probs[c] * 100.0f);
    }
    Serial.println();
    Serial.printf("    => RESULT: [%s]\n", MODEL_CLASS_LABELS[state.predictedClass]);
    Serial.println();
    
    Serial.println(F("[3] PERFORMANCE"));
    Serial.printf("    Inference: %d us\n", state.inferenceTime);
    Serial.printf("    Free Heap: %d bytes\n", ESP.getFreeHeap());
    Serial.printf("    Inference #: %d\n", state.inferenceCount);
    Serial.println(F("============================================================"));
}