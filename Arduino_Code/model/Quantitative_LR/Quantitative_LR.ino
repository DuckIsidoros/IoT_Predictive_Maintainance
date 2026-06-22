#include <Wire.h>
#include <math.h>
#include <arduinoFFT.h>
#include "model_parameters.h"

#define I2C_SDA 21
#define I2C_SCL 22
#define IMU_ADDRESS 0x68
#define IMU_REG_PWR_MGMT_1 0x6B
#define IMU_REG_ACCEL_CONFIG 0x1C
#define IMU_REG_ACCEL_XOUT_H 0x3B
#define IMU_REG_CONFIG 0x1A
#define IMU_REG_SMPLRT_DIV 0x19

const float ACCEL_SCALE = 1.0f / 16384.0f;
const int WINDOW_SIZE = 512;
const float SAMPLING_FREQ = 500.0f;

#define INPUT_SIZE MODEL_NUM_FEATURES
#define OUTPUT_SIZE MODEL_NUM_CLASSES

struct AccelData
{
    int16_t x;
    int16_t y;
    int16_t z;
};

unsigned long lastSampleTime = 0;
const unsigned long sampleIntervalUs = 2000;

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

// --- BENCHMARK VARIABLES ---
unsigned long feature_extraction_us = 0;
unsigned long inference_latency_us = 0;
size_t ram_baseline = 0;
size_t ram_after_setup = 0;
size_t ram_peak_during_inference = 0;

bool IMU_Init();
bool IMU_ReadAcceleration(AccelData &data);
void remove_gravity(const AccelData &raw, float &fx, float &fy, float &fz);
void accumulate_metrics(float fx, float fy, float fz);
void compute_features_and_infer();
void predict_logistic_regression(float rx, float ry, float rz, float bLow, float bMid, float bHigh);
void print_dashboard(float rx, float ry, float rz, float bLow, float bMid, float bHigh);
void softmax(float *x, int n);

void updateRawBuffers(float x, float y, float z)
{
    for (int i = 0; i < 19; i++)
    {
        rawX_buffer[i] = rawX_buffer[i + 1];
        rawY_buffer[i] = rawY_buffer[i + 1];
        rawZ_buffer[i] = rawZ_buffer[i + 1];
    }
    rawX_buffer[19] = x;
    rawY_buffer[19] = y;
    rawZ_buffer[19] = z;
}

void setup()
{
    // 1. Ghi nhận RAM Baseline nguyên bản ngay khi vừa khởi động chip
    ram_baseline = esp_get_free_heap_size();

    Serial.begin(115200);
    delay(2000);
    Serial.println(F("\n--- KHOI DONG HE THONG OFFLINE VỚI PROFILER BENCHMARK ---"));
    Serial.println(F("System Ready (WiFi Disabled)"));

    if (!IMU_Init())
    {
        Serial.println(F("CRITICAL ERROR: Khong tim thay hoac khong khoi tao duoc IMU MPU6050!"));
        while (1)
        {
            Serial.println(F("Dang thu lai ket noi I2C..."));
            delay(2000);
        }
    }

    Serial.println(F("IMU Initialization OK!"));
    Serial.println(F("Bat dau lay mau voi Fs = 500Hz..."));

    lastSampleTime = micros();

    // 2. Ghi nhận RAM tĩnh sau khi cấu hình pipeline xử lý tín hiệu hoàn tất
    ram_after_setup = esp_get_free_heap_size();
}

void loop()
{
    unsigned long currentTime = micros();

    if (currentTime - lastSampleTime >= sampleIntervalUs)
    {
        lastSampleTime += sampleIntervalUs;

        AccelData rawData;
        if (IMU_ReadAcceleration(rawData))
        {
            float fx = 0.0f, fy = 0.0f, fz = 0.0f;

            remove_gravity(rawData, fx, fy, fz);
            accumulate_metrics(fx, fy, fz);

            static int debug_counter = 0;
            if (++debug_counter >= 50)
            {
                Serial.print(".");
                debug_counter = 0;
            }

            if (windowReady)
            {
                windowReady = false;
                Serial.println(F("\n[OK] Da nhan du 512 mau o 500Hz! Dang chay FFT va LGR..."));
                compute_features_and_infer();

                print_counter++;
                if (print_counter >= 1)
                {
                    print_dashboard(last_rmsX, last_rmsY, last_rmsZ, last_bpLow, last_bpMid, last_bpHigh);
                    print_counter = 0;
                }
            }
        }
        else
        {
            static unsigned long last_error_time = 0;
            if (millis() - last_error_time > 2000)
            {
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
    Wire.setClock(400000);
    delay(100);

    Wire.beginTransmission(IMU_ADDRESS);
    if (Wire.endTransmission() != 0)
    {
        return false;
    }

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_PWR_MGMT_1);
    Wire.write(0x00);
    if (Wire.endTransmission() != 0)
        return false;
    delay(50);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_CONFIG);
    Wire.write(0x00);
    if (Wire.endTransmission() != 0)
        return false;
    delay(10);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_SMPLRT_DIV);
    Wire.write(15);
    if (Wire.endTransmission() != 0)
        return false;
    delay(10);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_CONFIG);
    Wire.write(0x00);
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

void remove_gravity(const AccelData &raw, float &fx, float &fy, float &fz)
{
    const float HPF_ALPHA = 0.9843f;

    static float p_rx = 0.0f, p_fx = 0.0f;
    static float p_ry = 0.0f, p_fy = 0.0f;
    static float p_rz = 0.0f, p_fz = 0.0f;

    float crx = (float)raw.x * ACCEL_SCALE;
    float cry = (float)raw.y * ACCEL_SCALE;
    float crz = (float)raw.z * ACCEL_SCALE;

    fx = HPF_ALPHA * (p_fx + crx - p_rx);
    fy = HPF_ALPHA * (p_fy + cry - p_ry);
    fz = HPF_ALPHA * (p_fz + crz - p_rz);

    p_rx = crx;
    p_fx = fx;
    p_ry = cry;
    p_fy = fy;
    p_rz = crz;
    p_fz = fz;

    updateRawBuffers(fx, fy, fz);
}

void accumulate_metrics(float fx, float fy, float fz)
{
    static int idx = 0;
    static float sSqX = 0.0f;
    static float sSqY = 0.0f;
    static float sSqZ = 0.0f;

    sSqX += fx * fx;
    sSqY += fy * fy;
    sSqZ += fz * fz;

    fft_vReal[idx] = fz;
    fft_vImag[idx] = 0.0f;

    idx++;

    if (idx >= WINDOW_SIZE)
    {
        idx = 0;

        latestSqX = sSqX;
        latestSqY = sSqY;
        latestSqZ = sSqZ;

        sSqX = 0.0f;
        sSqY = 0.0f;
        sSqZ = 0.0f;

        windowReady = true;
    }
}

void compute_features_and_infer()
{
    // A. Bấm giờ quá trình trích xuất đặc trưng (DSP)
    unsigned long start_feature = micros();

    float sSqX = latestSqX;
    float sSqY = latestSqY;
    float sSqZ = latestSqZ;

    float rmsX = sqrt(sSqX / (float)WINDOW_SIZE);
    float rmsY = sqrt(sSqY / (float)WINDOW_SIZE);
    float rmsZ = sqrt(sSqZ / (float)WINDOW_SIZE);

    FFT.windowing(FFT_WIN_TYP_HANN, FFT_FORWARD);
    FFT.compute(FFT_FORWARD);
    FFT.complexToMagnitude();

    float bpLow = 0.0f, bpMid = 0.0f, bpHigh = 0.0f;
    const float bin_width = SAMPLING_FREQ / (float)WINDOW_SIZE;

    for (int i = 0; i < WINDOW_SIZE / 2; i++)
    {
        if (i % 50 == 0)
            yield();

        float freq = (float)i * bin_width;
        float magSq = fft_vReal[i] * fft_vReal[i];

        if (freq >= 50.0f && freq < 80.0f)
            bpLow += magSq;
        else if (freq >= 110.0f && freq < 140.0f)
            bpMid += magSq;
        else if (freq >= 170.0f && freq <= 210.0f)
            bpHigh += magSq;
    }
    feature_extraction_us = micros() - start_feature;

    // B. Bấm giờ quá trình chạy Logistic Regression Inference
    unsigned long start_bench = micros();
    predict_logistic_regression(rmsX, rmsY, rmsZ, bpLow, bpMid, bpHigh);
    inference_latency_us = micros() - start_bench;

    // C. Theo dõi bộ nhớ RAM đỉnh (Peak) khi chạy tính toán
    size_t current_free_ram = esp_get_free_heap_size();
    if (ram_peak_during_inference == 0 || current_free_ram < ram_peak_during_inference)
    {
        ram_peak_during_inference = current_free_ram;
    }

    last_rmsX = rmsX;
    last_rmsY = rmsY;
    last_rmsZ = rmsZ;
    last_bpLow = bpLow;
    last_bpMid = bpMid;
    last_bpHigh = bpHigh;
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

void predict_logistic_regression(float rx, float ry, float rz, float bLow, float bMid, float bHigh)
{
    float x[INPUT_SIZE] = {rx, ry, rz, bLow, bMid, bHigh};

    // 1. Z-Score Normalization
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        x[i] = (x[i] - SCALER_MEAN[i]) / SCALER_STD[i];
    }

    // 2. Compute Raw Logits
    float logits[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        float sum = LGR_BIASES[i];
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            sum += LGR_WEIGHTS[i][j] * x[j];
        }
        logits[i] = sum;
    }

    // 3. Multiclass Probability Activation
    softmax(logits, OUTPUT_SIZE);

    // 4. Argmax Classification Selector
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
    Serial.println();
    Serial.println(F("====================================================================="));
    Serial.println(F("[1] EDGE SIGNAL PROCESSING (512-sample window @ 500Hz)"));
    Serial.printf("- RMS X: %.4fG  |  RMS Y: %.4fG  |  RMS Z: %.4fG\n", rx, ry, rz);
    Serial.printf("- BandPower Low: %.6f | Mid: %.6f | High: %.6f\n", bLow, bMid, bHigh);
    Serial.println();

    Serial.println(F("[2] ON-CHIP SOFTMAX INFERENCE (Edge One-vs-Rest Logistic Regression)"));
    for (int c = 0; c < OUTPUT_SIZE; c++)
    {
        int bar_length = (int)(class_probabilities[c] * 20.0f);
        if (bar_length > 20)
            bar_length = 20;
        if (bar_length < 0)
            bar_length = 0;

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

    Serial.println(F("[3] HARDWARE RESOURCE BENCHMARK (AIoT PROFILER)"));
    Serial.println(F("---------------------------------------------------------------------"));
    Serial.printf("  - DSP & Feature Extraction Time   : %lu us (%.2f ms)\n", feature_extraction_us, (float)feature_extraction_us / 1000.0f);
    Serial.printf("  - AI Model Inference Latency      : %lu us (%.2f ms)\n", inference_latency_us, (float)inference_latency_us / 1000.0f);
    Serial.printf("  - Total Processing Pipeline Time  : %.2f ms (Max Throughput: %.1f FPS)\n",
                  (float)(feature_extraction_us + inference_latency_us) / 1000.0f,
                  1000000.0f / (feature_extraction_us + inference_latency_us));
    Serial.println(F("---------------------------------------------------------------------"));
    Serial.printf("  - Pipeline Static RAM Overhead      : %d bytes\n", ram_baseline - ram_after_setup);
    Serial.printf("  - Runtime Dynamic Peak RAM Usage    : %d bytes\n", ram_baseline - ram_peak_during_inference);
    Serial.printf("  - Current Free Heap Memory          : %d bytes\n", esp_get_free_heap_size());
    Serial.println(F("====================================================================="));
}