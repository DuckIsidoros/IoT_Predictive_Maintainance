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

const float ACCEL_SCALE = 1.0f / 16384.0f;
const int WINDOW_SIZE = 256;
const float SAMPLING_FREQ = 200.0f;

struct AccelData
{
    int16_t x;
    int16_t y;
    int16_t z;
};

hw_timer_t *timer = NULL;

volatile bool samplingTriggered = false;
volatile bool windowReady = false;
volatile float latestSqX = 0.0f;
volatile float latestSqY = 0.0f;
volatile float latestSqZ = 0.0f;

static float fft_vReal[WINDOW_SIZE];
static float fft_vImag[WINDOW_SIZE];
ArduinoFFT<float> FFT = ArduinoFFT<float>(fft_vReal, fft_vImag, WINDOW_SIZE, SAMPLING_FREQ);

float class_probabilities[MODEL_NUM_CLASSES] = {0.0f};
int predicted_class_idx = 0;
unsigned long inference_latency_us = 0;

bool IMU_Init();
bool IMU_ReadAcceleration(AccelData &data);
void apply_iir_filter(const AccelData &raw, float &fx, float &fy, float &fz);
void accumulate_metrics(float fx, float fy, float fz);
void compute_features_and_infer();
void predict_logistic_regression(float rx, float ry, float rz, float bLow, float bMid, float bHigh);
void print_dashboard(float rx, float ry, float rz, float bLow, float bMid, float bHigh);

void IRAM_ATTR onTimer()
{
    samplingTriggered = true;
}

void setup()
{
    Serial.begin(115200);
    while (!Serial)
        ;

    if (!IMU_Init())
    {
        Serial.println(F("CRITICAL ERROR: IMU Hardware Initialization Failed. System Halted."));
        while (1)
            ;
    }

    timer = timerBegin(0, 80, true);
    timerAttachInterrupt(timer, &onTimer, true);
    timerAlarmWrite(timer, 5000, true);
    timerAlarmEnable(timer);
}

void loop()
{
    if (!samplingTriggered)
        return;
    samplingTriggered = false;

    AccelData rawData;
    if (IMU_ReadAcceleration(rawData))
    {
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;

        apply_iir_filter(rawData, fx, fy, fz);
        accumulate_metrics(fx, fy, fz);

        if (windowReady)
        {
            windowReady = false;
            compute_features_and_infer();
        }
    }
}

bool IMU_Init()
{
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000);
    delay(100);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_PWR_MGMT_1);
    Wire.write(0x00);
    if (Wire.endTransmission() != 0)
        return false;
    delay(50);

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

void apply_iir_filter(const AccelData &raw, float &fx, float &fy, float &fz)
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

    unsigned long start_bench = micros();
    predict_logistic_regression(rmsX, rmsY, rmsZ, bpLow, bpMid, bpHigh);
    inference_latency_us = micros() - start_bench;

    print_dashboard(rmsX, rmsY, rmsZ, bpLow, bpMid, bpHigh);
}

void predict_logistic_regression(float rx, float ry, float rz, float bLow, float bMid, float bHigh)
{
    float raw_features[6] = {rx, ry, rz, bLow, bMid, bHigh};
    float normalized_features[6];
    float logit_scores[MODEL_NUM_CLASSES] = {0.0f};

    for (int f = 0; f < 6; f++)
    {
        normalized_features[f] = (raw_features[f] - SCALER_MEAN[f]) / SCALER_STD[f];
    }

    float max_score = -999999.0f;
    for (int c = 0; c < MODEL_NUM_CLASSES; c++)
    {
        float score = LGR_BIASES[c];
        for (int f = 0; f < 6; f++)
        {
            score += normalized_features[f] * LGR_WEIGHTS[c][f];
        }
        logit_scores[c] = score;
        if (score > max_score)
            max_score = score;
    }

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

    float sum_exp = 0.0f;
    for (int c = 0; c < MODEL_NUM_CLASSES; c++)
    {
        class_probabilities[c] = exp(logit_scores[c] - max_score);
        sum_exp += class_probabilities[c];
    }

    if (sum_exp > 0.0f)
    {
        for (int c = 0; c < MODEL_NUM_CLASSES; c++)
        {
            class_probabilities[c] /= sum_exp;
        }
    }
    else
    {
        for (int c = 0; c < MODEL_NUM_CLASSES; c++)
        {
            class_probabilities[c] = 1.0f / (float)MODEL_NUM_CLASSES;
        }
    }
}

void print_dashboard(float rx, float ry, float rz, float bLow, float bMid, float bHigh)
{
    Serial.print("\033[2J");
    Serial.print("\033[H");

    Serial.println(F("====================================================================="));
    Serial.println(F("[1] EDGE SIGNAL PROCESSING (256-sample flat window - Gravity Removal)"));
    Serial.printf("- RMS X: %.4fG  |  RMS Y: %.4fG  |  RMS Z: %.4fG\n", rx, ry, rz);
    Serial.printf("- BandPower Low: %.4f | Mid: %.4f | High: %.4f\n", bLow, bMid, bHigh);
    Serial.println();

    Serial.println(F("[2] ON-CHIP SOFTMAX INFERENCE (Logistic Regression)"));
    for (int c = 0; c < MODEL_NUM_CLASSES; c++)
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
    Serial.printf(" => DIAGNOSIS: FAN RUNNING IN STATE -> [%s]\n", MODEL_CLASS_LABELS[predicted_class_idx]);
    Serial.println();

    Serial.println(F("[3] RESOURCE BENCHMARK"));
    Serial.printf("- AI Inference Latency: %d us\n", inference_latency_us);
    Serial.printf("- ESP32 Free Heap RAM : %d bytes\n", esp_get_free_heap_size());
    Serial.println(F("====================================================================="));
}
