/*
 * esp32_vibration_classifier.ino
 *
 * Pipeline:
 *   MPU6050 (raw accel) → IIR Filter → Window Buffer
 *   → Feature Extraction (RMS, BandPower, CrestFactor, Kurtosis)
 *   → Small Dense NN Inference → Serial output
 *
 * Hardware : ESP32 + MPU6050
 * Framework: Arduino IDE
 *
 * File dependencies (cùng thư mục):
 *   model_parameters.h   — weights, biases, scaler params
 *   inference.h          — forward pass engine
 */

#include <Wire.h>
#include <math.h>
#include "model_parameters.h"
#include "inference.h"
#include <WiFi.h>
#include <PubSubClient.h>

/* ═══════════════════════════════════════════════════════════════════════
 * WiFi & MQTT CONFIG
 * ═══════════════════════════════════════════════════════════════════════ */
const char *WIFI_SSID = "Hans";
const char *WIFI_PASSWORD = "succmanuts";
const char *MQTT_SERVER = "10.98.57.10";
const int MQTT_PORT = 1883;
const char *MQTT_TOPIC = "vibration/inference";

WiFiClient espClient;
PubSubClient mqttClient(espClient);

/* ═══════════════════════════════════════════════════════════════════════
 * 1. PIN & I2C CONFIG
 * ═══════════════════════════════════════════════════════════════════════ */
#define I2C_SDA 21
#define I2C_SCL 22
#define IMU_ADDRESS 0x68

#define IMU_REG_PWR_MGMT_1 0x6B
#define IMU_REG_ACCEL_CONFIG 0x1C
#define IMU_REG_ACCEL_XOUT_H 0x3B

/* ═══════════════════════════════════════════════════════════════════════
 * 2. SAMPLING & WINDOW CONFIG
 * ═══════════════════════════════════════════════════════════════════════ */
#define SAMPLING_RATE_HZ 500
#define WINDOW_SIZE 256
#define SAMPLING_PERIOD_US (1000000 / SAMPLING_RATE_HZ) // 2000 µs
#define ACCEL_SCALE (1.0f / 2048.0f)                    // ±6g → m/s²

/* ═══════════════════════════════════════════════════════════════════════
 * 3. IIR BAND-PASS FILTER CONFIG (2nd-order Butterworth ~10–240 Hz)
 *    Coefficients computed at fs = 500 Hz
 * ═══════════════════════════════════════════════════════════════════════ */
#define HPF_ALPHA 0.9936f
static float prev_raw[3] = {0.0f, 0.0f, 0.0f};
static float prev_filt[3] = {0.0f, 0.0f, 0.0f};

/* ═══════════════════════════════════════════════════════════════════════
 * 4. WINDOW BUFFERS
 * ═══════════════════════════════════════════════════════════════════════ */
static float buf_x[WINDOW_SIZE];
static float buf_y[WINDOW_SIZE];
static float buf_z[WINDOW_SIZE];
static uint16_t buf_idx = 0;

// Transient detection — skip window khi condition thay đổi đột ngột
static float prev_rms_z = 0.0f;
#define TRANSIENT_THRESHOLD 0.5f // >50% thay đổi RMS_Z → skip

// FIX BUG 2: Pre-compute Hann window và trig tables một lần duy nhất
static float hann[WINDOW_SIZE];
static float cos_table[WINDOW_SIZE]; // cos(2π*i/N) — dùng lại qua additive angle
static float sin_table[WINDOW_SIZE]; // sin(2π*i/N)
static bool tables_ready = false;

/* ═══════════════════════════════════════════════════════════════════════
 * 5. FREQUENCY BAND CONFIG (khớp feature_extraction.py)
 * ═══════════════════════════════════════════════════════════════════════ */
#define FREQ_RESOLUTION (float(SAMPLING_RATE_HZ) / float(WINDOW_SIZE))

#define LOW_BAND_LOW_HZ 50.0f
#define LOW_BAND_HIGH_HZ 80.0f
#define MID_BAND_LOW_HZ 110.0f
#define MID_BAND_HIGH_HZ 140.0f
#define HIGH_BAND_LOW_HZ 170.0f
#define HIGH_BAND_HIGH_HZ 210.0f

/* ═══════════════════════════════════════════════════════════════════════
 * 6. FORWARD DECLARATIONS
 * ═══════════════════════════════════════════════════════════════════════ */
bool imu_init();
bool imu_read_raw(int16_t *ax, int16_t *ay, int16_t *az);
void precompute_tables();
void extract_features(float *features);
float compute_rms(const float *buf, uint16_t n);
float compute_crest_factor(const float *buf, uint16_t n, float rms);
float compute_kurtosis(const float *buf, uint16_t n);
float compute_band_power_fast(const float *buf_windowed, uint16_t n,
                              float f_low, float f_high, bool include_high = false);
float hpf_filter(float x, uint8_t axis);
void printRumtimeMetrics();

/* ═══════════════════════════════════════════════════════════════════════
 * 7. WiFi & MQTT HELPERS
 * ═══════════════════════════════════════════════════════════════════════ */
void setup_wifi()
{
    Serial.println(F("Connecting to WiFi..."));
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int timeout = 0;
    while (WiFi.status() != WL_CONNECTED && timeout < 40)
    {
        delay(500);
        Serial.print(F("."));
        timeout++;
    }

    if (WiFi.status() == WL_CONNECTED)
    {
        Serial.println(F("\nWiFi connected!"));
        Serial.print(F("IP: "));
        Serial.println(WiFi.localIP());
    }
    else
    {
        Serial.println(F("\nWiFi FAILED. Continuing offline..."));
    }
}

void mqtt_reconnect()
{
    if (mqttClient.connected())
        return;
    Serial.println(F("Attempting MQTT connection..."));
    if (mqttClient.connect("ESP32_SmallDense"))
        Serial.println(F("MQTT Connected!"));
    else
    {
        Serial.print(F("MQTT failed, rc="));
        Serial.println(mqttClient.state());
    }
}

void publish_mqtt(const InferenceResult &result, const float *features)
{
    if (!mqttClient.connected())
        return;

    String payload = "{";
    payload += "\"timestamp\":" + String(millis()) + ",";
    payload += "\"prediction\":\"" + String(result.label) + "\",";
    payload += "\"confidence\":" + String(result.confidence, 4) + ",";
    payload += "\"scores\":{";
    for (uint8_t i = 0; i < MODEL_NUM_CLASSES; i++)
    {
        payload += "\"" + String(MODEL_CLASS_LABELS[i]) + "\":" + String(result.probs[i], 4);
        if (i < MODEL_NUM_CLASSES - 1)
            payload += ",";
    }
    payload += "},";
    payload += "\"features\":{";
    payload += "\"RMS_X\":" + String(features[0], 6) + ",";
    payload += "\"RMS_Y\":" + String(features[1], 6) + ",";
    payload += "\"RMS_Z\":" + String(features[2], 6) + ",";
    payload += "\"Band_Power_Z_Low\":" + String(features[3], 6) + ",";
    payload += "\"Band_Power_Z_Mid\":" + String(features[4], 6) + ",";
    payload += "\"Band_Power_Z_High\":" + String(features[5], 6) + ",";
    payload += "\"CrestFactor_Z\":" + String(features[6], 6) + ",";
    payload += "\"Kurtosis_Z\":" + String(features[7], 6);
    payload += "}}";

    if (mqttClient.beginPublish(MQTT_TOPIC, payload.length(), false))
    {
        mqttClient.print(payload);
        mqttClient.endPublish();
    }
    else
        Serial.println(F("[MQTT] Publish FAILED!"));
}

/* ═══════════════════════════════════════════════════════════════════════
 * 8. SETUP
 * ═══════════════════════════════════════════════════════════════════════ */
float hpf_filter(float x, uint8_t axis)
{
    float filt = HPF_ALPHA * (prev_filt[axis] + x - prev_raw[axis]);
    prev_raw[axis] = x;
    prev_filt[axis] = filt;
    return filt;
}
void setup()
{
    Serial.begin(115200);
    while (!Serial)
        delay(10);

    Serial.println(F("=============================================="));
    Serial.println(F("  ESP32 Vibration Classifier — NN Inference  "));
    Serial.println(F("=============================================="));

    setup_wifi();
    mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
    mqttClient.setBufferSize(2048);
    mqtt_reconnect();
    Serial.print(F("Window size   : "));
    Serial.println(WINDOW_SIZE);
    Serial.print(F("Sampling rate : "));
    Serial.print(SAMPLING_RATE_HZ);
    Serial.println(F(" Hz"));
    Serial.print(F("Freq res.     : "));
    Serial.print(FREQ_RESOLUTION, 3);
    Serial.println(F(" Hz/bin"));
    Serial.println(F("----------------------------------------------"));

    // Pre-compute Hann + trig tables (tránh tính lại mỗi window)
    precompute_tables();
    Serial.println(F("[OK] Lookup tables ready."));

    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000);

    if (!imu_init())
    {
        Serial.println(F("[ERROR] MPU6050 init failed. Check wiring."));
        while (true)
            delay(1000);
    }

    // Trong setup(), sau imu_init():
    Serial.println(F("Warming up HPF filter..."));
    for (int w = 0; w < 500; w++)
    {
        int16_t ax_raw, ay_raw, az_raw;
        imu_read_raw(&ax_raw, &ay_raw, &az_raw);
        float az = az_raw * ACCEL_SCALE;
        hpf_filter(az, 2); // chạy filter nhưng không lưu vào buffer
        delayMicroseconds(SAMPLING_PERIOD_US);
    }
    Serial.println(F("HPF ready."));
}

/* ═══════════════════════════════════════════════════════════════════════
 * 8. MAIN LOOP
 * ═══════════════════════════════════════════════════════════════════════ */
void loop()
{
    // MQTT keepalive
    if (WiFi.status() == WL_CONNECTED)
    {
        if (!mqttClient.connected())
            mqtt_reconnect();
        mqttClient.loop();
    }

    static uint32_t next_sample_us = micros();

    while (micros() < next_sample_us)
        ;
    next_sample_us += SAMPLING_PERIOD_US;

    int16_t ax_raw, ay_raw, az_raw;
    if (!imu_read_raw(&ax_raw, &ay_raw, &az_raw))
    {
        Serial.println(F("[WARN] IMU read failed, skipping."));
        return;
    }

    float ax = ax_raw * ACCEL_SCALE;
    float ay = ay_raw * ACCEL_SCALE;
    float az = az_raw * ACCEL_SCALE;

    buf_x[buf_idx] = ax;
    buf_y[buf_idx] = ay;
    buf_z[buf_idx] = hpf_filter(az, 2);
    buf_idx++;

    if (buf_idx >= WINDOW_SIZE)
    {
        buf_idx = 0;

        float features[MODEL_NUM_FEATURES];
        extract_features(features);

        // Transient detection — bỏ qua window nếu condition vừa thay đổi đột ngột
        float curr_rms_z = features[2]; // RMS_Z = index 2
        bool is_transient = false;
        if (prev_rms_z > 0.0f)
        {
            float delta_ratio = fabsf(curr_rms_z - prev_rms_z) / prev_rms_z;
            if (delta_ratio > TRANSIENT_THRESHOLD)
            {
                is_transient = true;
                Serial.printf("[TRANSIENT SKIP] RMS_Z jump: %.4f → %.4f (delta: %.1f%%)\n",
                              prev_rms_z, curr_rms_z, delta_ratio * 100.0f);
            }
        }
        prev_rms_z = curr_rms_z;

        if (is_transient)
            return;

        // DEBUG: In raw features mỗi window
        Serial.println(F("===== RAW FEATURES ====="));
        const char *names[8] = {
            "RMS_X        ", "RMS_Y        ", "RMS_Z        ",
            "BP_Z_Low     ", "BP_Z_Mid     ", "BP_Z_High    ",
            "CrestFactor_Z", "Kurtosis_Z   "};
        for (uint8_t i = 0; i < 8; i++)
        {
            Serial.print(names[i]);
            Serial.print(F(" = "));
            Serial.println(features[i], 6);
        }
        Serial.println(F("========================"));

        uint32_t infer_start = micros();

        InferenceResult result = inference(features);

        uint32_t infer_time_us = micros() - infer_start;
        float infer_time_ms = infer_time_us / 1000.0f;

        printInferenceResult(result);

        Serial.println(F("---- Performance ----"));
        Serial.printf("Inference Time : %.3f ms (%lu us)\n", infer_time_ms, infer_time_us);
        Serial.println(F("---------------------"));

        printRuntimeMetrics();

        publish_mqtt(result, features);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * 9. IMU FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════════ */
bool imu_init()
{
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_PWR_MGMT_1);
    Wire.write(0x00);
    if (Wire.endTransmission(true) != 0)
        return false;
    delay(100);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_CONFIG);
    Wire.write(0x18);
    if (Wire.endTransmission(true) != 0)
        return false;
    delay(10);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(0x75);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)IMU_ADDRESS, (uint8_t)1);
    if (!Wire.available())
        return false;
    uint8_t who = Wire.read();
    return (who == 0x68 || who == 0x70 || who == 0x71);
}

bool imu_read_raw(int16_t *ax, int16_t *ay, int16_t *az)
{
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_XOUT_H);
    if (Wire.endTransmission(false) != 0)
        return false;

    Wire.requestFrom((uint8_t)IMU_ADDRESS, (uint8_t)6);
    if (Wire.available() < 6)
        return false;

    *ax = (int16_t)((Wire.read() << 8) | Wire.read());
    *ay = (int16_t)((Wire.read() << 8) | Wire.read());
    *az = (int16_t)((Wire.read() << 8) | Wire.read());
    return true;
}
/* ═══════════════════════════════════════════════════════════════════════
 * 11. PRE-COMPUTE LOOKUP TABLES (gọi 1 lần trong setup)
 *     FIX: tính Hann + cos/sin table trước, tránh O(N²) cosf/sinf
 *          trong vòng lặp sampling
 * ═══════════════════════════════════════════════════════════════════════ */
void precompute_tables()
{
    for (uint16_t i = 0; i < WINDOW_SIZE; i++)
    {
        hann[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (WINDOW_SIZE - 1)));
        cos_table[i] = cosf(2.0f * M_PI * i / WINDOW_SIZE);
        sin_table[i] = sinf(2.0f * M_PI * i / WINDOW_SIZE);
    }
    tables_ready = true;
}

/* ═══════════════════════════════════════════════════════════════════════
 * 12. FEATURE EXTRACTION
 *     Thứ tự output khớp FEATURE_COLUMNS trong Python:
 *       [0] RMS_X          [4] Band_Power_Z_Mid
 *       [1] RMS_Y          [5] Band_Power_Z_High
 *       [2] RMS_Z          [6] CrestFactor_Z
 *       [3] Band_Power_Z_Low  [7] Kurtosis_Z
 * ═══════════════════════════════════════════════════════════════════════ */
void extract_features(float *features)
{

    // -- Time-domain --------------------------------------------------
    float rms_x = compute_rms(buf_x, WINDOW_SIZE);
    float rms_y = compute_rms(buf_y, WINDOW_SIZE);
    float rms_z = compute_rms(buf_z, WINDOW_SIZE);

    // Detrend accZ (khớp Python: acc_z - mean trước khi tính CF & Kurt)
    float mean_z = 0.0f;
    for (uint16_t i = 0; i < WINDOW_SIZE; i++)
        mean_z += buf_z[i];
    mean_z /= WINDOW_SIZE;

    float buf_z_detrend[WINDOW_SIZE];
    for (uint16_t i = 0; i < WINDOW_SIZE; i++)
    {
        buf_z_detrend[i] = buf_z[i] - mean_z;
    }

    float rms_z_dt = compute_rms(buf_z_detrend, WINDOW_SIZE);
    float crest_factor_z = compute_crest_factor(buf_z_detrend, WINDOW_SIZE, rms_z_dt);
    float kurtosis_z = compute_kurtosis(buf_z, WINDOW_SIZE);

    // -- Frequency-domain: apply Hann window once, then reuse ---------
    float buf_z_win[WINDOW_SIZE];
    for (uint16_t i = 0; i < WINDOW_SIZE; i++)
    {
        buf_z_win[i] = buf_z[i] * hann[i];
    }

    float bp_low = compute_band_power_fast(buf_z_win, WINDOW_SIZE,
                                           LOW_BAND_LOW_HZ, LOW_BAND_HIGH_HZ);
    float bp_mid = compute_band_power_fast(buf_z_win, WINDOW_SIZE,
                                           MID_BAND_LOW_HZ, MID_BAND_HIGH_HZ);
    float bp_high = compute_band_power_fast(buf_z_win, WINDOW_SIZE,
                                            HIGH_BAND_LOW_HZ, HIGH_BAND_HIGH_HZ, true);

    // -- Pack ----------------------------------------------------------
    features[0] = rms_x;
    features[1] = rms_y;
    features[2] = rms_z;
    features[3] = bp_low;
    features[4] = bp_mid;
    features[5] = bp_high;
    features[6] = crest_factor_z;
    features[7] = kurtosis_z;

    // Debug: in tổng năng lượng signal sau Hann
    float total_energy = 0.0f;
    for (uint16_t i = 0; i < WINDOW_SIZE; i++)
    {
        total_energy += buf_z_win[i] * buf_z_win[i];
    }
    Serial.print(F("Total Z energy after Hann: "));
    Serial.println(total_energy, 8);
}

/* ═══════════════════════════════════════════════════════════════════════
 * 13. MATH HELPERS
 * ═══════════════════════════════════════════════════════════════════════ */

float compute_rms(const float *buf, uint16_t n)
{
    float sum = 0.0f;
    for (uint16_t i = 0; i < n; i++)
        sum += buf[i] * buf[i];
    return sqrtf(sum / n);
}

float compute_crest_factor(const float *buf, uint16_t n, float rms)
{
    if (rms <= 0.0f)
        return 0.0f;
    float peak = 0.0f;
    for (uint16_t i = 0; i < n; i++)
    {
        float a = fabsf(buf[i]);
        if (a > peak)
            peak = a;
    }
    return peak / rms;
}

float compute_kurtosis(const float *buf, uint16_t n)
{
    if (n < 4)
        return 0.0f;
    float mean = 0.0f;
    for (uint16_t i = 0; i < n; i++)
        mean += buf[i];
    mean /= n;

    float m2 = 0.0f, m4 = 0.0f;
    for (uint16_t i = 0; i < n; i++)
    {
        float d = buf[i] - mean;
        float d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= n;
    m4 /= n;
    if (m2 < 1e-10f)
        return 0.0f;
    return (m4 / (m2 * m2)) - 3.0f; // Fisher kurtosis
}

/*
 * compute_band_power_fast()
 * FIX: nhận buf đã áp Hann window sẵn (tránh tính lại 3 lần).
 *      Dùng additive angle recurrence thay vì cosf/sinf mỗi sample
 *      → giảm từ O(N²·trig) xuống O(N·bins·add) per window.
 *
 *      Recurrence: cos((k)(i+1)·2π/N) = cos(kΔ)·cos(kiΔ) - sin(kΔ)·sin(kiΔ)
 *      với Δ = 2π/N — chỉ cần 1 lần cosf/sinf per bin (k), không per sample.
 */
float compute_band_power_fast(const float *buf_windowed, uint16_t n,
                              float f_low, float f_high, bool include_high)
{
    uint16_t k_low = (uint16_t)(f_low / FREQ_RESOLUTION);
    uint16_t k_high = (uint16_t)(f_high / FREQ_RESOLUTION);
    if (k_high > n / 2)
        k_high = n / 2;

    uint16_t k_end = include_high ? k_high + 1 : k_high;

    float band_power = 0.0f;
    float inv_n = 1.0f / n;

    for (uint16_t k = k_low; k < k_end; k++)
    {
        float delta = 2.0f * M_PI * k * inv_n;
        float cos_delta = cosf(delta);
        float sin_delta = sinf(delta);

        float c = 1.0f;
        float s = 0.0f;

        float re = 0.0f, im = 0.0f;
        for (uint16_t i = 0; i < n; i++)
        {
            re += buf_windowed[i] * c;
            im -= buf_windowed[i] * s;

            float c_new = c * cos_delta - s * sin_delta;
            float s_new = s * cos_delta + c * sin_delta;
            c = c_new;
            s = s_new;
        }

        float mag = sqrtf(re * re + im * im) * inv_n;
        band_power += mag * mag;
    }
    return band_power;
}

void printRuntimeMetrics()
{
    uint32_t free_heap = ESP.getFreeHeap();
    uint32_t min_free_heap = ESP.getMinFreeHeap();
    uint32_t heap_size = ESP.getHeapSize();

    Serial.println(F("---- Runtime Metrics ----"));
    Serial.printf("Heap Size      : %lu bytes\n", heap_size);
    Serial.printf("Free Heap      : %lu bytes\n", free_heap);
    Serial.printf("Min Free Heap  : %lu bytes\n", min_free_heap);
    Serial.printf("RAM Used Peak  : %lu bytes\n", heap_size - min_free_heap);
    Serial.println(F("-------------------------"));
}