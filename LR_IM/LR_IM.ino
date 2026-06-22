#include <Wire.h>
#include "VibrationClassifier.h"
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

#define I2C_SDA 21
#define I2C_SCL 22
#define IMU_ADDRESS 0x68
#define IMU_REG_PWR_MGMT_1 0x6B
#define IMU_REG_ACCEL_CONFIG 0x1C
#define IMU_REG_ACCEL_XOUT_H 0x3B

// Cấu hình dải đo ±16g khớp với config.py
const float ACCEL_SCALE = 1.0f / 2048.0f;
const float HPF_ALPHA = 0.9936f;
uint32_t last_recovery_time_ms = 0;

struct AccelData
{
    int16_t x;
    int16_t y;
    int16_t z;
};

// Biến trạng thái cho bộ lọc trục Z
float prev_raw_z = 0.0f, prev_filt_z = 0.0f;

// Định thời ngắt cứng
hw_timer_t *timer = NULL;
volatile bool samplingTriggered = false;
volatile uint32_t last_sample_time_us = 0;
uint32_t error_counter = 0;

// ==============================================================================
// EDGE AI REALTIME WINDOW BUFFER CONFIGURATION
// ==============================================================================
#define SAMPLING_RATE_HZ 500
#define BUFFER_SIZE 256
#define STEP_SIZE 128 // 50% overlap

// Frequency resolution — tính động, không hardcode
#define FREQ_RESOLUTION (float(SAMPLING_RATE_HZ) / float(BUFFER_SIZE))

// Band boundaries khớp Small_Dense.ino
#define LOW_BAND_LOW_HZ 50.0f
#define LOW_BAND_HIGH_HZ 80.0f
#define MID_BAND_LOW_HZ 110.0f
#define MID_BAND_HIGH_HZ 140.0f
#define HIGH_BAND_LOW_HZ 170.0f
#define HIGH_BAND_HIGH_HZ 210.0f

float ring_buffer_x[BUFFER_SIZE];
float ring_buffer_y[BUFFER_SIZE];
float ring_buffer_z[BUFFER_SIZE];
int buffer_index = 0;
int sample_chunk_counter = 0;
bool window_ready = false;

// Hann window + trig tables — precompute 1 lần trong setup()
static float hann_table[BUFFER_SIZE];
static bool tables_ready = false;

// Đối tượng thực thi mô hình và mảng chứa đặc trưng đầu ra
VibrationClassifier classifier;
float current_features[8];
float class_probabilities[MODEL_NUM_CLASSES];

// Transient detection — skip window khi condition thay đổi đột ngột
static float prev_rms_z = 0.0f;
#define TRANSIENT_THRESHOLD 0.5f // >50% thay đổi RMS_Z → skip

void IRAM_ATTR onTimer()
{
    samplingTriggered = true;
    last_sample_time_us = micros();
}

bool IMU_Init()
{
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000); // Khóa tốc độ I2C Fast Mode
    delay(50);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_PWR_MGMT_1);
    Wire.write(0x00);
    if (Wire.endTransmission() != 0)
        return false;

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_CONFIG);
    Wire.write(0x18); // Ép cứng dải đo vật lý ±16g
    if (Wire.endTransmission() != 0)
        return false;

    return true;
}

bool IMU_ReadAccel(AccelData &data)
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

// ==============================================================================
// DSP FUNCTIONS ON EDGE — khớp hoàn toàn Small_Dense.ino
// ==============================================================================

// Precompute Hann window 1 lần trong setup() — tránh tính lại mỗi window
void precompute_tables()
{
    for (uint16_t i = 0; i < BUFFER_SIZE; i++)
    {
        hann_table[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (BUFFER_SIZE - 1)));
    }
    tables_ready = true;
}

// Band power dùng Goertzel-style additive recurrence — khớp compute_band_power_fast()
// @param buf_windowed  Buffer Z đã áp Hann window
// @param f_low / f_high  Boundary Hz
// @param include_high  true → include bin k_high
float compute_band_power_fast(const float *buf_windowed, uint16_t n,
                              float f_low, float f_high, bool include_high = false)
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

        float c = 1.0f, s = 0.0f;
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

void extract_features_on_edge()
{
    // -- Bước 1: RMS X, Y, Z (raw, không detrend) ----------------------
    float sum_x2 = 0.0f, sum_y2 = 0.0f, sum_z2 = 0.0f, sum_z = 0.0f;
    for (int i = 0; i < BUFFER_SIZE; i++)
    {
        sum_x2 += ring_buffer_x[i] * ring_buffer_x[i];
        sum_y2 += ring_buffer_y[i] * ring_buffer_y[i];
        sum_z2 += ring_buffer_z[i] * ring_buffer_z[i];
        sum_z += ring_buffer_z[i];
    }
    current_features[0] = sqrtf(sum_x2 / BUFFER_SIZE); // RMS_X
    current_features[1] = sqrtf(sum_y2 / BUFFER_SIZE); // RMS_Y
    current_features[2] = sqrtf(sum_z2 / BUFFER_SIZE); // RMS_Z

    // -- Bước 2: Detrend Z → CrestFactor & Kurtosis -------------------
    float mean_z = sum_z / BUFFER_SIZE;
    float sum_dt2 = 0.0f, sum_dt4 = 0.0f, peak_dt = 0.0f;

    for (int i = 0; i < BUFFER_SIZE; i++)
    {
        float d = ring_buffer_z[i] - mean_z;
        float d2 = d * d;
        sum_dt2 += d2;
        sum_dt4 += d2 * d2;
        float ad = fabsf(d);
        if (ad > peak_dt)
            peak_dt = ad;
    }

    float rms_dt = sqrtf(sum_dt2 / BUFFER_SIZE);
    current_features[6] = (rms_dt > 0.0f) ? peak_dt / rms_dt : 0.0f; // CrestFactor_Z

    // Fisher kurtosis (khớp Small_Dense.ino: m4/m2² - 3)
    float m2 = sum_dt2 / BUFFER_SIZE;
    float m4 = sum_dt4 / BUFFER_SIZE;
    current_features[7] = (m2 > 1e-10f) ? (m4 / (m2 * m2)) - 3.0f : 0.0f; // Kurtosis_Z

    // -- Bước 3: Apply Hann window → Band Power FFT -------------------
    float buf_z_win[BUFFER_SIZE];
    for (int i = 0; i < BUFFER_SIZE; i++)
        buf_z_win[i] = ring_buffer_z[i] * hann_table[i];

    current_features[3] = compute_band_power_fast(buf_z_win, BUFFER_SIZE,
                                                  LOW_BAND_LOW_HZ, LOW_BAND_HIGH_HZ); // BP_Z_Low
    current_features[4] = compute_band_power_fast(buf_z_win, BUFFER_SIZE,
                                                  MID_BAND_LOW_HZ, MID_BAND_HIGH_HZ); // BP_Z_Mid
    current_features[5] = compute_band_power_fast(buf_z_win, BUFFER_SIZE,
                                                  HIGH_BAND_LOW_HZ, HIGH_BAND_HIGH_HZ, true); // BP_Z_High

    // Debug: in tổng năng lượng signal sau Hann
    float total_energy = 0.0f;
    for (int i = 0; i < BUFFER_SIZE; i++)
        total_energy += buf_z_win[i] * buf_z_win[i];
    Serial.print(F("Total Z energy after Hann: "));
    Serial.println(total_energy, 8);
}

/* ═══════════════════════════════════════════════════════════════════════
 * WiFi & MQTT HELPERS
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
        Serial.println(F("\nWiFi FAILED. Continuing offline..."));
}

void mqtt_reconnect()
{
    if (mqttClient.connected())
        return;
    Serial.println(F("Attempting MQTT connection..."));
    if (mqttClient.connect("ESP32_LR"))
        Serial.println(F("MQTT Connected!"));
    else
    {
        Serial.print(F("MQTT failed, rc="));
        Serial.println(mqttClient.state());
    }
}

void publish_mqtt(const String &prediction, const float *probs, const float *features)
{
    if (!mqttClient.connected())
        return;

    String payload = "{";
    payload += "\"timestamp\":" + String(millis()) + ",";
    payload += "\"prediction\":\"" + prediction + "\",";
    payload += "\"scores\":{";
    for (int i = 0; i < MODEL_NUM_CLASSES; i++)
    {
        payload += "\"" + String(MODEL_CLASS_LABELS[i]) + "\":" + String(probs[i], 4);
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

void setup()
{
    Serial.begin(115200);
    while (!Serial)
        ;

    Serial.println(F("=============================================="));
    Serial.println(F("  ESP32 Vibration Classifier — LR Inference  "));
    Serial.println(F("=============================================="));

    setup_wifi();
    mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
    mqttClient.setBufferSize(2048);
    mqtt_reconnect();

    Serial.print(F("Window size   : "));
    Serial.println(BUFFER_SIZE);
    Serial.print(F("Sampling rate : "));
    Serial.print(SAMPLING_RATE_HZ);
    Serial.println(F(" Hz"));
    Serial.print(F("Freq res.     : "));
    Serial.print(FREQ_RESOLUTION, 3);
    Serial.println(F(" Hz/bin"));
    Serial.print(F("Step size     : "));
    Serial.print(STEP_SIZE);
    Serial.println(F(" samples (50% overlap)"));
    Serial.println(F("----------------------------------------------"));

    if (!IMU_Init())
    {
        Serial.println("ERROR: IMU Initialization Failed");
        while (1)
            ;
    }

    // Precompute Hann window 1 lần — tránh tính lại mỗi window
    precompute_tables();
    Serial.println(F("[OK] Hann table ready."));

    // HPF warmup — flush transient khởi động (khớp Small_Dense.ino: 500 mẫu)
    Serial.println(F("Warming up HPF filter..."));
    for (int w = 0; w < 500; w++)
    {
        AccelData dummy;
        if (IMU_ReadAccel(dummy))
        {
            float raw_z = (float)dummy.z * ACCEL_SCALE;
            float filt_z = HPF_ALPHA * (prev_filt_z + raw_z - prev_raw_z);
            prev_raw_z = raw_z;
            prev_filt_z = filt_z;
        }
        delayMicroseconds(2000); // 500 Hz
    }
    Serial.println(F("HPF ready."));

    timer = timerBegin(1000000);
    timerAttachInterrupt(timer, &onTimer);
    timerAlarm(timer, 2000, true, 0); // 2000us = 500 Hz
}

void loop()
{
    // MQTT keepalive
    if (WiFi.status() == WL_CONNECTED)
    {
        if (!mqttClient.connected())
            mqtt_reconnect();
        mqttClient.loop();
    }

    if (samplingTriggered)
    {
        samplingTriggered = false;
        uint32_t sample_timestamp_us = last_sample_time_us;

        AccelData raw_counts;
        if (IMU_ReadAccel(raw_counts))
        {
            error_counter = 0;

            // Ép cấu trúc sang gia tốc vật lý thực tế dạng số thực
            float curr_raw_x = (float)raw_counts.x * ACCEL_SCALE;
            float curr_raw_y = (float)raw_counts.y * ACCEL_SCALE;
            float curr_raw_z = (float)raw_counts.z * ACCEL_SCALE;

            // Lọc thông cao IIR cục bộ trục Z (Triệt tiêu DC phục vụ luồng FFT)
            float curr_filt_z = HPF_ALPHA * (prev_filt_z + curr_raw_z - prev_raw_z);

            prev_raw_z = curr_raw_z;
            prev_filt_z = curr_filt_z;

            // ==============================================================================
            // NẠP DỮ LIỆU VÀO MẢNG TRƯỢT OVERLAP 50%
            // ==============================================================================
            ring_buffer_x[buffer_index] = curr_raw_x;
            ring_buffer_y[buffer_index] = curr_raw_y;
            ring_buffer_z[buffer_index] = curr_raw_z;

            buffer_index++;
            sample_chunk_counter++;

            // Khi bộ đệm đầy chu kỳ đầu tiên (128 mẫu)
            if (buffer_index >= BUFFER_SIZE)
            {
                buffer_index = 0; // Reset vòng đệm cuốn chiếu
                window_ready = true;
            }

            // Kích hoạt tính toán khi trượt đủ 64 mẫu mới (Overlap 50%)
            if (window_ready && sample_chunk_counter >= STEP_SIZE)
            {
                sample_chunk_counter = 0; // Đặt lại bộ đếm dịch chuyển

                // 1. Thực thi hàm DSP bóc tách đặc trưng 8 chiều cực nhanh trên RAM
                extract_features_on_edge();

                // DEBUG: In raw features mỗi window
                Serial.println(F("===== RAW FEATURES ====="));
                const char *feat_names[8] = {
                    "RMS_X        ", "RMS_Y        ", "RMS_Z        ",
                    "BP_Z_Low     ", "BP_Z_Mid     ", "BP_Z_High    ",
                    "CrestFactor_Z", "Kurtosis_Z   "};
                for (uint8_t i = 0; i < 8; i++)
                {
                    Serial.print(feat_names[i]);
                    Serial.print(F(" = "));
                    Serial.println(current_features[i], 6);
                }
                Serial.println(F("========================"));

                // 2. Transient detection — bỏ qua window nếu condition vừa thay đổi đột ngột
                float curr_rms_z = current_features[2];
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
                    return; // Không inference window này

                // 3. Chạy hàm Inference của mô hình Logistic Regression
                uint32_t infer_start = micros();
                String prediction = classifier.predict(current_features, class_probabilities);
                uint32_t infer_time_us = micros() - infer_start;
                float infer_time_ms = infer_time_us / 1000.0f;

                // 4. In kết quả chẩn đoán lỗi trực tiếp ra cổng Serial Monitor
                float timestamp_ms = (float)sample_timestamp_us / 1000.0f;
                Serial.printf("\n[%010.2f ms] >>> [EDGE DIAGNOSIS]: %s\n", timestamp_ms, prediction.c_str());
                for (int i = 0; i < MODEL_NUM_CLASSES; i++)
                {
                    Serial.printf("   - Prob [%s]: %.2f%%\n", MODEL_CLASS_LABELS[i], class_probabilities[i] * 100.0f);
                }

                Serial.println(F("---- Performance ----"));
                Serial.printf("Inference Time : %.3f ms (%lu us)\n", infer_time_ms, infer_time_us);
                Serial.println(F("---------------------"));

                // Runtime Metrics
                {
                    uint32_t free_heap     = ESP.getFreeHeap();
                    uint32_t min_free_heap = ESP.getMinFreeHeap();
                    uint32_t heap_size     = ESP.getHeapSize();
                    Serial.println(F("---- Runtime Metrics ----"));
                    Serial.printf("Heap Size      : %lu bytes\n", heap_size);
                    Serial.printf("Free Heap      : %lu bytes\n", free_heap);
                    Serial.printf("Min Free Heap  : %lu bytes\n", min_free_heap);
                    Serial.printf("RAM Used Peak  : %lu bytes\n", heap_size - min_free_heap);
                    Serial.println(F("-------------------------"));
                }

                // 5. Publish MQTT
                publish_mqtt(prediction, class_probabilities, current_features);
            }
        }
        else
        {
            // Khối logic xử lý lỗi I2C non-blocking cứu hộ phần cứng của bạn
            error_counter++;
            if (error_counter > 20)
            {
                uint32_t current_time_ms = millis();
                if (current_time_ms - last_recovery_time_ms > 100)
                {
                    last_recovery_time_ms = current_time_ms;
                    Wire.end();
                    Wire.begin(I2C_SDA, I2C_SCL);
                    Wire.setClock(400000);

                    Wire.beginTransmission(IMU_ADDRESS);
                    Wire.write(IMU_REG_PWR_MGMT_1);
                    Wire.write(0x00);
                    Wire.endTransmission();

                    Wire.beginTransmission(IMU_ADDRESS);
                    Wire.write(IMU_REG_ACCEL_CONFIG);
                    Wire.write(0x18);
                    Wire.endTransmission();

                    error_counter = 0;
                }
            }
        }
    }
}