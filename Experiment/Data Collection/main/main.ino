#include <Wire.h>
#include <esp_task_wdt.h>

// ─── Pin & I2C config ────────────────────────────────────────────────────────
#define I2C_SDA 21
#define I2C_SCL 22
#define IMU_ADDRESS 0x68

// ─── MPU6050 register map ────────────────────────────────────────────────────
#define REG_SMPLRT_DIV 0x19
#define REG_DLPF_CONFIG 0x1A
#define REG_ACCEL_CONFIG 0x1C
#define REG_ACCEL_XOUT_H 0x3B
#define REG_PWR_MGMT_1 0x6B

// ─── Sampling config ─────────────────────────────────────────────────────────
#define SAMPLE_PERIOD_US 2000 // 2 ms  →  500 Hz
#define SERIAL_BAUD 921600
#define I2C_CLOCK_HZ 400000
#define WIRE_TIMEOUT_MS 1  // must be < one sample period (2 ms)
#define ERROR_THRESHOLD 25 // 25 consecutive failures → reinit (50 ms)
#define INIT_MAX_RETRY 5

// ─── Scale & filter ──────────────────────────────────────────────────────────
// ±2g range → LSB sensitivity = 16384 LSB/g
const float ACCEL_SCALE = 1.0f / 16384.0f;

// HPF to remove gravity (DC component).
// α = 1 − (2π × f_cutoff / Fs) = 1 − (2π × 0.5 / 500) ≈ 0.9937
// Preserves the same ~0.5 Hz cutoff that the original 200 Hz code used.
const float HPF_ALPHA = 0.9936f;

// ─── Label ───────────────────────────────────────────────────────────────────
const char *CURRENT_LABEL = "obstruction";

// ─── ISR state (volatile, written only in ISR) ───────────────────────────────
volatile uint32_t isr_sample_count = 0;
volatile uint32_t isr_timestamp_us = 0;

// ─── Loop state ──────────────────────────────────────────────────────────────
static uint32_t processed_count = 0;
static uint32_t consecutive_errors = 0;
static hw_timer_t *timer = NULL;

// HPF state — only gravity-removal filter is kept
static float prev_raw_x = 0.0f, prev_filt_x = 0.0f;
static float prev_raw_y = 0.0f, prev_filt_y = 0.0f;
static float prev_raw_z = 0.0f, prev_filt_z = 0.0f;

// ─── Raw accelerometer reading ───────────────────────────────────────────────
struct AccelData
{
    int16_t x, y, z;
};

// ─────────────────────────────────────────────────────────────────────────────
// ISR — runs every 2 ms, does nothing except record count + timestamp
// ─────────────────────────────────────────────────────────────────────────────
void IRAM_ATTR onTimer()
{
    isr_sample_count++;
    isr_timestamp_us = micros();
}

// ─────────────────────────────────────────────────────────────────────────────
// IMU init — wake, set ±2g, set ODR to 500 Hz via SMPLRT_DIV
// ─────────────────────────────────────────────────────────────────────────────
bool IMU_Init()
{
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(I2C_CLOCK_HZ);
    Wire.setTimeout(WIRE_TIMEOUT_MS);
    delay(50);

    // Wake up (clear sleep bit)
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(REG_PWR_MGMT_1);
    Wire.write(0x00);
    if (Wire.endTransmission() != 0)
        return false;

    // DLPF = 0 (disabled) → internal sample rate = 8 kHz
    // With DLPF off, SMPLRT_DIV divides 8 kHz: 8000 / (1 + 15) = 500 Hz
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(REG_DLPF_CONFIG);
    Wire.write(0x00);
    if (Wire.endTransmission() != 0)
        return false;

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(REG_SMPLRT_DIV);
    Wire.write(0x0F); // 8000 / (1 + 15) = 500 Hz ODR
    if (Wire.endTransmission() != 0)
        return false;

    // Accel full-scale = ±2g
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(REG_ACCEL_CONFIG);
    Wire.write(0x00);
    if (Wire.endTransmission() != 0)
        return false;

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// I2C burst read — 6 bytes (X/Y/Z high+low)
// ─────────────────────────────────────────────────────────────────────────────
bool IMU_ReadAccel(AccelData &data)
{
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(REG_ACCEL_XOUT_H);
    if (Wire.endTransmission(false) != 0)
        return false;
    if (Wire.requestFrom(IMU_ADDRESS, (uint8_t)6) < 6)
        return false;

    data.x = (Wire.read() << 8) | Wire.read();
    data.y = (Wire.read() << 8) | Wire.read();
    data.z = (Wire.read() << 8) | Wire.read();
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Process one sample: apply gravity-removal HPF, print CSV row
// ─────────────────────────────────────────────────────────────────────────────
void processSample(const AccelData &raw, uint32_t timestamp_us)
{
    // Convert raw counts → g
    float cx = raw.x * ACCEL_SCALE;
    float cy = raw.y * ACCEL_SCALE;
    float cz = raw.z * ACCEL_SCALE;

    // High-pass filter: removes gravity (DC) from each axis
    // y[n] = α × (y[n-1] + x[n] − x[n-1])
    float fx = HPF_ALPHA * (prev_filt_x + cx - prev_raw_x);
    float fy = HPF_ALPHA * (prev_filt_y + cy - prev_raw_y);
    float fz = HPF_ALPHA * (prev_filt_z + cz - prev_raw_z);

    prev_raw_x = cx;
    prev_filt_x = fx;
    prev_raw_y = cy;
    prev_filt_y = fy;
    prev_raw_z = cz;
    prev_filt_z = fz;

    // Use double to avoid float precision loss after ~1000 s uptime
    double timestamp_ms = (double)timestamp_us / 1000.0;

    // Output: timestamp_ms, accX, accY, accZ
    Serial.print(timestamp_ms, 3);
    Serial.print(",");
    Serial.print(fx, 4);
    Serial.print(",");
    Serial.print(fy, 4);
    Serial.print(",");
    Serial.println(fz, 4);
}

// ─────────────────────────────────────────────────────────────────────────────
// setup
// ─────────────────────────────────────────────────────────────────────────────
void setup()
{
    Serial.begin(SERIAL_BAUD);
    // No "while (!Serial)" — blocks headless/deployed devices forever

    // Hardware watchdog: restart if loop freezes for > 3 s
    const esp_task_wdt_config_t wdt_config = {
        .timeout_ms = 3000,
        .idle_core_mask = 0,
        .trigger_panic = true
    };
    esp_task_wdt_init(&wdt_config);
    esp_task_wdt_add(NULL);

    // IMU init with retry
    uint8_t retry = 0;
    while (!IMU_Init())
    {
        Serial.printf("WARN: IMU init attempt %u failed\n", retry + 1);
        if (++retry >= INIT_MAX_RETRY)
        {
            Serial.println("FATAL: IMU init failed, restarting");
            ESP.restart();
        }
        delay(200);
    }

    // Hardware timer: fires every 2000 µs = 500 Hz
    timer = timerBegin(1000000); // 1 MHz tick resolution
    timerAttachInterrupt(timer, &onTimer);
    timerAlarm(timer, SAMPLE_PERIOD_US, true, 0);

    Serial.println("timestamp_ms,accX,accY,accZ"); // CSV header
}

// ─────────────────────────────────────────────────────────────────────────────
// loop
// ─────────────────────────────────────────────────────────────────────────────
void loop()
{
    esp_task_wdt_reset(); // feed watchdog every iteration

    // Atomic snapshot of ISR counter (uint32_t read is atomic on ESP32)
    uint32_t current_count = isr_sample_count;

    if (current_count == processed_count)
        return; // nothing new

    // Detect dropped samples
    uint32_t delta = current_count - processed_count;
    if (delta > 1)
    {
        Serial.printf("WARN: %lu sample(s) dropped\n", (unsigned long)(delta - 1));
    }

    processed_count = current_count;
    uint32_t snapshot_us = isr_timestamp_us; // safe uint32_t snapshot

    AccelData raw;
    if (IMU_ReadAccel(raw))
    {
        consecutive_errors = 0;
        processSample(raw, snapshot_us);
    }
    else
    {
        consecutive_errors++;
        if (consecutive_errors >= ERROR_THRESHOLD)
        {
            Serial.println("ERROR: IMU unresponsive, reinitializing");
            if (!IMU_Init())
            {
                Serial.println("FATAL: IMU reinit failed, restarting");
                ESP.restart();
            }
            consecutive_errors = 0;
        }
    }
}