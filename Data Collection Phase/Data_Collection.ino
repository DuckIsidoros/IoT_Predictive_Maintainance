#include <Wire.h>

// =========================
// MPU6050 Registers
// =========================
#define MPU6050_ADDR 0x68
#define PWR_MGMT_1 0x6B
#define ACCEL_CONFIG 0x1C
#define ACCEL_XOUT_H 0x3B

// =========================
// ESP32 I2C Pins
// =========================
#define SDA_PIN 21
#define SCL_PIN 22

// =========================
// Sampling Configuration
// =========================
#define SAMPLE_RATE_HZ 1000

const float ACCEL_SCALE = 1.0f / 16384.0f;

// =========================
// Timer Variables
// =========================
hw_timer_t *timer = NULL;

volatile bool sampleFlag = false;

portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

// =========================
// Timer ISR
// =========================
void IRAM_ATTR onTimer()
{
    portENTER_CRITICAL_ISR(&timerMux);
    sampleFlag = true;
    portEXIT_CRITICAL_ISR(&timerMux);
}

// =========================
// MPU6050 Init
// =========================
bool MPU6050_Init()
{
    Wire.begin(SDA_PIN, SCL_PIN);

    Wire.setClock(400000);

    delay(100);

    // Wake up MPU6050
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(PWR_MGMT_1);
    Wire.write(0x00);

    if (Wire.endTransmission() != 0)
        return false;

    delay(50);

    // Set accel range ±2g
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(ACCEL_CONFIG);
    Wire.write(0x00);

    if (Wire.endTransmission() != 0)
        return false;

    delay(50);

    return true;
}

// =========================
// Read Acceleration
// =========================
bool MPU6050_Read(float &ax, float &ay, float &az)
{
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(ACCEL_XOUT_H);

    if (Wire.endTransmission(false) != 0)
        return false;

    if (Wire.requestFrom(MPU6050_ADDR, 6) != 6)
        return false;

    int16_t rawX = (Wire.read() << 8) | Wire.read();
    int16_t rawY = (Wire.read() << 8) | Wire.read();
    int16_t rawZ = (Wire.read() << 8) | Wire.read();

    ax = rawX * ACCEL_SCALE;
    ay = rawY * ACCEL_SCALE;
    az = rawZ * ACCEL_SCALE;

    return true;
}

// =========================
// Setup
// =========================
void setup()
{
    Serial.begin(115200);

    while (!Serial)
    {
    }

    if (!MPU6050_Init())
    {
        Serial.println("MPU6050 Init Failed");

        while (1)
        {
        }
    }

    // Timer config
    // 80 MHz / 80 = 1 MHz
    // 1 tick = 1 us
    timer = timerBegin(0, 80, true);

    timerAttachInterrupt(timer, &onTimer, true);

    timerAlarmWrite(
        timer,
        1000000 / SAMPLE_RATE_HZ,
        true);

    timerAlarmEnable(timer);

    // CSV Header
    Serial.println("timestamp_us,ax,ay,az");
}

// =========================
// Main Loop
// =========================
void loop()
{
    if (sampleFlag)
    {
        portENTER_CRITICAL(&timerMux);
        sampleFlag = false;
        portEXIT_CRITICAL(&timerMux);

        float ax, ay, az;

        if (MPU6050_Read(ax, ay, az))
        {
            uint64_t timestamp = esp_timer_get_time();

            Serial.print(timestamp);

            Serial.print(",");

            Serial.print(ax, 6);

            Serial.print(",");

            Serial.print(ay, 6);

            Serial.print(",");

            Serial.println(az, 6);
        }
    }
}