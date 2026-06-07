#include <Wire.h>

#define I2C_SDA 21
#define I2C_SCL 22
#define IMU_ADDRESS 0x68
#define IMU_REG_PWR_MGMT_1 0x6B
#define IMU_REG_ACCEL_CONFIG 0x1C
#define IMU_REG_ACCEL_XOUT_H 0x3B

const float ACCEL_SCALE = 1.0f / 16384.0f; // Scale cho cụm +/-2g (Đơn vị: g)

struct AccelData
{
    int16_t x;
    int16_t y;
    int16_t z;
};

// High_Pass Filtering:
const float HPF_ALPHA = 0.9843f;

float prev_raw_x = 0.0f, prev_filt_x = 0.0f;
float prev_raw_y = 0.0f, prev_filt_y = 0.0f;
float prev_raw_z = 0.0f, prev_filt_z = 0.0f;

const char *CURRENT_LABEL = "healthy";

hw_timer_t *timer = NULL;
volatile bool samplingTriggered = false;

// ISR:
void IRAM_ATTR onTimer()
{
    samplingTriggered = true;
}

bool IMU_Init()
{
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(100000); // 100kHz
    delay(100);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_PWR_MGMT_1);
    Wire.write(0x00); // Wake up
    if (Wire.endTransmission() != 0)
        return false;

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_CONFIG);
    Wire.write(0x00); // Set full scale range to +/- 2g
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
        return false; //

    data.x = (Wire.read() << 8) | Wire.read(); //
    data.y = (Wire.read() << 8) | Wire.read(); //
    data.z = (Wire.read() << 8) | Wire.read();
    return true;
}

void setup()
{
    Serial.begin(115200);
    while (!Serial)
        ;

    if (!IMU_Init())
    {
        Serial.println("IMU Init Failed!");
        while (1)
            ;
    }

    // Cấu hình Hardware Timer chạy ở tần số 200Hz (Chu kỳ 5000 microgiây)
    timer = timerBegin(0, 80, true);
    timerAttachInterrupt(timer, &onTimer, true);
    timerAlarmWrite(timer, 5000, true); // 5000 us = 5 ms = 200 Hz
    timerAlarmEnable(timer);
}

void loop()
{
    if (samplingTriggered)
    {
        samplingTriggered = false; // Clear flag

        AccelData raw_counts;
        if (IMU_ReadAccel(raw_counts))
        {
            // Convert ADS Counts to g
            float curr_raw_x = (float)raw_counts.x * ACCEL_SCALE;
            float curr_raw_y = (float)raw_counts.y * ACCEL_SCALE;
            float curr_raw_z = (float)raw_counts.z * ACCEL_SCALE;

            // Algorithm for filtering the gravity:
            // IIR Algorithm 1: Y[n] = Alpha * (Y[n-1] + X[n] - X[n-1])
            float curr_filt_x = HPF_ALPHA * (prev_filt_x + curr_raw_x - prev_raw_x);
            float curr_filt_y = HPF_ALPHA * (prev_filt_y + curr_raw_y - prev_raw_y);
            float curr_filt_z = HPF_ALPHA * (prev_filt_z + curr_raw_z - prev_raw_z);

            prev_raw_x = curr_raw_x;
            prev_filt_x = curr_filt_x;
            prev_raw_y = curr_raw_y;
            prev_filt_y = curr_filt_y;
            prev_raw_z = curr_raw_z;
            prev_filt_z = curr_filt_z;

            // 3. Package and Print on Serial:
            Serial.print(curr_raw_x, 4);
            Serial.print(",");
            Serial.print(curr_filt_x, 4);
            Serial.print(",");
            Serial.print(curr_raw_y, 4);
            Serial.print(",");
            Serial.print(curr_filt_y, 4);
            Serial.print(",");
            Serial.print(curr_raw_z, 4);
            Serial.print(",");
            Serial.print(curr_filt_z, 4);
            Serial.print(",");
            Serial.println(CURRENT_LABEL);
        }
    }
}