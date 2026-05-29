#include <Wire.h>

#define I2C_SDA 21
#define I2C_SCL 22
#define IMU_ADDRESS 0x68
#define IMU_REG_PWR_MGMT_1 0x6B
#define IMU_REG_ACCEL_CONFIG 0x1C
#define IMU_REG_ACCEL_XOUT_H 0x3B

const float ACCEL_SCALE = 1.0f / 16384.0f; // Scale cho cụm +/-2g [cite: 165, 171]

struct AccelData
{
    int16_t x;
    int16_t y;
    int16_t z;
};

// Khai báo biến phục vụ Timer Interrupt
hw_timer_t *timer = NULL;
volatile bool samplingTriggered = false;

// Hàm ngắt Timer (ISR) - Chỉ bật cờ hiệu, không xử lý logic nặng tại đây
void IRAM_ATTR onTimer()
{
    samplingTriggered = true;
}

bool IMU_Init()
{
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(100000); // 100kHz [cite: 169]
    delay(100);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_PWR_MGMT_1);
    Wire.write(0x00); // Wake up [cite: 170]
    if (Wire.endTransmission() != 0)
        return false;
    delay(50);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_CONFIG);
    Wire.write(0x00); // Set FS: +/-2g [cite: 171]
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
    [cite:173]

        if (Wire.requestFrom(IMU_ADDRESS, 6) < 6) return false;
    [cite:174, 175]

        data.x = (Wire.read() << 8) | Wire.read();
    [cite:175] data.y = (Wire.read() << 8) | Wire.read();
    [cite:176] data.z = (Wire.read() << 8) | Wire.read();
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
    // ESP32 clock mặc định 80MHz, chia 80 => 1 tick = 1 microgiây
    timer = timerBegin(0, 80, true);
    timerAttachInterrupt(timer, &onTimer, true);
    timerAlarmWrite(timer, 5000, true); // 5000 us = 5 ms = 200 Hz
    timerAlarmEnable(timer);
}

void loop()
{
    if (samplingTriggered)
    {
        samplingTriggered = false; // Xóa cờ ngắt

        AccelData rawData;
        if (IMU_ReadAcceleration(rawData))
        {
            // Định dạng dữ liệu dạng CSV chuẩn để Edge Impulse CLI hoặc Python dễ đọc
            // Thao tác in Serial này tốn dưới 1ms, hoàn toàn an toàn trong chu kỳ 5ms
            Serial.print(rawData.x * ACCEL_SCALE);
            Serial.print(",");
            Serial.print(rawData.y * ACCEL_SCALE);
            Serial.print(",");
            Serial.println(rawData.z * ACCEL_SCALE);
        }
    }
}