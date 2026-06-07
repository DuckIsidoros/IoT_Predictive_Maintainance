#include <Wire.h>

#define I2C_SDA 21
#define I2C_SCL 22
#define IMU_ADDRESS 0x68
#define IMU_REG_PWR_MGMT_1 0x6B
#define IMU_REG_ACCEL_CONFIG 0x1C
#define IMU_REG_ACCEL_XOUT_H 0x3B

const float ACCEL_SCALE = 1.0f / 16384.0f; // Scale cho cụm +/-2g

struct AccelData
{
    int16_t x;
    int16_t y;
    int16_t z;
};

// Cấu hình Timer
hw_timer_t *timer = NULL;
volatile bool samplingTriggered = false;

// Biến phục vụ tính toán Feature (RMS) sơ khai để demo
const int WINDOW_SIZE = 200; // 200 mẫu ở tần số 200Hz = Đúng 1 giây dữ liệu
int sampleCount = 0;
float sumSqX = 0, sumSqY = 0, sumSqZ = 0;

void IRAM_ATTR onTimer()
{
    samplingTriggered = true;
}

bool IMU_Init()
{
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000); // ĐÃ SỬA: Tăng lên 400kHz (Fast Mode) để tối ưu timing
    delay(100);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_PWR_MGMT_1);
    Wire.write(0x00); // Wake up
    if (Wire.endTransmission() != 0)
        return false;
    delay(50);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_CONFIG);
    Wire.write(0x00); // Set FS: +/-2g
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

void setup()
{
    Serial.begin(115200); // Giữ 115200 để an toàn cho mọi máy tính, nhưng đã giảm tần suất in
    while (!Serial)
        ;

    if (!IMU_Init())
    {
        Serial.println("IMU Init Failed!");
        while (1)
            ;
    }

    // Cấu hình Hardware Timer chạy ở tần số 200Hz
    timer = timerBegin(0, 80, true);
    timerAttachInterrupt(timer, &onTimer, true);
    timerAlarmWrite(timer, 5000, true); // 5000 us = 5 ms
    timerAlarmEnable(timer);
}

void print_dashboard(float rmsX, float rmsY, float rmsZ)
{
    // Giải pháp 2: Gửi mã điều khiển ANSI để xóa màn hình và đưa con trỏ về góc trái trên cùng
    Serial.print("\033[2J");
    Serial.print("\033[H");

    // Giả lập kết quả Model (Vì bạn đang test deploy pipeline chưa tối ưu accuracy)
    float prob_normal = 0.85; // Giả lập mẫu
    float prob_fault = 1.0 - prob_normal;
    int bar_normal = prob_normal * 20;

    Serial.println(F("=================================================="));
    Serial.println(F("[1] DATA ACQUISITION (Cửa sổ dữ liệu: 200 mẫu - 1s)"));
    Serial.printf("- RMS X: %.3fG  |  RMS Y: %.3fG  |  RMS Z: %.3fG\n", rmsX, rmsY, rmsZ);
    Serial.println();

    Serial.println(F("[2] EDGE AI INFERENCE (Mô hình tuyến tính)"));
    Serial.print(F("- Trạng thái BÌNH THƯỜNG: ["));
    for (int i = 0; i < 20; i++)
        Serial.print(i < bar_normal ? "|" : ".");
    Serial.printf("] %.1f%%\n", prob_normal * 100);

    Serial.print(F("- Trạng thái LỖI/RUNG  : ["));
    for (int i = 0; i < 20; i++)
        Serial.print(i >= bar_normal ? "|" : ".");
    Serial.printf("] %.1f%%\n", prob_fault * 100);
    Serial.println();

    Serial.println(F("[3] HARDWARE BENCHMARK"));
    Serial.printf("- Free Heap RAM         : %d bytes\n", esp_get_free_heap_size());
    Serial.printf("- Minimum Free Heap     : %d bytes\n", esp_get_minimum_free_heap_size());
    Serial.println(F("=================================================="));
}

void loop()
{
    if (samplingTriggered)
    {
        samplingTriggered = false; // Xóa cờ ngắt

        AccelData rawData;
        if (IMU_ReadAcceleration(rawData))
        {
            // Chuyển đổi sang đơn vị G
            float ax = rawData.x * ACCEL_SCALE;
            float ay = rawData.y * ACCEL_SCALE;
            float az = rawData.z * ACCEL_SCALE;

            // Tích lũy bình phương để tính RMS (Root Mean Square)
            sumSqX += ax * ax;
            sumSqY += ay * ay;
            sumSqZ += az * az;
            sampleCount++;

            // Khi gom đủ 1 giây dữ liệu (200 mẫu)
            if (sampleCount >= WINDOW_SIZE)
            {
                // Tính toán RMS thực tế
                float rmsX = sqrt(sumSqX / WINDOW_SIZE);
                float rmsY = sqrt(sumSqY / WINDOW_SIZE);
                float rmsZ = sqrt(sumSqZ / WINDOW_SIZE);

                // Gọi hàm hiển thị giao diện tĩnh xóa-và-in-lại
                print_dashboard(rmsX, rmsY, rmsZ);

                // Reset bộ đếm cho chu kỳ 1 giây tiếp theo
                sampleCount = 0;
                sumSqX = 0;
                sumSqY = 0;
                sumSqZ = 0;
            }
        }
    }
}