#include <Wire.h>

#define I2C_SDA 21
#define I2C_SCL 22
#define IMU_ADDRESS 0x68
#define IMU_REG_PWR_MGMT_1 0x6B
#define IMU_REG_ACCEL_CONFIG 0x1C
#define IMU_REG_ACCEL_XOUT_H 0x3B

const float ACCEL_SCALE = 1.0f / 16384.0f;
const float HPF_ALPHA = 0.9843f;

struct AccelData {
    int16_t x;
    int16_t y;
    int16_t z;
};

float prev_raw_x = 0.0f, prev_filt_x = 0.0f;
float prev_raw_y = 0.0f, prev_filt_y = 0.0f;
float prev_raw_z = 0.0f, prev_filt_z = 0.0f;

const char *CURRENT_LABEL = "healthy";

// Sử dụng kiểu dữ liệu mới cho ESP32 Core 3.x
hw_timer_t *timer = NULL;
volatile bool samplingTriggered = false;
uint32_t error_counter = 0;

void IRAM_ATTR onTimer() {
    samplingTriggered = true;
}

bool IMU_Init() {
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000); // Nâng lên 400kHz để giải phóng Timing Budget
    delay(50);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_PWR_MGMT_1);
    Wire.write(0x00); // Wake up
    if (Wire.endTransmission() != 0) return false;

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_CONFIG);
    Wire.write(0x00); // +/- 2g
    if (Wire.endTransmission() != 0) return false;

    return true;
}

bool IMU_ReadAccel(AccelData &data) {
    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_XOUT_H);
    if (Wire.endTransmission(false) != 0) return false;

    if (Wire.requestFrom(IMU_ADDRESS, 6) < 6) return false;

    data.x = (Wire.read() << 8) | Wire.read();
    data.y = (Wire.read() << 8) | Wire.read();
    data.z = (Wire.read() << 8) | Wire.read();
    return true;
}

void setup() {
    Serial.begin(115200);
    while (!Serial);

    if (!IMU_Init()) {
        Serial.println("IMU Init Failed!");
        while (1); 
    }

    // Cập nhật API Timer mới theo chuẩn ESP32 Arduino Core v3.x
    // Tần số tick = 1,000,000 Hz (1 us/tick). 5000 ticks = 5ms = 200Hz
    timer = timerBegin(1000000); 
    timerAttachInterrupt(timer, &onTimer);
    timerAlarm(timer, 5000, true, 0); 
}

void loop() {
    if (samplingTriggered) {
        samplingTriggered = false;

        AccelData raw_counts;
        if (IMU_ReadAccel(raw_counts)) {
            error_counter = 0; // Reset counter khi đọc thành công

            float curr_raw_x = (float)raw_counts.x * ACCEL_SCALE;
            float curr_raw_y = (float)raw_counts.y * ACCEL_SCALE;
            float curr_raw_z = (float)raw_counts.z * ACCEL_SCALE;

            // IIR High-pass Filter
            float curr_filt_x = HPF_ALPHA * (prev_filt_x + curr_raw_x - prev_raw_x);
            float curr_filt_y = HPF_ALPHA * (prev_filt_y + curr_raw_y - prev_raw_y);
            float curr_filt_z = HPF_ALPHA * (prev_filt_z + curr_raw_z - prev_raw_z);

            prev_raw_x = curr_raw_x; prev_filt_x = curr_filt_x;
            prev_raw_y = curr_raw_y; prev_filt_y = curr_filt_y;
            prev_raw_z = curr_raw_z; prev_filt_z = curr_filt_z;

            // In dữ liệu (Khuyên dùng: Tăng baudrate Serial lên 230400 hoặc 460800 nếu log thêm data)
            Serial.print(curr_raw_x, 4);   Serial.print(",");
            Serial.print(curr_filt_x, 4);  Serial.print(",");
            Serial.print(curr_raw_y, 4);   Serial.print(",");
            Serial.print(curr_filt_y, 4);  Serial.print(",");
            Serial.print(curr_raw_z, 4);   Serial.print(",");
            Serial.print(curr_filt_z, 4);  Serial.print(",");
            Serial.println(CURRENT_LABEL);
        } else {
            error_counter++;
            // Nếu lỗi liên tục trong 1 giây (200 chu kỳ), tiến hành Re-init Bus
            if (error_counter > 200) {
                IMU_Init();
                error_counter = 0;
            }
        }
    }
}
