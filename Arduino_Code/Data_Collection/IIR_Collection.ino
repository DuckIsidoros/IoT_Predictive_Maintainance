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

hw_timer_t *timer = NULL;
volatile bool samplingTriggered = false;
volatile uint32_t last_sample_time_us = 0; // Lưu thời gian thực tính bằng microgiây
uint32_t error_counter = 0;

void IRAM_ATTR onTimer() {
    samplingTriggered = true;
    last_sample_time_us = micros(); // Lấy mốc thời gian chính xác tuyệt đối từ ngắt cứng
}

bool IMU_Init() {
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000); 
    delay(50);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_PWR_MGMT_1);
    Wire.write(0x00); 
    if (Wire.endTransmission() != 0) return false;

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_CONFIG);
    Wire.write(0x00); 
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
        Serial.println("ERROR: IMU Initialization Failed");
        while (1); 
    }

    timer = timerBegin(1000000); 
    timerAttachInterrupt(timer, &onTimer);
    timerAlarm(timer, 5000, true, 0); // Khóa cứng chu kỳ 5000 us = 5 ms (200Hz)
}

void loop() {
    if (samplingTriggered) {
        samplingTriggered = false;
        
        // Đọc snapshot thời gian từ ngắt để tránh bị lệch chu kỳ do toán tử phía sau
        uint32_t sample_timestamp_us = last_sample_time_us; 

        AccelData raw_counts;
        if (IMU_ReadAccel(raw_counts)) {
            error_counter = 0; 

            float curr_raw_x = (float)raw_counts.x * ACCEL_SCALE;
            float curr_raw_y = (float)raw_counts.y * ACCEL_SCALE;
            float curr_raw_z = (float)raw_counts.z * ACCEL_SCALE;

            float curr_filt_x = HPF_ALPHA * (prev_filt_x + curr_raw_x - prev_raw_x);
            float curr_filt_y = HPF_ALPHA * (prev_filt_y + curr_raw_y - prev_raw_y);
            float curr_filt_z = HPF_ALPHA * (prev_filt_z + curr_raw_z - prev_raw_z);

            prev_raw_x = curr_raw_x; prev_filt_x = curr_filt_x;
            prev_raw_y = curr_raw_y; prev_filt_y = curr_filt_y;
            prev_raw_z = curr_raw_z; prev_filt_z = curr_filt_z;

            // Chuyển đổi us sang ms dạng số thực để giữ nguyên độ chính xác cao cho validator
            float timestamp_ms = (float)sample_timestamp_us / 1000.0f;

            // In dữ liệu ra Serial theo đúng cấu trúc cột yêu cầu
            Serial.print(timestamp_ms, 3); Serial.print(",");
            Serial.print(curr_raw_x, 4);    Serial.print(",");
            Serial.print(curr_filt_x, 4);   Serial.print(",");
            Serial.print(curr_raw_y, 4);    Serial.print(",");
            Serial.print(curr_filt_y, 4);   Serial.print(",");
            Serial.print(curr_raw_z, 4);    Serial.print(",");
            Serial.print(curr_filt_z, 4);   Serial.print(",");
            Serial.println(CURRENT_LABEL);
        } else {
            error_counter++;
            if (error_counter > 200) {
                IMU_Init();
                error_counter = 0;
            }
        }
    }
}
