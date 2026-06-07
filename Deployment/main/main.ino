#include <Wire.h>
#include <math.h>
#include "model_parameters.h" // Chứa W1, b1, W2, b2, W3, b3 xuất từ Python

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

// Cấu hình Hardware Timer
hw_timer_t *timer = NULL;
volatile bool samplingTriggered = false;

// Bộ đệm tính toán RMS (Cửa sổ 1 giây = 200 mẫu ở tần số 200Hz)
const int WINDOW_SIZE = 200;
int sampleCount = 0;
float sumSqX = 0, sumSqY = 0, sumSqZ = 0;

// Các biến lưu kết quả suy luận để hiển thị
float prob_class0 = 0.0f;
float prob_class1 = 0.0f;
float prob_class2 = 0.0f;
int predicted_class = 0;
unsigned long last_inference_time = 0;

void IRAM_ATTR onTimer()
{
    samplingTriggered = true;
}

bool IMU_Init()
{
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000); // 400kHz Fast Mode tối ưu Bus Timing
    delay(100);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_PWR_MGMT_1);
    Wire.write(0x00); // Wake up cảm biến
    if (Wire.endTransmission() != 0)
        return false;
    delay(50);

    Wire.beginTransmission(IMU_ADDRESS);
    Wire.write(IMU_REG_ACCEL_CONFIG);
    Wire.write(0x00); // Thiết lập dải đo +/-2g
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

// --- MODULE THUẬT TOÁN ĐỒNG BỘ THEO FILE TRAIN ---

float relu(float x)
{
    return (x > 0.0f) ? x : 0.0f;
}

// Hàm suy luận Small Dense NN (MLP 3 Tầng)
void predict_dense_nn(float rmsX, float rmsY, float rmsZ)
{
    float inputs[3] = {rmsX, rmsY, rmsZ};

    // Khai báo mảng tĩnh cục bộ để tối ưu bộ nhớ Stack, không dùng malloc
    float h1[16] = {0};
    float h2[8] = {0};
    float scores[3] = {0};

    // Tầng ẩn 1: Tuyến tính -> Lớp ẩn 1 (3 inputs -> 16 nodes) + ReLU
    for (int j = 0; j < 16; j++)
    {
        float act = b1[j];
        for (int i = 0; i < 3; i++)
        {
            act += inputs[i] * W1[i][j]; // Lưu ý: W1 cấu trúc mảng trích xuất từ Python
        }
        h1[j] = relu(act);
    }

    // Tầng ẩn 2: Lớp ẩn 1 -> Lớp ẩn 2 (16 nodes -> 8 nodes) + ReLU
    for (int j = 0; j < 8; j++)
    {
        float act = b2[j];
        for (int i = 0; i < 16; i++)
        {
            act += h1[i] * W2[i][j];
        }
        h2[j] = relu(act);
    }

    // Tầng đầu ra: Lớp ẩn 2 -> Lớp đầu ra (8 nodes -> 3 classes)
    float max_score = -999999.0f;
    for (int j = 0; j < 3; j++)
    {
        float act = b3[j];
        for (int i = 0; i < 8; i++)
        {
            act += h2[i] * W3[i][j];
        }
        scores[j] = act;
        if (scores[j] > max_score)
            max_score = scores[j]; // Tìm max để chống tràn số (Stable Softmax)
    }

    // Hàm kích hoạt Softmax chuyển đổi điểm số (logits) thành xác suất phần trăm
    float sum_exp = 0.0f;
    float exps[3] = {0};
    for (int j = 0; j < 3; j++)
    {
        exps[j] = exp(scores[j] - max_score); // Trừ max_score để đảm bảo an toàn số học
        sum_exp += exps[j];
    }

    prob_class0 = exps[0] / sum_exp;
    prob_class1 = exps[1] / sum_exp;
    prob_class2 = exps[2] / sum_exp;

    // Phân loại argmax
    if (prob_class0 >= prob_class1 && prob_class0 >= prob_class2)
        predicted_class = 0;
    else if (prob_class1 >= prob_class0 && prob_class1 >= prob_class2)
        predicted_class = 1;
    else
        predicted_class = 2;
}

// --- MODULE HIỂN THỊ DASHBOARD TRỰC QUAN (GIẢI PHÁP 2) ---

void print_dashboard(float rmsX, float rmsY, float rmsZ)
{
    // Xóa màn hình cũ và đẩy con trỏ về gốc trái (Mã điều khiển ANSI)
    Serial.print("\033[2J");
    Serial.print("\033[H");

    // Chuẩn bị thanh tiến trình cho 3 nhãn phân loại (Scale 20 ký tự)
    int bar0 = prob_class0 * 20;
    int bar1 = prob_class1 * 20;
    int bar2 = prob_class2 * 20;

    Serial.println(F("=================================================="));
    Serial.println(F("[1] EDGE DATA ACQUISITION (Cửa sổ 200 mẫu - 1 Giây)"));
    Serial.printf("- RMS Trục X: %.3fG  |  RMS Trục Y: %.3fG  |  RMS Trục Z: %.3fG\n", rmsX, rmsY, rmsZ);
    Serial.println();

    Serial.println(F("[2] MULTI-LAYER PERCEPTRON (Small Dense NN trên Chip)"));

    // Class 0: Bình thường
    Serial.print(F("- Trạng thái BÌNH THƯỜNG   : ["));
    for (int i = 0; i < 20; i++)
        Serial.print(i < bar0 ? "|" : ".");
    Serial.printf("] %.1f%%\n", prob_class0 * 100);

    // Class 1: Lỗi kẹt/bẩn cánh quạt (Ví dụ tương ứng từ nhãn train của bạn)
    Serial.print(F("- Trạng thái LỖI KẸT CÁNH  : ["));
    for (int i = 0; i < 20; i++)
        Serial.print(i < bar1 ? "|" : ".");
    Serial.printf("] %.1f%%\n", prob_class1 * 100);

    // Class 2: Lỗi lỏng ốc/rung lắc khung
    Serial.print(F("- Trạng thái RUNG LẮC KHUNG: ["));
    for (int i = 0; i < 20; i++)
        Serial.print(i < bar2 ? "|" : ".");
    Serial.printf("] %.1f%%\n", prob_class2 * 100);
    Serial.println();

    Serial.print(F(" => KẾT LUẬN CUỐI CÙNG: "));
    if (predicted_class == 0)
        Serial.println(F("QUẠT HOẠT ĐỘNG TỐT"));
    else if (predicted_class == 1)
        Serial.println(F("CẢNH BÁO: LỖI KẸT CÁNH / QUÁ TẢI"));
    else
        Serial.println(F("CẢNH BÁO: RUNG LẮC CƠ HỌC MẠNH"));
    Serial.println();

    Serial.println(F("[3] HARDWARE TIMING & RESOURCES BENCHMARK"));
    Serial.printf("- Thời gian suy luận mạng NN : %d us (Microseconds)\n", last_inference_time);
    Serial.printf("- Flash Memory Footprint     : Thấp (Mảng hằng số lưu trong ROM)\n");
    Serial.printf("- Free Heap RAM hiện tại     : %d bytes\n", esp_get_free_heap_size());
    Serial.printf("- Minimum Free Heap (Peak)   : %d bytes\n", esp_get_minimum_free_heap_size());
    Serial.println(F("=================================================="));
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

    // Cấu hình Hardware Timer kích hoạt chu kỳ ngắt đúng 5ms (200Hz)
    timer = timerBegin(0, 80, true);
    timerAttachInterrupt(timer, &onTimer, true);
    timerAlarmWrite(timer, 5000, true);
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
            // Chuẩn hóa sang đơn vị Gia tốc trọng trường (G)
            float ax = rawData.x * ACCEL_SCALE;
            float ay = rawData.y * ACCEL_SCALE;
            float az = rawData.z * ACCEL_SCALE;

            // Tính toán Feature cuốn chiếu: Cộng dồn bình phương
            sumSqX += ax * ax;
            sumSqY += ay * ay;
            sumSqZ += az * az;
            sampleCount++;

            // Khi gom đủ 1 giây dữ liệu (Đủ 200 mẫu)
            if (sampleCount >= WINDOW_SIZE)
            {
                // Đạt mốc cửa sổ thời gian -> Tính toán căn bậc hai để ra RMS hoàn chỉnh
                float rmsX = sqrt(sumSqX / WINDOW_SIZE);
                float rmsY = sqrt(sumSqY / WINDOW_SIZE);
                float rmsZ = sqrt(sumSqZ / WINDOW_SIZE);

                // Thao tác đo đạc hiệu năng thực thi của thuật toán AI
                unsigned long start_time = micros();
                predict_dense_nn(rmsX, rmsY, rmsZ);
                last_inference_time = micros() - start_time;

                // Xuất giao diện Dashboard tĩnh ra màn hình Serial Terminal
                print_dashboard(rmsX, rmsY, rmsZ);

                // Reset các bộ tích lũy để chuẩn bị cho chu kỳ 1 giây tiếp theo
                sampleCount = 0;
                sumSqX = 0;
                sumSqY = 0;
                sumSqZ = 0;
            }
        }
    }
}