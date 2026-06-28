    /*
 * inference.h
 * Forward pass engine for Small Dense NN on ESP32 (Arduino IDE).
 *
 * Architecture : MLP 3-layer [8 -> 16 -> 8 -> 3]
 * Activation   : ReLU (hidden layers), Softmax (output layer)
 * Convention   : W[output_neuron][input_neuron]  — row-major dot product
 *
 * Usage:
 *   1. Include file này SAU khi include model_parameters.h
 *      #include "model_parameters.h"
 *      #include "inference.h"
 *
 *   2. Gọi inference() với mảng raw features (chưa scale):
 *      float raw[8] = { rms_x, rms_y, rms_z,
 *                        bp_low, bp_mid, bp_high,
 *                        crest_z, kurtosis_z };
 *      InferenceResult result = inference(raw);
 *      Serial.println(result.label);
 *      Serial.println(result.confidence);
 *
 * DO NOT MODIFY — regenerate from notebook export cell.
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include <Arduino.h>
#include <math.h>
#include "model_parameters.h"

/* ─────────────────────────────────────────────────────────────────────
 * Kiểu trả về của inference()
 * ───────────────────────────────────────────────────────────────────── */
struct InferenceResult
{
    uint8_t class_index;            // index của class thắng (0, 1, 2)
    const char *label;              // chuỗi nhãn: "healthy" / "imbalanced" / "obstruction"
    float confidence;               // xác suất softmax của class thắng (0.0 – 1.0)
    float probs[MODEL_NUM_CLASSES]; // xác suất softmax toàn bộ classes
};

/* ─────────────────────────────────────────────────────────────────────
 * Bước 1: Chuẩn hóa Z-score
 *   output[i] = (input[i] - SCALER_MEAN[i]) / SCALER_STD[i]
 * ───────────────────────────────────────────────────────────────────── */
static void standardize(const float *raw, float *scaled)
{
    for (uint8_t i = 0; i < MODEL_NUM_FEATURES; i++)
    {
        scaled[i] = (raw[i] - SCALER_MEAN[i]) / SCALER_STD[i];
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * Bước 2: Dense layer + ReLU
 *   out[j] = ReLU( sum_i(W[j][i] * in[i]) + B[j] )
 * ───────────────────────────────────────────────────────────────────── */
static void dense_relu(
    const float *in,
    float *out,
    const float W[][MODEL_NUM_FEATURES], // dùng cho Layer 1 (W1)
    const float *B,
    uint8_t in_dim,
    uint8_t out_dim)
{
    for (uint8_t j = 0; j < out_dim; j++)
    {
        float acc = B[j];
        for (uint8_t i = 0; i < in_dim; i++)
        {
            acc += W[j][i] * in[i];
        }
        out[j] = (acc > 0.0f) ? acc : 0.0f; // ReLU
    }
}

/* Overload cho Layer 2 (W2: 8x16) và Layer 3 (W3: 3x8) —
 * C++ không cho phép dùng cùng 1 hàm với kích thước array khác nhau,
 * nên dùng con trỏ 2D flatten qua macro helper bên dưới.             */
static void dense_layer(
    const float *in,
    float *out,
    const float *W_flat, // W đã flatten: W[j * in_dim + i]
    const float *B,
    uint8_t in_dim,
    uint8_t out_dim,
    bool use_relu)
{
    for (uint8_t j = 0; j < out_dim; j++)
    {
        float acc = B[j];
        for (uint8_t i = 0; i < in_dim; i++)
        {
            acc += W_flat[j * in_dim + i] * in[i];
        }
        if (use_relu)
        {
            out[j] = (acc > 0.0f) ? acc : 0.0f;
        }
        else
        {
            out[j] = acc; // logit thô, softmax áp sau
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * Bước 3: Softmax (numerically stable)
 *   Trừ max trước để tránh overflow khi exp() với giá trị lớn
 * ───────────────────────────────────────────────────────────────────── */
static void softmax(float *logits, float *probs, uint8_t n)
{
    // Tìm max
    float max_val = logits[0];
    for (uint8_t i = 1; i < n; i++)
    {
        if (logits[i] > max_val)
            max_val = logits[i];
    }

    // exp(x - max) và tính tổng
    float sum = 0.0f;
    for (uint8_t i = 0; i < n; i++)
    {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }

    // Normalize
    for (uint8_t i = 0; i < n; i++)
    {
        probs[i] /= sum;
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * Hàm chính: inference()
 *
 * @param raw_features  Mảng float[8] — đúng thứ tự:
 *                      [RMS_X, RMS_Y, RMS_Z,
 *                       Band_Power_Z_Low, Band_Power_Z_Mid, Band_Power_Z_High,
 *                       CrestFactor_Z, Kurtosis_Z]
 * @return InferenceResult  chứa class_index, label, confidence, probs[]
 * ───────────────────────────────────────────────────────────────────── */
InferenceResult inference(const float *raw_features)
{

    /* -- Layer 0: Chuẩn hóa Z-score ---------------------------------- */
    float scaled[MODEL_NUM_FEATURES];
    standardize(raw_features, scaled);

    /* -- Layer 1: Dense(16, ReLU)  input=8 --------------------------- */
    float h1[16];
    dense_layer(
        scaled,
        h1,
        (const float *)W1, // W1[16][8] flatten
        BIAS1,
        MODEL_NUM_FEATURES, // in_dim  = 8
        16,                 // out_dim = 16
        true                // ReLU
    );

    /* -- Layer 2: Dense(8, ReLU)  input=16 --------------------------- */
    float h2[8];
    dense_layer(
        h1,
        h2,
        (const float *)W2, // W2[8][16] flatten
        BIAS2,
        16,  // in_dim  = 16
        8,   // out_dim = 8
        true // ReLU
    );

    /* -- Layer 3: Dense(3, Softmax)  input=8 ------------------------- */
    float logits[MODEL_NUM_CLASSES];
    dense_layer(
        h2,
        logits,
        (const float *)W3, // W3[3][8] flatten
        BIAS3,
        8,                 // in_dim  = 8
        MODEL_NUM_CLASSES, // out_dim = 3
        false              // không ReLU — softmax áp sau
    );

    /* -- Softmax ------------------------------------------------------ */
    InferenceResult result;
    softmax(logits, result.probs, MODEL_NUM_CLASSES);

    /* -- Argmax: tìm class thắng ------------------------------------- */
    result.class_index = 0;
    result.confidence = result.probs[0];
    for (uint8_t i = 1; i < MODEL_NUM_CLASSES; i++)
    {
        if (result.probs[i] > result.confidence)
        {
            result.confidence = result.probs[i];
            result.class_index = i;
        }
    }
    result.label = MODEL_CLASS_LABELS[result.class_index];

    return result;
}

/* ─────────────────────────────────────────────────────────────────────
 * (Tuỳ chọn) Debug helper — in toàn bộ xác suất ra Serial
 * ───────────────────────────────────────────────────────────────────── */
void printInferenceResult(const InferenceResult &r)
{
    Serial.println(F("---- Inference Result ----"));
    for (uint8_t i = 0; i < MODEL_NUM_CLASSES; i++)
    {
        Serial.print(F("  "));
        Serial.print(MODEL_CLASS_LABELS[i]);
        Serial.print(F(": "));
        Serial.print(r.probs[i] * 100.0f, 2);
        Serial.println(F(" %"));
    }
    Serial.print(F("  >> Prediction : "));
    Serial.println(r.label);
    Serial.print(F("  >> Confidence : "));
    Serial.print(r.confidence * 100.0f, 2);
    Serial.println(F(" %"));
    Serial.println(F("--------------------------"));
}

#endif /* INFERENCE_H */