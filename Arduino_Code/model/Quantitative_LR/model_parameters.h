/*
 * model_parameters.h
 * Auto-generated header file for ESP32 edge deployment.
 * Architecture: Logistic Regression (One-vs-Rest)
 * Optimization: Memory-mapped to Flash ROM via 'const' definitions.
 * DO NOT MODIFY THIS FILE MANUALLY.
 */

#ifndef MODEL_PARAMETERS_H
#define MODEL_PARAMETERS_H

#include <stdint.h>

#define MODEL_NUM_CLASSES   3
#define MODEL_NUM_FEATURES  6

// ==============================================================================
// RUNTIME DATA NORMALIZATION (STANDARDSALER OVERHEAD SYNC)
// ==============================================================================
// Math constraint on MCU: x_scaled = (x_raw - MEAN) / STD
const float SCALER_MEAN[MODEL_NUM_FEATURES] = { 0.040071f, 0.045238f, 0.139867f, 0.000408f, 0.001035f, 0.001338f };
const float SCALER_STD[MODEL_NUM_FEATURES]  = { 0.026997f, 0.034539f, 0.087689f, 0.001787f, 0.002012f, 0.002505f };

// ==============================================================================
// LOGISTIC REGRESSION WEIGHT MATRICES & BIAS VECTORS
// ==============================================================================
// Layout: Row-major matrix representation [Classes][Features]
const float LGR_WEIGHTS[MODEL_NUM_CLASSES][MODEL_NUM_FEATURES] = {
    { -3.479707f, -0.990258f, -1.308461f, -0.196213f, -1.259891f, 1.772919f }, // Class 0 Target: healthy
    { 4.120498f, 0.307367f, -2.686278f, -3.389503f, 1.312136f, -0.855305f }, // Class 1 Target: imbalanced
    { -0.640791f, 0.682891f, 3.994739f, 3.585716f, -0.052245f, -0.917615f }, // Class 2 Target: obstruction
};

const float LGR_BIASES[MODEL_NUM_CLASSES] = { -3.343172f, 0.421916f, 2.921256f };

const char* const MODEL_CLASS_LABELS[MODEL_NUM_CLASSES] = {
    "healthy",
    "imbalanced",
    "obstruction",
};

#endif // MODEL_PARAMETERS_H
