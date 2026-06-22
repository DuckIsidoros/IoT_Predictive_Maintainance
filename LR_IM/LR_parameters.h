/*
 * model_parameters.h
 * Auto-generated header file for ESP32 edge deployment.
 * Architecture: Logistic Regression (Multi-class Sync)
 * Optimization: Memory-mapped to Flash ROM via 'const' definitions.
 * DO NOT MODIFY THIS FILE MANUALLY.
 */

#ifndef MODEL_PARAMETERS_H
#define MODEL_PARAMETERS_H

#include <stdint.h>

#define MODEL_NUM_CLASSES   3
#define MODEL_NUM_FEATURES  8

// ==============================================================================
// RUNTIME DATA NORMALIZATION (STANDARDSALER OVERHEAD SYNC)
// ==============================================================================
const float SCALER_MEAN[MODEL_NUM_FEATURES] = { 1.004497f, 0.124778f, 0.618019f, 0.009496f, 0.003965f, 0.013249f, 2.835896f, 0.238569f };
const float SCALER_STD[MODEL_NUM_FEATURES]  = { 0.015937f, 0.092785f, 0.415143f, 0.014911f, 0.004451f, 0.017485f, 0.840607f, 2.643092f };

// ==============================================================================
// LOGISTIC REGRESSION WEIGHT MATRICES & BIAS VECTORS
// ==============================================================================
const float LGR_WEIGHTS[MODEL_NUM_CLASSES][MODEL_NUM_FEATURES] = {
    { -0.672027f, -1.935274f, -1.459712f, -0.405163f, -0.486734f, -0.563509f, -1.210010f, -0.149194f }, // Class 0 Target: healthy
    { -0.714056f, -0.531482f, 3.433053f, 0.392004f, 0.621896f, 0.011698f, -0.339974f, -0.852964f }, // Class 1 Target: imbalanced
    { 1.386083f, 2.466757f, -1.973341f, 0.013158f, -0.135162f, 0.551812f, 1.549984f, 1.002159f }, // Class 2 Target: obstruction
};

const float LGR_BIASES[MODEL_NUM_CLASSES] = { -2.433475f, 0.250473f, 2.183002f };

const char* const MODEL_CLASS_LABELS[MODEL_NUM_CLASSES] = {
    "healthy",
    "imbalanced",
    "obstruction",
};

#endif // MODEL_PARAMETERS_H
