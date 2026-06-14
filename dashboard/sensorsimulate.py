import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR = Path(__file__).resolve().parents[1] / "reactdashboard" / "public" / "dashboard-data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Imbalance", "Bearing Fault", "Misalignment"]

def generate_samples(n=256, sample_rate=1000, fault="Normal"):
    t = np.arange(n) / sample_rate
    base_freq = 50

    signal = 0.05 * np.sin(2*np.pi*base_freq*t)
    if fault == "Imbalance":
        signal += 0.15 * np.sin(2*np.pi*(2*base_freq)*t)
    elif fault == "Bearing Fault":
        signal += 0.25 * np.sin(2*np.pi*300*t)
    elif fault == "Misalignment":
        signal += 0.10 * np.sin(2*np.pi*(3*base_freq)*t)

    noise = np.random.normal(0, 0.02, n)
    az = 1.0 + signal + noise
    ax = signal*0.5 + noise
    ay = signal*0.3 + noise
    return ax, ay, az

def compute_fft(samples, sample_rate=1000):
    windowed = samples * np.hanning(len(samples))
    fft_result = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(len(samples), d=1/sample_rate)
    mask = frequencies <= 300
    return frequencies[mask].tolist(), fft_result[mask].tolist()

def compute_rms(arr):
    return float(np.sqrt(np.mean(arr**2)))

def compute_band_power(magnitudes, frequencies):
    mags, freqs = np.array(magnitudes), np.array(frequencies)
    total = np.sum(mags**2) + 1e-9
    low  = float(np.sum(mags[freqs <= 50]**2) / total)
    mid  = float(np.sum(mags[(freqs > 50) & (freqs <= 200)]**2) / total)
    high = float(np.sum(mags[freqs > 200]**2) / total)
    return low, mid, high

def fake_inference(fault):
    scores = {c: 0.02 for c in CLASSES}
    scores[fault] = 1 - 0.02*(len(CLASSES)-1)
    for k in scores:
        scores[k] += np.random.uniform(-0.02, 0.02)
    total = sum(scores.values())
    scores = {k: max(0, v/total) for k, v in scores.items()}
    prediction = max(scores, key=scores.get)
    return prediction, scores[prediction], scores

def main():
    fault_cycle = ["Normal", "Normal", "Normal", "Imbalance", "Bearing Fault", "Normal"]
    i = 0

    while True:
        fault = fault_cycle[i % len(fault_cycle)]
        i += 1

        ax, ay, az = generate_samples(fault=fault)
        frequencies, magnitudes = compute_fft(az)
        rms_x, rms_y, rms_z = compute_rms(ax), compute_rms(ay), compute_rms(az)
        low, mid, high = compute_band_power(magnitudes, frequencies)
        band_level = "high" if high > 0.3 else "medium" if mid > 0.3 else "normal"
        prediction, confidence, scores = fake_inference(fault)

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw": {"ax": ax[:50].tolist(), "ay": ay[:50].tolist(), "az": az[:50].tolist()},
            "fft": {"frequencies": frequencies, "magnitudes": magnitudes},
            "inference": {"prediction": prediction, "confidence": confidence, "scores": scores},
            "status": {"fan_state": "Running", "sensor_health": "OK", "connection": "Connected"},
        }
        features = {
            "rms": {"x": rms_x, "y": rms_y, "z": rms_z},
            "band_power": {"low": low, "mid": mid, "high": high},
            "band_level": band_level,
        }

        (DATA_DIR / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (DATA_DIR / "features.json").write_text(json.dumps(features), encoding="utf-8")

        print(f"[{summary['timestamp']}] fault={fault} → {prediction} ({confidence:.2f})")
        time.sleep(2)

if __name__ == "__main__":
    main()