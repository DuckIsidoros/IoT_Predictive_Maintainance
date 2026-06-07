from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = PROJECT_ROOT / "features" / "features_dataset.csv"
PLOT_DIR = PROJECT_ROOT / "features" / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FEATURE_PATH)

features_to_plot = [
    "dominant_frequency",
    "max_magnitude",
    "mean_magnitude",
    "std_magnitude",
    "total_energy",
    "spectral_centroid",
    "spectral_bandwidth",
    "low_band_energy",
    "mid_band_energy",
    "high_band_energy",
]

for feature in features_to_plot:
    plt.figure(figsize=(8, 5))

    for label in sorted(df["label"].unique()):
        subset = df[df["label"] == label]
        plt.hist(subset[feature], bins=30, alpha=0.5, label=label)

    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    output_path = PLOT_DIR / f"{feature}_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

print(f"Saved feature distribution plots to: {PLOT_DIR}")