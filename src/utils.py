"""
utils.py
--------
Shared helpers for the Drone Sound Detection project.
These functions are imported by all other modules to avoid code duplication.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────
# Constants – change these if your folder layout differs
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
DRONE_DIR    = os.path.join(DATA_DIR, "drone")
NON_DRONE_DIR = os.path.join(DATA_DIR, "non_drone")

MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")

# Audio settings used consistently across the project
SAMPLE_RATE  = 22050   # Hz – standard for librosa
DURATION     = 3.0     # seconds – clips will be padded or trimmed to this length
N_MFCC       = 40      # number of MFCC coefficients to extract
N_MELS       = 128     # number of mel frequency bands for spectrograms
HOP_LENGTH   = 512     # samples between STFT frames
N_FFT        = 2048    # FFT window size

# Label mapping
LABEL_MAP  = {0: "non_drone", 1: "drone"}
LABEL_MAP_INV = {"non_drone": 0, "drone": 1}

RANDOM_SEED = 42        # for reproducibility


# ──────────────────────────────────────────────────────────────
# Directory helpers
# ──────────────────────────────────────────────────────────────
def ensure_dirs():
    """Create output directories if they don't already exist."""
    for d in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
    print("✅ Output directories ready.")


# ──────────────────────────────────────────────────────────────
# Metrics helpers
# ──────────────────────────────────────────────────────────────
def save_metrics(metrics_dict: dict, filename: str = "metrics.json"):
    """
    Save a dictionary of evaluation metrics to a JSON file.

    Parameters
    ----------
    metrics_dict : dict
        e.g. {"random_forest": {...}, "cnn": {...}}
    filename : str
        Output filename inside results/
    """
    ensure_dirs()
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"📄 Metrics saved to {path}")


def load_metrics(filename: str = "metrics.json") -> dict:
    """Load previously saved metrics from JSON."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────
def save_figure(fig: plt.Figure, name: str):
    """
    Save a matplotlib figure to the results/figures/ directory.

    Parameters
    ----------
    fig : plt.Figure
        The figure object to save.
    name : str
        File name (without path), e.g. 'waveform_drone.png'
    """
    ensure_dirs()
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"🖼️  Figure saved to {path}")


def set_plot_style():
    """Apply a consistent, clean style to all matplotlib plots."""
    plt.rcParams.update({
        "figure.facecolor": "#1e1e2e",
        "axes.facecolor":   "#1e1e2e",
        "axes.edgecolor":   "#cdd6f4",
        "axes.labelcolor":  "#cdd6f4",
        "xtick.color":      "#cdd6f4",
        "ytick.color":      "#cdd6f4",
        "text.color":       "#cdd6f4",
        "grid.color":       "#313244",
        "figure.dpi":       100,
        "font.size":        11,
    })
