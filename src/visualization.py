"""
visualization.py
----------------
Plots waveforms, spectrograms, and mel spectrograms for the drone detection
project. Saves all figures to results/figures/.

Run this file directly to generate a sample plot page:
    python src/visualization.py
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import (
    SAMPLE_RATE, N_FFT, HOP_LENGTH,
    save_figure, set_plot_style
)


# Apply custom dark-mode style to all plots
set_plot_style()


# ──────────────────────────────────────────────────────────────
# 1. Waveform plot – time domain
# ──────────────────────────────────────────────────────────────
def plot_waveform(y: np.ndarray,
                  sr: int = SAMPLE_RATE,
                  title: str = "Waveform",
                  save: bool = True) -> plt.Figure:
    """
    Plot the raw audio waveform (amplitude vs. time).

    The waveform shows how loud the sound is at each point in time.
    Drone sounds typically show a very regular oscillation pattern.

    Parameters
    ----------
    y : np.ndarray
        Audio waveform array.
    sr : int
        Sample rate.
    title : str
        Plot title.
    save : bool
        If True, save the figure to results/figures/.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#89b4fa")
    ax.set_title(title, fontsize=14, color="#cdd6f4")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save:
        fname = title.lower().replace(" ", "_") + "_waveform.png"
        save_figure(fig, fname)

    return fig


# ──────────────────────────────────────────────────────────────
# 2. STFT Spectrogram – frequency content over time
# ──────────────────────────────────────────────────────────────
def plot_spectrogram(y: np.ndarray,
                     sr: int = SAMPLE_RATE,
                     title: str = "Spectrogram",
                     save: bool = True) -> plt.Figure:
    """
    Plot a Short-Time Fourier Transform (STFT) spectrogram.

    The spectrogram shows which frequencies are present at each moment.
    Drone sounds produce a characteristic band of strong harmonics.

    Colour scale: brighter = louder (in decibels).
    """
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(D_db,
                                    sr=sr,
                                    hop_length=HOP_LENGTH,
                                    x_axis="time",
                                    y_axis="log",
                                    ax=ax,
                                    cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title, fontsize=14, color="#cdd6f4")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()

    if save:
        fname = title.lower().replace(" ", "_") + "_spectrogram.png"
        save_figure(fig, fname)

    return fig


# ──────────────────────────────────────────────────────────────
# 3. Mel Spectrogram – used as CNN input
# ──────────────────────────────────────────────────────────────
def plot_mel_spectrogram(y: np.ndarray,
                          sr: int = SAMPLE_RATE,
                          n_mels: int = 128,
                          title: str = "Mel Spectrogram",
                          save: bool = True) -> plt.Figure:
    """
    Plot a Mel-frequency spectrogram.

    Mel scale is a non-linear frequency scale that matches how humans
    perceive pitch. It compresses high frequencies and expands lows.
    This is the image we feed into the CNN model.
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr,
                                          n_mels=n_mels,
                                          n_fft=N_FFT,
                                          hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_db,
                                    sr=sr,
                                    hop_length=HOP_LENGTH,
                                    x_axis="time",
                                    y_axis="mel",
                                    ax=ax,
                                    cmap="inferno")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title, fontsize=14, color="#cdd6f4")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mel Frequency")
    fig.tight_layout()

    if save:
        fname = title.lower().replace(" ", "_") + "_mel.png"
        save_figure(fig, fname)

    return fig


# ──────────────────────────────────────────────────────────────
# 4. MFCC plot – 40 coefficient tracks
# ──────────────────────────────────────────────────────────────
def plot_mfccs(y: np.ndarray,
               sr: int = SAMPLE_RATE,
               n_mfcc: int = 40,
               title: str = "MFCCs",
               save: bool = True) -> plt.Figure:
    """
    Plot all 40 MFCC tracks as a 2-D heat map.

    Each row is one MFCC coefficient; colour shows its value at each
    time frame. The pattern is a fingerprint of the audio texture.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                   n_fft=N_FFT, hop_length=HOP_LENGTH)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfccs,
                                    sr=sr,
                                    hop_length=HOP_LENGTH,
                                    x_axis="time",
                                    ax=ax,
                                    cmap="coolwarm")
    fig.colorbar(img, ax=ax)
    ax.set_title(title, fontsize=14, color="#cdd6f4")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("MFCC Coefficient #")
    fig.tight_layout()

    if save:
        fname = title.lower().replace(" ", "_") + "_mfcc.png"
        save_figure(fig, fname)

    return fig


# ──────────────────────────────────────────────────────────────
# 5. Side-by-side comparison: drone vs. non-drone
# ──────────────────────────────────────────────────────────────
def compare_classes(drone_wave: np.ndarray,
                    non_drone_wave: np.ndarray,
                    sr: int = SAMPLE_RATE,
                    save: bool = True) -> plt.Figure:
    """
    Create a 2×2 grid showing waveform + mel spectrogram for each class.

    This is a great figure to include in a project report!
    """
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    titles = ["Drone – Waveform", "Non-Drone – Waveform",
              "Drone – Mel Spectrogram", "Non-Drone – Mel Spectrogram"]
    pairs  = [(drone_wave, "waveform"), (non_drone_wave, "waveform"),
              (drone_wave, "mel"),      (non_drone_wave, "mel")]

    for i, (wave, kind) in enumerate(pairs):
        ax = fig.add_subplot(gs[i // 2, i % 2])

        if kind == "waveform":
            librosa.display.waveshow(wave, sr=sr, ax=ax, color="#89b4fa")
            ax.set_ylabel("Amplitude")
        else:  # mel spectrogram
            mel   = librosa.feature.melspectrogram(y=wave, sr=sr)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            img   = librosa.display.specshow(mel_db, sr=sr, x_axis="time",
                                              y_axis="mel", ax=ax,
                                              cmap="inferno")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")

        ax.set_title(titles[i], fontsize=12, color="#cdd6f4")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.2)

    fig.suptitle("Drone vs. Non-Drone Audio Comparison",
                 fontsize=16, color="#cba6f7", y=1.01)
    fig.tight_layout()

    if save:
        save_figure(fig, "comparison_drone_vs_non_drone.png")

    return fig


# ──────────────────────────────────────────────────────────────
# Demo: generate all plots on first 2 files found
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data_loader import collect_file_paths, load_audio

    df = collect_file_paths()

    if df.empty:
        print("No audio files found. Add .wav files to data/drone and data/non_drone")
    else:
        # Pick one sample from each class
        drone_row     = df[df["class_name"] == "drone"].iloc[0]
        non_drone_row = df[df["class_name"] == "non_drone"].iloc[0]

        drone_wave     = load_audio(drone_row["filepath"])
        non_drone_wave = load_audio(non_drone_row["filepath"])

        print("Generating waveform plots …")
        plot_waveform(drone_wave,     title="Drone")
        plot_waveform(non_drone_wave, title="Non-Drone")

        print("Generating spectrogram plots …")
        plot_spectrogram(drone_wave,     title="Drone")
        plot_spectrogram(non_drone_wave, title="Non-Drone")

        print("Generating mel spectrogram plots …")
        plot_mel_spectrogram(drone_wave,     title="Drone")
        plot_mel_spectrogram(non_drone_wave, title="Non-Drone")

        print("Generating MFCC plots …")
        plot_mfccs(drone_wave, title="Drone MFCCs")

        print("Generating comparison figure …")
        compare_classes(drone_wave, non_drone_wave)

        print("\n✅ All figures saved to results/figures/")
        plt.show()
