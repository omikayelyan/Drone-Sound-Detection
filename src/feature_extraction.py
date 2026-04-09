"""
feature_extraction.py
---------------------
Extracts handcrafted audio features from raw waveforms.

Features extracted per clip:
 - MFCCs        : 40 coefficients capturing the "shape" of the sound spectrum
 - Chroma       : 12 pitch class energies (like musical notes)
 - Spectral Centroid : "centre of mass" of the spectrum (brightness)
 - Zero-Crossing Rate: how often the signal crosses zero (noisiness)

Why these features?
    Drone sounds have a characteristic buzzing pitch and rhythmic pattern.
    MFCCs are the most powerful feature for audio classification.
    Together these 54 values give a compact but rich description of any clip.
"""

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from utils import SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT


# ──────────────────────────────────────────────────────────────
# Step 1: Extract features from ONE waveform
# ──────────────────────────────────────────────────────────────
def extract_features(y: np.ndarray,
                     sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract a fixed-size feature vector from a single audio waveform.

    The vector contains (in order):
        [mfcc_mean × 40]  [mfcc_std × 40]
        [chroma_mean × 12]
        [spectral_centroid_mean × 1]
        [zero_crossing_rate_mean × 1]
    Total: 94 features

    Parameters
    ----------
    y : np.ndarray
        1-D audio waveform (output of data_loader.load_audio).
    sr : int
        Sample rate of the waveform.

    Returns
    -------
    np.ndarray
        Feature vector of shape (94,).
    """
    features = []

    # ── 1. MFCCs (Mel-Frequency Cepstral Coefficients) ──────────
    # MFCCs model how the human ear perceives sound.
    # We compute 40 coefficients over time frames, then take the
    # mean and standard deviation across all frames, giving 80 values.
    mfccs = librosa.feature.mfcc(y=y, sr=sr,
                                  n_mfcc=N_MFCC,
                                  n_fft=N_FFT,
                                  hop_length=HOP_LENGTH)
    features.append(np.mean(mfccs, axis=1))   # shape: (40,)
    features.append(np.std(mfccs,  axis=1))   # shape: (40,)

    # ── 2. Chroma Features ───────────────────────────────────────
    # Chroma represents the energy for each of the 12 musical pitch
    # classes (C, C#, D, …). Useful for distinguishing tonal sounds.
    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                          n_fft=N_FFT,
                                          hop_length=HOP_LENGTH)
    features.append(np.mean(chroma, axis=1))  # shape: (12,)

    # ── 3. Spectral Centroid ─────────────────────────────────────
    # The weighted mean frequency – think of it as "brightness".
    # High-pitched drones may have a higher centroid than low traffic.
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                  n_fft=N_FFT,
                                                  hop_length=HOP_LENGTH)
    features.append(np.mean(centroid))        # scalar

    # ── 4. Zero-Crossing Rate ────────────────────────────────────
    # Number of times the signal changes sign per second.
    # Noisy, buzzing sounds like drones tend to have a high ZCR.
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)
    features.append(np.mean(zcr))             # scalar

    # Flatten everything into a single 1-D vector
    return np.concatenate([np.atleast_1d(f) for f in features])


# ──────────────────────────────────────────────────────────────
# Step 2: Extract features from ALL waveforms
# ──────────────────────────────────────────────────────────────
def build_feature_matrix(waveforms: np.ndarray,
                          sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Run extract_features on every row of `waveforms`.

    Parameters
    ----------
    waveforms : np.ndarray, shape (N, audio_samples)
        All loaded audio clips.
    sr : int
        Sample rate.

    Returns
    -------
    np.ndarray, shape (N, 94)
        Feature matrix – one row per audio clip.
    """
    print(f"\n🔬 Extracting features from {len(waveforms)} clips …")
    X = []
    for y in tqdm(waveforms, desc="Feature extraction"):
        X.append(extract_features(y, sr))
    X = np.array(X)
    print(f"✅ Feature matrix shape: {X.shape}")
    return X


# ──────────────────────────────────────────────────────────────
# Step 3: Build mel spectrograms (used as CNN input)
# ──────────────────────────────────────────────────────────────
def compute_mel_spectrogram(y: np.ndarray,
                            sr: int = SAMPLE_RATE,
                            n_mels: int = 128,
                            fixed_length: int = 128) -> np.ndarray:
    """
    Convert a waveform into a 2-D mel spectrogram suitable for CNN input.

    Steps:
        1. Compute short-time Fourier transform (STFT)
        2. Map to mel frequency scale
        3. Convert to decibels (log scale – closer to human hearing)
        4. Resize to (n_mels × fixed_length) so all clips are the same shape

    Parameters
    ----------
    y : np.ndarray
        Audio waveform.
    sr : int
        Sample rate.
    n_mels : int
        Number of mel frequency bands (rows of the spectrogram).
    fixed_length : int
        Number of time frames (columns of the spectrogram).

    Returns
    -------
    np.ndarray, shape (n_mels, fixed_length)
        Log-amplitude mel spectrogram.
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr,
                                          n_mels=n_mels,
                                          n_fft=N_FFT,
                                          hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # convert to dB

    # Pad or trim along the time axis to reach fixed_length
    if mel_db.shape[1] < fixed_length:
        pad = fixed_length - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant")
    else:
        mel_db = mel_db[:, :fixed_length]

    return mel_db  # shape: (n_mels, fixed_length)


def build_spectrogram_array(waveforms: np.ndarray,
                            sr: int = SAMPLE_RATE,
                            n_mels: int = 128,
                            fixed_length: int = 128) -> np.ndarray:
    """
    Compute mel spectrograms for all waveforms.

    Returns
    -------
    np.ndarray, shape (N, n_mels, fixed_length, 1)
        4-D array where the last dimension (1) is the colour-channel,
        matching TensorFlow/Keras Conv2D input format.
    """
    print(f"\n🎛️  Computing mel spectrograms for {len(waveforms)} clips …")
    specs = []
    for y in tqdm(waveforms, desc="Mel spectrograms"):
        s = compute_mel_spectrogram(y, sr, n_mels, fixed_length)
        specs.append(s)
    specs = np.array(specs)           # (N, n_mels, time)
    specs = specs[..., np.newaxis]    # (N, n_mels, time, 1) – channel dim
    print(f"✅ Spectrogram array shape: {specs.shape}")
    return specs


# ──────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data_loader import collect_file_paths, load_dataset

    df = collect_file_paths()
    if not df.empty:
        waveforms, labels = load_dataset(df)

        # Classical ML features
        X = build_feature_matrix(waveforms)
        print(f"\nFeature vector length per clip: {X.shape[1]}")

        # CNN spectrogram features
        S = build_spectrogram_array(waveforms)
        print(f"Spectrogram array shape       : {S.shape}")
