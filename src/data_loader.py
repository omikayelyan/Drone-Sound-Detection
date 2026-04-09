"""
data_loader.py
--------------
Loads WAV audio files from the data/ directory, handles resampling,
padding/trimming to a fixed duration, and returns a clean DataFrame
ready for feature extraction.

Folder structure expected:
    data/
    ├── drone/       <- drone audio .wav files
    └── non_drone/   <- non-drone audio .wav files
"""

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm  # progress bar

# Import shared settings from our utilities module
from utils import (
    DRONE_DIR, NON_DRONE_DIR,
    SAMPLE_RATE, DURATION,
    LABEL_MAP_INV
)


# ──────────────────────────────────────────────────────────────
# Step 1: Collect all audio file paths and labels
# ──────────────────────────────────────────────────────────────
def collect_file_paths(drone_dir: str = DRONE_DIR,
                       non_drone_dir: str = NON_DRONE_DIR) -> pd.DataFrame:
    """
    Scan folder structure and build a DataFrame with columns:
        filepath  : full path to the .wav file
        label     : integer label (0 = non_drone, 1 = drone)
        class_name: string label ("drone" or "non_drone")

    Parameters
    ----------
    drone_dir : str
        Path to folder containing drone audio clips.
    non_drone_dir : str
        Path to folder containing non-drone audio clips.

    Returns
    -------
    pd.DataFrame
        One row per audio file found.
    """
    records = []

    for folder, class_name in [(drone_dir, "drone"),
                               (non_drone_dir, "non_drone")]:
        if not os.path.isdir(folder):
            print(f"⚠️  Folder not found: {folder} — skipping.")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".wav", ".mp3", ".flac"))]

        print(f"📂 Found {len(files)} files in '{class_name}' class.")

        for fname in files:
            records.append({
                "filepath":   os.path.join(folder, fname),
                "label":      LABEL_MAP_INV[class_name],
                "class_name": class_name
            })

    df = pd.DataFrame(records)

    if df.empty:
        print("❌ No audio files found. Did you download and place the dataset?")
        print("   Place .wav files in data/drone/ and data/non_drone/")
    else:
        print(f"\n✅ Total files: {len(df)}")
        print(df["class_name"].value_counts().to_string())

    return df


# ──────────────────────────────────────────────────────────────
# Step 2: Load a single audio file
# ──────────────────────────────────────────────────────────────
def load_audio(filepath: str,
               sample_rate: int = SAMPLE_RATE,
               duration: float = DURATION) -> np.ndarray:
    """
    Load an audio file with librosa, resample it to `sample_rate`,
    and pad or trim it to exactly `duration` seconds.

    Why pad/trim?
        ML models need fixed-size inputs. Padding short clips with
        silence and trimming long ones ensures consistency.

    Parameters
    ----------
    filepath : str
        Path to the audio file.
    sample_rate : int
        Target sample rate in Hz (default 22,050).
    duration : float
        Length to normalise every clip to (in seconds).

    Returns
    -------
    np.ndarray
        1-D array of audio samples, shape (sample_rate * duration,).
    """
    # librosa.load returns (waveform_array, achieved_sample_rate)
    # res_type='kaiser_fast' speeds up resampling
    try:
        y, sr = librosa.load(filepath,
                             sr=sample_rate,
                             duration=duration,
                             res_type="kaiser_fast")
    except Exception as e:
        print(f"⚠️  Could not load {filepath}: {e}")
        # Return silence so the pipeline doesn't crash
        return np.zeros(int(sample_rate * duration))

    # Target number of samples
    target_length = int(sample_rate * duration)

    if len(y) < target_length:
        # Pad with zeros (silence) at the end
        y = np.pad(y, (0, target_length - len(y)), mode="constant")
    else:
        # Trim to exactly target_length samples
        y = y[:target_length]

    return y


# ──────────────────────────────────────────────────────────────
# Step 3: Load ALL audio files
# ──────────────────────────────────────────────────────────────
def load_dataset(df: pd.DataFrame,
                 sample_rate: int = SAMPLE_RATE,
                 duration: float = DURATION):
    """
    Load every audio file listed in `df` and return waveforms + labels.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `collect_file_paths()`.
    sample_rate : int
        Target sample rate (Hz).
    duration : float
        Fixed clip duration (seconds).

    Returns
    -------
    waveforms : np.ndarray, shape (n_samples, n_audio_samples)
        All audio data loaded into memory.
    labels : np.ndarray, shape (n_samples,)
        Integer labels (0 or 1).
    """
    print(f"\n🔊 Loading {len(df)} audio files …")

    waveforms = []
    labels    = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading audio"):
        y = load_audio(row["filepath"], sample_rate, duration)
        waveforms.append(y)
        labels.append(row["label"])

    waveforms = np.array(waveforms)  # shape: (N, sample_rate*duration)
    labels    = np.array(labels)     # shape: (N,)

    print(f"✅ Dataset loaded: {waveforms.shape[0]} clips, "
          f"{waveforms.shape[1]} samples each.")
    return waveforms, labels


# ──────────────────────────────────────────────────────────────
# Quick sanity check (run this file directly to test)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Build the file index
    df = collect_file_paths()

    if not df.empty:
        # Load all audio into RAM
        waveforms, labels = load_dataset(df)

        # Preview
        print(f"\nWaveforms array shape : {waveforms.shape}")
        print(f"Labels array shape    : {labels.shape}")
        print(f"Drone samples         : {(labels == 1).sum()}")
        print(f"Non-drone samples     : {(labels == 0).sum()}")
