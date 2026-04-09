"""
tests/test_features.py
-----------------------
Basic unit tests for the feature extraction pipeline.

Run with:
    python -m pytest tests/ -v
"""

import sys, os
import numpy as np
import pytest

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_extraction import (
    extract_features,
    compute_mel_spectrogram,
    build_feature_matrix,
    build_spectrogram_array,
)
from utils import SAMPLE_RATE, DURATION


# ──────────────────────────────────────────────────────────────
# Fixtures: synthetic audio signals for testing
# ──────────────────────────────────────────────────────────────
@pytest.fixture
def silent_clip():
    """A clip of pure silence (all zeros)."""
    n_samples = int(SAMPLE_RATE * DURATION)
    return np.zeros(n_samples, dtype=np.float32)


@pytest.fixture
def sine_wave():
    """A 440 Hz sine wave (A4 musical note) for 3 seconds."""
    n_samples = int(SAMPLE_RATE * DURATION)
    t = np.linspace(0, DURATION, n_samples)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def random_noise():
    """White noise – random values in [-1, 1]."""
    n_samples = int(SAMPLE_RATE * DURATION)
    rng = np.random.default_rng(42)
    return rng.uniform(-1, 1, n_samples).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Test: extract_features output shape and type
# ──────────────────────────────────────────────────────────────
class TestExtractFeatures:

    def test_output_is_ndarray(self, sine_wave):
        """extract_features should return a numpy array."""
        feats = extract_features(sine_wave)
        assert isinstance(feats, np.ndarray), "Expected np.ndarray output"

    def test_feature_vector_length(self, sine_wave):
        """
        Feature vector should have the expected length.
        40 MFCC mean + 40 MFCC std + 12 chroma + 1 centroid + 1 ZCR = 94
        """
        feats = extract_features(sine_wave)
        assert feats.shape == (94,), f"Expected shape (94,), got {feats.shape}"

    def test_no_nan_values(self, sine_wave):
        """No NaN or Inf values should appear in the feature vector."""
        feats = extract_features(sine_wave)
        assert not np.any(np.isnan(feats)), "NaN values found in features"
        assert not np.any(np.isinf(feats)), "Inf values found in features"

    def test_silent_clip(self, silent_clip):
        """Feature extraction should not crash on silence."""
        feats = extract_features(silent_clip)
        assert feats.shape == (94,)

    def test_noise_clip(self, random_noise):
        """Feature extraction should work on white noise."""
        feats = extract_features(random_noise)
        assert feats.shape == (94,)


# ──────────────────────────────────────────────────────────────
# Test: compute_mel_spectrogram output shape
# ──────────────────────────────────────────────────────────────
class TestMelSpectrogram:

    def test_output_shape_default(self, sine_wave):
        """Default mel spectrogram should be (128, 128)."""
        mel = compute_mel_spectrogram(sine_wave)
        assert mel.shape == (128, 128), f"Expected (128, 128), got {mel.shape}"

    def test_custom_n_mels(self, sine_wave):
        """Custom n_mels should be respected."""
        mel = compute_mel_spectrogram(sine_wave, n_mels=64, fixed_length=100)
        assert mel.shape == (64, 100)

    def test_no_nan(self, sine_wave):
        """Mel spectrogram should not contain NaN."""
        mel = compute_mel_spectrogram(sine_wave)
        assert not np.any(np.isnan(mel))


# ──────────────────────────────────────────────────────────────
# Test: batch operations
# ──────────────────────────────────────────────────────────────
class TestBatchOperations:

    def test_build_feature_matrix_shape(self, sine_wave, silent_clip):
        """build_feature_matrix should return (N, 94)."""
        batch = np.stack([sine_wave, silent_clip], axis=0)
        X = build_feature_matrix(batch)
        assert X.shape == (2, 94)

    def test_build_spectrogram_array_shape(self, sine_wave, silent_clip):
        """build_spectrogram_array should return (N, 128, 128, 1)."""
        batch = np.stack([sine_wave, silent_clip], axis=0)
        S = build_spectrogram_array(batch, n_mels=128, fixed_length=128)
        assert S.shape == (2, 128, 128, 1)


# ──────────────────────────────────────────────────────────────
# Test: label mapping
# ──────────────────────────────────────────────────────────────
def test_label_map_consistency():
    """LABEL_MAP and LABEL_MAP_INV should be inverses of each other."""
    from utils import LABEL_MAP, LABEL_MAP_INV
    for k, v in LABEL_MAP.items():
        assert LABEL_MAP_INV[v] == k, f"Label map inconsistency at key={k}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
