"""
predict.py
----------
Test the trained models on your own audio file!

Usage:
    python src/predict.py path/to/your/audio.wav
"""

import sys
import os
import argparse
import joblib
import numpy as np

# Suppress TensorFlow logging warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_audio
from feature_extraction import extract_features, compute_mel_spectrogram
from utils import MODELS_DIR

def predict_audio(filepath: str):
    if not os.path.exists(filepath):
        print(f"❌ Error: File '{filepath}' not found.")
        return

    print(f"\n🎧 Loading audio: {filepath} ...")
    
    # 1. Load and process the audio
    y = load_audio(filepath)
    if y is None or len(y) == 0:
        print("❌ Error: Could not read audio file.")
        return
        
    print("🔬 Extracting features ...")
    
    # 2. Prepare features for Classical ML (Random Forest)
    feat_vector = extract_features(y)
    X_feat = feat_vector.reshape(1, -1)  # Reshape for single sample prediction
    
    # 3. Prepare features for Deep Learning (CNN)
    mel = compute_mel_spectrogram(y, n_mels=128, fixed_length=128)
    X_spec = mel.reshape(1, 128, 128, 1) # Reshape to (1, 128, 128, 1)
    # Scale CNN input
    if X_spec.max() > X_spec.min():
        X_spec = (X_spec - X_spec.min()) / (X_spec.max() - X_spec.min() + 1e-8)

    print("\n" + "="*40)
    print("  RESULTS")
    print("="*40)

    # ── 4a. Predict with Random Forest ──
    rf_path  = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    sc_path  = os.path.join(MODELS_DIR, "random_forest_scaler.pkl")
    
    if os.path.exists(rf_path) and os.path.exists(sc_path):
        rf     = joblib.load(rf_path)
        scaler = joblib.load(sc_path)
        
        # Scale the features
        X_feat_scaled = scaler.transform(X_feat)
        
        # Predict
        prob_rf = rf.predict_proba(X_feat_scaled)[0][1] # Probability of Class 1 (Drone)
        pred_rf = "🚁 Drone" if prob_rf >= 0.5 else "🌿 Non-Drone"
        
        print(f"🌲 Random Forest Prediction : {pred_rf} (Confidence: {prob_rf:.1%})")
    else:
        print("⚠️ Random Forest model not found. Run train_classical.py first.")

    # ── 4b. Predict with CNN ──
    cnn_path = os.path.join(MODELS_DIR, "cnn_model.keras")
    if not os.path.exists(cnn_path):
        cnn_path = os.path.join(MODELS_DIR, "cnn_model.h5")
        
    if os.path.exists(cnn_path):
        cnn = tf.keras.models.load_model(cnn_path)
        
        # Predict
        prob_cnn_arr = cnn.predict(X_spec, verbose=0)
        prob_cnn = float(prob_cnn_arr[0][0]) # Probability of Class 1 (Drone)
        pred_cnn = "🚁 Drone" if prob_cnn >= 0.5 else "🌿 Non-Drone"
        
        print(f"🧠 CNN Prediction           : {pred_cnn} (Confidence: {prob_cnn:.1%})")
    else:
        print("⚠️ CNN model not found. Run train_cnn.py first.")
        
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test drone sound detection on a custom audio file.")
    parser.add_argument("filepath", type=str, help="Path to the .wav or .mp3 audio file")
    
    args = parser.parse_args()
    predict_audio(args.filepath)
