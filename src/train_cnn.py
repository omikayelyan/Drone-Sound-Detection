"""
train_cnn.py
------------
Trains a Convolutional Neural Network (CNN) on mel spectrograms for
drone vs. non-drone audio classification using TensorFlow / Keras.

Why CNN on spectrograms?
    A mel spectrogram is a 2-D "image" of sound — rows = frequency bands,
    columns = time frames, brightness = energy.
    CNNs excel at detecting local spatial patterns in images, so they can
    learn the characteristic stripe patterns that drone rotors create.

Run:
    python src/train_cnn.py
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # quieter TensorFlow logs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader        import collect_file_paths, load_dataset
from feature_extraction import build_spectrogram_array
from utils              import (MODELS_DIR, RESULTS_DIR, FIGURES_DIR,
                                RANDOM_SEED, ensure_dirs, save_figure)
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────────────────────
# Step 1: Build the CNN architecture
# ──────────────────────────────────────────────────────────────
def build_cnn(input_shape: tuple = (128, 128, 1)) -> keras.Model:
    """
    Build a small but effective CNN for spectrogram classification.

    Architecture:
        Block 1: Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
        Block 2: Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
        Block 3: Conv2D(128) → BatchNorm → GlobalAvgPool
        Head:    Dense(128) → Dropout(0.5) → Dense(1, sigmoid)

    Design choices explained:
        Conv2D         : learns local frequency–time patterns
        BatchNorm      : stabilises training (like normalising activations)
        MaxPool        : downsamples, reducing computation + overfitting
        Dropout        : randomly zero-out neurons to prevent overfitting
        GlobalAvgPool  : replaces Flatten → fewer parameters, more robust
        sigmoid output : for binary classification (drone / not drone)

    Parameters
    ----------
    input_shape : tuple
        (n_mels, time_frames, channels) – default (128, 128, 1).

    Returns
    -------
    keras.Model (not yet compiled)
    """
    model = keras.Sequential(name="DroneCNN")

    # ── Input ────────────────────────────────────────────────
    model.add(layers.Input(shape=input_shape))

    # ── Block 1 (low-level patterns: edges, simple textures) ─
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu",
                             padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Block 2 (mid-level patterns: combinations of edges) ──
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu",
                             padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Block 3 (high-level patterns: drone-specific features) ─
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation="relu",
                             padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling2D())  # collapses H×W into 128 numbers

    # ── Classification head ──────────────────────────────────
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))  # output: P(drone)

    return model


# ──────────────────────────────────────────────────────────────
# Step 2: Compile the model
# ──────────────────────────────────────────────────────────────
def compile_model(model: keras.Model,
                  learning_rate: float = 1e-3) -> keras.Model:
    """
    Compile the model with Adam optimiser and binary cross-entropy loss.

    Loss function:
        binary_crossentropy is the standard loss for binary classification.
        It measures the difference between predicted probability and true label.

    Optimiser:
        Adam (Adaptive Moment Estimation) adjusts the learning rate per
        parameter automatically – a reliable default choice.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )
    model.summary()
    return model


# ──────────────────────────────────────────────────────────────
# Step 3: Callbacks (smart training helpers)
# ──────────────────────────────────────────────────────────────
def get_callbacks(model_path: str) -> list:
    """
    Return a list of Keras callbacks:

        EarlyStopping : stop training when val_loss stops improving
                        (prevents wasted time and overfitting)
        ModelCheckpoint : save the best model checkpoint automatically
        ReduceLROnPlateau : halve learning rate if val_loss plateaus
    """
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,          # wait 10 epochs before stopping
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,           # new_lr = old_lr * 0.5
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]


# ──────────────────────────────────────────────────────────────
# Step 4: Plot training history
# ──────────────────────────────────────────────────────────────
def plot_training_history(history: keras.callbacks.History,
                           save: bool = True) -> plt.Figure:
    """
    Plot loss and accuracy curves for train and validation sets.

    A good training run shows both curves converging without a large gap
    (large gap → overfitting; train diverges → learning rate too high).
    """
    from utils import set_plot_style
    set_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Loss
    axes[0].plot(history.history["loss"],     label="Train Loss",     color="#89b4fa")
    axes[0].plot(history.history["val_loss"], label="Val Loss",       color="#f38ba8")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history.history["accuracy"],     label="Train Acc", color="#a6e3a1")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc",   color="#fab387")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("CNN Training History", fontsize=15, color="#cba6f7")
    fig.tight_layout()

    if save:
        save_figure(fig, "cnn_training_history.png")

    return fig


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────
def main():
    ensure_dirs()
    print("=" * 60)
    print("  CNN TRAINING – Drone Sound Detection (Mel Spectrograms)")
    print("=" * 60)

    # 1. Load & compute spectrograms
    df = collect_file_paths()
    if df.empty:
        raise RuntimeError("No audio files found. Check data/drone/ and data/non_drone/")

    waveforms, labels = load_dataset(df)

    # Build spectrogram array: shape (N, 128, 128, 1)
    X = build_spectrogram_array(waveforms, n_mels=128, fixed_length=128)
    y = labels

    # 2. Normalise spectrogram values to [0, 1]
    X_min = X.min()
    X_max = X.max()
    X = (X - X_min) / (X_max - X_min + 1e-8)

    # 3. Train / validation / test split (70 / 15 / 15)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=RANDOM_SEED, stratify=y_temp)

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 4. Build and compile model
    cnn = build_cnn(input_shape=X_train.shape[1:])
    cnn = compile_model(cnn, learning_rate=1e-3)

    # 5. Train
    model_path = os.path.join(MODELS_DIR, "cnn_model.keras")
    cbs = get_callbacks(model_path)

    print("\n🏋️  Training CNN …")
    history = cnn.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=cbs,
        verbose=1
    )

    # 6. Plot history
    plot_training_history(history)

    # 7. Final test evaluation
    print("\n📊 CNN – Final Test Set Evaluation")
    print("─" * 50)
    results = cnn.evaluate(X_test, y_test, verbose=0)
    metric_names = cnn.metrics_names

    metrics = dict(zip(metric_names, results))
    for k, v in metrics.items():
        print(f"   {k:<12}: {v:.4f}")

    # Also compute sklearn-style metrics for the comparison table
    from sklearn.metrics import classification_report, confusion_matrix
    y_prob = cnn.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_test, y_pred,
                                   target_names=["non_drone", "drone"],
                                   output_dict=True)
    print(classification_report(y_test, y_pred,
                                 target_names=["non_drone", "drone"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # 8. Save metrics
    cnn_metrics = {
        "accuracy":          report["accuracy"],
        "precision":         report["drone"]["precision"],
        "recall":            report["drone"]["recall"],
        "f1":                report["drone"]["f1-score"],
        "confusion_matrix":  cm.tolist()
    }
    metrics_path = os.path.join(RESULTS_DIR, "cnn_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(cnn_metrics, f, indent=4)
    print(f"\n💾 CNN metrics saved to {metrics_path}")
    print("✅ CNN training complete!")

    return cnn, cnn_metrics


if __name__ == "__main__":
    main()
