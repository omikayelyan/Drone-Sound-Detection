"""
evaluate.py
-----------
Loads the trained models and produces a comprehensive comparison:
  - Accuracy, Precision, Recall, F1-score for all models
  - Confusion matrices (plotted as heatmaps)
  - Side-by-side bar chart comparing all metrics
  - Saves a combined metrics.json to results/

Run:
    python src/evaluate.py
"""

import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader        import collect_file_paths, load_dataset
from feature_extraction import build_feature_matrix, build_spectrogram_array
from utils              import (MODELS_DIR, RESULTS_DIR, FIGURES_DIR,
                                RANDOM_SEED, ensure_dirs,
                                save_figure, set_plot_style)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay)

set_plot_style()


# ──────────────────────────────────────────────────────────────
# Helper: plot confusion matrix as a pretty heatmap
# ──────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray,
                           model_name: str,
                           class_names: list = None,
                           save: bool = True) -> plt.Figure:
    """
    Render a confusion matrix as a colour-coded heatmap.

    How to read a confusion matrix (binary case):
        ┌──────────────┬──────────────┐
        │  True Neg    │  False Pos   │   ← predicted drone, actually not
        ├──────────────┼──────────────┤
        │  False Neg   │  True Pos    │   ← predicted not drone, actually drone
        └──────────────┴──────────────┘

    False Positives (FP): system raises alarm when no drone is present.
    False Negatives (FN): system misses an actual drone – more dangerous!
    """
    if class_names is None:
        class_names = ["Non-Drone", "Drone"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=0.5,
                ax=ax)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13,
                 color="#cdd6f4")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()

    if save:
        save_figure(fig, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")

    return fig


# ──────────────────────────────────────────────────────────────
# Helper: bar chart comparing all models
# ──────────────────────────────────────────────────────────────
def plot_model_comparison(metrics_dict: dict, save: bool = True) -> plt.Figure:
    """
    Create a grouped bar chart comparing Accuracy, Precision, Recall, F1
    across all trained models.

    Parameters
    ----------
    metrics_dict : dict
        e.g. {"Random Forest": {...}, "SVM": {...}, "CNN": {...}}
    """
    metric_keys  = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names  = list(metrics_dict.keys())

    x       = np.arange(len(metric_labels))
    n_models = len(model_names)
    width   = 0.25
    colours  = ["#89b4fa", "#a6e3a1", "#fab387"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (model_name, colour) in enumerate(zip(model_names, colours)):
        vals = [metrics_dict[model_name].get(k, 0) for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width,
                      label=model_name,
                      color=colour,
                      alpha=0.85,
                      edgecolor="white",
                      linewidth=0.5)
        # Annotate bar tops
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=9, color="#cdd6f4")

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, color="#cba6f7")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save:
        save_figure(fig, "model_comparison.png")

    return fig


# ──────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ──────────────────────────────────────────────────────────────
def main():
    ensure_dirs()
    print("=" * 60)
    print("  MODEL EVALUATION & COMPARISON")
    print("=" * 60)

    # ── 1. Load & prepare data ───────────────────────────────
    df = collect_file_paths()
    if df.empty:
        raise RuntimeError("No audio files found.")

    waveforms, labels = load_dataset(df)

    # Classical ML features
    X_feat = build_feature_matrix(waveforms)
    # CNN spectrogram features
    X_spec  = build_spectrogram_array(waveforms, n_mels=128, fixed_length=128)
    X_spec  = (X_spec - X_spec.min()) / (X_spec.max() - X_spec.min() + 1e-8)

    # ── 2. Reproduce same test split (must use same seed!) ───
    # Classical split
    X_feat_tr, X_feat_te, y_tr, y_te = train_test_split(
        X_feat, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels)

    # CNN split  (mirrors train_cnn.py)
    X_temp, X_spec_te, y_temp, y_spec_te = train_test_split(
        X_spec, labels, test_size=0.15, random_state=RANDOM_SEED, stratify=labels)

    # ── 3. Load models ───────────────────────────────────────
    all_metrics = {}

    # 3a. Random Forest
    rf_path  = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    sc_path  = os.path.join(MODELS_DIR, "random_forest_scaler.pkl")
    if os.path.exists(rf_path):
        rf      = joblib.load(rf_path)
        scaler  = joblib.load(sc_path)
        X_feat_te_s = scaler.transform(X_feat_te)
        y_pred_rf   = rf.predict(X_feat_te_s)

        report = classification_report(y_te, y_pred_rf,
                                        target_names=["non_drone", "drone"],
                                        output_dict=True)
        cm_rf  = confusion_matrix(y_te, y_pred_rf)

        print("\n📊 Random Forest Results")
        print(classification_report(y_te, y_pred_rf,
                                     target_names=["non_drone", "drone"]))
        print(f"Confusion Matrix:\n{cm_rf}\n")

        plot_confusion_matrix(cm_rf, "Random Forest")
        all_metrics["Random Forest"] = {
            "accuracy":  report["accuracy"],
            "precision": report["drone"]["precision"],
            "recall":    report["drone"]["recall"],
            "f1":        report["drone"]["f1-score"],
            "confusion_matrix": cm_rf.tolist()
        }
    else:
        print("⚠️  Random Forest model not found. Run train_classical.py first.")

    # 3b. SVM
    svm_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    sc_path2 = os.path.join(MODELS_DIR, "svm_scaler.pkl")
    if os.path.exists(svm_path):
        svm     = joblib.load(svm_path)
        scaler2 = joblib.load(sc_path2)
        X_feat_te_s2 = scaler2.transform(X_feat_te)
        y_pred_svm   = svm.predict(X_feat_te_s2)

        report2 = classification_report(y_te, y_pred_svm,
                                         target_names=["non_drone", "drone"],
                                         output_dict=True)
        cm_svm  = confusion_matrix(y_te, y_pred_svm)

        print("\n📊 SVM Results")
        print(classification_report(y_te, y_pred_svm,
                                     target_names=["non_drone", "drone"]))
        print(f"Confusion Matrix:\n{cm_svm}\n")

        plot_confusion_matrix(cm_svm, "SVM")
        all_metrics["SVM"] = {
            "accuracy":  report2["accuracy"],
            "precision": report2["drone"]["precision"],
            "recall":    report2["drone"]["recall"],
            "f1":        report2["drone"]["f1-score"],
            "confusion_matrix": cm_svm.tolist()
        }
    else:
        print("⚠️  SVM model not found. Run train_classical.py first.")

    # 3c. CNN
    cnn_path = os.path.join(MODELS_DIR, "cnn_model.keras")
    if not os.path.exists(cnn_path):
        cnn_path = os.path.join(MODELS_DIR, "cnn_model.h5")
    if os.path.exists(cnn_path):
        cnn = tf.keras.models.load_model(cnn_path)
        y_prob_cnn = cnn.predict(X_spec_te, verbose=0).flatten()
        y_pred_cnn = (y_prob_cnn >= 0.5).astype(int)

        report3 = classification_report(y_spec_te, y_pred_cnn,
                                         target_names=["non_drone", "drone"],
                                         output_dict=True)
        cm_cnn  = confusion_matrix(y_spec_te, y_pred_cnn)

        print("\n📊 CNN Results")
        print(classification_report(y_spec_te, y_pred_cnn,
                                     target_names=["non_drone", "drone"]))
        print(f"Confusion Matrix:\n{cm_cnn}\n")

        plot_confusion_matrix(cm_cnn, "CNN")
        all_metrics["CNN"] = {
            "accuracy":  report3["accuracy"],
            "precision": report3["drone"]["precision"],
            "recall":    report3["drone"]["recall"],
            "f1":        report3["drone"]["f1-score"],
            "confusion_matrix": cm_cnn.tolist()
        }
    else:
        print("⚠️  CNN model not found. Run train_cnn.py first.")

    # ── 4. Comparison table & chart ──────────────────────────
    if all_metrics:
        print("\n" + "=" * 60)
        print("  FINAL COMPARISON TABLE")
        print("=" * 60)
        print(f"{'Model':<20} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print("─" * 60)
        for name, m in all_metrics.items():
            print(f"{name:<20} {m['accuracy']:>9.3f} "
                  f"{m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f}")

        # Winner
        best_model = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
        print(f"\n🏆 Best model by F1-score: {best_model}")

        print("\n📝 Analysis:")
        print("   • CNN learns directly from mel spectrograms (raw patterns)")
        print("   • Random Forest uses handcrafted features (interpretable)")
        print("   • SVM also uses handcrafted features with a non-linear boundary")
        print("   • For real-world deployment with enough data, CNN is recommended")
        print("   • For small datasets or edge devices, Random Forest is preferred")

        # Bar chart
        plot_model_comparison(all_metrics)

        # Save combined metrics
        from utils import save_metrics
        save_metrics(all_metrics)
        print("\n✅ Evaluation complete! Check results/ for figures and metrics.json")

    plt.show()


if __name__ == "__main__":
    main()
