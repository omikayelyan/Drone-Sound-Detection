"""
train_classical.py
------------------
Trains and saves two classical machine-learning models:
  1. Random Forest  – an ensemble of decision trees
  2. Support Vector Machine (SVM) with RBF kernel

Both models use the 94-dimensional feature vectors produced by
feature_extraction.py (MFCCs + chroma + spectral centroid + ZCR).

Run:
    python src/train_classical.py
"""

import os, sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Make sure sibling modules are importable when running this file directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader       import collect_file_paths, load_dataset
from feature_extraction import build_feature_matrix
from utils             import MODELS_DIR, RANDOM_SEED, ensure_dirs


# ──────────────────────────────────────────────────────────────
# Step 1: Load data & extract features
# ──────────────────────────────────────────────────────────────
def prepare_data():
    """Load audio files and compute feature matrix."""
    df = collect_file_paths()
    if df.empty:
        raise RuntimeError("No audio files found. Check data/drone/ and data/non_drone/")

    waveforms, labels = load_dataset(df)
    X = build_feature_matrix(waveforms)
    return X, labels


# ──────────────────────────────────────────────────────────────
# Step 2: Pre-process – scale features
# ──────────────────────────────────────────────────────────────
def scale_features(X_train, X_test):
    """
    Standardise features to zero mean and unit variance.

    Why scale?
        SVM is very sensitive to feature scale: a feature with values 0–10,000
        (like spectral centroid in Hz) would dominate over one with values 0–1
        (like ZCR). StandardScaler ensures fair comparison.
        Random Forest doesn't need scaling, but it doesn't hurt.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit on train only!
    X_test_scaled  = scaler.transform(X_test)        # apply same transform to test
    return X_train_scaled, X_test_scaled, scaler


# ──────────────────────────────────────────────────────────────
# Step 3a: Train Random Forest
# ──────────────────────────────────────────────────────────────
def train_random_forest(X_train: np.ndarray,
                         y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Random Forest works by training many decision trees on random subsets
    of the data, then voting on the final prediction. This reduces
    overfitting compared to a single tree.

    Key hyperparameters:
        n_estimators : number of trees (more = more accurate but slower)
        max_depth    : maximum depth of each tree (None = grow until pure)
        random_state : seed for reproducibility
    """
    print("\n🌲 Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=200,        # 200 trees
        max_depth=None,          # grow fully (regularised by min_samples_leaf)
        min_samples_leaf=2,      # each leaf must have ≥2 training samples
        class_weight="balanced", # handle class imbalance if any
        n_jobs=-1,               # use all CPU cores
        random_state=RANDOM_SEED
    )
    rf.fit(X_train, y_train)

    # Cross-validation score on training set (5-fold)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1")
    print(f"   Random Forest 5-fold CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return rf


# ──────────────────────────────────────────────────────────────
# Step 3b: Train SVM
# ──────────────────────────────────────────────────────────────
def train_svm(X_train: np.ndarray,
              y_train: np.ndarray) -> SVC:
    """
    Train a Support Vector Machine with an RBF (Gaussian) kernel.

    SVM finds the hyperplane that best separates the two classes
    with maximum margin. The RBF kernel maps the features into a
    higher-dimensional space, allowing non-linear decision boundaries.

    Key hyperparameters:
        C     : regularisation – larger = less regularisation (fits tighter)
        gamma : RBF width – 'scale' = 1 / (n_features * X.var())
    """
    print("\n🤖 Training SVM …")
    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,        # enable predict_proba (needed for some metrics)
        class_weight="balanced",
        random_state=RANDOM_SEED
    )
    svm.fit(X_train, y_train)

    cv_scores = cross_val_score(svm, X_train, y_train, cv=5, scoring="f1")
    print(f"   SVM 5-fold CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return svm


# ──────────────────────────────────────────────────────────────
# Step 4: Evaluate on test set
# ──────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Print classification report and return metrics as a dict."""
    print(f"\n📊 {model_name} – Test Set Results")
    print("─" * 50)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred,
                                   target_names=["non_drone", "drone"],
                                   output_dict=True)

    # Pretty print
    print(classification_report(y_test, y_pred,
                                 target_names=["non_drone", "drone"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy":  report["accuracy"],
        "precision": report["drone"]["precision"],
        "recall":    report["drone"]["recall"],
        "f1":        report["drone"]["f1-score"],
        "confusion_matrix": cm.tolist()
    }


# ──────────────────────────────────────────────────────────────
# Step 5: Save models to disk
# ──────────────────────────────────────────────────────────────
def save_model(model, scaler, name: str):
    """
    Persist a trained model and its scaler using joblib.

    joblib is faster than pickle for large numpy arrays (like model weights).
    """
    ensure_dirs()
    model_path  = os.path.join(MODELS_DIR, f"{name}_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{name}_scaler.pkl")
    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)
    print(f"💾 Saved {name} model  → {model_path}")
    print(f"💾 Saved {name} scaler → {scaler_path}")


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CLASSICAL ML TRAINING – Drone Sound Detection")
    print("=" * 60)

    # 1. Load & featurise
    X, y = prepare_data()
    print(f"\nDataset summary: {X.shape[0]} samples, {X.shape[1]} features")

    # 2. Train / test split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y            # maintain class ratio in both splits
    )
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # 3. Scale
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    # 4a. Random Forest (doesn't need scaling, but we pass scaled for consistency)
    rf = train_random_forest(X_train_s, y_train)
    rf_metrics = evaluate_model(rf, X_test_s, y_test, "Random Forest")

    # 4b. SVM
    svm = train_svm(X_train_s, y_train)
    svm_metrics = evaluate_model(svm, X_test_s, y_test, "SVM")

    # 5. Save both models
    save_model(rf,  scaler, "random_forest")
    save_model(svm, scaler, "svm")

    print("\n✅ Classical ML training complete!")
    return rf, svm, rf_metrics, svm_metrics


if __name__ == "__main__":
    main()
