"""
SVM-only hyperparameter tuning for the SIFT + BoVW baseline.

This script assumes scripts/sift_kernel_svm.py has already generated fixed BoVW
features. It does not rerun SIFT extraction or k-means. It compares several SVM
kernels and hyperparameters for the report's tuning discussion.
"""

from __future__ import annotations

import os
from itertools import product
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# This script tunes only the SVM classifier. The BoVW vocabulary and feature
# matrices are fixed and reused from the main SIFT pipeline in sift_kernel_svm.py.
RANDOM_SEED = 42

FEATURES_DIR = "features"

TRAIN_FEATURES_PATH = os.path.join(FEATURES_DIR, "train_bovw_features.npy")
VAL_FEATURES_PATH = os.path.join(FEATURES_DIR, "val_bovw_features.npy")
TRAIN_LABELS_PATH = os.path.join(FEATURES_DIR, "train_labels.npy")
VAL_LABELS_PATH = os.path.join(FEATURES_DIR, "val_labels.npy")

GRID_SEARCH_RESULTS_PATH = os.path.join(
    FEATURES_DIR, "sift_svm_grid_search_results.csv"
)
BEST_GRID_CLASSIFIER_PATH = os.path.join(FEATURES_DIR, "sift_bovw_svc_best_grid.joblib")
BEST_GRID_LABEL_ENCODER_PATH = os.path.join(
    FEATURES_DIR, "sift_label_encoder_best_grid.joblib"
)

LINEAR_C_VALUES = [1]
RBF_C_VALUES = [0.1, 1, 10]
RBF_GAMMA_VALUES: list[str | float] = ["scale", "auto", 0.01, 0.1]
POLY_C_VALUES = [1]
POLY_DEGREE_VALUES = [2, 3]
POLY_GAMMA_VALUES: list[str | float] = ["scale"]
SIGMOID_C_VALUES = [1]
SIGMOID_GAMMA_VALUES: list[str | float] = ["scale", 0.01]


def validate_feature_files_exist() -> None:
    required_paths = [
        TRAIN_FEATURES_PATH,
        VAL_FEATURES_PATH,
        TRAIN_LABELS_PATH,
        VAL_LABELS_PATH,
    ]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]

    if missing_paths:
        raise FileNotFoundError(
            "Missing saved BoVW feature files. Run scripts/sift_kernel_svm.py first. "
            f"Missing: {missing_paths}"
        )


def load_saved_bovw_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    validate_feature_files_exist()

    train_features = np.load(TRAIN_FEATURES_PATH)
    val_features = np.load(VAL_FEATURES_PATH)
    train_labels = np.load(TRAIN_LABELS_PATH, allow_pickle=True).astype(str)
    val_labels = np.load(VAL_LABELS_PATH, allow_pickle=True).astype(str)

    if train_features.shape[1] != val_features.shape[1]:
        raise ValueError(
            "Train and validation feature dimensions do not match: "
            f"{train_features.shape[1]} vs {val_features.shape[1]}."
        )

    return train_features, val_features, train_labels, val_labels


def build_svm_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []

    for c_value in LINEAR_C_VALUES:
        grid.append({"kernel": "linear", "C": c_value, "gamma": None, "degree": None})

    for c_value, gamma in product(RBF_C_VALUES, RBF_GAMMA_VALUES):
        grid.append({"kernel": "rbf", "C": c_value, "gamma": gamma, "degree": None})

    for c_value, degree, gamma in product(
        POLY_C_VALUES, POLY_DEGREE_VALUES, POLY_GAMMA_VALUES
    ):
        grid.append({"kernel": "poly", "C": c_value, "gamma": gamma, "degree": degree})

    for c_value, gamma in product(SIGMOID_C_VALUES, SIGMOID_GAMMA_VALUES):
        grid.append(
            {"kernel": "sigmoid", "C": c_value, "gamma": gamma, "degree": None}
        )

    return grid


def build_svc(trial: dict[str, Any]) -> SVC:
    params = {
        "C": trial["C"],
        "kernel": trial["kernel"],
        "random_state": RANDOM_SEED,
    }

    if trial["gamma"] is not None:
        params["gamma"] = trial["gamma"]
    if trial["degree"] is not None:
        params["degree"] = trial["degree"]

    return SVC(**params)


def run_grid_search() -> tuple[SVC, LabelEncoder, pd.DataFrame]:
    train_features, val_features, train_labels, val_labels = load_saved_bovw_data()

    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    val_labels_encoded = label_encoder.transform(val_labels)

    grid = build_svm_grid()
    results: list[dict[str, Any]] = []
    best_model: SVC | None = None
    best_val_accuracy = -1.0

    print("Starting SVM-only grid search on saved SIFT + BoVW features")
    print("BoVW vocabulary and feature matrices are fixed and reused.")
    print(f"Train feature matrix shape: {train_features.shape}")
    print(f"Validation feature matrix shape: {val_features.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Total SVM configurations: {len(grid)}")

    for trial_number, trial in enumerate(grid, start=1):
        classifier = build_svc(trial)

        print(
            "\n"
            f"[{trial_number}/{len(grid)}] "
            f"kernel={trial['kernel']}, C={trial['C']}, "
            f"gamma={trial['gamma']}, degree={trial['degree']}"
        )

        classifier.fit(train_features, train_labels_encoded)

        train_predictions = classifier.predict(train_features)
        val_predictions = classifier.predict(val_features)
        train_accuracy = accuracy_score(train_labels_encoded, train_predictions)
        val_accuracy = accuracy_score(val_labels_encoded, val_predictions)

        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")

        results.append(
            {
                "kernel": trial["kernel"],
                "C": trial["C"],
                "gamma": trial["gamma"],
                "degree": trial["degree"],
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
        )

        if val_accuracy > best_val_accuracy:
            best_model = classifier
            best_val_accuracy = val_accuracy

    if best_model is None:
        raise ValueError("No SVM grid-search trials were run.")

    results_df = pd.DataFrame(results)
    sorted_results_df = results_df.sort_values(
        ["val_accuracy", "train_accuracy"],
        ascending=[False, False],
    ).reset_index(drop=True)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    sorted_results_df.to_csv(GRID_SEARCH_RESULTS_PATH, index=False)
    joblib.dump(best_model, BEST_GRID_CLASSIFIER_PATH)
    joblib.dump(label_encoder, BEST_GRID_LABEL_ENCODER_PATH)

    print("\nSVM grid search results sorted by validation accuracy:")
    print(sorted_results_df.to_string(index=False))
    print("\nBest SVM configuration:")
    print(sorted_results_df.head(1).to_string(index=False))
    print(f"\nSaved grid search results to: {GRID_SEARCH_RESULTS_PATH}")
    print(f"Saved best SVM model to: {BEST_GRID_CLASSIFIER_PATH}")
    print(f"Saved best label encoder to: {BEST_GRID_LABEL_ENCODER_PATH}")

    return best_model, label_encoder, sorted_results_df


def main() -> None:
    run_grid_search()


if __name__ == "__main__":
    main()
