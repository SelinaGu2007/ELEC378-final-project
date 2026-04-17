from __future__ import annotations

import os
from typing import Any, Optional

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# Stage flags
RUN_SPLIT = True
RUN_SANITY_CHECK = True
RUN_BUILD_BOVW = True
RUN_TRAIN = True
RUN_SUBMISSION = True


# Pipeline settings
RANDOM_SEED = 42
VAL_RATIO = 0.20
SANITY_CHECK_SAMPLE_SIZE = 20
VISUALIZE_SAMPLE = False

VOCAB_SIZE = 100
MAX_DESCRIPTORS_FOR_KMEANS = 50_000

C_VALUE = 1.0
KERNEL = "rbf"
GAMMA = "scale"


# Paths and constants
DATA_DIR = "data"

TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "train_images", "train_images")
TEST_IMAGE_DIR = os.path.join(DATA_DIR, "test_images", "test_images")
TRAIN_SPLIT_PATH = os.path.join(DATA_DIR, "train_split.csv")
VAL_SPLIT_PATH = os.path.join(DATA_DIR, "val_split.csv")

FEATURES_DIR = "features"
SUBMISSION_DIR = "submission"

TRAIN_FEATURES_PATH = os.path.join(FEATURES_DIR, "train_bovw_features.npy")
VAL_FEATURES_PATH = os.path.join(FEATURES_DIR, "val_bovw_features.npy")
TRAIN_LABELS_PATH = os.path.join(FEATURES_DIR, "train_labels.npy")
VAL_LABELS_PATH = os.path.join(FEATURES_DIR, "val_labels.npy")
TRAIN_FILE_NAMES_PATH = os.path.join(FEATURES_DIR, "train_file_names.npy")
VAL_FILE_NAMES_PATH = os.path.join(FEATURES_DIR, "val_file_names.npy")

KMEANS_MODEL_PATH = os.path.join(FEATURES_DIR, "bovw_kmeans.joblib")
CLASSIFIER_PATH = os.path.join(FEATURES_DIR, "sift_bovw_svc_rbf.joblib")
LABEL_ENCODER_PATH = os.path.join(FEATURES_DIR, "sift_label_encoder.joblib")
OUTPUT_SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission_sift_bovw_svc.csv")


def ensure_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


def validate_labeled_dataframe(df: pd.DataFrame, name: str) -> None:
    required_columns = {"file_name", "TARGET"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"{name} CSV must contain columns: {sorted(required_columns)}")


def create_sift_extractor() -> Any:
    return cv2.SIFT_create()


def load_image(file_name: str, image_dir: str) -> np.ndarray:
    image_path = os.path.join(image_dir, file_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as image:
        return np.array(image.convert("RGB"))


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def extract_sift_descriptors(
    gray_image: np.ndarray, sift_extractor: Any
) -> Optional[np.ndarray]:
    _, descriptors = sift_extractor.detectAndCompute(gray_image, None)
    return descriptors


# Stage 1: train/validation split
def split_train_val(
    input_csv: str = TRAIN_CSV_PATH,
    train_output_csv: str = TRAIN_SPLIT_PATH,
    val_output_csv: str = VAL_SPLIT_PATH,
    val_ratio: float = VAL_RATIO,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_file_exists(input_csv)
    df = pd.read_csv(input_csv)
    validate_labeled_dataframe(df, "Training")

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df["TARGET"],
        random_state=random_seed,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    os.makedirs(os.path.dirname(train_output_csv), exist_ok=True)
    train_df.to_csv(train_output_csv, index=False)
    val_df.to_csv(val_output_csv, index=False)

    print(f"Total images: {len(df)}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Saved training split to: {train_output_csv}")
    print(f"Saved validation split to: {val_output_csv}")

    return train_df, val_df


# Stage 2: SIFT sanity check
def run_sift_sanity_check(
    train_csv_path: str = TRAIN_CSV_PATH,
    image_dir: str = TRAIN_IMAGE_DIR,
    sample_size: int = SANITY_CHECK_SAMPLE_SIZE,
    visualize_sample: bool = VISUALIZE_SAMPLE,
) -> None:
    ensure_file_exists(train_csv_path)
    train_df = pd.read_csv(train_csv_path)
    validate_labeled_dataframe(train_df, "Training")

    sift_extractor = create_sift_extractor()
    sample_df = train_df.head(sample_size).copy()

    keypoint_counts: list[int] = []
    images_with_no_descriptors = 0
    example_printed = False

    for row in sample_df.itertuples(index=False):
        image = load_image(row.file_name, image_dir)
        gray_image = to_grayscale(image)
        keypoints, descriptors = sift_extractor.detectAndCompute(gray_image, None)

        keypoint_counts.append(len(keypoints))
        if descriptors is None:
            images_with_no_descriptors += 1

        if not example_printed:
            descriptor_shape = None if descriptors is None else descriptors.shape
            print("Example image result")
            print(f"  File name: {row.file_name}")
            print(f"  Label: {row.TARGET}")
            print(f"  Original image shape: {image.shape}")
            print(f"  Grayscale image shape: {gray_image.shape}")
            print(f"  Number of keypoints: {len(keypoints)}")
            print(f"  Descriptor shape: {descriptor_shape}")
            print(f"  Descriptors are None: {descriptors is None}")

            if visualize_sample:
                import matplotlib.pyplot as plt

                overlay = cv2.drawKeypoints(
                    image,
                    keypoints,
                    None,
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                )
                plt.figure(figsize=(6, 6))
                plt.imshow(overlay)
                plt.title(f"SIFT keypoints: {row.file_name}")
                plt.axis("off")
                plt.tight_layout()
                plt.show()

            example_printed = True

    if not keypoint_counts:
        print("No images were successfully processed.")
        return

    keypoint_array = np.array(keypoint_counts, dtype=np.float32)
    print("\nSample summary")
    print(f"  Number of images processed: {len(keypoint_counts)}")
    print(f"  Images with no detected descriptors: {images_with_no_descriptors}")
    print(f"  Min number of keypoints: {int(keypoint_array.min())}")
    print(f"  Max number of keypoints: {int(keypoint_array.max())}")
    print(f"  Mean number of keypoints: {float(keypoint_array.mean()):.2f}")


# Stage 3: BoVW features
def collect_image_descriptors(
    df: pd.DataFrame,
    sift_extractor: Any,
    image_dir: str,
) -> tuple[list[Optional[np.ndarray]], np.ndarray, np.ndarray, int]:
    descriptor_list: list[Optional[np.ndarray]] = []
    labels: list[str] = []
    file_names: list[str] = []
    images_with_no_descriptors = 0

    for row in df.itertuples(index=False):
        image = load_image(row.file_name, image_dir)
        gray_image = to_grayscale(image)
        descriptors = extract_sift_descriptors(gray_image, sift_extractor)

        descriptor_list.append(descriptors)
        labels.append(row.TARGET)
        file_names.append(row.file_name)

        if descriptors is None:
            images_with_no_descriptors += 1

    return (
        descriptor_list,
        np.array(labels, dtype=object),
        np.array(file_names, dtype=object),
        images_with_no_descriptors,
    )


def fit_visual_vocabulary(
    descriptor_list: list[Optional[np.ndarray]],
    vocab_size: int = VOCAB_SIZE,
    max_descriptors: int = MAX_DESCRIPTORS_FOR_KMEANS,
    random_seed: int = RANDOM_SEED,
) -> tuple[MiniBatchKMeans, int, bool]:
    valid_descriptors = [
        descriptors.astype(np.float32)
        for descriptors in descriptor_list
        if descriptors is not None and len(descriptors) > 0
    ]

    if not valid_descriptors:
        raise ValueError("No training descriptors were collected for vocabulary fitting.")

    all_descriptors = np.vstack(valid_descriptors)
    total_descriptors = int(all_descriptors.shape[0])
    subsampling_used = total_descriptors > max_descriptors

    if subsampling_used:
        rng = np.random.default_rng(random_seed)
        selected_indices = rng.choice(total_descriptors, size=max_descriptors, replace=False)
        kmeans_descriptors = all_descriptors[selected_indices]
    else:
        kmeans_descriptors = all_descriptors

    n_clusters = min(vocab_size, int(kmeans_descriptors.shape[0]))
    if n_clusters < vocab_size:
        print(
            f"Warning: only {kmeans_descriptors.shape[0]} descriptors available, "
            f"so vocabulary size is reduced from {vocab_size} to {n_clusters}."
        )

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_seed,
        batch_size=1024,
        n_init=10,
    )
    kmeans.fit(kmeans_descriptors)

    return kmeans, total_descriptors, subsampling_used


def encode_bovw_histogram(
    descriptors: Optional[np.ndarray],
    kmeans: MiniBatchKMeans,
    vocab_size: Optional[int] = None,
) -> np.ndarray:
    effective_vocab_size = int(kmeans.n_clusters) if vocab_size is None else vocab_size
    histogram = np.zeros(effective_vocab_size, dtype=np.float32)

    if descriptors is None or len(descriptors) == 0:
        return histogram

    visual_words = kmeans.predict(descriptors.astype(np.float32))
    counts = np.bincount(visual_words, minlength=effective_vocab_size).astype(np.float32)

    total_count = float(counts.sum())
    if total_count > 0.0:
        counts /= total_count

    histogram[: counts.shape[0]] = counts
    return histogram


def encode_dataset(
    descriptor_list: list[Optional[np.ndarray]],
    kmeans: MiniBatchKMeans,
    vocab_size: Optional[int] = None,
) -> np.ndarray:
    histograms = [
        encode_bovw_histogram(descriptors, kmeans, vocab_size=vocab_size)
        for descriptors in descriptor_list
    ]
    return np.vstack(histograms).astype(np.float32)


def build_bovw_features(
    train_split_path: str = TRAIN_SPLIT_PATH,
    val_split_path: str = VAL_SPLIT_PATH,
    image_dir: str = TRAIN_IMAGE_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ensure_file_exists(train_split_path)
    ensure_file_exists(val_split_path)

    train_df = pd.read_csv(train_split_path)
    val_df = pd.read_csv(val_split_path)
    validate_labeled_dataframe(train_df, "Training split")
    validate_labeled_dataframe(val_df, "Validation split")

    sift_extractor = create_sift_extractor()

    print(f"Number of train images in split: {len(train_df)}")
    print(f"Number of val images in split: {len(val_df)}")

    train_descriptor_list, train_labels, train_file_names, train_no_desc = (
        collect_image_descriptors(
            df=train_df,
            sift_extractor=sift_extractor,
            image_dir=image_dir,
        )
    )
    val_descriptor_list, val_labels, val_file_names, val_no_desc = collect_image_descriptors(
        df=val_df,
        sift_extractor=sift_extractor,
        image_dir=image_dir,
    )

    print(f"Number of processed train images: {len(train_descriptor_list)}")
    print(f"Number of processed val images: {len(val_descriptor_list)}")
    print(f"Number of train images with no descriptors: {train_no_desc}")
    print(f"Number of val images with no descriptors: {val_no_desc}")

    kmeans, total_train_descriptors, subsampling_used = fit_visual_vocabulary(
        descriptor_list=train_descriptor_list
    )
    effective_vocab_size = int(kmeans.n_clusters)

    print(f"Total number of training descriptors collected: {total_train_descriptors}")
    print(f"Subsampling used for k-means: {subsampling_used}")
    print(f"Final vocabulary size: {effective_vocab_size}")

    train_features = encode_dataset(train_descriptor_list, kmeans, effective_vocab_size)
    val_features = encode_dataset(val_descriptor_list, kmeans, effective_vocab_size)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    np.save(TRAIN_FEATURES_PATH, train_features)
    np.save(VAL_FEATURES_PATH, val_features)
    np.save(TRAIN_LABELS_PATH, train_labels)
    np.save(VAL_LABELS_PATH, val_labels)
    np.save(TRAIN_FILE_NAMES_PATH, train_file_names)
    np.save(VAL_FILE_NAMES_PATH, val_file_names)
    joblib.dump(kmeans, KMEANS_MODEL_PATH)

    print(f"Final train feature matrix shape: {train_features.shape}")
    print(f"Final val feature matrix shape: {val_features.shape}")
    print(f"Saved BoVW features to: {FEATURES_DIR}")
    print(f"Saved BoVW vocabulary model to: {KMEANS_MODEL_PATH}")

    return train_features, val_features, train_labels, val_labels


# Stage 4 and 5: classifier training and evaluation
def load_saved_bovw_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for path in [TRAIN_FEATURES_PATH, VAL_FEATURES_PATH, TRAIN_LABELS_PATH, VAL_LABELS_PATH]:
        ensure_file_exists(path)

    train_features = np.load(TRAIN_FEATURES_PATH)
    val_features = np.load(VAL_FEATURES_PATH)
    train_labels = np.load(TRAIN_LABELS_PATH, allow_pickle=True)
    val_labels = np.load(VAL_LABELS_PATH, allow_pickle=True)
    return train_features, val_features, train_labels, val_labels


def validate_feature_data(
    train_features: np.ndarray,
    val_features: np.ndarray,
) -> None:
    if train_features.shape[1] != val_features.shape[1]:
        raise ValueError(
            "Train and validation feature dimensions do not match: "
            f"{train_features.shape[1]} vs {val_features.shape[1]}."
        )


def evaluate_classifier(
    classifier: SVC,
    label_encoder: LabelEncoder,
    train_features: np.ndarray,
    val_features: np.ndarray,
    train_labels_encoded: np.ndarray,
    val_labels_encoded: np.ndarray,
) -> dict[str, Any]:
    train_predictions = classifier.predict(train_features)
    val_predictions = classifier.predict(val_features)

    return {
        "train_accuracy": accuracy_score(train_labels_encoded, train_predictions),
        "val_accuracy": accuracy_score(val_labels_encoded, val_predictions),
        "classification_report": classification_report(
            val_labels_encoded,
            val_predictions,
            target_names=label_encoder.classes_.tolist(),
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(val_labels_encoded, val_predictions),
    }


def train_sift_classifier(
    train_features: Optional[np.ndarray] = None,
    val_features: Optional[np.ndarray] = None,
    train_labels: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
) -> tuple[SVC, LabelEncoder, dict[str, Any]]:
    if any(item is None for item in [train_features, val_features, train_labels, val_labels]):
        train_features, val_features, train_labels, val_labels = load_saved_bovw_features()

    assert train_features is not None
    assert val_features is not None
    assert train_labels is not None
    assert val_labels is not None

    validate_feature_data(train_features, val_features)

    train_labels = train_labels.astype(str)
    val_labels = val_labels.astype(str)

    train_unique_labels = np.unique(train_labels)
    val_unique_labels = np.unique(val_labels)

    print(f"Train feature matrix shape: {train_features.shape}")
    print(f"Validation feature matrix shape: {val_features.shape}")
    print(f"Unique train labels: {len(train_unique_labels)}")
    print(f"Unique validation labels: {len(val_unique_labels)}")

    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    val_labels_encoded = label_encoder.transform(val_labels)

    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(
        "Classifier hyperparameters: "
        f"SVC(C={C_VALUE}, kernel='{KERNEL}', gamma='{GAMMA}', random_state={RANDOM_SEED})"
    )

    classifier = SVC(
        C=C_VALUE,
        kernel=KERNEL,
        gamma=GAMMA,
        random_state=RANDOM_SEED,
    )
    classifier.fit(train_features, train_labels_encoded)

    metrics = evaluate_classifier(
        classifier=classifier,
        label_encoder=label_encoder,
        train_features=train_features,
        val_features=val_features,
        train_labels_encoded=train_labels_encoded,
        val_labels_encoded=val_labels_encoded,
    )

    os.makedirs(FEATURES_DIR, exist_ok=True)
    joblib.dump(classifier, CLASSIFIER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    print(f"Training accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
    print("\nValidation classification report:")
    print(metrics["classification_report"])
    print("Validation confusion matrix:")
    print(metrics["confusion_matrix"])
    print(f"Saved classifier to: {CLASSIFIER_PATH}")
    print(f"Saved label encoder to: {LABEL_ENCODER_PATH}")

    return classifier, label_encoder, metrics


# Stage 6: submission
def make_kaggle_submission(
    sample_submission_path: Optional[str] = None,
    test_image_dir: str = TEST_IMAGE_DIR,
) -> str:
    submission_path = SAMPLE_SUBMISSION_PATH if sample_submission_path is None else sample_submission_path

    ensure_file_exists(submission_path)
    ensure_file_exists(KMEANS_MODEL_PATH)
    ensure_file_exists(CLASSIFIER_PATH)
    ensure_file_exists(LABEL_ENCODER_PATH)

    submission_df = pd.read_csv(submission_path)
    required_columns = {"ID", "TARGET"}
    if not required_columns.issubset(submission_df.columns):
        raise ValueError(
            f"Sample submission must contain columns: {sorted(required_columns)}"
        )

    sift_extractor = create_sift_extractor()
    kmeans: MiniBatchKMeans = joblib.load(KMEANS_MODEL_PATH)
    classifier: SVC = joblib.load(CLASSIFIER_PATH)
    label_encoder: LabelEncoder = joblib.load(LABEL_ENCODER_PATH)

    print(f"Number of test rows loaded: {len(submission_df)}")

    features: list[np.ndarray] = []

    for row in submission_df.itertuples(index=False):
        file_name = f"{row.ID}.jpg"
        image = load_image(file_name, test_image_dir)
        gray_image = to_grayscale(image)
        descriptors = extract_sift_descriptors(gray_image, sift_extractor)
        histogram = encode_bovw_histogram(descriptors, kmeans)
        features.append(histogram)

    test_features = np.vstack(features).astype(np.float32)
    predicted_numeric_labels = classifier.predict(test_features)
    predicted_string_labels = label_encoder.inverse_transform(predicted_numeric_labels)

    submission_df["TARGET"] = predicted_string_labels
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(OUTPUT_SUBMISSION_PATH, index=False)

    print(f"Final submission shape: {submission_df.shape}")
    print(f"Output file path: {OUTPUT_SUBMISSION_PATH}")

    return OUTPUT_SUBMISSION_PATH


def main() -> None:
    train_features: Optional[np.ndarray] = None
    val_features: Optional[np.ndarray] = None
    train_labels: Optional[np.ndarray] = None
    val_labels: Optional[np.ndarray] = None

    if RUN_SPLIT:
        split_train_val()

    if RUN_SANITY_CHECK:
        run_sift_sanity_check()

    if RUN_BUILD_BOVW:
        train_features, val_features, train_labels, val_labels = build_bovw_features()

    if RUN_TRAIN:
        train_sift_classifier(train_features, val_features, train_labels, val_labels)

    if RUN_SUBMISSION:
        make_kaggle_submission()


if __name__ == "__main__":
    main()
