from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm



RUN_SPLIT = True
RUN_SANITY_CHECK = True
RUN_BUILD_FEATURES = True
RUN_TRAIN = True
RUN_SUBMISSION = True


RANDOM_SEED = 42
VAL_RATIO = 0.20
SANITY_CHECK_SAMPLE_SIZE = 20
VISUALIZE_SAMPLE = False

# Raw feature settings
RAW_IMAGE_SIZE = (32, 32)   # start small for speed

# SVM settings
C_VALUE = 1
KERNEL = "linear"
GAMMA = "scale"


# Paths
PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_DIR / "raw_data"
PROCESSED_DATA_DIR = PROJECT_DIR / "processed_data"
FEATURES_DIR = PROJECT_DIR / "features"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TRAIN_IMAGE_DIR = PROJECT_DIR / "train_images"
TEST_IMAGE_DIR = PROJECT_DIR / "test_images"

TRAIN_CSV_PATH = RAW_DATA_DIR / "train.csv"
TRAIN_SPLIT_PATH = PROCESSED_DATA_DIR / "train_split.csv"
VAL_SPLIT_PATH = PROCESSED_DATA_DIR / "val_split.csv"
SAMPLE_SUBMISSION_PATH = PROJECT_DIR / "sample_submission.csv"
FALLBACK_SAMPLE_SUBMISSION_PATH = SUBMISSION_DIR / "sample_submission.csv"

TRAIN_FEATURES_PATH = FEATURES_DIR / "train_raw_features.npy"
VAL_FEATURES_PATH = FEATURES_DIR / "val_raw_features.npy"
TRAIN_LABELS_PATH = FEATURES_DIR / "train_labels.npy"
VAL_LABELS_PATH = FEATURES_DIR / "val_labels.npy"
TRAIN_FILE_NAMES_PATH = FEATURES_DIR / "train_file_names.npy"
VAL_FILE_NAMES_PATH = FEATURES_DIR / "val_file_names.npy"

SCALER_PATH = FEATURES_DIR / "raw_feature_scaler.joblib"
CLASSIFIER_PATH = FEATURES_DIR / "raw_kernel_svc.joblib"
LABEL_ENCODER_PATH = FEATURES_DIR / "raw_label_encoder.joblib"
OUTPUT_SUBMISSION_PATH = SUBMISSION_DIR / "submission_raw_kernel_svc.csv"


# Shared helpers
def ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def resolve_sample_submission_path() -> Path:
    if SAMPLE_SUBMISSION_PATH.exists():
        return SAMPLE_SUBMISSION_PATH
    if FALLBACK_SAMPLE_SUBMISSION_PATH.exists():
        return FALLBACK_SAMPLE_SUBMISSION_PATH
    raise FileNotFoundError(
        f"Could not find sample submission CSV at {SAMPLE_SUBMISSION_PATH} "
        f"or {FALLBACK_SAMPLE_SUBMISSION_PATH}."
    )


def validate_labeled_dataframe(df: pd.DataFrame, name: str) -> None:
    required_columns = {"file_name", "TARGET"}
    if df.empty:
        raise ValueError(f"{name} CSV is empty.")
    if not required_columns.issubset(df.columns):
        raise ValueError(f"{name} CSV must contain columns: {sorted(required_columns)}")


def load_image(file_name: str, image_dir: Path) -> np.ndarray:
    image_path = image_dir / file_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as image:
        return np.array(image.convert("RGB"))


def extract_raw_feature_vector(
    image: np.ndarray,
    output_size: tuple[int, int] = RAW_IMAGE_SIZE,
) -> np.ndarray:
    """
    Full preprocessing pipeline:
    RGB -> resize -> flatten -> float32 in [0, 1]
    """
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize(output_size)
    resized_array = np.array(resized_image, dtype=np.float32)

    feature_vector = resized_array.reshape(-1) / 255.0
    return feature_vector


# Stage 1: train/validation split

def split_train_val(
    input_csv: Path = TRAIN_CSV_PATH,
    train_output_csv: Path = TRAIN_SPLIT_PATH,
    val_output_csv: Path = VAL_SPLIT_PATH,
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

    train_output_csv.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_output_csv, index=False)
    val_df.to_csv(val_output_csv, index=False)

    print(f"Total images: {len(df)}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Saved training split to: {train_output_csv}")
    print(f"Saved validation split to: {val_output_csv}")

    return train_df, val_df


# Stage 2: raw sanity check

def run_raw_sanity_check(
    train_csv_path: Path = TRAIN_CSV_PATH,
    image_dir: Path = TRAIN_IMAGE_DIR,
    sample_size: int = SANITY_CHECK_SAMPLE_SIZE,
    visualize_sample: bool = VISUALIZE_SAMPLE,
) -> None:
    ensure_file_exists(train_csv_path)
    train_df = pd.read_csv(train_csv_path)
    validate_labeled_dataframe(train_df, "Training")

    sample_df = train_df.head(sample_size).copy()

    feature_lengths: list[int] = []
    example_printed = False

    for row in sample_df.itertuples(index=False):
        try:
            image = load_image(row.file_name, image_dir)
        except FileNotFoundError as error:
            print(f"Skipping missing file: {error}")
            continue

        feature_vector = extract_raw_feature_vector(image)

        feature_lengths.append(len(feature_vector))

        if not example_printed:
            print("Example image result")
            print(f"  File name: {row.file_name}")
            print(f"  Label: {row.TARGET}")
            print(f"  Original image shape: {image.shape}")
            print(f"  Feature vector length: {len(feature_vector)}")
            print(f"  Feature min: {feature_vector.min():.4f}")
            print(f"  Feature max: {feature_vector.max():.4f}")
            print(f"  Feature mean: {feature_vector.mean():.4f}")

            if visualize_sample:
                try:
                    import matplotlib.pyplot as plt
                except ImportError:
                    print("Matplotlib is not installed, so visualization is skipped.")
                else:
                    plt.figure(figsize=(4, 4))
                    plt.imshow(image)
                    plt.title("Original RGB Image")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.show()

            example_printed = True

    if not feature_lengths:
        print("No images were successfully processed.")
        return

    feature_length_array = np.array(feature_lengths, dtype=np.int32)
    print("\nSample summary")
    print(f"  Number of images processed: {len(feature_lengths)}")
    print(f"  Feature length (all samples should match): {int(feature_length_array[0])}")


# Stage 3: raw features

def collect_raw_features(
    df: pd.DataFrame,
    image_dir: Path,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_list: list[np.ndarray] = []
    labels: list[str] = []
    file_names: list[str] = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=split_name):
        try:
            image = load_image(row.file_name, image_dir)
        except FileNotFoundError as error:
            print(f"[{split_name}] Skipping missing image: {error}")
            continue

        feature_vector = extract_raw_feature_vector(image)

        feature_list.append(feature_vector)
        labels.append(row.TARGET)
        file_names.append(row.file_name)

    if not feature_list:
        raise ValueError(f"No features were collected for split '{split_name}'.")

    return (
        np.vstack(feature_list).astype(np.float32),
        np.array(labels, dtype=object),
        np.array(file_names, dtype=object),
    )


def build_raw_features(
    train_split_path: Path = TRAIN_SPLIT_PATH,
    val_split_path: Path = VAL_SPLIT_PATH,
    image_dir: Path = TRAIN_IMAGE_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ensure_file_exists(train_split_path)
    ensure_file_exists(val_split_path)

    train_df = pd.read_csv(train_split_path)
    val_df = pd.read_csv(val_split_path)
    validate_labeled_dataframe(train_df, "Training split")
    validate_labeled_dataframe(val_df, "Validation split")

    print(f"Number of train images in split: {len(train_df)}")
    print(f"Number of val images in split: {len(val_df)}")

    train_features, train_labels, train_file_names = collect_raw_features(
        df=train_df,
        image_dir=image_dir,
        split_name="train",
    )
    val_features, val_labels, val_file_names = collect_raw_features(
        df=val_df,
        image_dir=image_dir,
        split_name="val",
    )

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TRAIN_FEATURES_PATH, train_features)
    np.save(VAL_FEATURES_PATH, val_features)
    np.save(TRAIN_LABELS_PATH, train_labels)
    np.save(VAL_LABELS_PATH, val_labels)
    np.save(TRAIN_FILE_NAMES_PATH, train_file_names)
    np.save(VAL_FILE_NAMES_PATH, val_file_names)

    print(f"Final train feature matrix shape: {train_features.shape}")
    print(f"Final val feature matrix shape: {val_features.shape}")
    print(f"Saved raw features to: {FEATURES_DIR}")

    return train_features, val_features, train_labels, val_labels


# Stage 4 and 5: classifier training and evaluation

def load_saved_raw_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    train_labels: np.ndarray,
    val_labels: np.ndarray,
) -> None:
    if train_features.ndim != 2:
        raise ValueError(f"Train features must be 2D, got shape {train_features.shape}.")
    if val_features.ndim != 2:
        raise ValueError(f"Validation features must be 2D, got shape {val_features.shape}.")
    if train_features.shape[1] != val_features.shape[1]:
        raise ValueError(
            "Train and validation feature dimensions do not match: "
            f"{train_features.shape[1]} vs {val_features.shape[1]}."
        )
    if train_features.shape[0] != len(train_labels):
        raise ValueError(
            "Number of training samples and training labels do not match: "
            f"{train_features.shape[0]} vs {len(train_labels)}."
        )
    if val_features.shape[0] != len(val_labels):
        raise ValueError(
            "Number of validation samples and validation labels do not match: "
            f"{val_features.shape[0]} vs {len(val_labels)}."
        )
    if train_features.shape[0] == 0:
        raise ValueError("Training feature matrix is empty.")
    if val_features.shape[0] == 0:
        raise ValueError("Validation feature matrix is empty.")


def evaluate_classifier(
    classifier: SVC,
    label_encoder: LabelEncoder,
    train_features_scaled: np.ndarray,
    val_features_scaled: np.ndarray,
    train_labels_encoded: np.ndarray,
    val_labels_encoded: np.ndarray,
) -> dict[str, object]:
    train_predictions = classifier.predict(train_features_scaled)
    val_predictions = classifier.predict(val_features_scaled)

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


def train_raw_classifier(
    train_features: Optional[np.ndarray] = None,
    val_features: Optional[np.ndarray] = None,
    train_labels: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
) -> tuple[SVC, LabelEncoder, StandardScaler, dict[str, object]]:
    if any(item is None for item in [train_features, val_features, train_labels, val_labels]):
        train_features, val_features, train_labels, val_labels = load_saved_raw_features()

    assert train_features is not None
    assert val_features is not None
    assert train_labels is not None
    assert val_labels is not None

    validate_feature_data(train_features, val_features, train_labels, val_labels)

    train_labels = train_labels.astype(str)
    val_labels = val_labels.astype(str)

    train_unique_labels = np.unique(train_labels)
    val_unique_labels = np.unique(val_labels)
    train_label_set = set(train_unique_labels.tolist())
    val_label_set = set(val_unique_labels.tolist())

    print(f"Train feature matrix shape: {train_features.shape}")
    print(f"Validation feature matrix shape: {val_features.shape}")
    print(f"Unique train labels: {len(train_unique_labels)}")
    print(f"Unique validation labels: {len(val_unique_labels)}")

    if train_label_set != val_label_set:
        print("Warning: training and validation class sets do not match exactly.")

    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)

    unseen_val_labels = sorted(val_label_set - set(label_encoder.classes_.tolist()))
    if unseen_val_labels:
        raise ValueError(
            "Validation set contains labels not seen in training: "
            f"{unseen_val_labels}"
        )

    val_labels_encoded = label_encoder.transform(val_labels)

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)

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
    classifier.fit(train_features_scaled, train_labels_encoded)

    metrics = evaluate_classifier(
        classifier=classifier,
        label_encoder=label_encoder,
        train_features_scaled=train_features_scaled,
        val_features_scaled=val_features_scaled,
        train_labels_encoded=train_labels_encoded,
        val_labels_encoded=val_labels_encoded,
    )

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, CLASSIFIER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Training accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
    print("\nValidation classification report:")
    print(metrics["classification_report"])
    print("Validation confusion matrix:")
    print(metrics["confusion_matrix"])
    print(f"Saved classifier to: {CLASSIFIER_PATH}")
    print(f"Saved label encoder to: {LABEL_ENCODER_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")

    return classifier, label_encoder, scaler, metrics


# Stage 6: submission

def make_kaggle_submission(
    sample_submission_path: Optional[Path] = None,
    test_image_dir: Path = TEST_IMAGE_DIR,
) -> Path:
    submission_path = (
        resolve_sample_submission_path()
        if sample_submission_path is None
        else sample_submission_path
    )

    ensure_file_exists(submission_path)
    ensure_file_exists(CLASSIFIER_PATH)
    ensure_file_exists(LABEL_ENCODER_PATH)
    ensure_file_exists(SCALER_PATH)

    submission_df = pd.read_csv(submission_path)
    required_columns = {"ID", "TARGET"}
    if not required_columns.issubset(submission_df.columns):
        raise ValueError(
            f"Sample submission must contain columns: {sorted(required_columns)}"
        )

    classifier: SVC = joblib.load(CLASSIFIER_PATH)
    label_encoder: LabelEncoder = joblib.load(LABEL_ENCODER_PATH)
    scaler: StandardScaler = joblib.load(SCALER_PATH)

    print(f"Number of test rows loaded: {len(submission_df)}")

    features: list[np.ndarray] = []
    error_count = 0

    for row in tqdm(submission_df.itertuples(index=False), total=len(submission_df), desc="test"):
        file_name = f"{row.ID}.jpg"
        try:
            image = load_image(file_name, test_image_dir)
            feature_vector = extract_raw_feature_vector(image)
        except FileNotFoundError as error:
            print(f"Missing test image: {error}")
            feature_vector = np.zeros(RAW_IMAGE_SIZE[0] * RAW_IMAGE_SIZE[1] * 3, dtype=np.float32)
            error_count += 1

        features.append(feature_vector)

    test_features = np.vstack(features).astype(np.float32)
    test_features_scaled = scaler.transform(test_features)

    predicted_numeric_labels = classifier.predict(test_features_scaled)
    predicted_string_labels = label_encoder.inverse_transform(predicted_numeric_labels)

    submission_df["TARGET"] = predicted_string_labels
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(OUTPUT_SUBMISSION_PATH, index=False)

    print(f"Number of missing test images skipped or errors encountered: {error_count}")
    print(f"Final submission shape: {submission_df.shape}")
    print(f"Output file path: {OUTPUT_SUBMISSION_PATH}")

    return OUTPUT_SUBMISSION_PATH


# Main

def main() -> None:
    train_features: Optional[np.ndarray] = None
    val_features: Optional[np.ndarray] = None
    train_labels: Optional[np.ndarray] = None
    val_labels: Optional[np.ndarray] = None

    if RUN_SPLIT:
        split_train_val()

    if RUN_SANITY_CHECK:
        run_raw_sanity_check()

    if RUN_BUILD_FEATURES:
        train_features, val_features, train_labels, val_labels = build_raw_features()

    if RUN_TRAIN:
        train_raw_classifier(train_features, val_features, train_labels, val_labels)

    if RUN_SUBMISSION:
        make_kaggle_submission()


if __name__ == "__main__":
    main()