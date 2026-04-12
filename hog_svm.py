import os
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
CSV_PATH = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

FILENAME_COL = "file_name"
LABEL_COL = "TARGET"

RANDOM_STATE = 42
IMG_SIZE = 64


def load_metadata():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows")
    print("Columns are:", df.columns.tolist())
    print(f"Number of classes: {df[LABEL_COL].nunique()}")
    print(df.head())
    return df

def load_image_gray(filename, img_dir, size=128):
    path = os.path.join(img_dir, filename)
    img = Image.open(path).convert("L")   # grayscale
    img = img.resize((size, size))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    return img


# Extract Hog features

def extract_hog_features(img):
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )
    return features


def extract_features_from_filenames(filenames, img_dir, size=128):
    X = []
    for i, fname in enumerate(filenames):
        if i % 1000 == 0:
            print(f"Processing image {i}/{len(filenames)}")
        img = load_image_gray(fname, img_dir=img_dir, size=size)
        feats = extract_hog_features(img)
        X.append(feats)
    return np.array(X, dtype=np.float32)

#main pipeline

def main():
    print("Loading metadata...")
    df = load_metadata()

    X_train_files, X_val_files, y_train, y_val = train_test_split(
        df[FILENAME_COL].values,
        df[LABEL_COL].values,
        test_size=0.2,
        stratify=df[LABEL_COL].values,
        random_state=RANDOM_STATE,
    )

    print(f"Train samples: {len(X_train_files)}")
    print(f"Validation samples: {len(X_val_files)}")

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)

    print("Extracting HOG features from training images...")
    X_train = extract_features_from_filenames(
        X_train_files,
        img_dir=TRAIN_IMG_DIR,
        size=IMG_SIZE
    )

    print("Extracting HOG features from validation images...")
    X_val = extract_features_from_filenames(
        X_val_files,
        img_dir=TRAIN_IMG_DIR,
        size=IMG_SIZE
    )

    print("Train feature shape:", X_train.shape)
    print("Validation feature shape:", X_val.shape)

    # standardize feature columns
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = LinearSVC(
    random_state=RANDOM_STATE,
    max_iter=3000,
    dual=False)

    print("Training HOG + SVM...")
    model.fit(X_train, y_train_encoded)

    print("Predicting on validation set...")
    y_val_pred = model.predict(X_val)

    acc = accuracy_score(y_val_encoded, y_val_pred)
    print(f"Validation accuracy: {acc:.4f}")

    # ===== Test prediction =====
    print("Loading sample submission file...")
    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    print("Submission columns:", submission.columns.tolist())
    print(submission.head())

    test_filenames = submission["ID"].apply(lambda x: f"{x}.jpg").values

    print("Extracting HOG features from test images...")
    X_test = extract_features_from_filenames(
        test_filenames,
        img_dir=TEST_IMG_DIR,
        size=IMG_SIZE
    )

    print("Test feature shape:", X_test.shape)

    X_test = scaler.transform(X_test)

    print("Predicting test set...")
    y_test_pred_encoded = model.predict(X_test)
    y_test_pred_labels = le.inverse_transform(y_test_pred_encoded)

    submission["TARGET"] = y_test_pred_labels

    output_path = os.path.join(BASE_DIR, "hog_svm.csv")
    submission.to_csv(output_path, index=False)

    print(f"Saved submission file to: {output_path}")


if __name__ == "__main__":
    main()