import os
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# =========================
# 1. PATH SETUP
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CHANGE THESE IF YOUR FILE NAMES / FOLDERS ARE DIFFERENT
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
CSV_PATH = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(BASE_DIR, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
FILENAME_COL = "file_name"
LABEL_COL = "TARGET"
RANDOM_STATE = 42
IMG_SIZE = 16   # start small for speed


# =========================
# 2. LOAD METADATA
# =========================

def load_metadata():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows")
    print("Columns are:", df.columns.tolist())
    print(f"Number of classes: {df[LABEL_COL].nunique()}")
    print(df.head())
    return df


# =========================
# 3. LOAD ONE IMAGE
# =========================

def load_image(filename, img_dir, size=32):
    path = os.path.join(img_dir, filename)
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    return img


# =========================
# 4. LOAD MANY IMAGES
# =========================

def load_images_from_filenames(filenames, img_dir, size=32):
    X = []
    for i, fname in enumerate(filenames):
        if i % 500 == 0:
            print(f"Loading image {i}/{len(filenames)}")
        img = load_image(fname, img_dir=img_dir, size=size)
        X.append(img)
    return np.array(X, dtype=np.float32)


# =========================
# 5. MAIN PIPELINE
# =========================

def main():
    print("Loading metadata...")
    df = load_metadata()

    # split filenames and labels
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
    print(f"Train samples: {len(X_train_files)}")
    print(f"Validation samples: {len(X_val_files)}")

    print("Loading training images...")
    X_train = load_images_from_filenames(X_train_files, size=IMG_SIZE, img_dir=TRAIN_IMG_DIR)

    print("Loading validation images...")
    X_val = load_images_from_filenames(X_val_files, size=IMG_SIZE, img_dir=TRAIN_IMG_DIR)

    print("Original train shape:", X_train.shape)
    print("Original val shape:", X_val.shape)

    # flatten images: (N, H, W, C) -> (N, H*W*C)
    X_train = X_train.reshape(len(X_train), -1)
    X_val = X_val.reshape(len(X_val), -1)

    print("Flattened train shape:", X_train.shape)
    print("Flattened val shape:", X_val.shape)

    # build SVM model
    model = LinearSVC(
        random_state=RANDOM_STATE,
        max_iter=5000
    )

    print("Training SVM...")
    model.fit(X_train, y_train_encoded)

    print("Predicting on validation set...")
    y_val_pred = model.predict(X_val)

    acc = accuracy_score(y_val_encoded, y_val_pred)
    print(f"Validation accuracy: {acc:.4f}")
    
    print("Loading sample submission file...")
    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    print("Submission columns:", submission.columns.tolist())
    print(submission.head())

    # Assume the test image file names are based on the IDs
    test_filenames = submission["ID"].apply(lambda x: f"{x}.jpg").values

    print("Loading test images...")
    X_test = load_images_from_filenames(test_filenames, img_dir=TEST_IMG_DIR, size=IMG_SIZE)

    print("Original test shape:", X_test.shape)

    X_test = X_test.reshape(len(X_test), -1)
    print("Flattened test shape:", X_test.shape)

    print("Predicting test set...")
    y_test_pred_encoded = model.predict(X_test)

    # convert numeric labels back to species names
    y_test_pred_labels = le.inverse_transform(y_test_pred_encoded)

    submission["TARGET"] = y_test_pred_labels

    output_path = os.path.join(BASE_DIR, "submission_svm.csv")
    submission.to_csv(output_path, index=False)

    print(f"Saved submission file to: {output_path}")


if __name__ == "__main__":
    main()