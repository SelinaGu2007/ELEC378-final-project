import os
import time
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from skimage.feature import canny
from tqdm import tqdm


# =========================
# 1. PATH SETUP
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
CSV_PATH = os.path.join(BASE_DIR, "train.csv")

FILENAME_COL = "file_name"
LABEL_COL = "TARGET"

RANDOM_STATE = 42
IMG_SIZE = 64 


# =========================
# 2. LOAD METADATA
# =========================

def load_metadata():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows")
    print("Columns are:", df.columns.tolist())
    print(f"Number of classes: {df[LABEL_COL].nunique()}")
    return df


# =========================
# 3. LOAD IMAGE AS GRAYSCALE
# =========================

def load_image_gray(filename, img_dir, size=64):
    path = os.path.join(img_dir, filename)
    img = Image.open(path).convert("L")
    img = img.resize((size, size))
    img = np.array(img, dtype=np.float32) / 255.0
    return img


# =========================
# 4. EDGE FEATURE EXTRACTION
# =========================

def extract_edge_features(img):
    edges = canny(
        img,
        sigma=1.2,
        low_threshold=0.1,
        high_threshold=0.25
    ).astype(np.float32)

    # edge map flatten
    feat = edges.flatten()

    # 加一点简单统计量
    edge_density = np.mean(edges)
    row_mean = np.mean(edges, axis=1)
    col_mean = np.mean(edges, axis=0)

    feat = np.concatenate([
        feat,
        np.array([edge_density], dtype=np.float32),
        row_mean.astype(np.float32),
        col_mean.astype(np.float32)
    ])

    return feat


def extract_features_from_filenames(filenames, img_dir, size=64, desc="Extracting edges"):
    X = []
    for fname in tqdm(filenames, desc=desc):
        img = load_image_gray(fname, img_dir=img_dir, size=size)
        feat = extract_edge_features(img)
        X.append(feat)
    return np.array(X, dtype=np.float32)


# =========================
# 5. MAIN PIPELINE
# =========================

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

    # ========= 小规模开发版 =========
    X_train_files = X_train_files[:5000]
    y_train = y_train[:5000]

    X_val_files = X_val_files[:200]
    y_val = y_val[:200]
    # ===============================

    print(f"Train samples: {len(X_train_files)}")
    print(f"Validation samples: {len(X_val_files)}")

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)

    print("Extracting edge features from training images...")
    X_train = extract_features_from_filenames(
        X_train_files,
        img_dir=TRAIN_IMG_DIR,
        size=IMG_SIZE,
        desc="Edge train"
    )

    print("Extracting edge features from validation images...")
    X_val = extract_features_from_filenames(
        X_val_files,
        img_dir=TRAIN_IMG_DIR,
        size=IMG_SIZE,
        desc="Edge val"
    )

    print("Train feature shape:", X_train.shape)
    print("Validation feature shape:", X_val.shape)

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # ========= kernel SVM =========
    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        decision_function_shape="ovr"
    )
    # ==============================

    print("Training Edge + RBF Kernel SVM...")
    start_time = time.time()

    model.fit(X_train, y_train_encoded)

    end_time = time.time()
    print(f"Kernel SVM training finished in {end_time - start_time:.2f} seconds")

    print("Predicting on validation set...")
    y_val_pred = model.predict(X_val)

    acc = accuracy_score(y_val_encoded, y_val_pred)
    print(f"Validation accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()