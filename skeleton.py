# The skeleton codes for the ELEC378 Final Project to load the dataset and help you get started
# Author: Rocky Ren, Harvey Chen, Matthew Karazincir
# Gooood luck!


# Before you start, you should do an Ananconda environment setup. Search it up online, it will make collaborations 
# so much easier. 
# The libraries in your conda environment can be transferred as a .yml file, which is great.
 
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

# Modify these paths to match your dataset. 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "butterfly_data")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
CSV_PATH = os.path.join(DATA_DIR, "Training_set.csv")

RANDOM_STATE = 42 # This is for the reproducibility of the train test split. 

# Loads the metadata from the csv file
def load_metadata():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} entries, {df['label'].nunique()} classes")
    return df

# Loads the images 
def load_image(filename, size):
    path = os.path.join(TRAIN_IMG_DIR, filename)
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.array(img)

# This loads one image and its label of your choice. 
# Input: the dataframe, the index of the imgae, and the size of the image. 
def load_image_label_pair(df, index, size=224):
    row = df.iloc[index]
    img = load_image(row["filename"], size)
    label = row["label"]
    return img, label


def main():
    print("Loading the ELEC378 Final Project Dataset")

    df = load_metadata()
    print(df.head())
    print("Loading csv done")

    img, label = load_image_label_pair(df, 0)
    print(f"Sample pair — label: {label}, image shape: {img.shape}")

    X_train_files, X_val_files, y_train, y_val = train_test_split(
        df["filename"].values,
        df["label"].values,
        test_size=0.2,
        stratify=df["label"].values,
        random_state=RANDOM_STATE,
    )
    print(f"Train: {len(X_train_files)}, Validation: {len(X_val_files)}")

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    img, label = load_image_label_pair(df, idx)
    print(f"Showing the {idx} th image: {label}")
    plt.imshow(img)
    plt.title(f"[{idx}] {label}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
