import os
import sys
import time
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.feature import hog
from skimage import color
from tqdm import tqdm
# directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH         = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_PATH  = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR    = os.path.join(BASE_DIR, "train")
TEST_IMG_DIR     = os.path.join(BASE_DIR, "test_images")
SUBMISSION_PATH  = os.path.join(BASE_DIR, "submission.csv")

# Cache paths
HOG_CACHE_PATH   = os.path.join(BASE_DIR, "hog_features.npz") # deprecated
MODEL_CACHE_PATH = os.path.join(BASE_DIR, "efficientnet_model_big.pth")

RANDOM_STATE = 42


# copied from skeleton.py
def load_metadata():
    df = pd.read_csv(CSV_PATH)
    print(df.columns)
    print(f"Loaded {len(df)} entries, {df['TARGET'].nunique()} classes")
    return df

def load_image(filename, size, img_dir=None):
    if img_dir is None:
        img_dir = TRAIN_IMG_DIR
    path = os.path.join(img_dir, filename)
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.array(img)

def load_image_label_pair(df, index, size=128):
    row = df.iloc[index]
    img = load_image(row["file_name"], size)
    label = row["TARGET"]
    return img, label


# ─deprecated and can be removed, this was copied over from earlier SVM models
from skimage.transform import rotate

def extract_hog_features(file_list, size=128,
                         orientations=9,
                         pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2),
                         img_dir=None,
                         augment=False):

    features = []

    for fname in tqdm(file_list, desc="HOG", unit="img"):
        img = load_image(fname, size, img_dir=img_dir)
        img_gray = color.rgb2gray(img)

        feat = hog(
            img_gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=True
        )
        features.append(feat)

        if augment:
            # horizontal flip
            img_flip = np.fliplr(img_gray)
            feat_flip = hog(
                img_flip,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm='L2-Hys',
                transform_sqrt=True,
                feature_vector=True
            )
            features.append(feat_flip)

            # rotation
            img_rot = rotate(img_gray, angle=10, mode='edge')
            feat_rot = hog(
                img_rot,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm='L2-Hys',
                transform_sqrt=True,
                feature_vector=True
            )
            features.append(feat_rot)

    features = np.array(features, dtype=np.float32)

    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features = features / norms

    return features


# ── HOG cache ─────────────────────────────────────────────────────
def get_hog_features(train_files, val_files, y_train, force_recompute=False):
    if not force_recompute and os.path.exists(HOG_CACHE_PATH):
        print(f"[Cache] Loading HOG features from {HOG_CACHE_PATH}")
        data = np.load(HOG_CACHE_PATH, allow_pickle=True)
        return data['X_train'], data['X_val'], data['y_train']

    print(f"\nExtracting HOG for {len(train_files)} train images (with augmentation)...")
    

    # TO:
    X_train_hog = extract_hog_features(train_files, augment=True)

    # AND REMOVE:
    y_train_aug = np.repeat(y_train, 3)

    print(f"Extracting HOG for {len(val_files)} val images...")
    X_val_hog = extract_hog_features(val_files, augment=False)

    np.savez_compressed(
        HOG_CACHE_PATH,
        X_train=X_train_hog,
        X_val=X_val_hog,
        y_train=y_train_aug
    )

    return X_train_hog, X_val_hog, y_train_aug

#handles data loading
class ButterflyDataset(Dataset):
    def __init__(self, file_names, labels, class_to_idx=None, transform=None):
        self.file_names = file_names
        self.transform = transform

        if class_to_idx is None:
            self.classes = sorted(list(set(labels)))
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = list(class_to_idx.keys())

        self.labels_idx = np.array([self.class_to_idx[l] for l in labels])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img = load_image(self.file_names[idx], size=224)
        label = self.labels_idx[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
    
class MBConvBlock(nn.Module):
    #implementation of MBconv used in CNN
    def __init__(self, in_channels, out_channels, expand_ratio, stride, se_ratio=0.25):
        super().__init__()
        self.stride = stride
        #skip connections if channel width between layers are the same
        self.use_residual = (stride == 1 and in_channels == out_channels)
        mid = in_channels * expand_ratio    
        #expands by using 1x1 convolution
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU()
        ) if expand_ratio != 1 else nn.Identity()
        #3x3 convolution to each channel separately.
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU()
        )

        # Squeeze-and-Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(mid, se_channels),    
            nn.SiLU(),
            nn.Linear(se_channels, mid),
            nn.Sigmoid()
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(mid, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        se_w = self.se(out).unsqueeze(-1).unsqueeze(-1)
        out = out * se_w
        out = self.pointwise(out)
        if self.use_residual:
            out = out + x
        return out


class ButterflyEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # wider stem
        # takes in 3 dimensional (RGB) information and quickly expands it into 40 features
        # use stride = 2 to cut image size in half as well. 
        self.stem = nn.Sequential(
            nn.Conv2d(3, 40, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.SiLU()
        )
        # Format: (in_channels, out_channels, expand_ratio, stride, num_repeats)
        # depth and width adjusted based on initial training results over a small amount of epochs.
        # cfg — starts at 40, ends at 384
        cfg = [
            (40,  20,  1, 1, 1),
            (20,  32,  6, 2, 2),
            (32,  48,  6, 2, 3),
            (48,  96,  6, 2, 4),
            (96,  136, 6, 1, 4),
            (136, 232, 6, 2, 4),
            (232, 384, 6, 1, 2),
        ]

        

        layers = []
        for in_c, out_c, exp, stride, n in cfg:
            for i in range(n):
                layers.append(MBConvBlock(
                    #only change the channel count on its first pass
                    in_c if i == 0 else out_c,
                    out_c, exp,
                    stride if i == 0 else 1
                ))
        self.blocks = nn.Sequential(*layers)

        # wider head with intermediate layer
        # head — expects 384 in
        #very deep features (384) and expand that into a huge space for model to choose features to evaluate
        self.head = nn.Sequential(
            nn.Conv2d(384, 1536, 1, bias=False),
            nn.BatchNorm2d(1536),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            #harsh dropout at first to be robust
            nn.Dropout(0.5),
            nn.Linear(1536, 512),
            nn.SiLU(),
            #lighter dropout later to not miss classes
            nn.Dropout(0.3),
            #finally steps down to number of classes
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        #goes from RGB through body(cfg) then out the head
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
    
#mixup and cutmix augmentation to increase robustness of features
def mixup_batch(imgs, labels, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0))
    mixed = lam * imgs + (1 - lam) * imgs[idx]
    return mixed, labels, labels[idx], lam

def cutmix_batch(imgs, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0))
    W, H = imgs.size(3), imgs.size(2)
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = np.clip(cx - cut_w//2, 0, W), np.clip(cx + cut_w//2, 0, W)
    y1, y2 = np.clip(cy - cut_h//2, 0, H), np.clip(cy + cut_h//2, 0, H)
    mixed = imgs.clone()
    mixed[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2-x1)*(y2-y1)/(W*H)
    return mixed, labels, labels[idx], lam
    
import copy
from torch.cuda.amp import autocast, GradScaler
#epochs to checkpoint model
SAVE_EPOCHS = {100, 150, 200, 250, 300}
def train_model(model, train_loader, val_loader, epochs=50):
    #need to use cuda or else it is too slow
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    #add label smoothing so the model can generalize and makes it not overconfident
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    #AdamW seems to be standard optimizer, add weight decay to keep weights small
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-5)
    # use warm restarts at certain epochs so that we can jump out of local minimas
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
# restarts every 20 epochs, then 40, then 80...
    #Added to hepl with compute time
    scaler = GradScaler()  

    train_losses = []
    val_losses = [] 
    best_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            #choose batch augmentation with equal probablility: mixup, cutmix, or none
            r = np.random.rand()
            if r < 0.33:
                imgs, la, lb, lam = mixup_batch(imgs, labels)
            elif r < 0.66:
                imgs, la, lb, lam = cutmix_batch(imgs, labels)
            else:
                la, lb, lam = labels, labels, 1.0
            #do things in 16 bit someitmes to save compute
            with autocast():
                outputs = model(imgs)
                #loss has to reflect cutmix and mixup augmentation
                loss = lam * criterion(outputs, la) + (1 - lam) * criterion(outputs, lb)
            #backpropogation using scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()


        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # validation so we turn off a lot the intensive compute and just tell it to predict things 
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        acc = correct / total
        

        
        

        # at the end of each epoch check if it is the best model:
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())  # store best weights
            print(f"  ↑ New best: {best_acc:.4f}")

        if (epoch + 1) in SAVE_EPOCHS and best_state is not None:
            ckpt_path = os.path.join(
                BASE_DIR,
                f"ckpt_epoch{epoch+1}_BEST_acc{best_acc:.4f}.pth"
            )
            torch.save(best_state, ckpt_path)
            print(f"  Saved BEST-so-far model at epoch {epoch+1} (acc={best_acc:.4f})")
        
    # step the scheduler forward at the end of each epoch
        scheduler.step()
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f}")

    # plot training and validation loss across epochs
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(os.path.join(BASE_DIR, "loss_curve.png"))
    plt.close()
    if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(best_state, MODEL_CACHE_PATH)
            print(f"[Final] Best model saved with acc={best_acc:.4f}")
    return model

# Basic submission just loads file and predicts it. 
def generate_submission(model, test_files, class_to_idx, submission_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    model.eval()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    preds = []

    transform = val_transform

    if submission_path is None:
        submission_path = SUBMISSION_PATH  # fallback

    with torch.no_grad():
        for fname in test_files:
            img = load_image(fname, size=224, img_dir=TEST_IMG_DIR)
            img = transform(img).unsqueeze(0).to(device)

            outputs = model(img)
            pred = torch.argmax(outputs, dim=1).item()
            preds.append(idx_to_class[pred])

    sub_df = pd.read_csv(SAMPLE_SUB_PATH)
    sub_df["TARGET"] = preds
    sub_df.to_csv(submission_path, index=False)

    print(f"Submission saved → {submission_path}")
TTA_SUBMISSION_PATH = os.path.join(BASE_DIR, "tta_submission.csv")

#Use TTA submission with augmented images to improve inference accuracy, ultimate submission was generated with this
def generate_submission_tta(model, test_files, class_to_idx, n_augments=10, submission_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if submission_path is None:
        submission_path = TTA_SUBMISSION_PATH
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    preds = []

    tta_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        for fname in tqdm(test_files, desc="TTA inference"):
            img = load_image(fname, size=224, img_dir=TEST_IMG_DIR)

            all_outputs = []

            # clean pass
            clean = val_transform(img).unsqueeze(0).to(device)
            with autocast():
                all_outputs.append(model(clean))

            # augmented passes
            for _ in range(n_augments):
                aug = tta_transform(img).unsqueeze(0).to(device)
                with autocast():
                    all_outputs.append(model(aug))
            #average of the predicitons
            avg_output = torch.stack(all_outputs).mean(0)
            pred = torch.argmax(avg_output, dim=1).item()
            preds.append(idx_to_class[pred])

    sub_df = pd.read_csv(SAMPLE_SUB_PATH)
    sub_df["TARGET"] = preds
    sub_df.to_csv(submission_path, index=False)
    print(f"TTA submission saved → {submission_path}")

# transformations to heavily augment data so the model can generalize
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),          
    transforms.RandomRotation(30),            
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  
    transforms.RandomGrayscale(p=0.1),       
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



FORCE_RETRAIN = True   # this can be set to just run inference after we alreadu trained out model
def main():
    df = load_metadata()
    #80-20 stratification split
    X_train_files, X_val_files, y_train, y_val = train_test_split(
        df["file_name"].values,
        df["TARGET"].values,
        test_size=0.2,
        stratify=df["TARGET"].values,
        random_state=RANDOM_STATE
    )

    train_dataset = ButterflyDataset(X_train_files, y_train, transform=train_transform)
    val_dataset = ButterflyDataset(
        X_val_files,
        y_val,
        class_to_idx=train_dataset.class_to_idx,
        transform=val_transform
    )
    # adjusted data loaders so persistent_workers=True to give speedup so they are alive between epochs.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                          num_workers=4, pin_memory=True, 
                          persistent_workers=True)  
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)
    model = ButterflyEfficientNet(len(train_dataset.classes))
    
    if os.path.exists(MODEL_CACHE_PATH) and not FORCE_RETRAIN:
        print("[Cache] Loading model...")
        model.load_state_dict(torch.load(MODEL_CACHE_PATH, weights_only=True))
    else:
        print("[Cache] Training new model...")
        model = train_model(model, train_loader, val_loader, epochs=300)
        
        print("[Cache] Model saved.")# in main()
    test_df = pd.read_csv(SAMPLE_SUB_PATH)
    test_files = test_df["ID"].values + ".jpg"  # → "test_000001.jpg"
    #generate submissions for all model in checkpointed epochs with tta and non tta submissions.
    for epoch_num in SAVE_EPOCHS:
        ckpt_files = [
            f for f in os.listdir(BASE_DIR)
            if f.startswith(f"ckpt_epoch{epoch_num}_BEST")
        ]

        if not ckpt_files:
            continue

        # if multiple exist, pick the one with highest acc
        ckpt_files.sort(reverse=True)  
        ckpt_path = os.path.join(BASE_DIR, ckpt_files[0])

        print(f"\nGenerating submission for {ckpt_files[0]}")

        model.load_state_dict(torch.load(ckpt_path, weights_only=True))



        test_df = pd.read_csv(SAMPLE_SUB_PATH)
        test_files = test_df["ID"].values + ".jpg"

        generate_submission(
            model,
            test_files,
            train_dataset.class_to_idx,
            submission_path=os.path.join(BASE_DIR, f"submission_epoch{epoch_num}.csv")
        )
        generate_submission_tta(
            model, test_files, train_dataset.class_to_idx,
            n_augments=5,
            submission_path=os.path.join(BASE_DIR, f"tta_submission_epoch{epoch_num}.csv")
        )
# but generate_submission_tta hardcodes TTA_SUBMISSION_PATH internally
# so add submission_path parameter same way you did for generate_submission
    
if __name__ == "__main__":
    main()
