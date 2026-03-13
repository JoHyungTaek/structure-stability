import os
import gc
import random
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import log_loss, accuracy_score

from configs.config import CFG, BASE_PATH, MODEL_PATH
from src.dataset import MultiViewDataset
from src.model import MultiViewClassifier

warnings.filterwarnings("ignore")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def label_to_int(x):
    if isinstance(x, str):
        x = x.strip().lower()
        if x == "stable":
            return 0
        if x == "unstable":
            return 1
    return int(float(x))


def smooth_labels(labels, smoothing=0.02):
    return labels * (1.0 - smoothing) + 0.5 * smoothing


def get_train_transform():
    size = CFG["IMG_SIZE"]
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.06,
            scale_limit=0.10,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.20,
            contrast_limit=0.20,
            p=0.6
        ),
        A.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.08,
            p=0.35
        ),
        A.GaussNoise(p=0.2),
        A.CoarseDropout(
            max_holes=6,
            max_height=size // 12,
            max_width=size // 12,
            min_holes=1,
            min_height=size // 32,
            min_width=size // 32,
            p=0.25
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_valid_transform():
    size = CFG["IMG_SIZE"]
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def build_dataloaders():
    train_df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
    dev_df = pd.read_csv(os.path.join(BASE_PATH, "dev.csv"))

    train_df["label"] = train_df["label"].apply(label_to_int)
    dev_df["label"] = dev_df["label"].apply(label_to_int)

    train_dataset = MultiViewDataset(
        df=train_df,
        image_root=os.path.join(BASE_PATH, "train"),
        transform=get_train_transform(),
        is_test=False,
    )

    valid_dataset = MultiViewDataset(
        df=dev_df,
        image_root=os.path.join(BASE_PATH, "dev"),
        transform=get_valid_transform(),
        is_test=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG["BATCH_SIZE"],
        shuffle=True,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
        drop_last=False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    losses = []
    optimizer.zero_grad()

    for step, (views, labels) in enumerate(tqdm(loader, desc="Train", leave=False), start=1):
        views = [v.to(device) for v in views]
        labels = labels.unsqueeze(1).to(device)
        labels = smooth_labels(labels, CFG["LABEL_SMOOTHING"])

        if CFG["AMP"]:
            with torch.cuda.amp.autocast():
                logits = model(views)
                loss = criterion(logits, labels)
                loss = loss / CFG["ACCUM_STEPS"]

            scaler.scale(loss).backward()

            if step % CFG["ACCUM_STEPS"] == 0 or step == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(views)
            loss = criterion(logits, labels)
            loss = loss / CFG["ACCUM_STEPS"]
            loss.backward()

            if step % CFG["ACCUM_STEPS"] == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad()

        losses.append(loss.item() * CFG["ACCUM_STEPS"])

    return float(np.mean(losses))


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    probs_all = []
    labels_all = []

    for views, labels in tqdm(loader, desc="Valid", leave=False):
        views = [v.to(device) for v in views]
        labels = labels.unsqueeze(1).to(device)

        logits = model(views)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        labels_np = labels.detach().cpu().numpy().reshape(-1)

        losses.append(loss.item())
        probs_all.extend(probs.tolist())
        labels_all.extend(labels_np.tolist())

    probs_all = np.clip(np.array(probs_all), 1e-7, 1 - 1e-7)
    labels_all = np.array(labels_all)

    valid_loss = float(np.mean(losses))
    valid_logloss = log_loss(labels_all, probs_all)
    valid_acc = accuracy_score(labels_all, (probs_all >= 0.5).astype(int))

    return valid_loss, valid_logloss, valid_acc


def main():
    seed_everything(CFG["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader = build_dataloaders()

    model = MultiViewClassifier(
        model_name=CFG["MODEL_NAME"],
        dropout=CFG["DROPOUT"]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["LEARNING_RATE"],
        weight_decay=CFG["WEIGHT_DECAY"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG["EPOCHS"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=CFG["AMP"])

    best_logloss = float("inf")
    patience_count = 0

    for epoch in range(1, CFG["EPOCHS"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        valid_loss, valid_logloss, valid_acc = validate(
            model, valid_loader, criterion, device
        )

        scheduler.step()

        print(f"[Epoch {epoch}/{CFG['EPOCHS']}]")
        print(f"Train Loss   : {train_loss:.4f}")
        print(f"Valid Loss   : {valid_loss:.4f}")
        print(f"Valid LogLoss: {valid_logloss:.6f}")
        print(f"Valid Acc    : {valid_acc:.4f}")

        if valid_logloss < best_logloss:
            best_logloss = valid_logloss
            patience_count = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Best model saved -> {MODEL_PATH}")
        else:
            patience_count += 1
            if patience_count >= CFG["PATIENCE"]:
                print("Early stopping triggered.")
                break

    print("=" * 50)
    print(f"Best Dev LogLoss: {best_logloss:.6f}")

    del model, train_loader, valid_loader
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()