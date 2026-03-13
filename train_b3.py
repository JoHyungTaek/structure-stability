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

from configs.config_b3 import CFG, BASE_PATH, MODEL_PATH
from src.dataset import MultiViewDataset
from src.model_b3 import MultiViewB3

warnings.filterwarnings("ignore")


def seed_everything(seed):
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


def get_train_transform():
    size = CFG["IMG_SIZE"]
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.08,
            rotate_limit=10,
            border_mode=0,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5
        ),
        A.GaussNoise(p=0.2),
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
        train_df,
        os.path.join(BASE_PATH, "train"),
        transform=get_train_transform(),
        is_test=False
    )

    valid_dataset = MultiViewDataset(
        dev_df,
        os.path.join(BASE_PATH, "dev"),
        transform=get_valid_transform(),
        is_test=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG["BATCH_SIZE"],
        shuffle=True,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True
    )

    return train_loader, valid_loader


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    losses = []

    for views, labels in tqdm(loader, desc="Train B3", leave=False):
        views = [v.to(device) for v in views]
        labels = labels.unsqueeze(1).to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=CFG["AMP"]):
            logits = model(views)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    probs_all = []
    labels_all = []

    for views, labels in tqdm(loader, desc="Valid B3", leave=False):
        views = [v.to(device) for v in views]
        labels = labels.unsqueeze(1).to(device)

        logits = model(views)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        labs = labels.cpu().numpy().reshape(-1)

        losses.append(loss.item())
        probs_all.extend(probs.tolist())
        labels_all.extend(labs.tolist())

    probs_all = np.clip(np.array(probs_all), 1e-7, 1 - 1e-7)
    labels_all = np.array(labels_all)

    return (
        float(np.mean(losses)),
        log_loss(labels_all, probs_all),
        accuracy_score(labels_all, (probs_all >= 0.5).astype(int))
    )


def main():
    seed_everything(CFG["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader = build_dataloaders()

    model = MultiViewB3(dropout=CFG["DROPOUT"]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["LEARNING_RATE"],
        weight_decay=CFG["WEIGHT_DECAY"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["EPOCHS"])
    scaler = torch.cuda.amp.GradScaler(enabled=CFG["AMP"])

    best_logloss = float("inf")
    patience_count = 0

    for epoch in range(1, CFG["EPOCHS"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        valid_loss, valid_logloss, valid_acc = validate(model, valid_loader, criterion, device)
        scheduler.step()

        print(f"[B3 Epoch {epoch}/{CFG['EPOCHS']}]")
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
    print(f"B3 Best Dev LogLoss: {best_logloss:.6f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()