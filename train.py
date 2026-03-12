import os
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

from configs.config import CFG, BASE_PATH, MODEL_SAVE_PATH
from src.dataset import MultiViewDataset
from src.model import MultiViewEfficientNet
from src.utils import seed_everything


def get_train_transform():
    return A.Compose([
        A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=10,
            p=0.5
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_valid_transform():
    return A.Compose([
        A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for views, labels in tqdm(loader, desc="Train", leave=False):
        views = [v.to(device) for v in views]
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(views)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    probs_list = []
    labels_list = []

    for views, labels in tqdm(loader, desc="Valid", leave=False):
        views = [v.to(device) for v in views]
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(views)
        loss = criterion(outputs, labels)

        probs = torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1)
        labels_np = labels.detach().cpu().numpy().reshape(-1)

        running_loss += loss.item()
        probs_list.extend(probs.tolist())
        labels_list.extend(labels_np.tolist())

    avg_loss = running_loss / len(loader)

    probs_arr = np.array(probs_list)
    labels_arr = np.array(labels_list)

    eps = 1e-7
    probs_arr = np.clip(probs_arr, eps, 1 - eps)

    val_logloss = log_loss(labels_arr, probs_arr)
    val_pred = (probs_arr >= 0.5).astype(int)
    val_acc = accuracy_score(labels_arr, val_pred)

    return avg_loss, val_logloss, val_acc


def main():
    seed_everything(CFG["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv_path = os.path.join(BASE_PATH, "train.csv")
    val_csv_path = os.path.join(BASE_PATH, "dev.csv")

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    train_dataset = MultiViewDataset(
        train_df,
        os.path.join(BASE_PATH, "train"),
        transform=get_train_transform(),
        is_test=False
    )

    val_dataset = MultiViewDataset(
        val_df,
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

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True
    )

    model = MultiViewEfficientNet(
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

    best_logloss = float("inf")

    for epoch in range(1, CFG["EPOCHS"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_logloss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"[Epoch {epoch}/{CFG['EPOCHS']}]")
        print(f"Train Loss   : {train_loss:.4f}")
        print(f"Valid Loss   : {val_loss:.4f}")
        print(f"Valid LogLoss: {val_logloss:.6f}")
        print(f"Valid Acc    : {val_acc:.4f}")

        if val_logloss < best_logloss:
            best_logloss = val_logloss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Best model saved -> {MODEL_SAVE_PATH}")

    print(f"Training finished. Best LogLoss: {best_logloss:.6f}")


if __name__ == "__main__":
    main()