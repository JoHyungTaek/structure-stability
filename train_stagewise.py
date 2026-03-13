import os
import gc
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
from sklearn.model_selection import StratifiedShuffleSplit

from configs.config import (
    CFG,
    PRETRAIN_MODEL_PATH,
    STAGE2_HEAD_MODEL_PATH,
    FINETUNE_MODEL_PATH,
)
from src.dataset import MultiViewDataset
from src.model import MultiViewClassifier
from src.utils import seed_everything, label_to_int

warnings.filterwarnings("ignore")


def get_stage1_train_transform():
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
            brightness_limit=0.18,
            contrast_limit=0.18,
            p=0.5
        ),
        A.GaussNoise(p=0.2),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


def get_stage2_train_transform():
    size = CFG["IMG_SIZE"]
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.10,
            contrast_limit=0.10,
            p=0.3
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


def get_valid_transform():
    size = CFG["IMG_SIZE"]
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


def build_stage1_loaders():
    base_path = CFG["BASE_PATH"]

    train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
    dev_df = pd.read_csv(os.path.join(base_path, "dev.csv"))

    train_df["label"] = train_df["label"].apply(label_to_int)
    dev_df["label"] = dev_df["label"].apply(label_to_int)

    train_dataset = MultiViewDataset(
        df=train_df,
        image_root=os.path.join(base_path, "train"),
        transform=get_stage1_train_transform(),
        is_test=False,
    )

    valid_dataset = MultiViewDataset(
        df=dev_df,
        image_root=os.path.join(base_path, "dev"),
        transform=get_valid_transform(),
        is_test=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG["STAGE1_BATCH_SIZE"],
        shuffle=True,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG["STAGE1_BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
    )

    return train_loader, valid_loader


def build_stage2_loaders():
    base_path = CFG["BASE_PATH"]

    dev_df = pd.read_csv(os.path.join(base_path, "dev.csv"))
    dev_df["label"] = dev_df["label"].apply(label_to_int)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=CFG["DEV_VALID_RATIO"],
        random_state=CFG["SEED"]
    )

    y = dev_df["label"].values
    train_idx, valid_idx = next(splitter.split(dev_df, y))

    dev_train_df = dev_df.iloc[train_idx].reset_index(drop=True)
    dev_valid_df = dev_df.iloc[valid_idx].reset_index(drop=True)

    train_dataset = MultiViewDataset(
        df=dev_train_df,
        image_root=os.path.join(base_path, "dev"),
        transform=get_stage2_train_transform(),
        is_test=False,
    )

    valid_dataset = MultiViewDataset(
        df=dev_valid_df,
        image_root=os.path.join(base_path, "dev"),
        transform=get_valid_transform(),
        is_test=False,
    )

    return dev_train_df, dev_valid_df, DataLoader(
        train_dataset,
        batch_size=CFG["STAGE2_HEAD_BATCH_SIZE"],
        shuffle=True,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
    ), DataLoader(
        valid_dataset,
        batch_size=CFG["STAGE2_HEAD_BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
    )


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    losses = []

    for views, labels in tqdm(loader, desc="Train", leave=False):
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


def run_stage(
    model,
    train_loader,
    valid_loader,
    save_path,
    epochs,
    lr,
    weight_decay,
    patience,
    stage_name,
    device
):
    criterion = nn.BCEWithLogitsLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=CFG["AMP"])

    best_logloss = float("inf")
    patience_count = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        valid_loss, valid_logloss, valid_acc = validate(
            model, valid_loader, criterion, device
        )

        scheduler.step()

        print(f"[{stage_name} Epoch {epoch}/{epochs}]")
        print(f"Train Loss   : {train_loss:.4f}")
        print(f"Valid Loss   : {valid_loss:.4f}")
        print(f"Valid LogLoss: {valid_logloss:.6f}")
        print(f"Valid Acc    : {valid_acc:.4f}")

        if valid_logloss < best_logloss:
            best_logloss = valid_logloss
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved -> {save_path}")
        else:
            patience_count += 1
            if patience_count >= patience:
                print("Early stopping triggered.")
                break

    print("=" * 50)
    print(f"{stage_name} Best LogLoss: {best_logloss:.6f}")
    return best_logloss


def main():
    seed_everything(CFG["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiViewClassifier(
        model_name=CFG["MODEL_NAME"],
        dropout=CFG["DROPOUT"]
    ).to(device)

    # Stage1
    stage1_train_loader, stage1_valid_loader = build_stage1_loaders()
    model.unfreeze_backbone()
    run_stage(
        model=model,
        train_loader=stage1_train_loader,
        valid_loader=stage1_valid_loader,
        save_path=PRETRAIN_MODEL_PATH,
        epochs=CFG["STAGE1_EPOCHS"],
        lr=CFG["STAGE1_LR"],
        weight_decay=CFG["STAGE1_WEIGHT_DECAY"],
        patience=CFG["STAGE1_PATIENCE"],
        stage_name="Stage1 Pretrain(train->dev)",
        device=device,
    )

    model.load_state_dict(torch.load(PRETRAIN_MODEL_PATH, map_location=device))

    # Stage2-A: head only
    _, _, stage2_train_loader, stage2_valid_loader = build_stage2_loaders()
    model.freeze_backbone()
    run_stage(
        model=model,
        train_loader=stage2_train_loader,
        valid_loader=stage2_valid_loader,
        save_path=STAGE2_HEAD_MODEL_PATH,
        epochs=CFG["STAGE2_HEAD_EPOCHS"],
        lr=CFG["STAGE2_HEAD_LR"],
        weight_decay=CFG["STAGE2_HEAD_WEIGHT_DECAY"],
        patience=CFG["STAGE2_HEAD_PATIENCE"],
        stage_name="Stage2A HeadOnly(dev-train->dev-val)",
        device=device,
    )

    model.load_state_dict(torch.load(STAGE2_HEAD_MODEL_PATH, map_location=device))

    # Stage2-B: last blocks only
    model.unfreeze_last_blocks(CFG["UNFREEZE_LAST_N_BLOCKS"])
    run_stage(
        model=model,
        train_loader=stage2_train_loader,
        valid_loader=stage2_valid_loader,
        save_path=FINETUNE_MODEL_PATH,
        epochs=CFG["STAGE2_FINE_EPOCHS"],
        lr=CFG["STAGE2_FINE_LR"],
        weight_decay=CFG["STAGE2_FINE_WEIGHT_DECAY"],
        patience=CFG["STAGE2_FINE_PATIENCE"],
        stage_name="Stage2B LastBlocks(dev-train->dev-val)",
        device=device,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()