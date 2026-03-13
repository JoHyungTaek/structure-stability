import os
import gc
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import log_loss, accuracy_score

from configs.config import CFG, TEACHER_MODEL_PATH
from src.utils import seed_everything, label_to_int
from src.dataset import TeacherTrainDataset
from src.model import TeacherMultiModalModel
from src.transforms import get_teacher_train_transform, get_teacher_valid_transform

warnings.filterwarnings("ignore")


def make_teacher_loaders():
    base_path = CFG["BASE_PATH"]
    df = pd.read_csv(os.path.join(base_path, "train.csv"))
    df["label"] = df["label"].apply(label_to_int)

    dataset = TeacherTrainDataset(
        df=df,
        image_root=os.path.join(base_path, "train"),
        transform=get_teacher_train_transform(CFG["TEACHER_IMG_SIZE"]),
        num_video_frames=CFG["NUM_VIDEO_FRAMES"],
    )

    valid_ratio = 0.15
    valid_size = int(len(dataset) * valid_ratio)
    train_size = len(dataset) - valid_size

    train_dataset, valid_dataset = random_split(
        dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(CFG["SEED"])
    )

    valid_dataset.dataset.transform = get_teacher_valid_transform(CFG["TEACHER_IMG_SIZE"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG["TEACHER_BATCH_SIZE"],
        shuffle=True,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG["TEACHER_BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
    )
    return train_loader, valid_loader


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    losses = []
    for views, video_frames, labels in tqdm(loader, desc="Teacher Train", leave=False):
        views = [v.to(device) for v in views]
        video_frames = video_frames.to(device)
        labels = labels.unsqueeze(1).to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=CFG["AMP"]):
            logits = model(views, video_frames)
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
    probs_all, labels_all = [], []

    for views, video_frames, labels in tqdm(loader, desc="Teacher Valid", leave=False):
        views = [v.to(device) for v in views]
        video_frames = video_frames.to(device)
        labels = labels.unsqueeze(1).to(device)

        logits = model(views, video_frames)
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

    train_loader, valid_loader = make_teacher_loaders()

    model = TeacherMultiModalModel(
        model_name=CFG["TEACHER_MODEL_NAME"],
        dropout=CFG["TEACHER_DROPOUT"]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["TEACHER_LR"],
        weight_decay=CFG["TEACHER_WEIGHT_DECAY"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["TEACHER_EPOCHS"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=CFG["AMP"])

    best_logloss = float("inf")
    patience = 0

    for epoch in range(1, CFG["TEACHER_EPOCHS"] + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_logloss, va_acc = validate(model, valid_loader, criterion, device)
        scheduler.step()

        print(f"[Teacher Epoch {epoch}/{CFG['TEACHER_EPOCHS']}]")
        print(f"Train Loss   : {tr_loss:.4f}")
        print(f"Valid Loss   : {va_loss:.4f}")
        print(f"Valid LogLoss: {va_logloss:.6f}")
        print(f"Valid Acc    : {va_acc:.4f}")

        if va_logloss < best_logloss:
            best_logloss = va_logloss
            patience = 0
            torch.save(model.state_dict(), TEACHER_MODEL_PATH)
            print(f"Best teacher saved -> {TEACHER_MODEL_PATH}")
        else:
            patience += 1
            if patience >= CFG["TEACHER_PATIENCE"]:
                print("Teacher early stopping.")
                break

    print(f"Best Teacher LogLoss: {best_logloss:.6f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()