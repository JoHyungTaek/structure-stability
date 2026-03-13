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
from sklearn.metrics import log_loss, accuracy_score

from configs.config import CFG, SOFT_LABEL_PATH, STUDENT_MODEL_DIR
from src.utils import seed_everything, label_to_int, ensure_dir
from src.dataset import StudentDataset
from src.model import StudentImageOnlyModel
from src.transforms import get_student_train_transform, get_student_valid_transform

warnings.filterwarnings("ignore")


def build_train_dev_frames():
    base_path = CFG["BASE_PATH"]

    train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
    dev_df = pd.read_csv(os.path.join(base_path, "dev.csv"))
    soft_df = pd.read_csv(SOFT_LABEL_PATH)

    train_df["label"] = train_df["label"].apply(label_to_int)
    dev_df["label"] = dev_df["label"].apply(label_to_int)

    train_df = train_df.merge(
        soft_df[["id", "soft_unstable_prob"]],
        on="id",
        how="left"
    )
    train_df["domain"] = "train"
    dev_df["soft_unstable_prob"] = np.nan
    dev_df["domain"] = "dev"

    return train_df, dev_df


def make_loaders(train_df, dev_df):
    base_path = CFG["BASE_PATH"]

    train_dataset = StudentDataset(
        df=train_df,
        image_root=os.path.join(base_path, "train"),
        transform=get_student_train_transform(CFG["STUDENT_IMG_SIZE"]),
        is_test=False,
    )

    valid_dataset = StudentDataset(
        df=dev_df,
        image_root=os.path.join(base_path, "dev"),
        transform=get_student_valid_transform(CFG["STUDENT_IMG_SIZE"]),
        is_test=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG["STUDENT_BATCH_SIZE"],
        shuffle=True,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG["STUDENT_BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
    )
    return train_loader, valid_loader


def distill_loss_fn(student_logits, soft_targets, temperature=2.0):
    soft_targets = soft_targets.unsqueeze(1)
    student_probs = torch.sigmoid(student_logits / temperature)
    teacher_probs = soft_targets
    loss = torch.mean((student_probs - teacher_probs) ** 2)
    return loss


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    losses = []

    for views, labels, soft_labels, domains in tqdm(loader, desc="Student Train", leave=False):
        views = [v.to(device) for v in views]
        labels = labels.unsqueeze(1).to(device)
        soft_labels = soft_labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=CFG["AMP"]):
            logits = model(views)
            hard_loss = criterion(logits, labels)

            train_mask = (soft_labels >= 0).float()
            if train_mask.sum() > 0:
                distill = distill_loss_fn(
                    logits[train_mask.bool()],
                    soft_labels[train_mask.bool()],
                    temperature=CFG["DISTILL_TEMP"]
                )
                loss = (1 - CFG["DISTILL_ALPHA"]) * hard_loss + CFG["DISTILL_ALPHA"] * distill
            else:
                loss = hard_loss

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

    for views, labels, soft_labels, domains in tqdm(loader, desc="Student Valid", leave=False):
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


def train_single_seed(seed):
    seed_everything(seed)
    ensure_dir(STUDENT_MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, dev_df = build_train_dev_frames()
    train_loader, valid_loader = make_loaders(train_df, dev_df)

    model = StudentImageOnlyModel(
        model_name=CFG["STUDENT_MODEL_NAME"],
        dropout=CFG["STUDENT_DROPOUT"]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["STUDENT_LR"],
        weight_decay=CFG["STUDENT_WEIGHT_DECAY"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["STUDENT_EPOCHS"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=CFG["AMP"])

    best_logloss = float("inf")
    patience = 0
    save_path = os.path.join(STUDENT_MODEL_DIR, f"student_seed{seed}.pth")

    for epoch in range(1, CFG["STUDENT_EPOCHS"] + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_logloss, va_acc = validate(model, valid_loader, criterion, device)
        scheduler.step()

        print(f"[Seed {seed} | Epoch {epoch}/{CFG['STUDENT_EPOCHS']}]")
        print(f"Train Loss   : {tr_loss:.4f}")
        print(f"Valid Loss   : {va_loss:.4f}")
        print(f"Valid LogLoss: {va_logloss:.6f}")
        print(f"Valid Acc    : {va_acc:.4f}")

        if va_logloss < best_logloss:
            best_logloss = va_logloss
            patience = 0
            torch.save(model.state_dict(), save_path)
            print(f"Best student saved -> {save_path}")
        else:
            patience += 1
            if patience >= CFG["STUDENT_PATIENCE"]:
                print("Student early stopping.")
                break

    print(f"Best Dev LogLoss (seed={seed}): {best_logloss:.6f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    for seed in CFG["SEEDS"][:CFG["N_SEEDS"]]:
        train_single_seed(seed)


if __name__ == "__main__":
    main()