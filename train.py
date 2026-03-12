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
from torch.utils.data import DataLoader, ConcatDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score

from configs.config import CFG, BASE_PATH, MODEL_DIR, OOF_PATH
from src.dataset import MultiModalStructureDataset
from src.model import MultiModalStructureModel

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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def label_to_int(x):
    if isinstance(x, str):
        x = x.strip().lower()
        if x == "stable":
            return 0
        if x == "unstable":
            return 1
    return int(float(x))


def get_train_transform():
    return A.Compose([
        A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=15,
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
    return A.Compose([
        A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def build_full_dataframe():
    train_csv_path = os.path.join(BASE_PATH, "train.csv")
    dev_csv_path = os.path.join(BASE_PATH, "dev.csv")

    train_df = pd.read_csv(train_csv_path).copy()
    dev_df = pd.read_csv(dev_csv_path).copy()

    train_df["label"] = train_df["label"].apply(label_to_int)
    dev_df["label"] = dev_df["label"].apply(label_to_int)

    train_df["image_root"] = os.path.join(BASE_PATH, "train")
    dev_df["image_root"] = os.path.join(BASE_PATH, "dev")

    full_df = pd.concat([train_df, dev_df], axis=0, ignore_index=True)
    return full_df


def collate_df_by_root(df):
    datasets = []
    for root_path in df["image_root"].unique():
        sub_df = df[df["image_root"] == root_path].reset_index(drop=True)

        dataset = MultiModalStructureDataset(
            df=sub_df.drop(columns=["image_root"]),
            image_root=root_path,
            transform=None,  # 아래 make_loader에서 재설정
            is_test=False,
            use_video=CFG["USE_VIDEO"],
            num_video_frames=CFG["NUM_VIDEO_FRAMES"],
        )
        datasets.append((sub_df, root_path))
    return datasets


def make_loader(df, transform, is_test=False, shuffle=False):
    datasets = []

    for root_path in df["image_root"].unique():
        sub_df = df[df["image_root"] == root_path].reset_index(drop=True)

        dataset = MultiModalStructureDataset(
            df=sub_df.drop(columns=["image_root"]),
            image_root=root_path,
            transform=transform,
            is_test=is_test,
            use_video=CFG["USE_VIDEO"],
            num_video_frames=CFG["NUM_VIDEO_FRAMES"],
        )
        datasets.append(dataset)

    concat_dataset = ConcatDataset(datasets)

    loader = DataLoader(
        concat_dataset,
        batch_size=CFG["BATCH_SIZE"],
        shuffle=shuffle,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
        drop_last=False,
    )
    return loader


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0

    for views, video_frames, labels in tqdm(loader, desc="Train", leave=False):
        views = [v.to(device) for v in views]
        video_frames = video_frames.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        if CFG["AMP"]:
            with torch.cuda.amp.autocast():
                outputs = model(views, video_frames)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(views, video_frames)
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

    for views, video_frames, labels in tqdm(loader, desc="Valid", leave=False):
        views = [v.to(device) for v in views]
        video_frames = video_frames.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(views, video_frames)
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

    return avg_loss, val_logloss, val_acc, probs_arr, labels_arr


def main():
    seed_everything(CFG["SEED"])
    ensure_dir(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_df = build_full_dataframe()

    y = full_df["label"].values
    oof_preds = np.zeros(len(full_df), dtype=np.float32)

    skf = StratifiedKFold(
        n_splits=CFG["N_SPLITS"],
        shuffle=True,
        random_state=CFG["SEED"]
    )

    for fold, (train_idx, valid_idx) in enumerate(skf.split(full_df, y), start=1):
        print("=" * 60)
        print(f"Fold {fold}/{CFG['N_SPLITS']}")

        train_fold_df = full_df.iloc[train_idx].reset_index(drop=True)
        valid_fold_df = full_df.iloc[valid_idx].reset_index(drop=True)

        train_loader = make_loader(
            train_fold_df,
            transform=get_train_transform(),
            is_test=False,
            shuffle=True,
        )
        valid_loader = make_loader(
            valid_fold_df,
            transform=get_valid_transform(),
            is_test=False,
            shuffle=False,
        )

        model = MultiModalStructureModel(
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
        patience_counter = 0
        best_model_path = os.path.join(MODEL_DIR, f"best_fold{fold}.pth")

        for epoch in range(1, CFG["EPOCHS"] + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )
            val_loss, val_logloss, val_acc, val_probs, val_labels = validate(
                model, valid_loader, criterion, device
            )
            scheduler.step()

            print(f"[Fold {fold} | Epoch {epoch}/{CFG['EPOCHS']}]")
            print(f"Train Loss   : {train_loss:.4f}")
            print(f"Valid Loss   : {val_loss:.4f}")
            print(f"Valid LogLoss: {val_logloss:.6f}")
            print(f"Valid Acc    : {val_acc:.4f}")

            if val_logloss < best_logloss:
                best_logloss = val_logloss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved -> {best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= CFG["PATIENCE"]:
                    print("Early stopping triggered.")
                    break

        print(f"Fold {fold} best logloss: {best_logloss:.6f}")

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        _, fold_logloss, fold_acc, fold_probs, fold_labels = validate(
            model, valid_loader, criterion, device
        )

        oof_preds[valid_idx] = fold_probs
        print(f"Reloaded Fold {fold} LogLoss: {fold_logloss:.6f}")
        print(f"Reloaded Fold {fold} Acc    : {fold_acc:.4f}")

        del model, train_loader, valid_loader
        gc.collect()
        torch.cuda.empty_cache()

    eps = 1e-7
    oof_preds = np.clip(oof_preds, eps, 1 - eps)
    oof_logloss = log_loss(y, oof_preds)
    oof_acc = accuracy_score(y, (oof_preds >= 0.5).astype(int))

    print("=" * 60)
    print(f"OOF LogLoss: {oof_logloss:.6f}")
    print(f"OOF Acc    : {oof_acc:.4f}")

    oof_df = full_df[["id", "label"]].copy()
    oof_df["unstable_prob"] = oof_preds
    oof_df["stable_prob"] = 1.0 - oof_df["unstable_prob"]
    oof_df.to_csv(OOF_PATH, index=False)
    print(f"OOF saved -> {OOF_PATH}")


if __name__ == "__main__":
    main()