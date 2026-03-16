import gc
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from configs.config import (
    CFG,
    PRETRAIN_MODEL_PATH,
    STAGE2_DIR,
    ENSEMBLE_META_PATH,
)
from src.dataset import MultiViewDataset
from src.model import MultiViewClassifier
from src.transforms import (
    get_stage1_train_transform,
    get_stage2_train_transform,
    get_valid_transform,
)
from src.utils import (
    seed_everything,
    label_to_int,
    binary_logloss,
    TemperatureScaler,
    save_json,
)

warnings.filterwarnings("ignore")


def build_stage1_loaders():
    base_path = CFG["BASE_PATH"]
    train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
    dev_df = pd.read_csv(os.path.join(base_path, "dev.csv"))

    train_df["label"] = train_df["label"].apply(label_to_int)
    dev_df["label"] = dev_df["label"].apply(label_to_int)

    train_dataset = MultiViewDataset(
        df=train_df,
        image_root=os.path.join(base_path, "train"),
        transform=get_stage1_train_transform(CFG["IMG_SIZE"]),
        is_test=False,
    )
    valid_dataset = MultiViewDataset(
        df=dev_df,
        image_root=os.path.join(base_path, "dev"),
        transform=get_valid_transform(CFG["IMG_SIZE"]),
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


def load_dev_df():
    dev_df = pd.read_csv(os.path.join(CFG["BASE_PATH"], "dev.csv"))
    dev_df["label"] = dev_df["label"].apply(label_to_int)
    return dev_df


def make_loader(df, image_root, transform, batch_size, shuffle):
    dataset = MultiViewDataset(
        df=df,
        image_root=image_root,
        transform=transform,
        is_test=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
    )


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    losses = []

    for views, labels in tqdm(loader, desc="Train", leave=False):
        views = [v.to(device) for v in views]
        labels = labels.unsqueeze(1).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=CFG["AMP"]):
            logits = model(views)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def validate(model, loader, criterion, device, temperature=1.0):
    model.eval()
    losses = []
    logits_all = []
    probs_all = []
    labels_all = []

    for views, labels in tqdm(loader, desc="Valid", leave=False):
        views = [v.to(device) for v in views]
        labels = labels.unsqueeze(1).to(device)

        logits = model(views)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits / temperature).detach().cpu().numpy().reshape(-1)
        logits_np = logits.detach().cpu().numpy().reshape(-1)
        labels_np = labels.detach().cpu().numpy().reshape(-1)

        losses.append(loss.item())
        logits_all.extend(logits_np.tolist())
        probs_all.extend(probs.tolist())
        labels_all.extend(labels_np.tolist())

    probs_all = np.array(probs_all)
    labels_all = np.array(labels_all)
    logits_all = np.array(logits_all)

    valid_loss = float(np.mean(losses))
    valid_logloss = binary_logloss(labels_all, probs_all, CFG["CLIP_MIN"], CFG["CLIP_MAX"])
    valid_acc = accuracy_score(labels_all, (probs_all >= 0.5).astype(int))

    return valid_loss, valid_logloss, valid_acc, logits_all, labels_all, probs_all


def run_stage(model, train_loader, valid_loader, save_path, epochs, lr, weight_decay, patience, stage_name, device, use_temp_scaling=False):
    criterion = nn.BCEWithLogitsLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG["AMP"])

    best_logloss = float("inf")
    best_temperature = 1.0
    patience_count = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        valid_loss, valid_logloss_raw, valid_acc_raw, logits_all, labels_all, probs_raw = validate(
            model, valid_loader, criterion, device, temperature=1.0
        )

        valid_logloss = valid_logloss_raw
        valid_acc = valid_acc_raw
        current_temperature = 1.0

        if use_temp_scaling and CFG["USE_TEMPERATURE_SCALING"]:
            scaler_obj = TemperatureScaler()
            current_temperature = scaler_obj.fit(logits_all, labels_all, CFG["CLIP_MIN"], CFG["CLIP_MAX"])
            probs_cal = scaler_obj.predict_proba(logits_all, CFG["CLIP_MIN"], CFG["CLIP_MAX"])
            valid_logloss = binary_logloss(labels_all, probs_cal, CFG["CLIP_MIN"], CFG["CLIP_MAX"])
            valid_acc = accuracy_score(labels_all, (probs_cal >= 0.5).astype(int))

        scheduler.step()

        print(f"[{stage_name} Epoch {epoch}/{epochs}]")
        print(f"Train Loss   : {train_loss:.4f}")
        print(f"Valid Loss   : {valid_loss:.4f}")
        print(f"Valid LogLoss(raw): {valid_logloss_raw:.6f}")
        print(f"Valid LogLoss(cal): {valid_logloss:.6f}")
        print(f"Valid Acc    : {valid_acc:.4f}")
        if use_temp_scaling:
            print(f"Temperature  : {current_temperature:.4f}")

        if valid_logloss < best_logloss:
            best_logloss = valid_logloss
            best_temperature = current_temperature
            patience_count = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "temperature": best_temperature,
                "best_logloss": best_logloss,
            }, save_path)
            print(f"Best model saved -> {save_path}")
        else:
            patience_count += 1
            if patience_count >= patience:
                print("Early stopping triggered.")
                break

    print("=" * 60)
    print(f"{stage_name} Best LogLoss: {best_logloss:.6f}")
    return best_logloss, best_temperature


def stage2_kfold_adaptation(device):
    dev_df = load_dev_df()
    image_root = os.path.join(CFG["BASE_PATH"], "dev")
    skf = StratifiedKFold(n_splits=CFG["STAGE2_N_SPLITS"], shuffle=True, random_state=CFG["SEED"])

    fold_scores = []
    fold_meta = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(dev_df, dev_df["label"].values)):
        print("#" * 80)
        print(f"Stage2 Fold {fold + 1}/{CFG['STAGE2_N_SPLITS']}")

        dev_train_df = dev_df.iloc[train_idx].reset_index(drop=True)
        dev_valid_df = dev_df.iloc[valid_idx].reset_index(drop=True)

        train_loader = make_loader(
            dev_train_df, image_root, get_stage2_train_transform(CFG["IMG_SIZE"]), CFG["STAGE2_HEAD_BATCH_SIZE"], True
        )
        valid_loader = make_loader(
            dev_valid_df, image_root, get_valid_transform(CFG["IMG_SIZE"]), CFG["STAGE2_HEAD_BATCH_SIZE"], False
        )

        model = MultiViewClassifier(model_name=CFG["MODEL_NAME"], dropout=CFG["DROPOUT"]).to(device)
        stage1_ckpt = torch.load(PRETRAIN_MODEL_PATH, map_location=device)
        if isinstance(stage1_ckpt, dict) and "model_state_dict" in stage1_ckpt:
            model.load_state_dict(stage1_ckpt["model_state_dict"])
        else:
            model.load_state_dict(stage1_ckpt)

        # Stage2-A: head only
        model.freeze_backbone()
        head_ckpt = STAGE2_DIR / f"fold{fold}_head_best.pth"
        run_stage(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            save_path=str(head_ckpt),
            epochs=CFG["STAGE2_HEAD_EPOCHS"],
            lr=CFG["STAGE2_HEAD_LR"],
            weight_decay=CFG["STAGE2_HEAD_WEIGHT_DECAY"],
            patience=CFG["STAGE2_HEAD_PATIENCE"],
            stage_name=f"Stage2A Fold{fold} HeadOnly(dev-train->dev-val)",
            device=device,
            use_temp_scaling=False,
        )

        head_bundle = torch.load(head_ckpt, map_location=device)
        model.load_state_dict(head_bundle["model_state_dict"])

        # Stage2-B: last blocks only
        model.unfreeze_last_blocks(CFG["UNFREEZE_LAST_N_BLOCKS"])
        fine_ckpt = STAGE2_DIR / f"fold{fold}_finetune_best.pth"
        best_score, best_temperature = run_stage(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            save_path=str(fine_ckpt),
            epochs=CFG["STAGE2_FINE_EPOCHS"],
            lr=CFG["STAGE2_FINE_LR"],
            weight_decay=CFG["STAGE2_FINE_WEIGHT_DECAY"],
            patience=CFG["STAGE2_FINE_PATIENCE"],
            stage_name=f"Stage2B Fold{fold} LastBlocks(dev-train->dev-val)",
            device=device,
            use_temp_scaling=True,
        )

        fold_scores.append(best_score)
        fold_meta.append({
            "fold": fold,
            "head_ckpt": str(head_ckpt),
            "finetune_ckpt": str(fine_ckpt),
            "temperature": float(best_temperature),
            "best_logloss": float(best_score),
            "n_train": int(len(dev_train_df)),
            "n_valid": int(len(dev_valid_df)),
        })

        del model
        gc.collect()
        torch.cuda.empty_cache()

    mean_score = float(np.mean(fold_scores)) if fold_scores else float("inf")
    meta = {
        "n_splits": CFG["STAGE2_N_SPLITS"],
        "mean_cv_logloss": mean_score,
        "folds": fold_meta,
    }
    save_json(meta, ENSEMBLE_META_PATH)
    print("=" * 80)
    print(f"Stage2 KFold mean LogLoss: {mean_score:.6f}")


def main():
    seed_everything(CFG["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiViewClassifier(model_name=CFG["MODEL_NAME"], dropout=CFG["DROPOUT"]).to(device)

    # Stage1: 공식 train -> dev
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
        use_temp_scaling=False,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Stage2: dev 내부 KFold adaptation
    stage2_kfold_adaptation(device)


if __name__ == "__main__":
    main()
