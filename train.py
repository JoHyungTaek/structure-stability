from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import MultiViewDataset, build_dataset_specs, read_split_dataframe
from src.engine import TemperatureScaler, build_optimizer, build_scheduler, train_one_epoch, valid_one_epoch
from src.model import MultiViewClassifier
from src.transforms import build_train_transform, build_valid_transform
from src.utils import build_run_name, ensure_dir, get_device, save_checkpoint, save_json, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--override", nargs="*", default=None)
    parser.add_argument("--only_stage", choices=["all", "stage1", "stage2"], default="all")
    parser.add_argument("--resume_stage1", type=str, default=None)
    return parser.parse_args()


def make_loader(df, image_root, transform, batch_size, num_workers, is_test=False, shuffle=False, drop_last=False):
    ds = MultiViewDataset(df=df, image_root=image_root, transform=transform, is_test=is_test)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def build_criterion(cfg):
    pos_weight = cfg["train"].get("pos_weight")
    if pos_weight is not None:
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=get_device()))
    return nn.BCEWithLogitsLoss()


def stage1_train(cfg, run_dir: Path, train_df: pd.DataFrame, dev_df: pd.DataFrame, specs: dict):
    device = get_device()
    model = MultiViewClassifier(
        model_name=cfg["model"]["name"],
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("dropout", 0.25),
    ).to(device)

    train_loader = make_loader(
        train_df,
        specs["train"].image_root,
        build_train_transform(cfg),
        cfg["train"]["batch_size"],
        cfg["num_workers"],
        shuffle=True,
        drop_last=True,
    )
    valid_loader = make_loader(
        dev_df,
        specs["dev"].image_root,
        build_valid_transform(cfg),
        cfg["train"]["batch_size"],
        cfg["num_workers"],
        shuffle=False,
    )

    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, len(train_loader), cfg["train"]["stage1_epochs"], cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("use_amp", True) and device.type == "cuda")

    best_score = float("inf")
    best_path = run_dir / "stage1_best.pth"
    history = []
    patience = 0

    freeze_epochs = cfg["train"].get("freeze_backbone_epochs", 0)
    if freeze_epochs > 0:
        model.freeze_backbone()

    for epoch in range(cfg["train"]["stage1_epochs"]):
        if epoch == freeze_epochs:
            model.unfreeze_backbone()
            optimizer = build_optimizer(model, cfg)
            scheduler = build_scheduler(optimizer, len(train_loader), cfg["train"]["stage1_epochs"] - epoch, cfg)

        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, cfg, prefix=f"stage1-train-{epoch+1}")
        va = valid_one_epoch(model, valid_loader, criterion, device, cfg, prefix=f"stage1-valid-{epoch+1}")

        row = {
            "epoch": epoch + 1,
            "train_loss": tr.loss,
            "train_logloss": tr.logloss,
            "valid_loss": va.loss,
            "valid_logloss": va.logloss,
        }
        history.append(row)
        print(row)

        if va.logloss < best_score:
            best_score = va.logloss
            patience = 0
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_logloss": best_score,
                    "cfg": cfg,
                },
                best_path,
            )
        else:
            patience += 1
            if patience >= cfg["train"].get("early_stopping_patience", 3):
                break

    pd.DataFrame(history).to_csv(run_dir / "stage1_history.csv", index=False)
    return best_path, best_score


def stage2_dev_folds(cfg, run_dir: Path, dev_df: pd.DataFrame, specs: dict, stage1_checkpoint: Path):
    device = get_device()
    skf = StratifiedKFold(
        n_splits=cfg["train"]["dev_folds"],
        shuffle=True,
        random_state=cfg["seed"],
    )

    fold_dir = ensure_dir(run_dir / "stage2_folds")
    fold_records = []
    oof_logits = np.zeros(len(dev_df), dtype=np.float32)
    oof_targets = dev_df[MultiViewDataset(dev_df, specs["dev"].image_root, is_test=False)._find_label_col()].map(
        lambda x: 1 if str(x).strip().lower() == "unstable" else (0 if str(x).strip().lower() == "stable" else float(x))
    ).values.astype(np.float32)

    target_col = MultiViewDataset(dev_df, specs["dev"].image_root, is_test=False)._find_label_col()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(dev_df, dev_df[target_col])):
        print(f"\n===== Stage2 Fold {fold} =====")
        tr_df = dev_df.iloc[tr_idx].reset_index(drop=True)
        va_df = dev_df.iloc[va_idx].reset_index(drop=True)

        model = MultiViewClassifier(
            model_name=cfg["model"]["name"],
            pretrained=False,
            dropout=cfg["model"].get("dropout", 0.25),
        ).to(device)
        checkpoint = torch.load(stage1_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)

        train_loader = make_loader(
            tr_df,
            specs["dev"].image_root,
            build_train_transform(cfg),
            cfg["train"]["batch_size"],
            cfg["num_workers"],
            shuffle=True,
            drop_last=True,
        )
        valid_loader = make_loader(
            va_df,
            specs["dev"].image_root,
            build_valid_transform(cfg),
            cfg["train"]["batch_size"],
            cfg["num_workers"],
            shuffle=False,
        )

        criterion = build_criterion(cfg)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, len(train_loader), cfg["train"]["stage2_epochs"], cfg)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("use_amp", True) and device.type == "cuda")

        best_score = float("inf")
        best_state = None
        best_logits = None
        patience = 0
        history = []

        for epoch in range(cfg["train"]["stage2_epochs"]):
            tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, cfg, prefix=f"stage2-train-f{fold}-e{epoch+1}")
            va = valid_one_epoch(model, valid_loader, criterion, device, cfg, prefix=f"stage2-valid-f{fold}-e{epoch+1}")
            history.append({
                "fold": fold,
                "epoch": epoch + 1,
                "train_logloss": tr.logloss,
                "valid_logloss": va.logloss,
            })
            print(history[-1])
            if va.logloss < best_score:
                best_score = va.logloss
                patience = 0
                best_state = copy.deepcopy(model.state_dict())
                probs = np.clip(va.y_pred, 1e-6, 1 - 1e-6)
                best_logits = np.log(probs / (1 - probs))
            else:
                patience += 1
                if patience >= cfg["train"].get("early_stopping_patience", 3):
                    break

        fold_ckpt_path = fold_dir / f"fold_{fold}.pth"
        save_checkpoint(
            {
                "model": best_state,
                "fold": fold,
                "best_logloss": best_score,
                "cfg": cfg,
            },
            fold_ckpt_path,
        )
        pd.DataFrame(history).to_csv(fold_dir / f"fold_{fold}_history.csv", index=False)

        fold_records.append({
            "fold": fold,
            "path": str(fold_ckpt_path),
            "best_logloss": best_score,
        })

        oof_logits[va_idx] = best_logits

    scaler_model = TemperatureScaler()
    temperature = 1.0
    if cfg["inference"].get("temperature_scaling", True):
        temperature = scaler_model.fit(oof_logits, oof_targets, device=device)

    artifacts = {
        "stage1_checkpoint": str(stage1_checkpoint),
        "folds": fold_records,
        "temperature": temperature,
    }
    save_json(artifacts, run_dir / "artifacts.json")
    pd.DataFrame(fold_records).to_csv(run_dir / "stage2_summary.csv", index=False)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg["seed"])

    specs = build_dataset_specs(cfg)
    train_df = read_split_dataframe(specs["train"])
    dev_df = read_split_dataframe(specs["dev"])

    run_name = build_run_name(cfg)
    run_dir = ensure_dir(Path(cfg["paths"]["output_root"]) / run_name)
    save_json(cfg, run_dir / "resolved_config.json")

    stage1_ckpt = Path(args.resume_stage1) if args.resume_stage1 else run_dir / "stage1_best.pth"

    if args.only_stage in ["all", "stage1"] and args.resume_stage1 is None:
        stage1_ckpt, stage1_score = stage1_train(cfg, run_dir, train_df, dev_df, specs)
        print(f"Stage1 best logloss: {stage1_score:.6f}")

    if args.only_stage in ["all", "stage2"]:
        if not stage1_ckpt.exists():
            raise FileNotFoundError(f"Stage1 checkpoint not found: {stage1_ckpt}")
        stage2_dev_folds(cfg, run_dir, dev_df, specs, stage1_ckpt)
        print(f"Saved stage2 artifacts under: {run_dir}")


if __name__ == "__main__":
    main()
