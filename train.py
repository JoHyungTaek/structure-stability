from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import StructureDataset, load_split_dataframe
from src.engine import BinarySmoothBCE, TemperatureScaler, run_train_epoch, run_valid_epoch
from src.model import MultiViewStabilityModel
from src.transforms import build_train_transform, build_valid_transform
from src.utils import apply_overrides, ensure_dir, flatten_dict, load_yaml, save_json, save_yaml, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--override", nargs="*", default=None)
    return parser.parse_args()


def make_loaders(train_df, valid_df, cfg):
    image_size = cfg["model"]["image_size"]
    train_ds = StructureDataset(train_df, transform=build_train_transform(image_size, cfg), test_mode=False)
    valid_ds = StructureDataset(valid_df, transform=build_valid_transform(image_size), test_mode=False)
    common = dict(num_workers=cfg.get("num_workers", 2), pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=False, **common)
    valid_loader = DataLoader(valid_ds, batch_size=cfg["infer"]["batch_size"], shuffle=False, drop_last=False, **common)
    return train_loader, valid_loader


def build_model(cfg):
    return MultiViewStabilityModel(
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"].get("pretrained", True),
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        attention_heads=cfg["model"]["attention_heads"],
    )


def train_one_fold(fold, train_df, valid_df, cfg, out_dir: Path, device):
    train_loader, valid_loader = make_loaders(train_df, valid_df, cfg)
    model = build_model(cfg).to(device)

    criterion = BinarySmoothBCE(
        smoothing=cfg["train"].get("label_smoothing", 0.0),
        pos_weight=cfg["train"].get("pos_weight", 1.0),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    total_steps = max(len(train_loader) * cfg["train"]["epochs"] // cfg["train"]["grad_accum_steps"], 1)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg["train"]["min_lr"])

    best_logloss = float("inf")
    best_path = out_dir / f"fold{fold}_best.pt"
    patience = 0
    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_res = run_train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            device,
            mixed_precision=cfg["train"].get("mixed_precision", True),
            grad_accum_steps=cfg["train"].get("grad_accum_steps", 1),
            gradient_clip=cfg["train"].get("gradient_clip", None),
        )
        valid_res, valid_logits = run_valid_epoch(model, valid_loader, criterion, device)

        log_row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_res.loss,
            "train_logloss": train_res.logloss,
            "valid_loss": valid_res.loss,
            "valid_logloss": valid_res.logloss,
        }
        history.append(log_row)
        print(log_row)

        if valid_res.logloss < best_logloss:
            best_logloss = valid_res.logloss
            patience = 0
            scaler = TemperatureScaler()
            temperature = scaler.fit(valid_logits, valid_res.y_true, device=str(device))
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "best_logloss": best_logloss,
                    "temperature": temperature,
                    "fold": fold,
                },
                best_path,
            )
        else:
            patience += 1
            if patience >= cfg["train"].get("early_stopping", 999):
                print(f"Fold {fold}: early stopping")
                break

    return history


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)
    seed_everything(cfg["seed"])

    out_dir = ensure_dir(cfg["paths"]["output_root"])
    save_yaml(out_dir / "used_config.yaml", cfg)
    print("\n".join(flatten_dict(cfg)))

    data_root = cfg["paths"]["data_root"]
    train_df = load_split_dataframe(data_root, "train")
    if cfg["train"].get("use_dev", True):
        dev_df = load_split_dataframe(data_root, "dev")
        all_df = pd.concat([train_df, dev_df], ignore_index=True)
    else:
        all_df = train_df.copy()

    all_df["source"] = all_df["split"].map({"train": 0, "dev": 1}).astype(int)
    all_df["stratify_key"] = all_df["target"].astype(str) + "_" + all_df["source"].astype(str)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu")
    print(f"device: {device}")

    skf = StratifiedKFold(n_splits=cfg["train"]["n_folds"], shuffle=True, random_state=cfg["seed"])
    histories = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(all_df, all_df["stratify_key"])):
        print(f"\n===== Fold {fold} =====")
        tr_df = all_df.iloc[tr_idx].reset_index(drop=True)
        va_df = all_df.iloc[va_idx].reset_index(drop=True)
        histories.extend(train_one_fold(fold, tr_df, va_df, cfg, out_dir, device))

    hist_df = pd.DataFrame(histories)
    hist_df.to_csv(out_dir / "history.csv", index=False)
    summary = hist_df.groupby("fold")["valid_logloss"].min().to_dict()
    save_json(out_dir / "cv_summary.json", {"best_valid_logloss_by_fold": summary})
    print("Done")


if __name__ == "__main__":
    main()
