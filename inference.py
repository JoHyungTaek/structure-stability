from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import MultiViewDataset, build_dataset_specs, read_split_dataframe
from src.engine import predict_logits
from src.model import MultiViewClassifier
from src.transforms import build_tta_transforms
from src.utils import build_run_name, ensure_dir, get_device, infer_submission_columns, seed_everything, sigmoid_np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--override", nargs="*", default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--output_name", type=str, default="submission.csv")
    return parser.parse_args()


def make_loader(df, image_root, transform, batch_size, num_workers, is_test=False):
    ds = MultiViewDataset(df=df, image_root=image_root, transform=transform, is_test=is_test)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg["seed"])
    device = get_device()

    run_dir = Path(args.run_dir) if args.run_dir else Path(cfg["paths"]["output_root"]) / build_run_name(cfg)
    artifacts_path = run_dir / "artifacts.json"
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")

    with open(artifacts_path, "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    specs = build_dataset_specs(cfg)
    test_df = read_split_dataframe(specs["test"])
    sample_sub_path = Path(cfg["paths"]["data_root"]) / cfg["paths"]["sample_submission_csv"]
    sample_sub = pd.read_csv(sample_sub_path)

    tta_transforms = build_tta_transforms(cfg) if cfg["inference"].get("use_tta", True) else build_tta_transforms(cfg)[:1]
    all_fold_probs = []

    for fold_info in artifacts["folds"]:
        fold_ckpt = Path(fold_info["path"])
        if not fold_ckpt.exists():
            raise FileNotFoundError(f"Fold checkpoint not found: {fold_ckpt}")

        model = MultiViewClassifier(
            model_name=cfg["model"]["name"],
            pretrained=False,
            dropout=cfg["model"].get("dropout", 0.25),
        ).to(device)

        checkpoint = torch.load(fold_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()

        fold_logits_tta = []
        fold_ids = None
        for transform in tta_transforms:
            test_loader = make_loader(
                test_df,
                specs["test"].image_root,
                transform,
                cfg["train"]["batch_size"],
                cfg["num_workers"],
                is_test=True,
            )
            logits, ids = predict_logits(model, test_loader, device, cfg)
            fold_ids = ids
            fold_logits_tta.append(logits)

        mean_logits = np.mean(fold_logits_tta, axis=0)
        temperature = float(artifacts.get("temperature", 1.0))
        mean_logits = mean_logits / max(temperature, 1e-6)
        probs = sigmoid_np(mean_logits)
        all_fold_probs.append(probs)

    final_probs = np.mean(all_fold_probs, axis=0)
    final_probs = np.clip(final_probs, 1e-6, 1 - 1e-6)

    id_col, pred_col = infer_submission_columns(sample_sub)
    if pred_col is None:
        raise ValueError("Could not infer prediction column in sample submission")

    out_df = sample_sub.copy()
    out_df[pred_col] = final_probs
    out_path = run_dir / args.output_name
    out_df.to_csv(out_path, index=False)
    print(f"Saved submission to: {out_path}")


if __name__ == "__main__":
    main()
