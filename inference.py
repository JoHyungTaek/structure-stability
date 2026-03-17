from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import StructureDataset, load_split_dataframe
from src.model import MultiViewStabilityModel
from src.transforms import apply_tta, build_valid_transform
from src.utils import apply_overrides, ensure_dir, flatten_dict, load_yaml, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--override", nargs="*", default=None)
    return parser.parse_args()


def build_model(cfg):
    return MultiViewStabilityModel(
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        attention_heads=cfg["model"]["attention_heads"],
    )


@torch.no_grad()
def predict_logits(model, loader, device, tta_modes=None):
    model.eval()

    if tta_modes is None or len(tta_modes) == 0:
        tta_modes = ["none"]

    all_logits = []

    for batch in tqdm(loader, total=len(loader)):
        front = batch["front"].to(device, non_blocking=(device.type == "cuda"))
        top = batch["top"].to(device, non_blocking=(device.type == "cuda"))

        tta_logits = []
        for mode in tta_modes:
            front_aug, top_aug = apply_tta(front, top, mode)
            logits = model(front_aug, top_aug)
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            tta_logits.append(logits.detach().cpu().numpy())

        batch_logits = np.mean(np.stack(tta_logits, axis=0), axis=0)
        all_logits.append(batch_logits)

    return np.concatenate(all_logits, axis=0)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)
    seed_everything(cfg["seed"])
    print("\n".join(flatten_dict(cfg)))

    out_dir = ensure_dir(cfg["paths"]["output_root"])
    data_root = cfg["paths"]["data_root"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu"
    )
    print(f"device: {device}")

    test_df = load_split_dataframe(data_root, "test")
    test_ds = StructureDataset(
        test_df,
        transform=build_valid_transform(cfg["model"]["image_size"]),
        test_mode=True,
    )

    use_pin_memory = bool(torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda")
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["infer"]["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=use_pin_memory,
    )

    ckpt_paths = sorted(Path(out_dir).glob(cfg["infer"]["checkpoint_glob"]))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {out_dir}")

    all_probs = []

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")

        model = build_model(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        logits = predict_logits(
            model,
            test_loader,
            device,
            tta_modes=cfg["infer"].get("tta", ["none"]),
        )

        temperature = float(ckpt.get("temperature", 1.0))
        logits = logits / max(temperature, 1e-6)

        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

        all_probs.append(probs)
        print(f"loaded: {ckpt_path.name} | temperature={temperature:.4f}")

    unstable_prob = np.mean(np.stack(all_probs, axis=0), axis=0)
    unstable_prob = np.clip(unstable_prob, 1e-6, 1.0 - 1e-6)
    stable_prob = 1.0 - unstable_prob

    sub = pd.read_csv(Path(data_root) / "sample_submission.csv")
    sub["unstable_prob"] = unstable_prob
    sub["stable_prob"] = stable_prob

    save_path = Path(out_dir) / "submission.csv"
    sub.to_csv(save_path, index=False)
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()