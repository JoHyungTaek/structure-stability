from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import log_loss


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_binary_logloss(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), dtype=np.float32)
    y_pred = np.asarray(list(y_pred), dtype=np.float32)
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    return float(log_loss(y_true, y_pred, labels=[0, 1]))


def infer_submission_columns(df) -> tuple[str, str | None]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    id_candidates = ["id", "sample_id"]
    id_col = None
    for cand in id_candidates:
        if cand in lower_map:
            id_col = lower_map[cand]
            break
    if id_col is None:
        id_col = cols[0]

    prob_candidates = [
        "unstable",
        "probability",
        "target",
        "label",
        "score",
        "prediction",
    ]
    pred_col = None
    for cand in prob_candidates:
        if cand in lower_map:
            pred_col = lower_map[cand]
            break

    if pred_col is None:
        remaining = [c for c in cols if c != id_col]
        pred_col = remaining[0] if remaining else None

    return id_col, pred_col


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_run_name(cfg: Dict, suffix: str = "") -> str:
    model_name = cfg["model"]["name"].split(".")[0]
    img_size = cfg["model"]["image_size"]
    seed = cfg["seed"]
    base = f"{model_name}_img{img_size}_seed{seed}"
    return f"{base}_{suffix}" if suffix else base


def cosine_warmup_lr_lambda(current_step: int, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    if total_steps <= 0:
        return 1.0
    if current_step < warmup_steps:
        return float(current_step) / max(1, warmup_steps)
    progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def save_checkpoint(state: Dict, checkpoint_path: str | Path) -> None:
    torch.save(state, checkpoint_path)


def to_python_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def mean_of_list(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))
