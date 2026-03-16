import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import log_loss


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def label_to_int(x):
    if isinstance(x, str):
        x = x.strip().lower()
        if x == "stable":
            return 0
        if x == "unstable":
            return 1
    return int(float(x))


def clip_probs(probs, clip_min=1e-6, clip_max=1 - 1e-6):
    return np.clip(np.asarray(probs, dtype=np.float64), clip_min, clip_max)


def binary_logloss(y_true, probs, clip_min=1e-6, clip_max=1 - 1e-6):
    probs = clip_probs(probs, clip_min, clip_max)
    return log_loss(np.asarray(y_true).astype(int), probs)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, labels, clip_min=1e-6, clip_max=1 - 1e-6):
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)

        def objective(x):
            t = float(np.clip(x[0], 0.05, 10.0))
            probs = 1.0 / (1.0 + np.exp(-(logits / t)))
            probs = np.clip(probs, clip_min, clip_max)
            return log_loss(labels, probs)

        res = minimize(objective, x0=np.array([1.0]), method="Nelder-Mead")
        self.temperature = float(np.clip(res.x[0], 0.05, 10.0))
        return self.temperature

    def predict_proba(self, logits, clip_min=1e-6, clip_max=1 - 1e-6):
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-(logits / self.temperature)))
        return np.clip(probs, clip_min, clip_max)
