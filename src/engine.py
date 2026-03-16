from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.transforms import apply_tta
from src.utils import AverageMeter


@dataclass
class EpochResult:
    loss: float
    logloss: float
    y_true: np.ndarray
    y_prob: np.ndarray


class BinarySmoothBCE(nn.Module):
    def __init__(self, smoothing: float = 0.0, pos_weight: float = 1.0):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing > 0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: np.ndarray, targets: np.ndarray, device: str = "cpu") -> float:
        logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
        targets_t = torch.tensor(targets, dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self(logits_t), targets_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach().cpu().item())


def _batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {"front": batch["front"].to(device), "top": batch["top"].to(device), "id": batch["id"]}
    if "target" in batch:
        out["target"] = batch["target"].to(device)
    return out


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler,
    device: torch.device,
    mixed_precision: bool = True,
    grad_accum_steps: int = 1,
    gradient_clip: float | None = None,
) -> EpochResult:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
    meter = AverageMeter()
    y_true, y_prob = [], []

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, leave=False)
    for step, batch in enumerate(pbar, start=1):
        batch = _batch_to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            logits = model(batch["front"], batch["top"])
            loss = criterion(logits, batch["target"])
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if step % grad_accum_steps == 0:
            if gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = batch["target"].detach().cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(targets.tolist())
        meter.update(loss.item() * grad_accum_steps, len(targets))
        pbar.set_description(f"train loss={meter.avg:.4f}")

    y_true = np.array(y_true)
    y_prob = np.clip(np.array(y_prob), 1e-6, 1 - 1e-6)
    ll = log_loss(y_true, np.vstack([1 - y_prob, y_prob]).T, labels=[0, 1])
    return EpochResult(meter.avg, ll, y_true, y_prob)


def run_valid_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[EpochResult, np.ndarray]:
    model.eval()
    meter = AverageMeter()
    y_true, y_prob, y_logits = [], [], []

    with torch.no_grad():
        pbar = tqdm(loader, leave=False)
        for batch in pbar:
            batch = _batch_to_device(batch, device)
            logits = model(batch["front"], batch["top"])
            loss = criterion(logits, batch["target"])
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            targets = batch["target"].detach().cpu().numpy()
            y_logits.extend(logits.detach().cpu().numpy().tolist())
            y_prob.extend(probs.tolist())
            y_true.extend(targets.tolist())
            meter.update(loss.item(), len(targets))
            pbar.set_description(f"valid loss={meter.avg:.4f}")

    y_true = np.array(y_true)
    y_prob = np.clip(np.array(y_prob), 1e-6, 1 - 1e-6)
    ll = log_loss(y_true, np.vstack([1 - y_prob, y_prob]).T, labels=[0, 1])
    return EpochResult(meter.avg, ll, y_true, y_prob), np.array(y_logits)


def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device, tta_modes: List[str] | None = None) -> np.ndarray:
    model.eval()
    tta_modes = tta_modes or ["none"]
    outputs = []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            batch = _batch_to_device(batch, device)
            logits_per_tta = []
            for mode in tta_modes:
                front, top = apply_tta(batch["front"], batch["top"], mode)
                logits = model(front, top)
                logits_per_tta.append(logits)
            logits = torch.stack(logits_per_tta).mean(dim=0)
            outputs.extend(logits.detach().cpu().numpy().tolist())
    return np.array(outputs)
