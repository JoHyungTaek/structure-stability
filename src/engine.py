from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from .utils import AverageMeter, compute_binary_logloss, cosine_warmup_lr_lambda, sigmoid_np


@dataclass
class EpochResult:
    loss: float
    logloss: float
    y_true: np.ndarray
    y_pred: np.ndarray
    ids: List[str]


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.clamp(self.temperature, min=0.05, max=10.0)
        return logits / temperature

    def fit(self, logits: np.ndarray, labels: np.ndarray, device: torch.device, max_iter: int = 200) -> float:
        self.to(device)
        logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
        labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits_t), labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(torch.clamp(self.temperature.detach(), min=0.05, max=10.0).item())


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone"):
            encoder_params.append(param)
        else:
            head_params.append(param)

    params = [
        {"params": encoder_params, "lr": cfg["train"]["encoder_lr"]},
        {"params": head_params, "lr": cfg["train"]["lr"]},
    ]
    return torch.optim.AdamW(params, weight_decay=cfg["train"]["weight_decay"])


def build_scheduler(optimizer, steps_per_epoch: int, epochs: int, cfg: dict):
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * cfg["train"]["scheduler"].get("warmup_ratio", 0.1))
    base_lr = cfg["train"]["lr"]
    min_lr = cfg["train"]["scheduler"].get("min_lr", 1e-6)
    min_ratio = min_lr / max(base_lr, 1e-12)

    lr_lambda = lambda step: cosine_warmup_lr_lambda(step, total_steps, warmup_steps, min_ratio)
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _iterate_batches(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    optimizer=None,
    scheduler=None,
    scaler=None,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 5.0,
    use_amp: bool = True,
    print_prefix: str = "train",
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)
    losses = AverageMeter()
    all_logits = []
    all_targets = []
    all_ids = []

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc=print_prefix, leave=False)
    for step, batch in enumerate(progress):
        front = batch["front"].to(device, non_blocking=True)
        top = batch["top"].to(device, non_blocking=True)
        ids = batch["id"]
        targets = batch.get("target")
        if targets is not None:
            targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                logits = model(front, top)
                loss = criterion(logits, targets) if targets is not None else None
                if loss is not None and is_train and grad_accum_steps > 1:
                    loss = loss / grad_accum_steps

            if is_train:
                scaler.scale(loss).backward()
                if (step + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

        probs = torch.sigmoid(logits.detach()).cpu().numpy()
        all_logits.append(logits.detach().cpu().numpy())
        all_ids.extend(list(ids))
        if targets is not None:
            target_np = targets.detach().cpu().numpy()
            all_targets.append(target_np)
            display_loss = float(loss.detach().cpu().item()) * (grad_accum_steps if is_train and grad_accum_steps > 1 else 1)
            losses.update(display_loss, len(ids))
            progress.set_postfix(loss=f"{losses.avg:.5f}")

    logits_np = np.concatenate(all_logits) if all_logits else np.array([])
    if all_targets:
        y_true = np.concatenate(all_targets)
        y_pred = sigmoid_np(logits_np)
        ll = compute_binary_logloss(y_true, y_pred)
    else:
        y_true = np.array([])
        y_pred = sigmoid_np(logits_np)
        ll = float("nan")

    return EpochResult(
        loss=float(losses.avg),
        logloss=ll,
        y_true=y_true,
        y_pred=y_pred,
        ids=all_ids,
    )


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, cfg, prefix="train"):
    return _iterate_batches(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        grad_accum_steps=cfg["train"].get("grad_accum_steps", 1),
        max_grad_norm=cfg["train"].get("max_grad_norm", 5.0),
        use_amp=cfg.get("use_amp", True),
        print_prefix=prefix,
    )


def valid_one_epoch(model, loader, criterion, device, cfg, prefix="valid"):
    return _iterate_batches(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scheduler=None,
        scaler=torch.cuda.amp.GradScaler(enabled=False),
        grad_accum_steps=1,
        max_grad_norm=0.0,
        use_amp=cfg.get("use_amp", True),
        print_prefix=prefix,
    )


def predict_logits(model, loader, device, cfg, transforms=None) -> Tuple[np.ndarray, list[str]]:
    model.eval()
    logits_list = []
    ids_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="predict", leave=False):
            front = batch["front"].to(device, non_blocking=True)
            top = batch["top"].to(device, non_blocking=True)
            logits = model(front, top)
            logits_list.append(logits.detach().cpu().numpy())
            ids_list.extend(list(batch["id"]))

    logits_np = np.concatenate(logits_list) if logits_list else np.array([])
    return logits_np, ids_list
