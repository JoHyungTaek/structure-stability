from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss
from tqdm import tqdm


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(1, self.count)

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n


class BinarySmoothBCE(nn.Module):
    def __init__(self, smoothing=0.0, pos_weight=None):
        super().__init__()
        self.smoothing = float(smoothing)
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        targets = targets.float()
        if self.smoothing > 0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing

        if self.pos_weight is not None:
            return F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=self.pos_weight.to(logits.device),
            )
        return F.binary_cross_entropy_with_logits(logits, targets)


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits, targets, device="cpu"):
        device = torch.device(device)
        self.to(device)

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


def _prob_from_logits(logits):
    return 1.0 / (1.0 + np.exp(-logits))


def run_train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scheduler,
    device,
    mixed_precision=True,
    grad_accum_steps=1,
    gradient_clip=1.0,
):
    model.train()
    meter = AverageMeter()

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    use_amp = bool(mixed_precision and device_type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    all_logits = []
    all_targets = []

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, total=len(loader))
    for step, batch in enumerate(pbar):
        front = batch["front"].to(device, non_blocking=(device_type == "cuda"))
        top = batch["top"].to(device, non_blocking=(device_type == "cuda"))
        target = batch["target"].float().to(device, non_blocking=(device_type == "cuda"))

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(front, top)
            loss = criterion(logits, target) / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if gradient_clip is not None:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

        meter.update(loss.item() * grad_accum_steps, front.size(0))
        all_logits.append(logits.detach().cpu())
        all_targets.append(target.detach().cpu())

        pbar.set_postfix(loss=f"{meter.avg:.4f}")

    all_logits = torch.cat(all_logits).numpy()
    all_targets = torch.cat(all_targets).numpy()
    ll = log_loss(all_targets, _prob_from_logits(all_logits), labels=[0, 1])

    return SimpleNamespace(
        loss=meter.avg,
        logloss=ll,
        logits=all_logits,
        y_true=all_targets,
    )


@torch.no_grad()
def run_valid_epoch(model, loader, criterion, device):
    model.eval()
    meter = AverageMeter()

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    all_logits = []
    all_targets = []

    for batch in tqdm(loader, total=len(loader)):
        front = batch["front"].to(device, non_blocking=(device_type == "cuda"))
        top = batch["top"].to(device, non_blocking=(device_type == "cuda"))
        target = batch["target"].float().to(device, non_blocking=(device_type == "cuda"))

        logits = model(front, top)
        loss = criterion(logits, target)

        meter.update(loss.item(), front.size(0))
        all_logits.append(logits.detach().cpu())
        all_targets.append(target.detach().cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_targets = torch.cat(all_targets).numpy()
    ll = log_loss(all_targets, _prob_from_logits(all_logits), labels=[0, 1])

    result = SimpleNamespace(
        loss=meter.avg,
        logloss=ll,
        logits=all_logits,
        y_true=all_targets,
    )
    return result, all_logits