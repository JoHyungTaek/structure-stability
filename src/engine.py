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


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits, targets, device='cuda'):
        device = torch.device(device)
        self.to(device)
        logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
        targets_t = torch.tensor(targets, dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.05, max_iter=100)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self(logits_t), targets_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach().cpu().item())


def _prob_from_logits(logits):
    return 1.0 / (1.0 + np.exp(-logits))


def compute_loss(outputs, batch, cfg):
    hard_target = batch['target'].float()
    soft_target = batch['soft_target']
    target = torch.where(torch.isnan(soft_target), hard_target, soft_target)
    loss_main = F.binary_cross_entropy_with_logits(outputs['logit'], target)

    motion_tgt = torch.stack([batch['max_diff_first'], batch['mean_diff_prev']], dim=1)
    valid_motion = ~torch.isnan(motion_tgt).any(dim=1)
    if valid_motion.any():
        pred = outputs['motion_reg'][valid_motion]
        tgt = motion_tgt[valid_motion]
        tgt_norm = torch.stack([tgt[:, 0] / 10.0, tgt[:, 1] / 0.15], dim=1).clamp(min=0.0, max=2.0)
        loss_motion = F.smooth_l1_loss(pred, tgt_norm)
    else:
        loss_motion = outputs['motion_reg'].sum() * 0.0

    onset = batch['onset_bucket']
    valid_onset = onset >= 0
    loss_onset = F.cross_entropy(outputs['onset_logit'][valid_onset], onset[valid_onset]) if valid_onset.any() else outputs['onset_logit'].sum() * 0.0
    sev = batch['severity_bucket']
    valid_sev = sev >= 0
    loss_sev = F.cross_entropy(outputs['severity_logit'][valid_sev], sev[valid_sev]) if valid_sev.any() else outputs['severity_logit'].sum() * 0.0

    loss = (
        loss_main
        + cfg['train'].get('aux_motion_weight', 0.18) * loss_motion
        + cfg['train'].get('aux_onset_weight', 0.08) * loss_onset
        + cfg['train'].get('aux_severity_weight', 0.08) * loss_sev
    )
    metrics = {
        'loss': float(loss.detach().cpu().item()),
        'loss_main': float(loss_main.detach().cpu().item()),
        'loss_motion': float(loss_motion.detach().cpu().item()),
        'loss_onset': float(loss_onset.detach().cpu().item()),
        'loss_sev': float(loss_sev.detach().cpu().item()),
    }
    return loss, metrics


def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def run_train_epoch(model, loader, optimizer, scheduler, device, cfg):
    model.train()
    meter = AverageMeter()
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['train'].get('mixed_precision', True) and device.type == 'cuda')
    all_logits, all_targets = [], []
    optimizer.zero_grad(set_to_none=True)
    grad_accum_steps = int(cfg['train'].get('grad_accum_steps', 1))
    gradient_clip = cfg['train'].get('gradient_clip', 1.0)

    pbar = tqdm(loader, total=len(loader))
    for step, batch in enumerate(pbar):
        batch = _to_device(batch, device)
        with torch.amp.autocast('cuda', enabled=cfg['train'].get('mixed_precision', True) and device.type == 'cuda'):
            outputs = model(batch['front'], batch['top'], batch['geom_feat'])
            loss, metrics = compute_loss(outputs, batch, cfg)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()
        if (step + 1) % grad_accum_steps == 0:
            if gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        meter.update(loss.item() * grad_accum_steps, batch['front'].size(0))
        all_logits.append(outputs['logit'].detach().cpu())
        all_targets.append(batch['target'].detach().cpu())
        pbar.set_postfix(loss=f"{metrics['loss']:.4f}")

    all_logits = torch.cat(all_logits).numpy()
    all_targets = torch.cat(all_targets).numpy()
    ll = log_loss(all_targets, _prob_from_logits(all_logits), labels=[0, 1])
    return SimpleNamespace(loss=meter.avg, logloss=ll, logits=all_logits, y_true=all_targets)


@torch.no_grad()
def run_valid_epoch(model, loader, device, cfg):
    model.eval()
    meter = AverageMeter()
    all_logits, all_targets = [], []

    for batch in tqdm(loader, total=len(loader)):
        batch = _to_device(batch, device)
        outputs = model(batch['front'], batch['top'], batch['geom_feat'])
        loss, _ = compute_loss(outputs, batch, cfg)
        meter.update(loss.item(), batch['front'].size(0))
        all_logits.append(outputs['logit'].detach().cpu())
        all_targets.append(batch['target'].detach().cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_targets = torch.cat(all_targets).numpy()
    ll = log_loss(all_targets, _prob_from_logits(all_logits), labels=[0, 1])
    return SimpleNamespace(loss=meter.avg, logloss=ll, logits=all_logits, y_true=all_targets)
