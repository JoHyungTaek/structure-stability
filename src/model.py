from __future__ import annotations

import timm
import torch
import torch.nn as nn


class CrossViewAttentionFusion(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.front_to_top = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.top_to_front = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, front_tokens: torch.Tensor, top_tokens: torch.Tensor):
        f2t, _ = self.front_to_top(front_tokens, top_tokens, top_tokens)
        t2f, _ = self.top_to_front(top_tokens, front_tokens, front_tokens)
        front_tokens = self.norm1(front_tokens + f2t)
        top_tokens = self.norm2(top_tokens + t2f)
        return front_tokens, top_tokens


class MultiViewStabilityModel(nn.Module):
    def __init__(
        self,
        backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.25,
        attention_heads: int = 8,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 320, 320)
            out = self.backbone(dummy)
            if out.ndim != 4:
                raise ValueError("This implementation expects a CNN-style backbone returning [B,C,H,W].")
            feature_dim = out.shape[1]

        self.token_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fusion = CrossViewAttentionFusion(hidden_dim, heads=attention_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if feat.ndim == 2:
            return feat.unsqueeze(1)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)
        return tokens

    def forward(self, front: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        front_tokens = self._encode(front)
        top_tokens = self._encode(top)

        front_tokens, top_tokens = self.fusion(front_tokens, top_tokens)

        front_vec = front_tokens.mean(dim=1)
        top_vec = top_tokens.mean(dim=1)
        diff = torch.abs(front_vec - top_vec)
        prod = front_vec * top_vec
        summ = front_vec + top_vec
        maximum = torch.maximum(front_vec, top_vec)

        fused = torch.cat([front_vec, top_vec, diff, prod, summ, maximum], dim=1)
        fused = self.dropout(fused)
        return self.head(fused).squeeze(1)
