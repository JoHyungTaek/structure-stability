from __future__ import annotations

import timm
import torch
import torch.nn as nn


class DualViewPhysicsModel(nn.Module):
    def __init__(
        self,
        backbone: str = 'convnextv2_base.fcmae_ft_in22k_in1k',
        pretrained: bool = True,
        hidden_dim: int = 384,
        dropout: float = 0.25,
        attention_heads: int = 8,
        geom_dim: int = 17,
    ):
        super().__init__()
        self.front_backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='avg')
        self.top_backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='avg')

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 384, 384)
            feat_dim = int(self.front_backbone(dummy).shape[-1])

        self.front_proj = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.top_proj = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.geom_proj = nn.Sequential(nn.LayerNorm(geom_dim), nn.Linear(geom_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.5))

        self.view_embed = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=2)
        fused_dim = hidden_dim * 6 + hidden_dim // 2
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )
        self.motion_head = nn.Sequential(nn.Linear(fused_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))
        self.onset_head = nn.Sequential(nn.Linear(fused_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4))
        self.severity_head = nn.Sequential(nn.Linear(fused_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4))

    def _encode(self, backbone, proj, x: torch.Tensor) -> torch.Tensor:
        feat = backbone(x)
        return proj(feat)

    def forward(self, front: torch.Tensor, top: torch.Tensor, geom_feat: torch.Tensor):
        front_vec = self._encode(self.front_backbone, self.front_proj, front)
        top_vec = self._encode(self.top_backbone, self.top_proj, top)
        tokens = torch.stack([front_vec + self.view_embed[0], top_vec + self.view_embed[1]], dim=1)
        fused_tokens = self.fusion(tokens)
        fused_mean = fused_tokens.mean(dim=1)
        diff = torch.abs(front_vec - top_vec)
        prod = front_vec * top_vec
        maximum = torch.maximum(front_vec, top_vec)
        geom_emb = self.geom_proj(geom_feat)
        fused = torch.cat([front_vec, top_vec, fused_mean, diff, prod, maximum, geom_emb], dim=1)
        fused = self.dropout(fused)
        return {
            'logit': self.classifier(fused).squeeze(1),
            'motion_reg': self.motion_head(fused),
            'onset_logit': self.onset_head(fused),
            'severity_logit': self.severity_head(fused),
        }
