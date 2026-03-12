import torch
import torch.nn as nn
import timm


class MultiModalStructureModel(nn.Module):
    def __init__(self, model_name="efficientnet_b3", dropout=0.3, num_classes=1):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        feature_dim = self.backbone.num_features
        self.feature_dim = feature_dim

        # front, top, video_mean, video_max, motion
        fusion_dim = feature_dim * 5

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, num_classes),
        )

    def encode_image(self, x):
        return self.backbone(x)

    def forward(self, views, video_frames):
        # views[0], views[1] => [B, C, H, W]
        front_feat = self.encode_image(views[0])
        top_feat = self.encode_image(views[1])

        # video_frames => [B, T, C, H, W]
        b, t, c, h, w = video_frames.shape
        video_frames = video_frames.view(b * t, c, h, w)
        video_feats = self.encode_image(video_frames)           # [B*T, F]
        video_feats = video_feats.view(b, t, self.feature_dim) # [B, T, F]

        video_mean = video_feats.mean(dim=1)
        video_max = video_feats.max(dim=1).values
        motion_feat = torch.abs(video_feats[:, -1, :] - video_feats[:, 0, :])

        combined = torch.cat(
            [front_feat, top_feat, video_mean, video_max, motion_feat],
            dim=1
        )

        logits = self.classifier(combined)
        return logits