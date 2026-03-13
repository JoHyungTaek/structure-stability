import torch
import torch.nn as nn
import timm


class TeacherMultiModalModel(nn.Module):
    def __init__(self, model_name="efficientnet_b3", dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.feature_dim = self.backbone.num_features

        fusion_dim = self.feature_dim * 5

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def encode(self, x):
        return self.backbone(x)

    def forward(self, views, video_frames):
        front_feat = self.encode(views[0])
        top_feat = self.encode(views[1])

        b, t, c, h, w = video_frames.shape
        video_frames = video_frames.view(b * t, c, h, w)
        video_feat = self.encode(video_frames).view(b, t, self.feature_dim)

        video_mean = video_feat.mean(dim=1)
        video_max = video_feat.max(dim=1).values
        video_motion = torch.abs(video_feat[:, -1, :] - video_feat[:, 0, :])

        fused = torch.cat(
            [front_feat, top_feat, video_mean, video_max, video_motion],
            dim=1
        )
        return self.classifier(fused)


class StudentImageOnlyModel(nn.Module):
    def __init__(self, model_name="convnext_small", dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, views):
        front_feat = self.backbone(views[0])
        top_feat = self.backbone(views[1])
        fused = torch.cat([front_feat, top_feat], dim=1)
        return self.classifier(fused)