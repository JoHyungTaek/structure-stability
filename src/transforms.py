from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _normalize_and_tensor():
    return [
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ]


def build_train_transform(image_size, cfg):
    aug_cfg = cfg["augment"]

    return A.Compose(
        [
            A.Resize(image_size, image_size),

            A.HorizontalFlip(p=aug_cfg.get("hflip", 0.5)),

            A.Affine(
                scale=(0.92, 1.08),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-10, 10),
                shear=(-3, 3),
                p=aug_cfg.get("shift_scale_rotate", 0.45),
                border_mode=0,
            ),

            A.ColorJitter(
                brightness=aug_cfg.get("color_jitter", 0.25),
                contrast=aug_cfg.get("color_jitter", 0.25),
                saturation=aug_cfg.get("color_jitter", 0.25),
                hue=min(0.08, aug_cfg.get("color_jitter", 0.25) / 4),
                p=0.55,
            ),

            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
                ],
                p=aug_cfg.get("blur", 0.12),
            ),

            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.04, 0.10),
                hole_width_range=(0.04, 0.10),
                fill=0,
                p=aug_cfg.get("coarse_dropout", 0.08),
            ),

            *_normalize_and_tensor(),
        ],
        additional_targets={"image_top": "image"},
    )


def build_valid_transform(image_size):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            *_normalize_and_tensor(),
        ],
        additional_targets={"image_top": "image"},
    )


def apply_tta(front, top, mode="none"):
    if mode == "none":
        return front, top
    if mode == "hflip":
        return front.flip(-1), top.flip(-1)
    raise ValueError(f"Unknown TTA mode: {mode}")


def build_train_transforms(cfg):
    return build_train_transform(cfg["model"]["image_size"], cfg)


def build_valid_transforms(cfg):
    return build_valid_transform(cfg["model"]["image_size"])


def build_tta_transforms(cfg, tta_name="none"):
    return build_valid_transform(cfg["model"]["image_size"])