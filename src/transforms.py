from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _normalize_and_tensor():
    return [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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
                p=aug_cfg.get("shift_scale_rotate", 0.55),
                border_mode=0,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.22,
                contrast_limit=0.22,
                p=0.7,
            ),

            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=12,
                val_shift_limit=8,
                p=0.35,
            ),

            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
                ],
                p=aug_cfg.get("blur", 0.18),
            ),

            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.04, 0.12),
                hole_width_range=(0.04, 0.12),
                fill=0,
                p=aug_cfg.get("coarse_dropout", 0.12),
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
    if mode == "vflip":
        return front.flip(-2), top.flip(-2)
    if mode == "transpose":
        return front.transpose(-1, -2), top.transpose(-1, -2)
    raise ValueError(f"Unknown TTA mode: {mode}")