from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transform(image_size: int, cfg: dict) -> A.Compose:
    aug = cfg.get("augment", {})
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=aug.get("hflip", 0.5)),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.08,
                rotate_limit=10,
                border_mode=0,
                p=aug.get("shift_scale_rotate", 0.5),
            ),
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15),
                    A.CLAHE(clip_limit=2.0),
                ],
                p=aug.get("color_jitter", 0.4),
            ),
            A.OneOf(
                [A.MotionBlur(blur_limit=3), A.GaussianBlur(blur_limit=(3, 5)), A.MedianBlur(blur_limit=3)],
                p=aug.get("blur", 0.15),
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=int(image_size * 0.08),
                max_width=int(image_size * 0.08),
                min_holes=1,
                fill_value=0,
                p=aug.get("coarse_dropout", 0.2),
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_valid_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def apply_tta(front, top, mode: str):
    if mode == "none":
        return front, top
    if mode == "hflip":
        return front.flip(-1), top.flip(-1)
    raise ValueError(f"Unsupported TTA mode: {mode}")
