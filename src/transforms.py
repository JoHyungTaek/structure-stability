from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(cfg: dict):
    aug = cfg["augment"]["train"]
    img_size = cfg["model"]["image_size"]
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=aug.get("horizontal_flip", 0.5)),
            A.VerticalFlip(p=aug.get("vertical_flip", 0.0)),
            A.ShiftScaleRotate(
                shift_limit=aug.get("shift_limit", 0.04),
                scale_limit=aug.get("scale_limit", 0.08),
                rotate_limit=aug.get("rotate_limit", 8),
                border_mode=0,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1.0),
                    A.HueSaturationValue(p=1.0),
                    A.CLAHE(p=1.0),
                ],
                p=aug.get("color_jitter", 0.5),
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=aug.get("blur", 0.15),
            ),
            A.GaussNoise(p=aug.get("noise", 0.15)),
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(0.03, 0.12),
                hole_width_range=(0.03, 0.12),
                fill=0,
                p=aug.get("coarse_dropout", 0.15),
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_valid_transform(cfg: dict):
    img_size = cfg["model"]["image_size"]
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_tta_transforms(cfg: dict):
    img_size = cfg["model"]["image_size"]
    tta = [build_valid_transform(cfg)]
    valid_cfg = cfg["augment"]["valid"]

    if valid_cfg.get("tta_hflip", True):
        tta.append(
            A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]
            )
        )

    if valid_cfg.get("tta_vflip", False):
        tta.append(
            A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.VerticalFlip(p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]
            )
        )

    return tta
