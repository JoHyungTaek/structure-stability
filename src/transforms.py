from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


class TopRectify(A.ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        from .physics import rectify_top_image
        return rectify_top_image(img)


def _normalize_and_tensor():
    return [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]


def _center_crop_front_and_top():
    return [
        A.Crop(x_min=140, y_min=110, x_max=500, y_max=610),
        A.Resize(384, 384),
    ]


def build_train_transform(image_size, cfg):
    aug_cfg = cfg['augment']
    return A.Compose(
        [
            A.OneOf([
                A.NoOp(),
                A.Lambda(name='top_rectify_noop'),
            ], p=1.0),
            A.Crop(x_min=140, y_min=110, x_max=500, y_max=610),
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=aug_cfg.get('hflip', 0.5)),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.08,
                rotate_limit=10,
                border_mode=0,
                p=aug_cfg.get('shift_scale_rotate', 0.5),
            ),
            A.RandomBrightnessContrast(brightness_limit=0.22, contrast_limit=0.22, p=0.7),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=8, p=0.35),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
            ], p=aug_cfg.get('blur', 0.18)),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.04, 0.12),
                hole_width_range=(0.04, 0.12),
                fill=0,
                p=aug_cfg.get('coarse_dropout', 0.15),
            ),
            *_normalize_and_tensor(),
        ],
        additional_targets={'image_top': 'image'},
    )


def build_valid_transform(image_size):
    return A.Compose(
        [
            A.Crop(x_min=140, y_min=110, x_max=500, y_max=610),
            A.Resize(image_size, image_size),
            *_normalize_and_tensor(),
        ],
        additional_targets={'image_top': 'image'},
    )


def apply_tta(front, top, mode='none'):
    if mode == 'none':
        return front, top
    if mode == 'hflip':
        return front.flip(-1), top.flip(-1)
    if mode == 'vflip':
        return front.flip(-2), top.flip(-2)
    if mode == 'transpose':
        return front.transpose(-1, -2), top.transpose(-1, -2)
    raise ValueError(f'Unknown TTA mode: {mode}')
