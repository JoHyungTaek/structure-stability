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


# train.py가 build_train_transform(image_size, cfg) 형태로 호출하므로 여기에 맞춤
def build_train_transform(image_size, cfg):
    aug_cfg = cfg["augment"]

    return A.Compose([
        A.Resize(image_size, image_size),

        A.HorizontalFlip(p=aug_cfg["hflip"]),

        A.ColorJitter(
            brightness=aug_cfg["color_jitter"],
            contrast=aug_cfg["color_jitter"],
            saturation=aug_cfg["color_jitter"],
            hue=min(0.1, aug_cfg["color_jitter"] / 4),
            p=0.6,
        ),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
        ], p=aug_cfg["blur"]),

        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            rotate=(-12, 12),
            p=aug_cfg["shift_scale_rotate"],
        ),

        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(0.05, 0.15),
            hole_width_range=(0.05, 0.15),
            fill=0,
            p=aug_cfg["coarse_dropout"],
        ),

        *_normalize_and_tensor(),
    ])


# train.py / inference.py가 build_valid_transform(image_size) 형태로 호출하므로 여기에 맞춤
def build_valid_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        *_normalize_and_tensor(),
    ])


# 나중에 inference 쪽 TTA에서 쓸 수 있게 같이 맞춰둠
def build_tta_transform(image_size, tta_name="none"):
    transforms = [
        A.Resize(image_size, image_size),
    ]

    if tta_name == "hflip":
        transforms.append(A.HorizontalFlip(p=1.0))

    transforms.extend(_normalize_and_tensor())
    return A.Compose(transforms)


# 호환용 alias
def build_train_transforms(cfg):
    return build_train_transform(cfg["model"]["image_size"], cfg)


def build_valid_transforms(cfg):
    return build_valid_transform(cfg["model"]["image_size"])


def build_tta_transforms(cfg, tta_name="none"):
    return build_tta_transform(cfg["model"]["image_size"], tta_name)