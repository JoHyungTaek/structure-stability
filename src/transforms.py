import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_teacher_train_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.06,
            scale_limit=0.10,
            rotate_limit=15,
            border_mode=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.18,
            contrast_limit=0.18,
            p=0.5,
        ),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_teacher_valid_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_student_train_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.25),
        A.RandomRotate90(p=0.35),
        A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.12,
            rotate_limit=18,
            border_mode=0,
            p=0.55,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.22,
            contrast_limit=0.22,
            p=0.6,
        ),
        A.GaussNoise(p=0.2),
        A.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.08,
            p=0.35,
        ),
        A.CoarseDropout(
            max_holes=6,
            max_height=img_size // 12,
            max_width=img_size // 12,
            p=0.25,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_student_valid_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_student_tta_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])