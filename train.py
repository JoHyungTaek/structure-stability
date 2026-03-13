import os
import gc
import random
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import log_loss, accuracy_score

from configs.config import CFG, BASE_PATH, MODEL_PATH
from src.dataset import MultiViewDataset
from src.model import MultiViewClassifier

warnings.filterwarnings("ignore")


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_to_int(x):

    if isinstance(x, str):
        x = x.strip().lower()
        if x == "stable":
            return 0
        if x == "unstable":
            return 1

    return int(float(x))


def train_transform():

    size = CFG["IMG_SIZE"]

    return A.Compose([

        A.Resize(size, size),

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.08,
            rotate_limit=10,
            border_mode=0,
            p=0.5
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5
        ),

        A.GaussNoise(p=0.2),

        A.Normalize(
            mean=(0.485,0.456,0.406),
            std=(0.229,0.224,0.225)
        ),

        ToTensorV2()

    ])


def valid_transform():

    size = CFG["IMG_SIZE"]

    return A.Compose([

        A.Resize(size, size),

        A.Normalize(
            mean=(0.485,0.456,0.406),
            std=(0.229,0.224,0.225)
        ),

        ToTensorV2()

    ])


def build_loaders():

    train_df = pd.read_csv(os.path.join(BASE_PATH,"train.csv"))
    dev_df = pd.read_csv(os.path.join(BASE_PATH,"dev.csv"))

    train_df["label"] = train_df["label"].apply(label_to_int)
    dev_df["label"] = dev_df["label"].apply(label_to_int)

    train_ds = MultiViewDataset(
        train_df,
        os.path.join(BASE_PATH,"train"),
        transform=train_transform()
    )

    valid_ds = MultiViewDataset(
        dev_df,
        os.path.join(BASE_PATH,"dev"),
        transform=valid_transform()
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["BATCH_SIZE"],
        shuffle=True,
        num_workers=CFG["NUM_WORKERS"]
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=CFG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG["NUM_WORKERS"]
    )

    return train_loader, valid_loader