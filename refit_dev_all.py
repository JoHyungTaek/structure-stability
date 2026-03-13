import os
import gc
import warnings

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.config import (
    CFG,
    FINETUNE_MODEL_PATH,
    FINAL_MODEL_PATH,
)
from src.dataset import MultiViewDataset
from src.model import MultiViewClassifier
from src.utils import seed_everything, label_to_int

warnings.filterwarnings("ignore")


def get_train_transform():
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
            brightness_limit=0.18,
            contrast_limit=0.18,
            p=0.5
        ),
        A.GaussNoise(p=0.2),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


def build_loader():
    base_path = CFG["BASE_PATH"]
    dev_df = pd.read_csv(os.path.join(base_path, "dev.csv"))
    dev_df["label"] = dev_df["label"].apply(label_to_int)

    dataset = MultiViewDataset(
        df=dev_df,
        image_root=os.path.join(base_path, "dev"),
        transform=get_train_transform(),
        is_test=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=CFG["REFIT_BATCH_SIZE"],
        shuffle=True,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
    )
    return loader


def main():
    seed_everything(CFG["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiViewClassifier(
        model_name=CFG["MODEL_NAME"],
        dropout=CFG["DROPOUT"]
    ).to(device)

    model.load_state_dict(torch.load(FINETUNE_MODEL_PATH, map_location=device))

    loader = build_loader()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["REFIT_LR"],
        weight_decay=CFG["REFIT_WEIGHT_DECAY"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=CFG["AMP"])

    model.train()
    for epoch in range(1, CFG["REFIT_EPOCHS"] + 1):
        losses = []
        for views, labels in tqdm(loader, desc=f"Refit Epoch {epoch}", leave=False):
            views = [v.to(device) for v in views]
            labels = labels.unsqueeze(1).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=CFG["AMP"]):
                logits = model(views)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

        print(f"[Refit Epoch {epoch}/{CFG['REFIT_EPOCHS']}] Loss: {sum(losses)/len(losses):.4f}")

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"Final refit model saved -> {FINAL_MODEL_PATH}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()