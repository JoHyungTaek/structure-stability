import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.config import (
    CFG,
    FINAL_MODEL_PATH,
    FINETUNE_MODEL_PATH,
    SUBMISSION_PATH,
)
from src.dataset import MultiViewDataset
from src.model import MultiViewClassifier


def get_test_transform():
    size = CFG["IMG_SIZE"]
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


def get_test_transform_hflip():
    size = CFG["IMG_SIZE"]
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


def make_test_loader(test_df, transform):
    dataset = MultiViewDataset(
        df=test_df,
        image_root=os.path.join(CFG["BASE_PATH"], "test"),
        transform=transform,
        is_test=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=CFG["REFIT_BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
    )
    return loader


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []

    for views in tqdm(loader, desc="Inference", leave=False):
        views = [v.to(device) for v in views]
        logits = model(views)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        preds.extend(probs.tolist())

    return np.array(preds, dtype=np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_sub = pd.read_csv(os.path.join(CFG["BASE_PATH"], "sample_submission.csv"))

    ckpt_path = FINAL_MODEL_PATH if os.path.exists(FINAL_MODEL_PATH) else FINETUNE_MODEL_PATH

    model = MultiViewClassifier(
        model_name=CFG["MODEL_NAME"],
        dropout=CFG["DROPOUT"]
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded model -> {ckpt_path}")

    loader_org = make_test_loader(sample_sub, get_test_transform())
    preds_org = predict(model, loader_org, device)

    if CFG["TTA"]:
        loader_flip = make_test_loader(sample_sub, get_test_transform_hflip())
        preds_flip = predict(model, loader_flip, device)
        preds = (preds_org + preds_flip) / 2.0
    else:
        preds = preds_org

    preds = np.clip(preds, 1e-7, 1 - 1e-7)

    submission = sample_sub.copy()
    submission["unstable_prob"] = preds
    submission["stable_prob"] = 1.0 - preds
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"{SUBMISSION_PATH} 생성 완료")


if __name__ == "__main__":
    main()