import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.config_b3 import BASE_PATH, SUBMISSION_PATH
from src.dataset import MultiViewDataset
from src.model_b3 import MultiViewB3
from src.model_convnext import MultiViewConvNext


B3_MODEL_PATH = "best_model_b3.pth"
CONVNEXT_MODEL_PATH = "best_model_convnext.pth"
IMG_SIZE = 384
BATCH_SIZE = 8
NUM_WORKERS = 2
TTA = True


def get_test_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_test_transform_hflip():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def make_test_loader(df, transform):
    dataset = MultiViewDataset(
        df=df,
        image_root=os.path.join(BASE_PATH, "test"),
        transform=transform,
        is_test=True
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []

    for views in tqdm(loader, desc="Inference", leave=False):
        views = [v.to(device) for v in views]
        logits = model(views)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        preds.extend(probs.tolist())

    return np.array(preds, dtype=np.float32)


def predict_with_tta(model, df, device):
    loader_org = make_test_loader(df, get_test_transform())
    pred_org = predict(model, loader_org, device)

    if not TTA:
        return pred_org

    loader_flip = make_test_loader(df, get_test_transform_hflip())
    pred_flip = predict(model, loader_flip, device)

    return (pred_org + pred_flip) / 2.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_sub = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))

    preds_all = []

    if os.path.exists(B3_MODEL_PATH):
        model_b3 = MultiViewB3(dropout=0.4).to(device)
        model_b3.load_state_dict(torch.load(B3_MODEL_PATH, map_location=device))
        pred_b3 = predict_with_tta(model_b3, sample_sub, device)
        preds_all.append(pred_b3)

    if os.path.exists(CONVNEXT_MODEL_PATH):
        model_conv = MultiViewConvNext(dropout=0.4).to(device)
        model_conv.load_state_dict(torch.load(CONVNEXT_MODEL_PATH, map_location=device))
        pred_conv = predict_with_tta(model_conv, sample_sub, device)
        preds_all.append(pred_conv)

    if len(preds_all) == 0:
        raise FileNotFoundError("앙상블할 모델 파일이 없습니다.")

    preds = np.mean(preds_all, axis=0)
    preds = np.clip(preds, 1e-7, 1 - 1e-7)

    submission = sample_sub.copy()
    submission["unstable_prob"] = preds
    submission["stable_prob"] = 1.0 - preds
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"{SUBMISSION_PATH} 생성 완료")


if __name__ == "__main__":
    main()