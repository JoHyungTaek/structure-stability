import os
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.config import CFG, BASE_PATH, MODEL_SAVE_PATH
from src.dataset import MultiViewDataset
from src.model import MultiViewEfficientNet


def get_test_transform():
    return A.Compose([
        A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


@torch.no_grad()
def inference(model, loader, device):
    model.eval()
    preds = []

    for views in tqdm(loader, desc="Inference", leave=False):
        views = [v.to(device) for v in views]
        outputs = model(views)
        probs = torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1)
        preds.extend(probs.tolist())

    return preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")
    test_df = pd.read_csv(sample_sub_path)

    test_dataset = MultiViewDataset(
        test_df,
        os.path.join(BASE_PATH, "test"),
        transform=get_test_transform(),
        is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True
    )

    model = MultiViewEfficientNet(
        model_name=CFG["MODEL_NAME"],
        dropout=CFG["DROPOUT"]
    ).to(device)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    preds = inference(model, test_loader, device)

    # 두 번째 컬럼에 예측값 넣기
    target_col = test_df.columns[1]
    test_df[target_col] = preds
    test_df.to_csv("submission.csv", index=False)

    print("submission.csv 생성 완료")


if __name__ == "__main__":
    main()