import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.config import CFG, BASE_PATH, MODEL_DIR, SUBMISSION_PATH
from src.dataset import MultiModalStructureDataset
from src.model import MultiModalStructureModel


def get_test_transform():
    return A.Compose([
        A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_test_transform_hflip():
    return A.Compose([
        A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def make_test_loader(test_df, transform):
    dataset = MultiModalStructureDataset(
        df=test_df,
        image_root=os.path.join(BASE_PATH, "test"),
        transform=transform,
        is_test=True,
        use_video=CFG["USE_VIDEO"],
        num_video_frames=CFG["NUM_VIDEO_FRAMES"],
    )

    loader = DataLoader(
        dataset,
        batch_size=CFG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
        drop_last=False,
    )
    return loader


@torch.no_grad()
def inference_one_model(model, loader, device):
    model.eval()
    preds = []

    for views, video_frames in tqdm(loader, desc="Inference", leave=False):
        views = [v.to(device) for v in views]
        video_frames = video_frames.to(device)

        outputs = model(views, video_frames)
        probs = torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1)
        preds.extend(probs.tolist())

    return np.array(preds, dtype=np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")
    test_df = pd.read_csv(sample_sub_path)

    loader_org = make_test_loader(test_df, get_test_transform())
    loader_flip = make_test_loader(test_df, get_test_transform_hflip()) if CFG["TTA"] else None

    fold_preds = []

    for fold in range(1, CFG["N_SPLITS"] + 1):
        model_path = os.path.join(MODEL_DIR, f"best_fold{fold}.pth")
        if not os.path.exists(model_path):
            print(f"Skip fold {fold}: {model_path} not found")
            continue

        print(f"Load model -> {model_path}")

        model = MultiModalStructureModel(
            model_name=CFG["MODEL_NAME"],
            dropout=CFG["DROPOUT"]
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))

        pred_org = inference_one_model(model, loader_org, device)

        if CFG["TTA"] and loader_flip is not None:
            pred_flip = inference_one_model(model, loader_flip, device)
            pred = (pred_org + pred_flip) / 2.0
        else:
            pred = pred_org

        fold_preds.append(pred)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    if len(fold_preds) == 0:
        raise FileNotFoundError("추론 가능한 fold 모델이 없습니다. 먼저 train.py를 실행하세요.")

    preds = np.mean(fold_preds, axis=0)
    preds = np.clip(preds, 1e-7, 1 - 1e-7)

    submission = test_df.copy()

    if "unstable_prob" in submission.columns and "stable_prob" in submission.columns:
        submission["unstable_prob"] = preds
        submission["stable_prob"] = 1.0 - preds
    else:
        prob_cols = [c for c in submission.columns if c != "id"]
        if len(prob_cols) >= 2:
            submission[prob_cols[0]] = preds
            submission[prob_cols[1]] = 1.0 - preds
        else:
            raise ValueError("sample_submission.csv의 확률 컬럼 구조를 확인하세요.")

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"{SUBMISSION_PATH} 생성 완료")


if __name__ == "__main__":
    main()