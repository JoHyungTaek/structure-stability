import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from configs.config import CFG, STUDENT_MODEL_DIR, SUBMISSION_PATH
from src.dataset import StudentDataset
from src.model import StudentImageOnlyModel
from src.transforms import get_student_valid_transform, get_student_tta_transform


def make_test_loader(transform):
    base_path = CFG["BASE_PATH"]
    sample_sub = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))

    dataset = StudentDataset(
        df=sample_sub,
        image_root=os.path.join(base_path, "test"),
        transform=transform,
        is_test=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=CFG["STUDENT_BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
    )
    return sample_sub, loader


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_sub, loader_org = make_test_loader(
        get_student_valid_transform(CFG["STUDENT_IMG_SIZE"])
    )

    loader_flip = None
    if CFG["TTA"]:
        _, loader_flip = make_test_loader(
            get_student_tta_transform(CFG["STUDENT_IMG_SIZE"])
        )

    preds_all = []

    for seed in CFG["SEEDS"][:CFG["N_SEEDS"]]:
        ckpt = os.path.join(STUDENT_MODEL_DIR, f"student_seed{seed}.pth")
        if not os.path.exists(ckpt):
            print(f"Skip seed {seed}: {ckpt} not found")
            continue

        model = StudentImageOnlyModel(
            model_name=CFG["STUDENT_MODEL_NAME"],
            dropout=CFG["STUDENT_DROPOUT"]
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        pred_org = predict(model, loader_org, device)
        if CFG["TTA"] and loader_flip is not None:
            pred_flip = predict(model, loader_flip, device)
            pred = (pred_org + pred_flip) / 2.0
        else:
            pred = pred_org

        preds_all.append(pred)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    if len(preds_all) == 0:
        raise FileNotFoundError("student checkpoint가 없습니다. 먼저 train_student.py를 실행하세요.")

    preds = np.mean(preds_all, axis=0)
    preds = np.clip(preds, 1e-7, 1 - 1e-7)

    submission = sample_sub.copy()
    submission["unstable_prob"] = preds
    submission["stable_prob"] = 1.0 - preds
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"{SUBMISSION_PATH} saved")


if __name__ == "__main__":
    main()