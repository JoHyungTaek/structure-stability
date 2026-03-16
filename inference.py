import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from configs.config import (
    CFG,
    STAGE2_DIR,
    ENSEMBLE_META_PATH,
    FINAL_MODEL_PATH,
    SUBMISSION_PATH,
)
from src.dataset import MultiViewDataset
from src.model import MultiViewClassifier
from src.transforms import get_valid_transform, get_tta_hflip_transform
from src.utils import clip_probs, load_json


def make_test_loader(test_df, transform):
    dataset = MultiViewDataset(
        df=test_df,
        image_root=os.path.join(CFG["BASE_PATH"], "test"),
        transform=transform,
        is_test=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=CFG["STAGE2_FINE_BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
    )
    return loader


@torch.no_grad()
def predict(model, loader, device, temperature=1.0):
    model.eval()
    preds = []
    for views in tqdm(loader, desc="Inference", leave=False):
        views = [v.to(device) for v in views]
        logits = model(views)
        probs = torch.sigmoid(logits / temperature).detach().cpu().numpy().reshape(-1)
        preds.extend(probs.tolist())
    return np.array(preds, dtype=np.float32)


def infer_single_checkpoint(ckpt_path, test_df, device, temperature=1.0):
    model = MultiViewClassifier(model_name=CFG["MODEL_NAME"], dropout=CFG["DROPOUT"]).to(device)
    bundle = torch.load(ckpt_path, map_location=device)
    if isinstance(bundle, dict) and "model_state_dict" in bundle:
        model.load_state_dict(bundle["model_state_dict"])
        temperature = float(bundle.get("temperature", temperature))
    else:
        model.load_state_dict(bundle)

    loader_org = make_test_loader(test_df, get_valid_transform(CFG["IMG_SIZE"]))
    preds_org = predict(model, loader_org, device, temperature=temperature)

    if CFG["TTA"]:
        loader_flip = make_test_loader(test_df, get_tta_hflip_transform(CFG["IMG_SIZE"]))
        preds_flip = predict(model, loader_flip, device, temperature=temperature)
        preds = (preds_org + preds_flip) / 2.0
    else:
        preds = preds_org

    del model
    torch.cuda.empty_cache()
    return preds


def fill_submission(sample_sub, unstable_probs):
    unstable_probs = clip_probs(unstable_probs, CFG["CLIP_MIN"], CFG["CLIP_MAX"])
    stable_probs = 1.0 - unstable_probs
    submission = sample_sub.copy()

    cols_lower = {c.lower(): c for c in submission.columns}

    if CFG["POSITIVE_LABEL_COL"] in submission.columns and CFG["NEGATIVE_LABEL_COL"] in submission.columns:
        submission[CFG["POSITIVE_LABEL_COL"]] = unstable_probs
        submission[CFG["NEGATIVE_LABEL_COL"]] = stable_probs
        return submission

    if "unstable_prob" in cols_lower and "stable_prob" in cols_lower:
        submission[cols_lower["unstable_prob"]] = unstable_probs
        submission[cols_lower["stable_prob"]] = stable_probs
        return submission

    pred_cols = [c for c in submission.columns if c.lower() not in {"id", "sample_id"}]
    if len(pred_cols) == 2:
        col0, col1 = pred_cols[0], pred_cols[1]
        if "stable" in col0.lower() or col0.lower().startswith("0"):
            submission[col0] = stable_probs
            submission[col1] = unstable_probs
        else:
            submission[col0] = unstable_probs
            submission[col1] = stable_probs
        return submission

    if len(pred_cols) == 1:
        submission[pred_cols[0]] = unstable_probs
        return submission

    raise ValueError("sample_submission 컬럼 구조를 자동으로 해석하지 못했습니다. config.py의 POSITIVE_LABEL_COL / NEGATIVE_LABEL_COL을 확인하세요.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_sub = pd.read_csv(os.path.join(CFG["BASE_PATH"], "sample_submission.csv"))

    preds_list = []

    if os.path.exists(ENSEMBLE_META_PATH):
        meta = load_json(ENSEMBLE_META_PATH)
        for fold_info in meta["folds"]:
            ckpt_path = fold_info["finetune_ckpt"]
            temperature = float(fold_info.get("temperature", 1.0))
            print(f"Loaded fold checkpoint -> {ckpt_path} | T={temperature:.4f}")
            preds = infer_single_checkpoint(ckpt_path, sample_sub, device, temperature)
            preds_list.append(preds)
    elif os.path.exists(FINAL_MODEL_PATH):
        print(f"Loaded single model -> {FINAL_MODEL_PATH}")
        preds_list.append(infer_single_checkpoint(FINAL_MODEL_PATH, sample_sub, device, 1.0))
    else:
        raise FileNotFoundError("학습된 모델이 없습니다. 먼저 train_stagewise.py를 실행하세요.")

    preds = np.mean(preds_list, axis=0)
    preds = clip_probs(preds, CFG["CLIP_MIN"], CFG["CLIP_MAX"])

    submission = fill_submission(sample_sub, preds)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"{SUBMISSION_PATH} 생성 완료")


if __name__ == "__main__":
    main()
