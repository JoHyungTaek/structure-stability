import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from configs.config import CFG, TEACHER_MODEL_PATH, SOFT_LABEL_PATH
from src.utils import label_to_int
from src.dataset import TeacherTrainDataset
from src.model import TeacherMultiModalModel
from src.transforms import get_teacher_valid_transform


@torch.no_grad()
def infer_teacher(model, loader, device):
    model.eval()
    preds = []

    for views, video_frames, labels in tqdm(loader, desc="Soft Labeling", leave=False):
        views = [v.to(device) for v in views]
        video_frames = video_frames.to(device)

        logits = model(views, video_frames)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        preds.extend(probs.tolist())

    return np.array(preds, dtype=np.float32)


def main():
    base_path = CFG["BASE_PATH"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
    train_df["label"] = train_df["label"].apply(label_to_int)

    dataset = TeacherTrainDataset(
        df=train_df,
        image_root=os.path.join(base_path, "train"),
        transform=get_teacher_valid_transform(CFG["TEACHER_IMG_SIZE"]),
        num_video_frames=CFG["NUM_VIDEO_FRAMES"],
    )

    loader = DataLoader(
        dataset,
        batch_size=CFG["TEACHER_BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
    )

    model = TeacherMultiModalModel(
        model_name=CFG["TEACHER_MODEL_NAME"],
        dropout=CFG["TEACHER_DROPOUT"]
    ).to(device)
    model.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=device))

    preds = infer_teacher(model, loader, device)

    out = train_df[["id", "label"]].copy()
    out["soft_unstable_prob"] = preds
    out.to_csv(SOFT_LABEL_PATH, index=False)
    print(f"Saved -> {SOFT_LABEL_PATH}")


if __name__ == "__main__":
    main()