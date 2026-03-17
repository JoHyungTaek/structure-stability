from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


LABEL_MAP = {"unstable": 1, "stable": 0, 1: 1, 0: 0, "1": 1, "0": 0}


def _find_csv(data_root: Path, filename: str) -> Path:
    path = data_root / filename
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing required file: {path}")


def _normalize_label(v):
    if pd.isna(v):
        return None
    if v in LABEL_MAP:
        return LABEL_MAP[v]
    if isinstance(v, str):
        v = v.strip().lower()
        if v in LABEL_MAP:
            return LABEL_MAP[v]
    raise ValueError(f"Unknown label value: {v}")


def load_split_dataframe(data_root: str | Path, split: str) -> pd.DataFrame:
    data_root = Path(data_root)

    if split == "train":
        df = pd.read_csv(_find_csv(data_root, "train.csv"))
        df["split"] = "train"
    elif split == "dev":
        df = pd.read_csv(_find_csv(data_root, "dev.csv"))
        df["split"] = "dev"
    elif split == "test":
        df = pd.read_csv(_find_csv(data_root, "sample_submission.csv"))
        df = df[["id"]].copy()
        df["split"] = "test"
    else:
        raise ValueError(split)

    df["id"] = df["id"].astype(str)

    if "label" in df.columns:
        df["target"] = df["label"].apply(_normalize_label)
    else:
        df["target"] = None

    df["front_path"] = df["id"].apply(lambda x: str(data_root / split / x / "front.png"))
    df["top_path"] = df["id"].apply(lambda x: str(data_root / split / x / "top.png"))
    df["video_path"] = df["id"].apply(lambda x: str(data_root / split / x / "simulation.mp4"))
    return df


class StructureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, test_mode: bool = False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.test_mode = test_mode

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _read_image(path: str):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        front = self._read_image(row.front_path)
        top = self._read_image(row.top_path)

        if self.transform is None:
            raise ValueError("transform is required")

        transformed = self.transform(image=front, image_top=top)
        front = transformed["image"]
        top = transformed["image_top"]

        sample = {
            "id": row.id,
            "front": front,
            "top": top,
        }

        if not self.test_mode:
            sample["target"] = torch.tensor(float(row.target), dtype=torch.float32)

        return sample