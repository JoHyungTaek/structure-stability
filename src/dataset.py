from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetSpec:
    csv_path: Path
    image_root: Path
    split_name: str


class MultiViewDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str | Path,
        transform=None,
        is_test: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.transform = transform
        self.is_test = is_test
        self.id_col = self._find_id_col()
        self.label_col = None if is_test else self._find_label_col()

    def _find_id_col(self) -> str:
        cols = list(self.df.columns)
        lower_map = {c.lower(): c for c in cols}
        for cand in ["id", "sample_id"]:
            if cand in lower_map:
                return lower_map[cand]
        return cols[0]

    def _find_label_col(self) -> str:
        cols = list(self.df.columns)
        lower_map = {c.lower(): c for c in cols}
        for cand in ["label", "target", "unstable"]:
            if cand in lower_map:
                return lower_map[cand]
        for c in cols:
            if c != self.id_col:
                return c
        raise ValueError("No label column found")

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _label_to_float(label) -> float:
        if isinstance(label, str):
            value = label.strip().lower()
            if value == "stable":
                return 0.0
            if value == "unstable":
                return 1.0
        return float(label)

    @staticmethod
    def _load_image(path: str | Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _resolve_paths(self, sample_id: str) -> tuple[Path, Path]:
        front_path = self.image_root / sample_id / "front.png"
        top_path = self.image_root / sample_id / "top.png"
        return front_path, top_path

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sample_id = str(row[self.id_col])
        front_path, top_path = self._resolve_paths(sample_id)

        front_img = self._load_image(front_path)
        top_img = self._load_image(top_path)

        if self.transform is not None:
            front_img = self.transform(image=front_img)["image"]
            top_img = self.transform(image=top_img)["image"]

        item = {
            "id": sample_id,
            "front": front_img,
            "top": top_img,
        }

        if not self.is_test:
            label = self._label_to_float(row[self.label_col])
            item["target"] = torch.tensor(label, dtype=torch.float32)

        return item


def build_dataset_specs(cfg: dict) -> dict[str, DatasetSpec]:
    root = Path(cfg["paths"]["data_root"])
    specs = {
        "train": DatasetSpec(
            csv_path=root / cfg["paths"]["train_csv"],
            image_root=root / cfg["paths"]["train_image_dir"],
            split_name="train",
        ),
        "dev": DatasetSpec(
            csv_path=root / cfg["paths"]["dev_csv"],
            image_root=root / cfg["paths"]["dev_image_dir"],
            split_name="dev",
        ),
        "test": DatasetSpec(
            csv_path=root / cfg["paths"]["test_csv"],
            image_root=root / cfg["paths"]["test_image_dir"],
            split_name="test",
        ),
    }
    return specs


def read_split_dataframe(spec: DatasetSpec) -> pd.DataFrame:
    if not spec.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {spec.csv_path}")
    return pd.read_csv(spec.csv_path)
