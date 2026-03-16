from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from .physics import load_geometry_features

LABEL_MAP = {'unstable': 1, 'stable': 0, 1: 1, 0: 0, '1': 1, '0': 0}


def _find_csv(data_root: Path, filename: str) -> Path:
    path = data_root / filename
    if path.exists():
        return path
    raise FileNotFoundError(f'Missing required file: {path}')


def _normalize_label(v):
    if pd.isna(v):
        return None
    if v in LABEL_MAP:
        return LABEL_MAP[v]
    if isinstance(v, str):
        v = v.strip().lower()
        if v in LABEL_MAP:
            return LABEL_MAP[v]
    raise ValueError(f'Unknown label value: {v}')


def load_split_dataframe(data_root: str | Path, split: str, motion_csv: str | Path | None = None) -> pd.DataFrame:
    data_root = Path(data_root)
    if split == 'train':
        df = pd.read_csv(_find_csv(data_root, 'train.csv'))
        df['split'] = 'train'
    elif split == 'dev':
        df = pd.read_csv(_find_csv(data_root, 'dev.csv'))
        df['split'] = 'dev'
    elif split == 'test':
        df = pd.read_csv(_find_csv(data_root, 'sample_submission.csv'))[['id']].copy()
        df['split'] = 'test'
    else:
        raise ValueError(split)

    df['id'] = df['id'].astype(str)
    if 'label' in df.columns:
        df['target'] = df['label'].apply(_normalize_label)
    else:
        df['target'] = None

    df['front_path'] = df['id'].apply(lambda x: str(data_root / split / x / 'front.png'))
    df['top_path'] = df['id'].apply(lambda x: str(data_root / split / x / 'top.png'))
    df['video_path'] = df['id'].apply(lambda x: str(data_root / split / x / 'simulation.mp4'))
    df['source_domain'] = 0 if split == 'train' else (1 if split == 'dev' else -1)

    if motion_csv is not None and split == 'train' and Path(motion_csv).exists():
        motion_df = pd.read_csv(motion_csv)
        motion_df['id'] = motion_df['id'].astype(str)
        df = df.merge(motion_df, on='id', how='left', suffixes=('', '_motion'))
    else:
        for col in ['max_diff_first', 'mean_diff_first', 'max_diff_prev', 'mean_diff_prev', 'first_move_thr2', 'first_move_thr5', 'first_move_thr10', 'severity_bucket', 'onset_bucket', 'soft_target']:
            df[col] = pd.NA
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

        if self.transform is not None:
            transformed = self.transform(image=front, image_top=top)
            front = transformed['image']
            top = transformed['image_top']
        else:
            raise ValueError('transform is required')

        geom_feat = torch.tensor(load_geometry_features(row.front_path, row.top_path), dtype=torch.float32)

        sample = {
            'id': row.id,
            'front': front,
            'top': top,
            'geom_feat': geom_feat,
            'source_domain': torch.tensor(int(getattr(row, 'source_domain', -1)), dtype=torch.long),
        }
        if not self.test_mode:
            sample['target'] = torch.tensor(float(row.target), dtype=torch.float32)
            sample['soft_target'] = torch.tensor(float(row.soft_target) if pd.notna(row.soft_target) else float('nan'), dtype=torch.float32)
            sample['max_diff_first'] = torch.tensor(float(row.max_diff_first) if pd.notna(row.max_diff_first) else float('nan'), dtype=torch.float32)
            sample['mean_diff_prev'] = torch.tensor(float(row.mean_diff_prev) if pd.notna(row.mean_diff_prev) else float('nan'), dtype=torch.float32)
            sample['severity_bucket'] = torch.tensor(int(row.severity_bucket) if pd.notna(row.severity_bucket) else -1, dtype=torch.long)
            sample['onset_bucket'] = torch.tensor(int(row.onset_bucket) if pd.notna(row.onset_bucket) else -1, dtype=torch.long)
        return sample
