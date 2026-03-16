from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


@dataclass
class MotionExtractionConfig:
    resize: Tuple[int, int] = (64, 64)
    thr_low: float = 2.0
    thr_mid: float = 5.0
    thr_high: float = 10.0


def _first_hit(arr: np.ndarray, thr: float) -> int:
    idx = np.where(arr > thr)[0]
    return int(idx[0] + 1) if len(idx) else -1


def _severity_bucket(max_diff_first: float) -> int:
    if max_diff_first < 2.0:
        return 0
    if max_diff_first < 5.0:
        return 1
    if max_diff_first < 10.0:
        return 2
    return 3


def _onset_bucket(first_move_thr2: int, first_move_thr5: int) -> int:
    onset = first_move_thr5 if first_move_thr5 >= 0 else first_move_thr2
    if onset < 0:
        return 3
    if onset < 10:
        return 0
    if onset < 20:
        return 1
    return 2


def _soft_target_from_motion(label_int: int, max_diff_first: float, mean_diff_prev: float) -> float:
    motion_score = 0.65 * min(max_diff_first / 10.0, 1.5) + 0.35 * min(mean_diff_prev / 0.15, 1.5)
    motion_score = min(max(motion_score, 0.0), 1.5)
    if label_int == 0:
        return float(np.clip(0.02 + 0.10 * min(motion_score, 1.0), 0.02, 0.15))
    return float(np.clip(0.65 + 0.30 * min(motion_score, 1.0), 0.65, 0.98))


def extract_motion_targets(data_root: str | Path, out_csv: str | Path, cfg: MotionExtractionConfig | None = None) -> pd.DataFrame:
    cfg = cfg or MotionExtractionConfig()
    data_root = Path(data_root)
    train_df = pd.read_csv(data_root / 'train.csv')
    rows: List[Dict[str, float | int | str]] = []

    for sid, label in tqdm(train_df[['id', 'label']].itertuples(index=False), total=len(train_df), desc='extract-motion'):
        video_path = data_root / 'train' / str(sid) / 'simulation.mp4'
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ok, first = cap.read()
        if not ok:
            cap.release()
            continue

        first_gray = cv2.cvtColor(cv2.resize(first, cfg.resize), cv2.COLOR_BGR2GRAY).astype(np.float32)
        prev_gray = first_gray
        mad_to_first: List[float] = []
        mad_prev: List[float] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(frame, cfg.resize), cv2.COLOR_BGR2GRAY).astype(np.float32)
            mad_to_first.append(float(np.mean(np.abs(gray - first_gray))))
            mad_prev.append(float(np.mean(np.abs(gray - prev_gray))))
            prev_gray = gray
        cap.release()

        arr_first = np.asarray(mad_to_first, dtype=np.float32)
        arr_prev = np.asarray(mad_prev, dtype=np.float32)
        max_diff_first = float(arr_first.max()) if len(arr_first) else 0.0
        mean_diff_first = float(arr_first.mean()) if len(arr_first) else 0.0
        max_diff_prev = float(arr_prev.max()) if len(arr_prev) else 0.0
        mean_diff_prev = float(arr_prev.mean()) if len(arr_prev) else 0.0
        first_move_thr2 = _first_hit(arr_first, cfg.thr_low) if len(arr_first) else -1
        first_move_thr5 = _first_hit(arr_first, cfg.thr_mid) if len(arr_first) else -1
        first_move_thr10 = _first_hit(arr_first, cfg.thr_high) if len(arr_first) else -1
        label_int = 1 if str(label).lower() == 'unstable' else 0

        rows.append({
            'id': str(sid),
            'label_int': label_int,
            'frames': total,
            'fps': fps,
            'max_diff_first': max_diff_first,
            'mean_diff_first': mean_diff_first,
            'max_diff_prev': max_diff_prev,
            'mean_diff_prev': mean_diff_prev,
            'first_move_thr2': first_move_thr2,
            'first_move_thr5': first_move_thr5,
            'first_move_thr10': first_move_thr10,
            'severity_bucket': _severity_bucket(max_diff_first),
            'onset_bucket': _onset_bucket(first_move_thr2, first_move_thr5),
            'soft_target': _soft_target_from_motion(label_int, max_diff_first, mean_diff_prev),
        })

    out_df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_df


def _center_crop(img: np.ndarray, view: str) -> np.ndarray:
    h, w = img.shape[:2]
    if view == 'front':
        x1, y1, x2, y2 = int(0.22 * w), int(0.18 * h), int(0.78 * w), int(0.92 * h)
    else:
        x1, y1, x2, y2 = int(0.27 * w), int(0.27 * h), int(0.73 * w), int(0.73 * h)
    return img[y1:y2, x1:x2]


def rectify_top_image(img: np.ndarray) -> np.ndarray:
    crop = _center_crop(img, 'top')
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=max(40, min(gray.shape) // 2))
    if lines is None:
        return crop
    angles = []
    for line in lines[:30]:
        theta = float(line[0][1])
        deg = np.rad2deg(theta) - 90.0
        while deg <= -45:
            deg += 90
        while deg > 45:
            deg -= 90
        angles.append(deg)
    if not angles:
        return crop
    rot_deg = float(np.median(angles))
    h, w = crop.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
    rotated = cv2.warpAffine(crop, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return rotated


def _build_structure_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thr.mean() > 127:
        thr = 255 - thr
    kernel = np.ones((5, 5), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    return thr


def compute_geometry_features(front_img: np.ndarray, top_img: np.ndarray) -> np.ndarray:
    front = _center_crop(front_img, 'front')
    top = rectify_top_image(top_img)

    feats: List[float] = []
    for img, view in [(front, 'front'), (top, 'top')]:
        mask = _build_structure_mask(img)
        h, w = mask.shape[:2]
        area = float((mask > 0).mean())
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            feats.extend([0.0] * 7)
            continue
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        bbox_w = max(1, x_max - x_min + 1)
        bbox_h = max(1, y_max - y_min + 1)
        cx = float(xs.mean() / w)
        cy = float(ys.mean() / h)
        support_band = mask[int(h * 0.88):, :]
        support_width = float((support_band.max(axis=0) > 0).mean()) if support_band.size else 0.0
        top_band = mask[: max(1, int(h * 0.25)), :]
        bottom_band = mask[int(h * 0.75):, :]
        top_mass = float((top_band > 0).mean())
        bottom_mass = float((bottom_band > 0).mean())
        left_mass = float((mask[:, : w // 2] > 0).mean())
        right_mass = float((mask[:, w // 2 :] > 0).mean())
        aspect = float(bbox_h / max(1, bbox_w))
        balance = float(abs(left_mass - right_mass))
        top_heavy = float(top_mass / max(bottom_mass, 1e-6))
        feats.extend([area, bbox_w / w, bbox_h / h, cx, cy, support_width, aspect if view == 'front' else balance + top_heavy])

    front_aspect = feats[6]
    top_mass_imbalance = feats[13]
    collapse_margin = feats[5] - abs(feats[3] - 0.5) - 0.35 * max(0.0, front_aspect - 1.8)
    feats.extend([front_aspect, top_mass_imbalance, collapse_margin])
    return np.asarray(feats, dtype=np.float32)


@lru_cache(maxsize=200000)
def load_geometry_features(front_path: str, top_path: str) -> np.ndarray:
    front = cv2.cvtColor(cv2.imread(front_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    top = cv2.cvtColor(cv2.imread(top_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return compute_geometry_features(front, top)


def build_geometry_clusters(df: pd.DataFrame, n_clusters: int = 24) -> pd.Series:
    feature_rows: List[np.ndarray] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc='geometry-cluster'):
        feat = load_geometry_features(row.front_path, row.top_path)
        feature_rows.append(feat)
    x = np.stack(feature_rows, axis=0)
    n_clusters = int(min(max(4, n_clusters), len(df)))
    if len(df) <= n_clusters:
        return pd.Series(np.arange(len(df)), index=df.index)
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=min(512, len(df)), random_state=42, n_init='auto')
    groups = km.fit_predict(x)
    return pd.Series(groups, index=df.index)
