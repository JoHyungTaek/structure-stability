import os
import cv2
import torch
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    def __init__(self, df, image_root, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _label_to_float(self, label):
        if isinstance(label, str):
            label = label.strip().lower()
            if label == "stable":
                return 0.0
            if label == "unstable":
                return 1.0
        return float(label)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row["id"])
        sample_dir = os.path.join(self.image_root, sample_id)

        front = self._load_image(os.path.join(sample_dir, "front.png"))
        top = self._load_image(os.path.join(sample_dir, "top.png"))

        if self.transform is not None:
            front = self.transform(image=front)["image"]
            top = self.transform(image=top)["image"]

        if self.is_test:
            return [front, top]

        label = torch.tensor(self._label_to_float(row["label"]), dtype=torch.float32)
        return [front, top], label