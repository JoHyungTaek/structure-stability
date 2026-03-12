import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    def __init__(self, df, image_root, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.is_test = is_test

        # 컬럼명 자동 감지
        self.id_col = "id" if "id" in self.df.columns else self.df.columns[0]

        if not self.is_test:
            if "label" in self.df.columns:
                self.label_col = "label"
            elif "target" in self.df.columns:
                self.label_col = "target"
            else:
                self.label_col = self.df.columns[1]

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _label_to_int(self, label):
        if isinstance(label, str):
            label = label.strip().lower()
            if label == "stable":
                return 0.0
            if label == "unstable":
                return 1.0
        return float(label)

    def __getitem__(self, idx):
        sample_id = str(self.df.iloc[idx][self.id_col])

        front_path = os.path.join(self.image_root, sample_id, "front.png")
        top_path = os.path.join(self.image_root, sample_id, "top.png")

        front_img = self._load_image(front_path)
        top_img = self._load_image(top_path)

        if self.transform:
            front_img = self.transform(image=front_img)["image"]
            top_img = self.transform(image=top_img)["image"]

        views = [front_img, top_img]

        if self.is_test:
            return views

        label = self._label_to_int(self.df.iloc[idx][self.label_col])
        return views, label