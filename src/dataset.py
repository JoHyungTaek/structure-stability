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

    def _label_to_int(self, label):
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

        front_path = os.path.join(sample_dir, "front.png")
        top_path = os.path.join(sample_dir, "top.png")

        front_img = self._load_image(front_path)
        top_img = self._load_image(top_path)

        if self.transform is not None:
            front_img = self.transform(image=front_img)["image"]
            top_img = self.transform(image=top_img)["image"]

        if self.is_test:
            return [front_img, top_img]

        label = self._label_to_int(row["label"])
        label = torch.tensor(label, dtype=torch.float32)

        return [front_img, top_img], label