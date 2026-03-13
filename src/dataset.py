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

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def label_to_float(self, label):
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

        folder = os.path.join(self.image_root, sample_id)

        front = self.load_image(os.path.join(folder, "front.png"))
        top = self.load_image(os.path.join(folder, "top.png"))

        if self.transform:
            front = self.transform(image=front)["image"]
            top = self.transform(image=top)["image"]

        if self.is_test:
            return [front, top]

        label = torch.tensor(self.label_to_float(row["label"]), dtype=torch.float32)

        return [front, top], label