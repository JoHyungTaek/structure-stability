import os
from PIL import Image
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    def __init__(self, df, image_root, transform=None, is_test=False):
        self.df = df
        self.image_root = image_root
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_id = self.df.iloc[idx, 0]

        front_path = os.path.join(self.image_root, sample_id, "front.png")
        side_path = os.path.join(self.image_root, sample_id, "side.png")

        front_img = Image.open(front_path).convert("RGB")
        side_img = Image.open(side_path).convert("RGB")

        if self.transform:
            front_img = self.transform(front_img)
            side_img = self.transform(side_img)

        views = [front_img, side_img]

        if self.is_test:
            return views

        label = self.df.iloc[idx, 1]
        return views, label
