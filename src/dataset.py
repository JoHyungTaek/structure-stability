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

        # 이미지 경로
        front_path = os.path.join(self.image_root, sample_id, "front.png")
        top_path = os.path.join(self.image_root, sample_id, "top.png")

        # 이미지 로드
        front_img = Image.open(front_path).convert("RGB")
        top_img = Image.open(top_path).convert("RGB")

        # Transform 적용
        if self.transform:
            front_img = self.transform(front_img)
            top_img = self.transform(top_img)

        views = [front_img, top_img]

        # 테스트 데이터
        if self.is_test:
            return views

        # 학습 데이터
        label = self.df.iloc[idx, 1]

        return views, label
