import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TeacherTrainDataset(Dataset):
    def __init__(self, df, image_root, transform=None, num_video_frames=6):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.num_video_frames = num_video_frames

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _sample_frame_indices(self, total_frames, num_samples):
        if total_frames <= 0:
            return [0] * num_samples
        return np.linspace(0, total_frames - 1, num_samples).astype(int).tolist()

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = self._sample_frame_indices(total_frames, self.num_video_frames)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                if len(frames) > 0:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        while len(frames) < self.num_video_frames:
            frames.append(frames[-1].copy())
        return frames

    def _apply_transform(self, image):
        if self.transform is None:
            return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return self.transform(image=image)["image"]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row["id"])
        label = float(row["label"])

        sample_dir = os.path.join(self.image_root, sample_id)
        front = self._load_image(os.path.join(sample_dir, "front.png"))
        top = self._load_image(os.path.join(sample_dir, "top.png"))
        frames = self._load_video_frames(os.path.join(sample_dir, "simulation.mp4"))

        front = self._apply_transform(front)
        top = self._apply_transform(top)
        frames = torch.stack([self._apply_transform(x) for x in frames], dim=0)

        label = torch.tensor(label, dtype=torch.float32)
        return [front, top], frames, label


class StudentDataset(Dataset):
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
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _apply_transform(self, image):
        if self.transform is None:
            return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return self.transform(image=image)["image"]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row["id"])
        sample_dir = os.path.join(self.image_root, sample_id)

        front = self._load_image(os.path.join(sample_dir, "front.png"))
        top = self._load_image(os.path.join(sample_dir, "top.png"))

        front = self._apply_transform(front)
        top = self._apply_transform(top)

        if self.is_test:
            return [front, top]

        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        soft_label = None
        if "soft_unstable_prob" in row.index and not np.isnan(row["soft_unstable_prob"]):
            soft_label = torch.tensor(float(row["soft_unstable_prob"]), dtype=torch.float32)

        domain = row["domain"] if "domain" in row.index else "train"

        if soft_label is None:
            soft_label = torch.tensor(-1.0, dtype=torch.float32)

        return [front, top], label, soft_label, domain