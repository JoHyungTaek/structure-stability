import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiModalStructureDataset(Dataset):
    def __init__(self, df, image_root, transform=None, is_test=False, use_video=True, num_video_frames=6):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.is_test = is_test
        self.use_video = use_video
        self.num_video_frames = num_video_frames

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

    def _sample_frame_indices(self, total_frames, num_samples):
        if total_frames <= 0:
            return [0] * num_samples
        if total_frames < num_samples:
            indices = np.linspace(0, total_frames - 1, num_samples)
        else:
            indices = np.linspace(0, total_frames - 1, num_samples)
        return indices.astype(int).tolist()

    def _load_video_frames(self, video_path):
        if not os.path.exists(video_path):
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

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
                    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
                    frames.append(dummy)
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        while len(frames) < self.num_video_frames:
            frames.append(frames[-1].copy())

        return frames

    def _apply_transform(self, image):
        if self.transform is None:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            return image
        return self.transform(image=image)["image"]

    def __getitem__(self, idx):
        sample_id = str(self.df.iloc[idx][self.id_col])

        sample_dir = os.path.join(self.image_root, sample_id)
        front_path = os.path.join(sample_dir, "front.png")
        top_path = os.path.join(sample_dir, "top.png")
        video_path = os.path.join(sample_dir, "simulation.mp4")

        front_img = self._load_image(front_path)
        top_img = self._load_image(top_path)

        front_img = self._apply_transform(front_img)
        top_img = self._apply_transform(top_img)

        if self.use_video:
            video_frames = self._load_video_frames(video_path)

            if video_frames is None:
                c, h, w = front_img.shape
                video_tensor = torch.zeros(self.num_video_frames, c, h, w, dtype=front_img.dtype)
            else:
                transformed_frames = [self._apply_transform(frame) for frame in video_frames]
                video_tensor = torch.stack(transformed_frames, dim=0)  # [T, C, H, W]
        else:
            c, h, w = front_img.shape
            video_tensor = torch.zeros(self.num_video_frames, c, h, w, dtype=front_img.dtype)

        if self.is_test:
            return [front_img, top_img], video_tensor

        label = self._label_to_int(self.df.iloc[idx][self.label_col])
        return [front_img, top_img], video_tensor, label