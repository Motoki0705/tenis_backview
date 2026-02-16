from __future__ import annotations

from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class YoloFrameDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        image_size: int = 960,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = image_size
        self.image_paths = sorted(self.images_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_paths[index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"画像を読み込めません: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        labels: list[list[float]] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                labels.append([float(value) for value in line.split()])

        target_tensor = torch.tensor(labels, dtype=torch.float32)
        if target_tensor.numel() == 0:
            target_tensor = torch.zeros((0, 5), dtype=torch.float32)

        return {
            "image": image_tensor,
            "targets": target_tensor,
            "image_path": str(image_path),
        }
