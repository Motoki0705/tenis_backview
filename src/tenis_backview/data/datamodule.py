from __future__ import annotations

from pathlib import Path

import lightning as pl
from torch.utils.data import DataLoader

from tenis_backview.data.dataset import YoloFrameDataset


class YoloFrameDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str | Path,
        image_size: int = 960,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = YoloFrameDataset(
            images_dir=self.data_root / "images/train",
            labels_dir=self.data_root / "labels/train",
            image_size=self.image_size,
        )
        self.val_dataset = YoloFrameDataset(
            images_dir=self.data_root / "images/val",
            labels_dir=self.data_root / "labels/val",
            image_size=self.image_size,
        )

    @staticmethod
    def _collate_fn(batch: list[dict]) -> dict:
        return {
            "images": [item["image"] for item in batch],
            "targets": [item["targets"] for item in batch],
            "image_paths": [item["image_path"] for item in batch],
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )
