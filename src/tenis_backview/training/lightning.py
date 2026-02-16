from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightning as pl
from ultralytics import YOLO

from tenis_backview.configs.settings import TrainConfig


class YoloLightningModule(pl.LightningModule):
    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = YOLO(model_name_or_path)

    def fit_with_ultralytics(
        self,
        *,
        data_yaml: str,
        output_dir: str,
        run_name: str,
        epochs: int,
        batch: int,
        imgsz: int,
        device: str | None,
        workers: int,
    ):
        return self.model.train(
            data=data_yaml,
            project=output_dir,
            name=run_name,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            workers=workers,
        )


@dataclass(slots=True)
class TrainSummary:
    run_dir: str
    best_weights: str


class LightningYoloTrainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config

    def train(self) -> TrainSummary:
        pl.seed_everything(self.config.seed, workers=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        module = YoloLightningModule(self.config.base_model)
        module.fit_with_ultralytics(
            data_yaml=str(self.config.data_yaml),
            output_dir=str(self.config.output_dir),
            run_name=self.config.run_name,
            epochs=self.config.epochs,
            batch=self.config.batch,
            imgsz=self.config.imgsz,
            device=self.config.device,
            workers=self.config.workers,
        )

        run_dir = self.config.output_dir / self.config.run_name
        best_weights = run_dir / "weights" / "best.pt"
        return TrainSummary(run_dir=str(run_dir), best_weights=str(best_weights))
