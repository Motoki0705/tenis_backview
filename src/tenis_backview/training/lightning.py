from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        fraction: float,
        batch_loss_trace: list[float],
    ):
        def _on_train_batch_end(trainer: Any) -> None:
            tloss = getattr(trainer, "tloss", None)
            if tloss is None:
                return
            if hasattr(tloss, "detach"):
                value = float(tloss.detach().sum().cpu().item())
            elif isinstance(tloss, (list, tuple)):
                value = float(sum(float(item) for item in tloss))
            else:
                value = float(tloss)
            batch_loss_trace.append(value)

        self.model.add_callback("on_train_batch_end", _on_train_batch_end)
        return self.model.train(
            data=data_yaml,
            project=output_dir,
            name=run_name,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            workers=workers,
            fraction=fraction,
        )


@dataclass(slots=True)
class TrainSummary:
    run_dir: str
    best_weights: str
    first_batch_loss: float | None
    last_batch_loss: float | None
    loss_decreased: bool


class LightningYoloTrainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config

    def train(self) -> TrainSummary:
        pl.seed_everything(self.config.seed, workers=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        module = YoloLightningModule(self.config.base_model)
        batch_loss_trace: list[float] = []
        module.fit_with_ultralytics(
            data_yaml=str(self.config.data_yaml),
            output_dir=str(self.config.output_dir),
            run_name=self.config.run_name,
            epochs=self.config.epochs,
            batch=self.config.batch,
            imgsz=self.config.imgsz,
            device=self.config.device,
            workers=self.config.workers,
            fraction=self.config.fraction,
            batch_loss_trace=batch_loss_trace,
        )

        run_dir = self.config.output_dir / self.config.run_name
        best_weights = run_dir / "weights" / "best.pt"

        first_loss = batch_loss_trace[0] if batch_loss_trace else None
        last_loss = batch_loss_trace[-1] if batch_loss_trace else None
        loss_decreased = False
        if batch_loss_trace:
            if len(batch_loss_trace) < 3:
                loss_decreased = (
                    first_loss is not None and last_loss is not None and last_loss < first_loss
                )
            else:
                window = min(5, len(batch_loss_trace))
                head_mean = sum(batch_loss_trace[:window]) / window
                tail_mean = sum(batch_loss_trace[-window:]) / window
                loss_decreased = tail_mean < head_mean

        return TrainSummary(
            run_dir=str(run_dir),
            best_weights=str(best_weights),
            first_batch_loss=first_loss,
            last_batch_loss=last_loss,
            loss_decreased=loss_decreased,
        )
