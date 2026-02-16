from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from tenis_backview.configs.settings import BuildDatasetConfig, TrainConfig
from tenis_backview.generate_dataset.builder import build_tenis_backview_yolo_dataset
from tenis_backview.training.lightning import LightningYoloTrainer


@hydra.main(version_base=None, config_path="../configs", config_name="train_player")
def main(cfg: DictConfig) -> None:
    if bool(cfg.prepare_dataset):
        build_config = BuildDatasetConfig(
            dataset_ref=str(cfg.dataset_ref),
            dataset_dir=Path(cfg.dataset_dir),
            output_dir=Path(cfg.data_yolo_dir),
            person_detector_model=Path(cfg.person_model),
            frame_stride=max(1, int(cfg.frame_stride)),
            ball_box_size_px=max(2, int(cfg.ball_box_size)),
            max_player_match_distance_px=max(1.0, float(cfg.max_player_match_distance)),
        )
        summary = build_tenis_backview_yolo_dataset(build_config)
        print(
            "dataset prepared: "
            f"player_images={summary.player_images_written}, player_labels={summary.player_labels_written}, "
            f"ball_images={summary.ball_images_written}, ball_labels={summary.ball_labels_written}"
        )

    data_yaml = Path(cfg.data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} が見つかりません。prepare_dataset=true で先に生成してください。"
        )

    train_config = TrainConfig(
        base_model=str(cfg.base_model),
        data_yaml=data_yaml,
        output_dir=Path(cfg.output_dir),
        run_name=str(cfg.run_name),
        epochs=max(1, int(cfg.epochs)),
        batch=max(1, int(cfg.batch)),
        imgsz=max(64, int(cfg.imgsz)),
        fraction=min(1.0, max(0.001, float(cfg.fraction))),
        device=cfg.device,
        workers=max(0, int(cfg.workers)),
        seed=int(cfg.seed),
    )

    trainer = LightningYoloTrainer(train_config)
    result = trainer.train()
    print(f"run_dir: {result.run_dir}")
    print(f"best_weights: {result.best_weights}")
    print(f"first_batch_loss: {result.first_batch_loss}")
    print(f"last_batch_loss: {result.last_batch_loss}")
    print(f"loss_decreased: {result.loss_decreased}")


if __name__ == "__main__":
    main()
