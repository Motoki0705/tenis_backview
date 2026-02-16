from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from tenis_backview.configs.settings import BuildDatasetConfig
from tenis_backview.generate_dataset.builder import build_tenis_backview_yolo_dataset


@hydra.main(version_base=None, config_path="../configs", config_name="prepare_dataset")
def main(cfg: DictConfig) -> None:
    config = BuildDatasetConfig(
        dataset_ref=str(cfg.dataset_ref),
        dataset_dir=Path(cfg.dataset_dir),
        output_dir=Path(cfg.output_dir),
        person_detector_model=Path(cfg.person_model),
        frame_stride=max(1, int(cfg.frame_stride)),
        ball_box_size_px=max(2, int(cfg.ball_box_size)),
        max_player_match_distance_px=max(1.0, float(cfg.max_player_match_distance)),
    )

    summary = build_tenis_backview_yolo_dataset(config)
    print(
        "dataset generated: "
        f"player_images={summary.player_images_written}, player_labels={summary.player_labels_written}, "
        f"ball_images={summary.ball_images_written}, ball_labels={summary.ball_labels_written}"
    )
    print(f"output: {summary.output_dir}")


if __name__ == "__main__":
    main()
