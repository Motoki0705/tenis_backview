from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from tenis_backview.visualization.overlay import visualize_video_with_yolo


@hydra.main(version_base=None, config_path="../configs", config_name="visualize")
def main(cfg: DictConfig) -> None:
    output = visualize_video_with_yolo(
        model_path=Path(cfg.model),
        input_video=Path(cfg.video),
        output_video=Path(cfg.output),
        conf=float(cfg.conf),
        device=cfg.device,
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
