from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class BuildDatasetConfig:
    dataset_ref: str = "gastonarielfrancois/tenis-backview"
    dataset_dir: Path = Path("data/tenis-backview")
    output_dir: Path = Path("data/tenis-backview/yolo")
    person_detector_model: Path = Path("checkpoints/player/model.pt")
    frame_stride: int = 1
    ball_box_size_px: int = 14
    max_player_match_distance_px: float = 160.0


@dataclass(slots=True)
class TrainConfig:
    base_model: str = "yolo11n.pt"
    data_yaml: Path = Path("data/tenis-backview/yolo/dataset.yaml")
    output_dir: Path = Path("outputs/train")
    run_name: str = "player_ball_finetune"
    epochs: int = 30
    batch: int = 16
    imgsz: int = 960
    device: str | None = None
    workers: int = 4
    seed: int = 42


@dataclass(slots=True)
class VisualizeConfig:
    model_path: Path = Path("outputs/train/player_ball_finetune/weights/best.pt")
    input_video: Path = Path("data/sample.mp4")
    output_video: Path = Path("outputs/visualize/sample_overlay.mp4")
    conf: float = 0.25
    device: str | None = None
