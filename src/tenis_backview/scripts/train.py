from __future__ import annotations

import argparse
from pathlib import Path

from tenis_backview.configs.settings import BuildDatasetConfig, TrainConfig
from tenis_backview.generate_dataset.builder import build_tenis_backview_yolo_dataset
from tenis_backview.training.lightning import LightningYoloTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO player/ball fine-tuning")
    parser.add_argument("--prepare-dataset", action="store_true")
    parser.add_argument("--dataset-ref", type=str, default="gastonarielfrancois/tenis-backview")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/tenis-backview"))
    parser.add_argument("--data-yolo-dir", type=Path, default=Path("data/tenis-backview/yolo"))
    parser.add_argument("--person-model", type=Path, default=Path("checkpoints/player/model.pt"))
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--ball-box-size", type=int, default=14)
    parser.add_argument("--max-player-match-distance", type=float, default=160.0)
    parser.add_argument("--base-model", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train"))
    parser.add_argument("--run-name", type=str, default="player_ball_finetune")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.prepare_dataset:
        build_config = BuildDatasetConfig(
            dataset_ref=args.dataset_ref,
            dataset_dir=args.dataset_dir,
            output_dir=args.data_yolo_dir,
            person_detector_model=args.person_model,
            frame_stride=max(1, args.frame_stride),
            ball_box_size_px=max(2, args.ball_box_size),
            max_player_match_distance_px=max(1.0, args.max_player_match_distance),
        )
        summary = build_tenis_backview_yolo_dataset(build_config)
        print(f"dataset prepared: images={summary.images_written}, labels={summary.labels_written}")

    data_yaml = args.data_yolo_dir / "dataset.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} が見つかりません。--prepare-dataset を付けるか事前生成してください。"
        )

    train_config = TrainConfig(
        base_model=args.base_model,
        data_yaml=data_yaml,
        output_dir=args.output_dir,
        run_name=args.run_name,
        epochs=max(1, args.epochs),
        batch=max(1, args.batch),
        imgsz=max(64, args.imgsz),
        device=args.device,
        workers=max(0, args.workers),
        seed=args.seed,
    )

    trainer = LightningYoloTrainer(train_config)
    result = trainer.train()
    print(f"run_dir: {result.run_dir}")
    print(f"best_weights: {result.best_weights}")


if __name__ == "__main__":
    main()
