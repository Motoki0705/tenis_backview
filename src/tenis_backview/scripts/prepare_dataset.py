from __future__ import annotations

import argparse
from pathlib import Path

from tenis_backview.configs.settings import BuildDatasetConfig
from tenis_backview.generate_dataset.builder import build_tenis_backview_yolo_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="tenis-backview データセットをYOLO学習用に生成")
    parser.add_argument("--dataset-ref", type=str, default="gastonarielfrancois/tenis-backview")
    parser.add_argument("--dataset-dir", type=str, default="data/tenis-backview")
    parser.add_argument("--output-dir", type=str, default="data/tenis-backview/yolo")
    parser.add_argument("--person-model", type=str, default="checkpoints/player/model.pt")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--ball-box-size", type=int, default=14)
    parser.add_argument("--max-player-match-distance", type=float, default=160.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BuildDatasetConfig(
        dataset_ref=args.dataset_ref,
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        person_detector_model=Path(args.person_model),
        frame_stride=max(1, args.frame_stride),
        ball_box_size_px=max(2, args.ball_box_size),
        max_player_match_distance_px=max(1.0, args.max_player_match_distance),
    )

    summary = build_tenis_backview_yolo_dataset(config)
    print(f"dataset generated: images={summary.images_written}, labels={summary.labels_written}")
    print(f"output: {summary.output_dir}")


if __name__ == "__main__":
    main()
