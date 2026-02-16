from __future__ import annotations

import argparse
from pathlib import Path

from tenis_backview.visualization.overlay import visualize_video_with_yolo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="推論結果を動画へオーバーレイして可視化")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=Path("data/sample.mp4"))
    parser.add_argument("--output", type=Path, default=Path("outputs/visualize/sample_overlay.mp4"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = visualize_video_with_yolo(
        model_path=args.model,
        input_video=args.video,
        output_video=args.output,
        conf=args.conf,
        device=args.device,
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
