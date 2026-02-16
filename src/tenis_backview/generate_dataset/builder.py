from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import kagglehub
import pandas as pd
from ultralytics import YOLO

from tenis_backview.configs.settings import BuildDatasetConfig
from tenis_backview.data.type import BoundingBox, Point2D


@dataclass(slots=True)
class BuildSummary:
    player_images_written: int
    player_labels_written: int
    ball_images_written: int
    ball_labels_written: int
    output_dir: str


def _download_dataset(destination: Path, dataset_ref: str) -> Path:
    source = Path(kagglehub.dataset_download(dataset_ref))
    destination.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        target = destination / item.name
        if target.exists():
            if target.is_file():
                continue
            shutil.rmtree(target)
        if item.is_file():
            shutil.copy2(item, target)
        else:
            shutil.copytree(item, target)

    return destination


def _parse_frame_index(frame_name: str) -> int:
    return int(str(frame_name).split("_")[-1])


def _to_yolo_line(class_id: int, box: BoundingBox, width: int, height: int) -> str:
    x_center = ((box.x1 + box.x2) / 2) / width
    y_center = ((box.y1 + box.y2) / 2) / height
    box_w = (box.x2 - box.x1) / width
    box_h = (box.y2 - box.y1) / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def _ball_box(point: Point2D, size_px: int, width: int, height: int) -> BoundingBox:
    half = size_px / 2
    return BoundingBox(
        x1=max(0.0, point.x - half),
        y1=max(0.0, point.y - half),
        x2=min(float(width - 1), point.x + half),
        y2=min(float(height - 1), point.y + half),
    )


def _choose_player_bbox(
    point: Point2D,
    boxes: list[BoundingBox],
    max_distance_px: float,
) -> BoundingBox | None:
    if not boxes:
        return None

    containing = [box for box in boxes if box.contains(point)]
    candidates = containing if containing else boxes

    def distance_sq(box: BoundingBox) -> float:
        dx = box.center_x - point.x
        dy = box.center_y - point.y
        return dx * dx + dy * dy

    best = min(candidates, key=distance_sq)
    if containing:
        return best

    if distance_sq(best) <= max_distance_px * max_distance_px:
        return best
    return None


def _load_points(player_csv: Path, ball_csv: Path) -> tuple[dict[int, list[Point2D]], dict[int, Point2D]]:
    player_df = pd.read_csv(player_csv)
    ball_df = pd.read_csv(ball_csv)

    player_points: dict[int, list[Point2D]] = {}
    ball_points: dict[int, Point2D] = {}

    for row in player_df.itertuples(index=False):
        frame_index = _parse_frame_index(str(row.frame))
        points: list[Point2D] = []
        for x_name, y_name in (("player_l_x", "player_l_y"), ("player_r_x", "player_r_y")):
            x = getattr(row, x_name)
            y = getattr(row, y_name)
            if pd.notna(x) and pd.notna(y):
                points.append(Point2D(float(x), float(y)))
        player_points[frame_index] = points

    for row in ball_df.itertuples(index=False):
        frame_index = _parse_frame_index(str(row.frame))
        if pd.notna(row.ball_x) and pd.notna(row.ball_y):
            ball_points[frame_index] = Point2D(float(row.ball_x), float(row.ball_y))

    return player_points, ball_points


def _video_split(video_name: str) -> str:
    video_id = int(video_name.replace("video", "").replace(".mp4", ""))
    if video_id <= 8:
        return "train"
    if video_id == 9:
        return "val"
    return "test"


def build_tenis_backview_yolo_dataset(config: BuildDatasetConfig) -> BuildSummary:
    dataset_root = _download_dataset(config.dataset_dir, config.dataset_ref)
    output_root = config.output_dir

    player_root = output_root / "player"
    ball_root = output_root / "ball"

    for split in ("train", "val", "test"):
        (player_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (player_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (ball_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (ball_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    person_detector = YOLO(str(config.person_detector_model))

    player_images_written = 0
    player_labels_written = 0
    ball_images_written = 0
    ball_labels_written = 0

    video_files = sorted(dataset_root.glob("video*.mp4"))
    for video_path in video_files:
        base_name = video_path.stem
        split = _video_split(video_path.name)

        player_csv = dataset_root / f"{base_name}_player.csv"
        ball_csv = dataset_root / f"{base_name}_ball.csv"
        if not player_csv.exists() or not ball_csv.exists():
            continue

        player_points, ball_points = _load_points(player_csv, ball_csv)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % config.frame_stride != 0:
                frame_index += 1
                continue

            height, width = frame.shape[:2]
            results = person_detector.predict(
                source=frame,
                conf=0.2,
                classes=[0],
                verbose=False,
            )

            person_boxes: list[BoundingBox] = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    person_boxes.append(BoundingBox(float(x1), float(y1), float(x2), float(y2)))

            player_labels: list[str] = []
            ball_labels: list[str] = []

            for player_center in player_points.get(frame_index, []):
                matched = _choose_player_bbox(
                    point=player_center,
                    boxes=person_boxes,
                    max_distance_px=config.max_player_match_distance_px,
                )
                if matched is not None:
                    player_labels.append(_to_yolo_line(0, matched, width, height))

            ball_center = ball_points.get(frame_index)
            if ball_center is not None:
                ball_box = _ball_box(ball_center, config.ball_box_size_px, width, height)
                ball_labels.append(_to_yolo_line(0, ball_box, width, height))

            frame_name = f"{base_name}_frame_{frame_index:06d}"

            player_image_path = player_root / "images" / split / f"{frame_name}.jpg"
            player_label_path = player_root / "labels" / split / f"{frame_name}.txt"
            cv2.imwrite(str(player_image_path), frame)
            player_label_path.write_text("\n".join(player_labels), encoding="utf-8")
            player_images_written += 1
            player_labels_written += 1

            ball_image_path = ball_root / "images" / split / f"{frame_name}.jpg"
            ball_label_path = ball_root / "labels" / split / f"{frame_name}.txt"
            cv2.imwrite(str(ball_image_path), frame)
            ball_label_path.write_text("\n".join(ball_labels), encoding="utf-8")
            ball_images_written += 1
            ball_labels_written += 1

            frame_index += 1

        cap.release()

    player_yaml = player_root / "dataset.yaml"
    player_yaml.write_text(
        "\n".join(
            [
                f"path: {player_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: player",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ball_yaml = ball_root / "dataset.yaml"
    ball_yaml.write_text(
        "\n".join(
            [
                f"path: {ball_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: ball",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return BuildSummary(
        player_images_written=player_images_written,
        player_labels_written=player_labels_written,
        ball_images_written=ball_images_written,
        ball_labels_written=ball_labels_written,
        output_dir=str(output_root),
    )
