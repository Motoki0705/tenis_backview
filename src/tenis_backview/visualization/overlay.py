from __future__ import annotations

from pathlib import Path

import cv2

from tenis_backview.inference.yolo_inf import YoloInference


def visualize_video_with_yolo(
    *,
    model_path: str | Path,
    input_video: str | Path,
    output_video: str | Path,
    conf: float = 0.25,
    device: str | None = None,
) -> str:
    detector = YoloInference(model_path=model_path, conf=conf, device=device)

    input_path = Path(input_video)
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けません: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = detector.predict_frame(frame)
            for det in detections:
                x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
                x2, y2 = int(det.bbox.x2), int(det.bbox.y2)
                color = (0, 255, 0) if det.class_id == 0 else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{det.class_name}:{det.confidence:.2f}",
                    (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return str(output_path)
