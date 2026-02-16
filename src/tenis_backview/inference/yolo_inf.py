from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from ultralytics import YOLO

from tenis_backview.data.type import BoundingBox, Detection


class YoloInference:
    def __init__(
        self,
        model_path: str | Path,
        *,
        conf: float = 0.25,
        device: str | None = None,
        classes: Iterable[int] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))
        self.conf = conf
        self.device = device
        self.classes = list(classes) if classes is not None else None

    def predict_frame(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []

        names = result.names if isinstance(result.names, dict) else {}
        detections: list[Detection] = []
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            class_id = int(box.cls.item())
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(names.get(class_id, class_id)),
                    confidence=float(box.conf.item()),
                    bbox=BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                    ),
                )
            )
        return detections
