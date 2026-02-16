from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Point2D:
    x: float
    y: float


@dataclass(slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    def contains(self, point: Point2D) -> bool:
        return self.x1 <= point.x <= self.x2 and self.y1 <= point.y <= self.y2


@dataclass(slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


@dataclass(slots=True)
class FrameAnnotation:
    frame_name: str
    player_left: Point2D | None
    player_right: Point2D | None
    ball: Point2D | None
