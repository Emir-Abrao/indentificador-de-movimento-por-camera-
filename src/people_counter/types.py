"""Validated values shared by detection, tracking and counting."""

from dataclasses import dataclass
from math import isfinite

Point = tuple[float, float]


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        if not all(isfinite(v) for v in (self.x1, self.y1, self.x2, self.y2)):
            raise ValueError("Box coordinates must be finite")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("Box must have positive width and height")

    @property
    def center(self) -> Point:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def anchor(self) -> Point:
        """Bottom centre is a useful ground-plane proxy for doorway counting."""
        return ((self.x1 + self.x2) / 2, self.y2)

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def moved(self, dx: float, dy: float) -> "Box":
        return Box(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def iou(self, other: "Box") -> float:
        intersection = max(0, min(self.x2, other.x2) - max(self.x1, other.x1)) * max(
            0, min(self.y2, other.y2) - max(self.y1, other.y1)
        )
        return intersection / (self.area + other.area - intersection)


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float

    def __post_init__(self):
        if not isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("Detection score must be in [0, 1]")


@dataclass(frozen=True)
class Observation:
    track_id: int
    box: Box
    score: float


@dataclass(frozen=True)
class Crossing:
    sequence: int
    track_id: int
    direction: str
    frame_index: int
    media_seconds: float
    x: float
    y: float
