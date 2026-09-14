"""Strict TOML configuration. Unknown keys fail instead of being silently ignored."""

import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    detector: str = "yolo"
    model: str = "yolo11n.pt"
    confidence: float = 0.45
    image_size: int = 640
    device: str = "cpu"
    max_detections: int = 100
    max_width: int = 1280
    min_hits: int = 3
    max_age: int = 15
    match_distance: float = 0.12
    line_start: tuple[float, float] = (0.1, 0.55)
    line_end: tuple[float, float] = (0.9, 0.55)
    deadband: float = 0.015
    stable_frames: int = 3
    cooldown_frames: int = 15
    max_crossing_gap: int = 3
    initial_occupancy: int = 0
    max_reconnects: int = 3
    reconnect_delay: float = 1.0
    network_timeout_ms: int = 5000
    snapshot_interval: int = 30
    camera_label: str = "camera-01"

    def __post_init__(self):
        if self.detector not in {"yolo", "hog"}:
            raise ValueError("detector must be 'yolo' or 'hog'")
        for name in ("model", "device", "camera_label"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip() or len(v) > 200:
                raise ValueError(f"{name} must be a non-empty string of up to 200 characters")
        for name in (
            "image_size",
            "max_detections",
            "max_width",
            "min_hits",
            "max_age",
            "stable_frames",
            "max_crossing_gap",
            "network_timeout_ms",
            "snapshot_interval",
        ):
            v = getattr(self, name)
            if type(v) is not int or v <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("cooldown_frames", "initial_occupancy", "max_reconnects"):
            v = getattr(self, name)
            if type(v) is not int or v < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name, lo, hi in (
            ("confidence", 0.01, 1),
            ("match_distance", 0.001, 1),
            ("deadband", 0, 0.25),
            ("reconnect_delay", 0, 60),
        ):
            v = getattr(self, name)
            if type(v) not in (int, float) or not math.isfinite(v) or not lo <= v <= hi:
                raise ValueError(f"{name} must be between {lo} and {hi}")
        if self.max_detections > 300 or self.max_width > 4096 or self.image_size > 2048:
            raise ValueError("Resolution/detection limits exceed the supported resource budget")
        for name in ("line_start", "line_end"):
            point = getattr(self, name)
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"{name} must contain exactly two normalized coordinates")
            if any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                for v in point
            ):
                raise ValueError(f"{name} coordinates must be in [0, 1]")
            object.__setattr__(self, name, tuple(point))
        if math.dist(self.line_start, self.line_end) < 0.05:
            raise ValueError("Counting line must span at least 5% of the normalized image")


def load_settings(path: str | Path | None, **overrides) -> Settings:
    values = {}
    if path is not None:
        with Path(path).open("rb") as file:
            values = tomllib.load(file)
    allowed = {f.name for f in fields(Settings)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Settings(**values)
