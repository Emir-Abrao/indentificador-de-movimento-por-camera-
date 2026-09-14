import math

import pytest

from people_counter.config import Settings, load_settings
from people_counter.types import Box, Detection


@pytest.mark.parametrize("coordinates", [(0, 0, 0, 1), (1, 0, 0, 1), (0, 0, 1, math.nan)])
def test_invalid_boxes(coordinates):
    with pytest.raises(ValueError):
        Box(*coordinates)


def test_geometry():
    a, b = Box(0, 0, 10, 10), Box(5, 0, 15, 10)
    assert a.area == 100 and a.center == (5, 5) and a.anchor == (5, 10)
    assert a.iou(b) == pytest.approx(1 / 3)
    assert a.iou(a.moved(100, 0)) == 0


@pytest.mark.parametrize("score", [-1, 1.1, math.nan])
def test_invalid_detection_score(score):
    with pytest.raises(ValueError):
        Detection(Box(0, 0, 1, 1), score)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence": math.nan},
        {"min_hits": 0},
        {"min_hits": True},
        {"detector": "motion"},
        {"deadband": -0.1},
        {"max_reconnects": -1},
        {"line_start": (0.9, 0.55)},
        {"line_start": (0, 2)},
        {"line_start": (0,)},
        {"initial_occupancy": -1},
        {"camera_label": ""},
        {"max_detections": 301},
        {"image_size": 9999},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_config_read_override_and_unknown(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("confidence = 0.5\nline_start = [0.2, 0.6]\n")
    settings = load_settings(path, confidence=0.7)
    assert settings.confidence == 0.7 and settings.line_start == (0.2, 0.6)
    path.write_text("confidnce = 0.5")
    with pytest.raises(ValueError, match="Unknown"):
        load_settings(path)
