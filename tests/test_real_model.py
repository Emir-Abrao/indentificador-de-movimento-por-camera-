"""Opt-in real model test. Default suite never downloads weights or opens cameras."""

import os
from pathlib import Path

import cv2
import pytest

from people_counter.config import Settings
from people_counter.detection import YoloDetector
from people_counter.tracking import Tracker


@pytest.mark.model
@pytest.mark.skipif(
    not os.environ.get("PEOPLE_COUNTER_MODEL"),
    reason="Set PEOPLE_COUNTER_MODEL to a trusted local YOLO weights file",
)
def test_real_yolo_people_and_stable_ids():
    from ultralytics.utils import ASSETS

    path = Path(os.environ["PEOPLE_COUNTER_MODEL"])
    assert path.is_file(), "Model test never auto-downloads weights"
    image = cv2.imread(str(ASSETS / "bus.jpg"))
    assert image is not None, "Ultralytics bus sample must be installed"
    detector = YoloDetector(Settings(model=str(path)))
    detections = detector.detect(image)
    assert len(detections) >= 2, "Expected at least two people in the real sample"
    height, width = image.shape[:2]
    tracker = Tracker(min_hits=2)
    tracker.update(detections, 0, width, height)
    first = tracker.update(detector.detect(image), 1, width, height)
    second = tracker.update(detector.detect(image), 2, width, height)
    assert len(first) >= 2
    assert [o.track_id for o in first] == [o.track_id for o in second]
    assert tracker.total_confirmed == len(first)
