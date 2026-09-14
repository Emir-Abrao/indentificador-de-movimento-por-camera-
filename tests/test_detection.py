from types import SimpleNamespace

import numpy as np
import pytest

from people_counter.config import Settings
from people_counter.detection import HogDetector, YoloDetector, build_detector, validate_frame


class FakeBoxes:
    def __init__(self):
        self.xyxy = np.array(
            [
                [-2, 4, 100, 200],
                [0, 0, 10, 10],
                [0, 0, 10, 10],
                [0, 0, 0, 20],
                [0, 0, float("nan"), 20],
            ]
        )
        self.conf = np.array([0.9, 0.95, 0.1, 0.9, 0.9])
        self.cls = np.array([0, 2, 0, 0, 0])

    def cpu(self):
        return self

    def numpy(self):
        return self


class FakeModel:
    names = {0: "person", 2: "car"}

    def __init__(self, *args, **kwargs):
        self.args = None

    def predict(self, **kwargs):
        self.args = kwargs
        return [SimpleNamespace(boxes=FakeBoxes())]


def test_yolo_adapter_person_filter_clip_and_actual_call(frame):
    detector = YoloDetector(Settings(), True, FakeModel)
    results = detector.detect(frame)
    assert len(results) == 1 and results[0].box.x1 == 0
    assert detector.model.args["classes"] == [0]
    assert detector.model.args["save"] is False
    assert detector.model.args["device"] == "cpu"


def test_download_requires_explicit_opt_in():
    with pytest.raises(FileNotFoundError):
        YoloDetector(Settings(model="not-a-model.pt"), model_factory=FakeModel)
    with pytest.raises(FileNotFoundError):
        YoloDetector(Settings(model="untrusted.pt"), True, FakeModel)


def test_missing_person_class():
    class BadModel(FakeModel):
        names = {0: "truck"}

    with pytest.raises(ValueError, match="person"):
        YoloDetector(Settings(), True, BadModel)


def test_empty_model_result(frame):
    class Empty(FakeModel):
        names = ["person"]

        def predict(self, **kwargs):
            return []

    assert YoloDetector(Settings(), True, Empty).detect(frame) == []


def test_hog_actual_inference_on_blank_frame(frame):
    detector = build_detector(Settings(detector="hog"))
    assert isinstance(detector, HogDetector)
    assert detector.detect(frame) == []
    assert detector.detect(frame[:50, :50]) == []


@pytest.mark.parametrize("value", [None, np.zeros((10, 10)), np.zeros((2, 2, 4), dtype=np.uint8)])
def test_bad_frame(value):
    with pytest.raises(ValueError):
        validate_frame(value)
