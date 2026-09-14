"""Replaceable people detectors. No silent substitution of real inference by mocks."""

import math
from pathlib import Path

import cv2
import numpy as np

from .config import Settings
from .types import Box, Detection


def validate_frame(frame):
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or not frame.size
    ):
        raise ValueError("Expected a non-empty uint8 BGR frame")


class YoloDetector:
    def __init__(self, settings: Settings, allow_download=False, model_factory=None):
        path = Path(settings.model)
        if not path.is_file() and not (allow_download and settings.model == "yolo11n.pt"):
            raise FileNotFoundError(
                "Model not found. Use --allow-model-download for the official yolo11n.pt "
                "or --model with an existing trusted weights file."
            )
        if model_factory is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    'Install YOLO support: pip install -e ".[desktop,yolo]"'
                ) from exc
            model_factory = YOLO
        self.model = model_factory(settings.model, task="detect")
        names = self.model.names
        if isinstance(names, list):
            names = dict(enumerate(names))
        people = [int(k) for k, value in names.items() if str(value).lower() == "person"]
        if len(people) != 1:
            raise ValueError("Model must expose exactly one class named 'person'")
        self.person_class = people[0]
        self.settings = settings

    def detect(self, frame) -> list[Detection]:
        validate_frame(frame)
        results = self.model.predict(
            source=frame,
            classes=[self.person_class],
            conf=self.settings.confidence,
            imgsz=self.settings.image_size,
            device=self.settings.device,
            max_det=self.settings.max_detections,
            verbose=False,
            save=False,
        )
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes.cpu().numpy()
        height, width = frame.shape[:2]
        detections = []
        for coordinates, score, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls, strict=True):
            if int(class_id) != self.person_class or float(score) < self.settings.confidence:
                continue
            x1, y1, x2, y2 = map(float, coordinates)
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2, float(score))):
                continue
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 > x1 and y2 > y1:
                detections.append(Detection(Box(x1, y1, x2, y2), float(score)))
        return detections[: self.settings.max_detections]


class HogDetector:
    """Offline baseline. HOG/SVM score is not a calibrated confidence probability."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame) -> list[Detection]:
        validate_frame(frame)
        height, width = frame.shape[:2]
        scale = min(1.0, 800 / width)
        resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        if resized.shape[0] < 128 or resized.shape[1] < 64:
            return []
        boxes, weights = self.hog.detectMultiScale(
            resized, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        if not len(boxes):
            return []
        scores = [
            1 / (1 + math.exp(-max(-50, min(50, float(v)))))
            for v in np.asarray(weights).reshape(-1)
        ]
        selected = cv2.dnn.NMSBoxes(boxes.tolist(), scores, self.settings.confidence, 0.4)
        detections = []
        for idx in np.asarray(selected).reshape(-1)[: self.settings.max_detections]:
            x, y, w, h = map(float, boxes[int(idx)])
            detections.append(
                Detection(
                    Box(
                        x / scale,
                        y / scale,
                        min(width, (x + w) / scale),
                        min(height, (y + h) / scale),
                    ),
                    scores[int(idx)],
                )
            )
        return detections


def build_detector(settings: Settings, allow_download=False):
    if settings.detector == "yolo":
        return YoloDetector(settings, allow_download)
    return HogDetector(settings)
