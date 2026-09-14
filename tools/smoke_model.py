"""Opt-in real inference smoke test; no webcam and no downloads without --download.

Run after installing .[desktop,yolo]. Builds a short file from the bundled sample,
then exercises decoding -> YOLO -> tracking -> overlay -> encoding -> SQLite.
This validates integration, NOT counting accuracy on moving people in a real site.
"""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from people_counter.app import run
from people_counter.config import Settings
from people_counter.detection import YoloDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    settings = Settings(model=args.model)
    detector = YoloDetector(settings, args.download)
    from ultralytics.utils import ASSETS

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    if image is None:
        raise RuntimeError("Ultralytics bus.jpg fixture not found")
    height, width = image.shape[:2]
    detections = detector.detect(image)
    assert len(detections) >= 2, "Expected multiple people on the sample"
    with TemporaryDirectory(prefix="people-counter-smoke-") as directory:
        source, output, db = [
            Path(directory) / name for name in ("input.avi", "annotated.mp4", "counts.db")
        ]
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height))
        assert writer.isOpened(), "MJPEG writer unavailable"
        try:
            for _ in range(8):
                writer.write(image)
        finally:
            writer.release()
        result = run(
            settings,
            source=str(source),
            database=db,
            headless=True,
            output_video=str(output),
            detector=detector,
        )
        assert result["frames"] == 8 and result["visible"] >= 2
        assert result["entries"] == result["exits"] == 0
        reader = cv2.VideoCapture(str(output))
        try:
            assert reader.isOpened() and int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
        finally:
            reader.release()
        with sqlite3.connect(db) as connection:
            assert connection.execute("SELECT status FROM sessions").fetchone()[0] == "completed"
        result["model_sha256"] = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
        result["fixture"] = "Ultralytics bus.jpg repeated for 8 frames; not a real motion video"
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
