"""Bounded capture recovery. Source strings (possibly credentials) never enter logs/DB."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import cv2

from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class Frame:
    image: object
    index: int
    media_seconds: float
    discontinuity: bool


class Capture:
    def __init__(self, source: str, settings: Settings, factory=None, sleep=time.sleep):
        self.settings, self.sleep = settings, sleep
        self.factory = factory or cv2.VideoCapture
        self.source = int(source) if source.isdecimal() else source
        self.network = isinstance(self.source, str) and urlsplit(source).scheme.lower() in {
            "rtsp",
            "rtsps",
            "http",
            "https",
        }
        self.is_file = isinstance(self.source, str) and not self.network
        if self.is_file and not Path(source).is_file():
            raise FileNotFoundError(
                "Video file not found; provide a camera index or an existing file"
            )
        self.cap = None
        self.fps = 30.0
        self.reconnects = 0

    def _open(self):
        if self.network:
            timeout = self.settings.network_timeout_ms
            self.cap = self.factory(
                self.source,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    timeout,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    timeout,
                ],
            )
        else:
            self.cap = self.factory(self.source)
        if self.cap.isOpened():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = float(fps) if 0 < fps < 241 else 30.0
            return True
        return False

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __iter__(self):
        index, failures, boundary = 0, 0, False
        start = time.monotonic()
        try:
            while True:
                if self.cap is None and not self._open():
                    ok, image = False, None
                else:
                    ok, image = self.cap.read()
                if ok and image is not None and image.size:
                    timestamp = index / self.fps if self.is_file else time.monotonic() - start
                    yield Frame(image, index, timestamp, boundary)
                    index += 1
                    failures, boundary = 0, False
                    continue
                if self.is_file:
                    if index == 0:
                        raise RuntimeError(
                            "Video could not be decoded (empty or unsupported codec)"
                        )
                    break
                self.close()
                failures += 1
                if failures > self.settings.max_reconnects:
                    raise RuntimeError("Camera unavailable after bounded reconnection attempts")
                self.reconnects += 1
                boundary = True
                log.warning(
                    "Capture interrupted; retry %s/%s", failures, self.settings.max_reconnects
                )
                self.sleep(self.settings.reconnect_delay)
        finally:
            self.close()
