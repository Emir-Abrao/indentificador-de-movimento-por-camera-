"""One camera per process; fail closed on detector/persistence errors."""

import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from .capture import Capture
from .config import Settings
from .counting import LineCounter
from .detection import build_detector, validate_frame
from .overlay import render
from .storage import Store
from .tracking import Tracker


def run(
    settings: Settings,
    source="0",
    database="data/counts.db",
    headless=False,
    max_frames: int | None = None,
    output_video: str | None = None,
    allow_download=False,
    detector=None,
    capture=None,
):
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if output_video and Path(output_video).suffix.lower() != ".mp4":
        raise ValueError("Output video must use the .mp4 extension")
    if (
        not headless
        and sys.platform.startswith("linux")
        and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    ):
        raise RuntimeError("No graphical display. Use --headless or run on a desktop")
    detector = detector or build_detector(settings, allow_download)
    capture = capture or Capture(str(source), settings)
    tracker = Tracker(settings.min_hits, settings.max_age, settings.match_distance)
    counter = LineCounter(settings)
    store = Store(database, settings.camera_label, asdict(settings))
    writer = None
    status = "failed"
    frames, shape, fps, last_index = 0, None, 0.0, -1
    final = {}
    try:
        if not headless:
            try:
                cv2.namedWindow("People Counter", cv2.WINDOW_NORMAL)
            except cv2.error as exc:
                raise RuntimeError(
                    "GUI unavailable: install the desktop extra or use --headless"
                ) from exc
        for packet in capture:
            started = time.perf_counter()
            validate_frame(packet.image)
            frame = packet.image
            h, w = frame.shape[:2]
            if w > settings.max_width:
                frame = cv2.resize(
                    frame, (settings.max_width, max(1, round(h * settings.max_width / w)))
                )
            height, width = frame.shape[:2]
            current_shape = (width, height)
            if packet.discontinuity or (shape is not None and shape != current_shape):
                tracker.reset_associations()
                counter.reset_associations()
                if writer and shape != current_shape:
                    raise RuntimeError(
                        "Resolution changed during recording; recording stopped safely"
                    )
            shape = current_shape
            observations = tracker.update(detector.detect(frame), packet.index, width, height)
            events = counter.update(
                observations, packet.index, width, height, packet.media_seconds, set(tracker.tracks)
            )
            elapsed = time.perf_counter() - started
            instant = 1 / max(elapsed, 1e-6)
            fps = instant if not fps else 0.9 * fps + 0.1 * instant
            frames += 1
            last_index = packet.index
            final = dict(
                session_id=store.session_id,
                frames=frames,
                visible=len(observations),
                entries=counter.entries,
                exits=counter.exits,
                net_flow=counter.net_flow,
                estimated_occupancy=counter.occupancy,
                underflow_events=counter.underflow_events,
                confirmed_trajectories=tracker.total_confirmed,
                processing_fps=round(fps, 2),
                reconnects=capture.reconnects,
                untracked_detections=tracker.untracked_detections,
            )
            store.record(
                events, packet.index, final if frames % settings.snapshot_interval == 0 else None
            )
            if output_video or not headless:
                image = render(frame, observations, counter, settings, fps, tracker.total_confirmed)
                if output_video and writer is None:
                    path = Path(output_video)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # Only a newly reserved destination may be used by OpenCV's writer.
                    with path.open("xb"):
                        pass
                    writer = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*"mp4v"), capture.fps, current_shape
                    )
                    if not writer.isOpened():
                        raise RuntimeError("MP4 encoder unavailable; recording could not start")
                if writer:
                    writer.write(image)
                if not headless:
                    cv2.imshow("People Counter", image)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        break
                    if cv2.getWindowProperty("People Counter", cv2.WND_PROP_VISIBLE) < 1:
                        break
            if max_frames is not None and frames >= max_frames:
                break
        status = "completed"
        return final
    except cv2.error as exc:
        raise RuntimeError("OpenCV capture/render/codec failure; check device and codecs") from exc
    except KeyboardInterrupt:
        status = "interrupted"
        return final
    finally:
        capture.close()
        if writer is not None:
            writer.release()
        if not headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        try:
            if final:
                store.record([], last_index, final)
        finally:
            store.close(status)
