import numpy as np
import pytest

from people_counter.capture import Capture
from people_counter.config import Settings


class FakeCap:
    def __init__(self, frames=(), opened=True, fps=30):
        self.frames, self.opened, self.fps = iter(frames), opened, fps
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, _):
        return self.fps

    def read(self):
        item = next(self.frames, None)
        return item is not None, item

    def release(self):
        self.released = True


def test_camera_failure_bounded_releases_every_handle():
    caps = []

    def factory(*_):
        caps.append(FakeCap(opened=False))
        return caps[-1]

    capture = Capture("0", Settings(max_reconnects=2), factory, lambda _: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        list(capture)
    assert len(caps) == 3 and all(cap.released for cap in caps)


def test_reconnect_marks_boundary_and_monotonic_frames(frame):
    caps = [FakeCap([frame]), FakeCap([frame])]
    source = Capture("0", Settings(max_reconnects=1), lambda *_: caps.pop(0), lambda _: None)
    iterator = iter(source)
    first, second = next(iterator), next(iterator)
    assert first.index == 0 and not first.discontinuity
    assert second.index == 1 and second.discontinuity
    assert second.media_seconds >= first.media_seconds
    iterator.close()


def test_file_eof_no_retry_and_timestamp_uses_source_fps(tmp_path, frame):
    file = tmp_path / "test.avi"
    file.touch()
    cap = FakeCap([frame, frame], fps=20)
    source = Capture(str(file), Settings(), lambda *_: cap)
    frames = list(source)
    assert len(frames) == 2 and frames[1].media_seconds == 0.05
    assert source.reconnects == 0 and cap.released


def test_empty_file_and_missing_file_errors(tmp_path):
    file = tmp_path / "test.avi"
    with pytest.raises(FileNotFoundError):
        Capture(str(file), Settings())
    file.touch()
    cap = FakeCap()
    with pytest.raises(RuntimeError, match="decoded"):
        list(Capture(str(file), Settings(), lambda *_: cap))
    assert cap.released


def test_network_timeouts_passed_to_backend(frame):
    args = []

    def factory(*values):
        args.append(values)
        return FakeCap([frame], fps=float("nan"))

    capture = Capture("rtsp://user:secret@camera/live", Settings(), factory)
    iterator = iter(capture)
    assert next(iterator).image is frame
    assert len(args[0]) == 3 and capture.fps == 30
    iterator.close()


def test_empty_frames_are_capture_failures():
    cap = FakeCap([np.zeros((0, 0, 3), dtype=np.uint8)])
    capture = Capture("0", Settings(max_reconnects=0), lambda *_: cap)
    with pytest.raises(RuntimeError):
        list(capture)
    assert cap.released
