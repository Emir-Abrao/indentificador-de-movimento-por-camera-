import json
import sqlite3

import cv2
import numpy as np
import pytest

from people_counter.app import run
from people_counter.capture import Frame
from people_counter.cli import main
from people_counter.config import Settings
from people_counter.counting import LineCounter
from people_counter.overlay import render

from .conftest import detection, observation


class MemoryCapture:
    fps = 30
    reconnects = 0

    def __init__(self, frame, count=12):
        self.frame, self.count, self.closed = frame, count, False

    def __iter__(self):
        for i in range(self.count):
            yield Frame(self.frame.copy(), i, i / 30, False)

    def close(self):
        self.closed = True


class SequenceDetector:
    def __init__(self):
        self.frame = -1

    def detect(self, _):
        self.frame += 1
        return [detection(300, 140 + self.frame * 18)]


def test_pipeline_count_and_database(frame, tmp_path):
    capture = MemoryCapture(frame)
    db = tmp_path / "counts.db"
    result = run(
        Settings(stable_frames=2, min_hits=2),
        database=db,
        headless=True,
        capture=capture,
        detector=SequenceDetector(),
    )
    assert result["frames"] == 12 and result["entries"] == 1
    assert result["visible"] == result["confirmed_trajectories"] == 1
    assert capture.closed
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM crossings").fetchone()[0] == 1
        assert connection.execute("SELECT status FROM sessions").fetchone()[0] == "completed"
        assert connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 1


def test_detector_failure_releases_camera_and_marks_failed(frame, tmp_path):
    class Broken:
        def detect(self, _):
            raise RuntimeError("Detector fault")

    db, capture = tmp_path / "counts.db", MemoryCapture(frame)
    with pytest.raises(RuntimeError, match="Detector fault"):
        run(Settings(), database=db, headless=True, capture=capture, detector=Broken())
    assert capture.closed
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT status FROM sessions").fetchone()[0] == "failed"


def test_keyboard_interrupt_cleanup(frame, tmp_path):
    class Interrupted:
        def detect(self, _):
            raise KeyboardInterrupt

    capture = MemoryCapture(frame)
    assert (
        run(
            Settings(),
            database=tmp_path / "counts.db",
            headless=True,
            capture=capture,
            detector=Interrupted(),
        )
        == {}
    )
    assert capture.closed


def test_green_rectangles_are_rendered_without_mutating_source(frame):
    settings = Settings()
    result = render(frame, [observation(200, 300)], LineCounter(settings), settings, 30, 1)
    assert np.array_equal(result[270, 190], [0, 255, 0])
    assert frame.sum() == 0


def test_real_opencv_video_io_and_hog_inference(tmp_path, frame):
    source = tmp_path / "blank.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 15, (640, 480))
    assert writer.isOpened(), "MJPEG codec required for the integration test"
    for _ in range(5):
        writer.write(frame)
    writer.release()
    output = tmp_path / "annotated.mp4"
    result = run(
        Settings(detector="hog"),
        source=str(source),
        headless=True,
        database=tmp_path / "counts.db",
        output_video=str(output),
    )
    assert result["frames"] == 5 and result["visible"] == 0
    reader = cv2.VideoCapture(str(output))
    assert reader.isOpened() and int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    ok, annotated = reader.read()
    reader.release()
    assert ok and annotated.sum() > 0


def test_existing_recording_never_overwritten(frame, tmp_path):
    path = tmp_path / "keep.mp4"
    path.write_bytes(b"important")
    with pytest.raises(FileExistsError):
        run(
            Settings(),
            database=tmp_path / "counts.db",
            headless=True,
            output_video=str(path),
            detector=SequenceDetector(),
            capture=MemoryCapture(frame),
        )
    assert path.read_bytes() == b"important"


def test_max_frames_and_resize(tmp_path):
    frame = np.zeros((720, 1920, 3), dtype=np.uint8)
    seen = []

    class Detector:
        def detect(self, image):
            seen.append(image.shape)
            return []

    capture = MemoryCapture(frame)
    result = run(
        Settings(max_width=640),
        headless=True,
        database=tmp_path / "counts.db",
        detector=Detector(),
        capture=capture,
        max_frames=2,
    )
    assert result["frames"] == 2 and seen == [(240, 640, 3), (240, 640, 3)]
    assert capture.closed


def test_camera_discontinuity_resets_association_without_false_count(frame, tmp_path):
    class Discontinuous(MemoryCapture):
        def __iter__(self):
            for i in range(12):
                yield Frame(self.frame, i, i / 30, i == 7)

    result = run(
        Settings(stable_frames=2, min_hits=1),
        headless=True,
        database=tmp_path / "counts.db",
        detector=SequenceDetector(),
        capture=Discontinuous(frame),
    )
    assert result["entries"] == 0 and result["confirmed_trajectories"] == 2


def test_cli_doctor_and_validation(tmp_path, capsys, monkeypatch):
    assert main(["doctor"]) == 0
    assert "python" in json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "export",
                "--database",
                str(tmp_path / "absent.db"),
                "--output",
                str(tmp_path / "out.csv"),
            ]
        )
        == 2
    )
    monkeypatch.delenv("NO_CAMERA_VAR", raising=False)
    assert main(["run", "--source-env", "NO_CAMERA_VAR"]) == 2


def test_invalid_output_and_limit():
    with pytest.raises(ValueError):
        run(Settings(), headless=True, max_frames=0)
    with pytest.raises(ValueError):
        run(Settings(), headless=True, output_video="bad.avi")


def test_gui_quit_path_without_physical_display(frame, tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":test")
    windows = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: windows.append(args[0]))
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *args: ord("q"))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: windows.clear())
    capture = MemoryCapture(frame)
    result = run(
        Settings(), database=tmp_path / "counts.db", detector=SequenceDetector(), capture=capture
    )
    assert result["frames"] == 1 and capture.closed and not windows


def test_gui_missing_backend_explained(frame, tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":test")

    def broken(*args):
        raise cv2.error("No GUI")

    monkeypatch.setattr(cv2, "namedWindow", broken)
    monkeypatch.setattr(cv2, "destroyAllWindows", broken)
    capture = MemoryCapture(frame)
    with pytest.raises(RuntimeError, match="GUI unavailable"):
        run(
            Settings(),
            database=tmp_path / "counts.db",
            detector=SequenceDetector(),
            capture=capture,
        )
    assert capture.closed


def test_cli_success_and_export(tmp_path, monkeypatch, capsys):
    from people_counter import app
    from people_counter.storage import Store

    monkeypatch.setattr(app, "run", lambda *args: {"frames": 10})
    assert main(["run", "--detector", "hog", "--headless"]) == 0
    assert json.loads(capsys.readouterr().out)["frames"] == 10
    path = tmp_path / "counts.db"
    store = Store(path, "test", {})
    store.close()
    assert main(["export", "--database", str(path), "--output", str(tmp_path / "out.csv")]) == 0
