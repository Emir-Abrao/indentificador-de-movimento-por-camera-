import pytest

from people_counter.tracking import Tracker

from .conftest import detection


def test_confirmation_and_stationary_id():
    tracker = Tracker(min_hits=3)
    assert tracker.update([detection(200, 200)], 0, 640, 480) == []
    assert tracker.update([detection(200, 200)], 1, 640, 480) == []
    for frame in range(2, 50):
        visible = tracker.update([detection(200, 200)], frame, 640, 480)
        assert [o.track_id for o in visible] == [1]
    assert tracker.total_confirmed == 1


def test_two_people_keep_ids_when_detector_order_changes():
    tracker = Tracker(min_hits=1)
    first = tracker.update([detection(100, 150), detection(500, 200)], 0, 640, 480)
    second = tracker.update([detection(496, 200), detection(104, 150)], 1, 640, 480)
    assert len(second) == 2
    assert first[0].track_id == second[0].track_id
    assert second[0].box.anchor[0] == 104
    assert second[1].box.anchor[0] == 496


def test_global_assignment_and_velocity_at_crossing():
    tracker = Tracker(min_hits=1)
    for frame in range(10):
        left, right = 100 + frame * 15, 385 - frame * 15
        observations = tracker.update(
            [detection(right, 230), detection(left, 200)], frame, 640, 480
        )
    assert len(observations) == 2
    assert tracker.total_confirmed == 2
    assert observations[0].box.anchor == (250, 230)
    assert observations[1].box.anchor == (235, 200)


def test_missed_tracks_not_visible_then_reassociated():
    tracker = Tracker(min_hits=1, max_age=4)
    tracker.update([detection(100, 200)], 0, 640, 480)
    assert not tracker.update([], 1, 640, 480)
    assert tracker.update([detection(102, 200)], 2, 640, 480)[0].track_id == 1
    assert tracker.total_confirmed == 1


def test_expired_track_gets_new_id():
    tracker = Tracker(min_hits=1, max_age=2)
    tracker.update([detection(100, 200)], 0, 640, 480)
    tracker.update([], 1, 640, 480)
    tracker.update([], 2, 640, 480)
    assert tracker.update([detection(100, 200)], 3, 640, 480)[0].track_id == 2


def test_gate_rejects_teleport_and_clear_preserves_totals():
    tracker = Tracker(min_hits=1)
    tracker.update([detection(30, 200)], 0, 640, 480)
    assert tracker.update([detection(600, 200)], 1, 640, 480)[0].track_id == 2
    tracker.reset_associations()
    assert tracker.update([detection(600, 200)], 2, 640, 480)[0].track_id == 3
    assert tracker.total_confirmed == 3


def test_confirmation_requires_consecutive_frames():
    tracker = Tracker(min_hits=2)
    tracker.update([detection(100, 200)], 0, 640, 480)
    tracker.update([], 1, 640, 480)
    assert not tracker.update([detection(100, 200)], 2, 640, 480)
    assert tracker.update([detection(100, 200)], 3, 640, 480)


def test_bad_frame_order():
    tracker = Tracker()
    tracker.update([], 1, 640, 480)
    with pytest.raises(ValueError):
        tracker.update([], 1, 640, 480)
    with pytest.raises(ValueError):
        Tracker(min_hits=0)


def test_memory_capacity_is_bounded_and_drops_are_observable():
    tracker = Tracker(min_hits=1, max_tracks=2)
    visible = tracker.update(
        [detection(30, 200), detection(300, 200), detection(600, 200)], 0, 640, 480
    )
    assert len(visible) == len(tracker.tracks) == 2
    assert tracker.untracked_detections == 1
    with pytest.raises(ValueError):
        Tracker(max_tracks=0)
