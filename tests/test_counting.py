import pytest

from people_counter.config import Settings
from people_counter.counting import LineCounter

from .conftest import observation


def counter(**kwargs):
    return LineCounter(
        Settings(
            line_start=(0.2, 0.5),
            line_end=(0.8, 0.5),
            deadband=0.02,
            stable_frames=2,
            cooldown_frames=4,
            **kwargs,
        )
    )


def feed(c, ys, x=300, start=0, track_id=1):
    events = []
    for frame, y in enumerate(ys, start):
        events.extend(
            c.update([observation(x, y, track_id)], frame, 640, 480, frame / 30, {track_id})
        )
    return events


def test_in_crossing_once_not_every_frame():
    c = counter()
    events = feed(c, [200, 200, 220, 245, 265, 270, 280, 290, 290])
    assert len(events) == 1 and events[0].direction == "in"
    assert events[0].y == pytest.approx(0.5)
    assert c.entries == c.occupancy == 1 and c.exits == 0


def test_out_crossing_and_underflow_is_reported():
    c = counter()
    events = feed(c, [280, 280, 260, 245, 220, 215])
    assert events[0].direction == "out"
    assert c.exits == 1 and c.occupancy == 0 and c.underflow_events == 1
    assert c.net_flow == -1


def test_initial_occupancy():
    c = counter(initial_occupancy=5)
    feed(c, [280, 280, 220, 220])
    assert c.occupancy == 4 and c.underflow_events == 0


def test_jitter_deadband_and_spawning_on_line_do_not_count():
    c = counter()
    assert not feed(c, [240, 239, 241, 238, 242, 243, 237] * 20)
    assert c.entries == c.exits == 0


def test_stationary_person_on_one_side_not_an_entry():
    c = counter()
    assert not feed(c, [280] * 50)


def test_crossing_outside_segment_ignored():
    c = counter()
    assert not feed(c, [200, 200, 280, 280], x=50)


def test_long_gap_cannot_invent_crossing():
    c = counter()
    feed(c, [200, 200])
    assert not feed(c, [280, 280], start=20)


def test_short_gap_crossing():
    c = counter()
    feed(c, [200, 200])
    assert len(feed(c, [280, 280], start=3)) == 1


def test_reentry_after_cooldown():
    c = counter()
    events = feed(c, [200, 200, 280, 280, 280, 280, 220, 220, 220, 220, 280, 280])
    assert [e.direction for e in events] == ["in", "out", "in"]
    assert c.occupancy == 1


def test_jitter_outside_band_needs_stability():
    c = counter()
    feed(c, [200, 200])
    assert not feed(c, [260, 220, 260, 220, 260, 220], start=2)


def test_simultaneous_opposite_passages():
    c = counter(initial_occupancy=1)
    events = []
    for frame, (one, two) in enumerate([(200, 280), (200, 280), (280, 200), (280, 200)]):
        events.extend(
            c.update(
                [observation(200, one, 1), observation(400, two, 2)],
                frame,
                640,
                480,
                frame / 30,
                {1, 2},
            )
        )
    assert {e.direction for e in events} == {"in", "out"}
    assert c.occupancy == 1


def test_reset_and_pruning():
    c = counter()
    feed(c, [200, 200, 280, 280])
    c.reset_associations()
    assert c.entries == 1 and not c.states
    feed(c, [200, 200], start=10)
    c.update([], 12, 640, 480, 1, set())
    assert not c.states


def test_vertical_line_and_reversed_direction():
    c = LineCounter(
        Settings(line_start=(0.5, 0.1), line_end=(0.5, 0.9), stable_frames=1, deadband=0)
    )
    c.update([observation(350, 200)], 0, 640, 480, 0, {1})
    event = c.update([observation(280, 200)], 1, 640, 480, 0.1, {1})[0]
    assert event.direction == "in" and event.x == pytest.approx(0.5)


def test_simultaneous_balance_independent_of_detector_order():
    for reverse in (False, True):
        c = counter()
        for frame, (one, two) in enumerate([(200, 280), (200, 280), (280, 200), (280, 200)]):
            people = [observation(200, one, 1), observation(400, two, 2)]
            c.update(people[::-1] if reverse else people, frame, 640, 480, frame / 30, {1, 2})
        assert c.entries == c.exits == 1 and c.occupancy == 0


def test_intersection_degenerate_and_missing_anchor():
    assert LineCounter._intersection(None, (0, 0), (0, 0), (1, 0)) is None
    assert LineCounter._intersection((0, 0), (1, 0), (0, 0), (1, 0)) is None
