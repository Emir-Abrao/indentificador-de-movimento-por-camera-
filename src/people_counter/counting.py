"""Finite directed-line crossing with deadband, stability and gap suppression."""

import math
from dataclasses import dataclass

from .config import Settings
from .types import Crossing, Observation, Point


@dataclass
class _State:
    side: int = 0
    anchor: Point | None = None
    pending_side: int = 0
    pending_count: int = 0
    last_seen: int = -1
    last_event: int = -1_000_000


class LineCounter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.states: dict[int, _State] = {}
        self.entries = self.exits = self.sequence = self.underflow_events = 0
        self.occupancy = settings.initial_occupancy

    @property
    def net_flow(self):
        return self.entries - self.exits

    def reset_associations(self):
        self.states.clear()

    def update(
        self,
        observations: list[Observation],
        frame: int,
        width: int,
        height: int,
        media_seconds: float,
        active_ids: set[int],
    ) -> list[Crossing]:
        self.states = {k: s for k, s in self.states.items() if k in active_ids}
        a = (self.settings.line_start[0] * width, self.settings.line_start[1] * height)
        b = (self.settings.line_end[0] * width, self.settings.line_end[1] * height)
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        band = self.settings.deadband * min(width, height)
        events = []
        for observation in observations:
            p = observation.box.anchor
            distance = (dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / length
            side = 1 if distance > band else (-1 if distance < -band else 0)
            state = self.states.setdefault(observation.track_id, _State())
            if frame - state.last_seen > self.settings.max_crossing_gap:
                # We cannot infer a passage during a long detection/camera gap.
                state.side, state.anchor = 0, None
                state.pending_side, state.pending_count = 0, 0
            state.last_seen = frame
            if not side:
                state.pending_count, state.pending_side = 0, 0
                continue
            if side == state.side:
                state.anchor = p
                state.pending_count, state.pending_side = 0, 0
                continue
            if state.pending_side == side:
                state.pending_count += 1
            else:
                state.pending_side, state.pending_count = side, 1
            if state.pending_count < self.settings.stable_frames:
                continue
            crossing_point = self._intersection(state.anchor, p, a, b) if state.side else None
            if (
                crossing_point is not None
                and frame - state.last_event >= self.settings.cooldown_frames
            ):
                direction = "in" if side > 0 else "out"
                self.entries += int(direction == "in")
                self.exits += int(direction == "out")
                self.sequence += 1
                events.append(
                    Crossing(
                        self.sequence,
                        observation.track_id,
                        direction,
                        frame,
                        media_seconds,
                        crossing_point[0] / width,
                        crossing_point[1] / height,
                    )
                )
                state.last_event = frame
            state.side, state.anchor = side, p
            state.pending_count, state.pending_side = 0, 0
        # Same-frame observations have no reliable ordering. Apply the net change
        # once so occupancy does not depend on detector output ordering.
        balance = self.occupancy + sum(1 if e.direction == "in" else -1 for e in events)
        self.underflow_events += max(0, -balance)
        self.occupancy = max(0, balance)
        return events

    @staticmethod
    def _intersection(p: Point | None, q: Point, a: Point, b: Point) -> Point | None:
        if p is None:
            return None
        dx, dy = b[0] - a[0], b[1] - a[1]
        first = dx * (p[1] - a[1]) - dy * (p[0] - a[0])
        second = dx * (q[1] - a[1]) - dy * (q[0] - a[0])
        if first == second:
            return None
        t = first / (first - second)
        intersection = (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
        along = ((intersection[0] - a[0]) * dx + (intersection[1] - a[1]) * dy) / (
            dx * dx + dy * dy
        )
        return intersection if 0 <= t <= 1 and 0 <= along <= 1 else None
