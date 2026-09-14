"""Global assignment + constant-velocity prediction; no face/appearance recognition."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import Box, Detection, Observation


@dataclass
class _Track:
    track_id: int
    box: Box
    score: float
    last_frame: int
    streak: int = 1
    confirmed: bool = False
    velocity: tuple[float, float] = (0.0, 0.0)

    def predict(self, frame: int) -> Box:
        dt = frame - self.last_frame
        return self.box.moved(self.velocity[0] * dt, self.velocity[1] * dt)


class Tracker:
    def __init__(self, min_hits=3, max_age=15, match_distance=0.12, max_tracks=300):
        if min_hits < 1 or max_age < 1 or not 0 < match_distance <= 1:
            raise ValueError("Invalid tracker configuration")
        self.min_hits, self.max_age, self.match_distance = min_hits, max_age, match_distance
        if not 1 <= max_tracks <= 300:
            raise ValueError("max_tracks must be between 1 and 300")
        self.max_tracks = max_tracks
        self.tracks: dict[int, _Track] = {}
        self.next_id = 1
        self.total_confirmed = 0
        self.untracked_detections = 0
        self.last_frame = -1

    def reset_associations(self):
        """Do not reuse IDs or erase session totals after a camera discontinuity."""
        self.tracks.clear()

    def update(self, detections: list[Detection], frame: int, width: int, height: int):
        if frame <= self.last_frame or width <= 0 or height <= 0:
            raise ValueError("Frames must be increasing with positive dimensions")
        self.last_frame = frame
        for key in list(self.tracks):
            if frame - self.tracks[key].last_frame > self.max_age:
                del self.tracks[key]
        tracks = list(self.tracks.values())
        unmatched = set(range(len(detections)))
        gate = math.hypot(width, height) * self.match_distance
        if tracks and detections:
            # Dummy columns explicitly represent non-assignment. Invalid pairs cannot
            # steal an otherwise valid match during Hungarian optimization.
            costs = np.full((len(tracks), len(detections) + len(tracks)), 1.1)
            for i, track in enumerate(tracks):
                predicted = track.predict(frame)
                for j, detection in enumerate(detections):
                    distance = math.dist(predicted.center, detection.box.center)
                    ratio = max(
                        predicted.area / detection.box.area, detection.box.area / predicted.area
                    )
                    costs[i, j] = (
                        0.7 * distance / gate + 0.3 * (1 - predicted.iou(detection.box))
                        if distance <= gate and ratio <= 4
                        else 1e6
                    )
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols, strict=True):
                if j >= len(detections) or costs[i, j] >= 1.1:
                    continue
                track, detection = tracks[i], detections[j]
                dt = frame - track.last_frame
                old, new = track.box.center, detection.box.center
                measured = ((new[0] - old[0]) / dt, (new[1] - old[1]) / dt)
                track.velocity = tuple(
                    0.7 * m + 0.3 * v for m, v in zip(measured, track.velocity, strict=True)
                )
                track.streak = track.streak + 1 if dt == 1 else 1
                track.box, track.score, track.last_frame = detection.box, detection.score, frame
                if not track.confirmed and track.streak >= self.min_hits:
                    track.confirmed = True
                    self.total_confirmed += 1
                unmatched.remove(j)
        for index, j in enumerate(sorted(unmatched)):
            if len(self.tracks) >= self.max_tracks:
                self.untracked_detections += len(unmatched) - index
                break
            detection = detections[j]
            confirmed = self.min_hits == 1
            self.tracks[self.next_id] = _Track(
                self.next_id, detection.box, detection.score, frame, confirmed=confirmed
            )
            self.next_id += 1
            self.total_confirmed += int(confirmed)
        # Missing tracks remain internally, but are never drawn or counted as visible.
        return [
            Observation(t.track_id, t.box, t.score)
            for t in self.tracks.values()
            if t.confirmed and t.last_frame == frame
        ]
