import numpy as np
import pytest

from people_counter.types import Box, Detection, Observation


@pytest.fixture
def frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def detection(x, y, score=0.9):
    return Detection(Box(x - 10, y - 50, x + 10, y), score)


def observation(x, y, track_id=1):
    return Observation(track_id, Box(x - 10, y - 50, x + 10, y), 0.9)
