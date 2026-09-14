"""Live overlay: green bounding boxes, anonymous IDs, directional line and metrics."""

import cv2

from .config import Settings


def render(frame, observations, counter, settings: Settings, fps: float, unique_tracks: int):
    image = frame.copy()
    height, width = image.shape[:2]
    a = tuple(round(v * d) for v, d in zip(settings.line_start, (width, height), strict=True))
    b = tuple(round(v * d) for v, d in zip(settings.line_end, (width, height), strict=True))
    cv2.line(image, a, b, (0, 200, 255), 2)
    mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = max(1, (dx * dx + dy * dy) ** 0.5)
    end = (round(mid[0] - 35 * dy / length), round(mid[1] + 35 * dx / length))
    cv2.arrowedLine(image, mid, end, (0, 200, 255), 2, tipLength=0.25)
    for observation in observations:
        box = observation.box
        cv2.rectangle(
            image, (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2)), (0, 255, 0), 2
        )
        cv2.putText(
            image,
            f"Pessoa #{observation.track_id} | {observation.score:.2f}",
            (max(0, round(box.x1)), max(18, round(box.y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(image, tuple(round(v) for v in box.anchor), 4, (0, 255, 0), -1)
    labels = [
        f"VISIVEIS {len(observations)} | ENTRADAS {counter.entries} | SAIDAS {counter.exits}",
        f"OCUPACAO EST. {counter.occupancy} | TRAJETORIAS {unique_tracks} | FPS {fps:.1f}",
        f"{settings.detector.upper()} | Q/ESC: sair | seta: entrada | "
        f"alertas: {counter.underflow_events}",
    ]
    cv2.rectangle(image, (0, 0), (width, min(height, 86)), (20, 28, 24), -1)
    for index, text in enumerate(labels):
        cv2.putText(
            image,
            text,
            (10, 23 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (235, 245, 240),
            1,
            cv2.LINE_AA,
        )
    return image
