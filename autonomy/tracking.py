"""Pure tracking and motion-control logic (no camera or ML dependencies)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass(frozen=True)
class Control:
    forward: float = 0.0
    theta: float = 0.0
    wrist_delta: float = 0.0
    mode: str = "search"


def closest(detections: Iterable[Detection]) -> Detection | None:
    """Choose the closest-looking object: largest image area, then lowest edge."""
    return max(detections, key=lambda d: (d.area, d.y2), default=None)


def associated(previous: Detection, detections: Iterable[Detection], width: int, height: int) -> Detection | None:
    """Find the observation belonging to a locked target, ignoring label aliases."""
    candidates = list(detections)
    if not candidates or width <= 0 or height <= 0:
        return None
    px, py = previous.center
    diagonal = max(1.0, (width * width + height * height) ** 0.5)

    def metrics(item):
        left, top = max(previous.x1, item.x1), max(previous.y1, item.y1)
        right, bottom = min(previous.x2, item.x2), min(previous.y2, item.y2)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        union = previous.area + item.area - intersection
        iou = intersection / union if union else 0.0
        x, y = item.center
        distance = ((x - px) ** 2 + (y - py) ** 2) ** 0.5 / diagonal
        return iou, distance

    ranked = [(metrics(item), item) for item in candidates]
    (iou, distance), best = max(ranked, key=lambda pair: (pair[0][0], -pair[0][1]))
    return best if iou >= 0.10 or distance <= 0.18 else None


def track(
    detection: Detection | None,
    width: int,
    height: int,
    *,
    target_height_ratio: float = 0.28,
    center_deadband: float = 0.06,
    distance_deadband: float = 0.025,
    max_forward: float = 0.07,
    max_theta: float = 9.0,
    min_theta: float = 4.0,
    wrist_gain: float = 0.035,
    patrol_theta: float = 4.0,
    ignore_distance: bool = False,
    steer_band: float = 0.0,
    vertical_band: float = 0.0,
) -> Control:
    """Base/wrist command toward `detection`.

    With `steer_band` > 0 (and ignore_distance), the base keeps driving while the target is within that horizontal
    error, steering proportionally and slowing as the error grows, instead of stopping to turn in place whenever the
    target leaves the centre deadband. `vertical_band` > 0 also slows the drive as the target sits off-centre
    vertically, so the wrist can keep it centred instead of it sliding off the bottom of the frame. Mode "approach"
    still means centred within `center_deadband`; steering while driving reports mode "steer".
    """
    if detection is None or width <= 0 or height <= 0:
        return Control(theta=patrol_theta, mode="search")

    cx, cy = detection.center
    x_error = (cx - width / 2) / (width / 2)
    y_error = (cy - height / 2) / (height / 2)
    height_ratio = max(0.0, detection.y2 - detection.y1) / height

    if abs(x_error) <= center_deadband:
        theta = 0.0
    else:
        theta = max(min_theta, min(max_theta, abs(x_error) * max_theta)) * (1 if x_error > 0 else -1)
    distance_error = target_height_ratio - height_ratio
    centered = abs(x_error) <= center_deadband
    if ignore_distance and steer_band > 0 and abs(x_error) <= steer_band:
        slow = 1.0 - abs(x_error) / steer_band
        if vertical_band > 0:
            slow *= max(0.0, 1.0 - abs(y_error) / vertical_band)
        # Moving wheels need no minimum turn rate: steer proportionally (full rate at the band edge).
        steer = 0.0 if centered else max(-max_theta, min(max_theta, x_error / steer_band * max_theta))
        return Control(forward=max_forward * slow, theta=steer, wrist_delta=y_error * wrist_gain,
                       mode="approach" if centered else "steer")
    forward = 0.0
    mode = "align"
    if centered and ignore_distance:
        forward = max_forward
        mode = "approach"
    elif centered and distance_error > distance_deadband:
        forward = min(max_forward, max_forward * distance_error / target_height_ratio)
        mode = "approach"
    elif centered and distance_error < -distance_deadband:
        # Do not back over unknown obstacles; stop if the target is too close.
        mode = "too_close"
    elif centered:
        mode = "arrived"

    return Control(forward=forward, theta=theta, wrist_delta=y_error * wrist_gain, mode=mode)
