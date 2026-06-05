from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class PoseIndex:
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


POSE_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (PoseIndex.LEFT_EAR, PoseIndex.LEFT_SHOULDER),
    (PoseIndex.RIGHT_EAR, PoseIndex.RIGHT_SHOULDER),
    (PoseIndex.LEFT_SHOULDER, PoseIndex.RIGHT_SHOULDER),
    (PoseIndex.LEFT_SHOULDER, PoseIndex.LEFT_ELBOW),
    (PoseIndex.LEFT_ELBOW, PoseIndex.LEFT_WRIST),
    (PoseIndex.RIGHT_SHOULDER, PoseIndex.RIGHT_ELBOW),
    (PoseIndex.RIGHT_ELBOW, PoseIndex.RIGHT_WRIST),
    (PoseIndex.LEFT_SHOULDER, PoseIndex.LEFT_HIP),
    (PoseIndex.RIGHT_SHOULDER, PoseIndex.RIGHT_HIP),
    (PoseIndex.LEFT_HIP, PoseIndex.RIGHT_HIP),
    (PoseIndex.LEFT_HIP, PoseIndex.LEFT_KNEE),
    (PoseIndex.LEFT_KNEE, PoseIndex.LEFT_ANKLE),
    (PoseIndex.RIGHT_HIP, PoseIndex.RIGHT_KNEE),
    (PoseIndex.RIGHT_KNEE, PoseIndex.RIGHT_ANKLE),
)


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0
    presence: float = 0.0

    def as_pixel(self, width: int, height: int) -> Tuple[int, int]:
        return int(self.x * width), int(self.y * height)


@dataclass
class PoseResult:
    landmarks: List[Landmark] = field(default_factory=list)
    valid: bool = False
    timestamp: float = 0.0
    reason: str = ""
    min_visibility: float = 0.0
    backend: str = "unknown"


@dataclass
class PostureAnalysis:
    timestamp: float
    valid: bool
    status: str
    head_angle_deg: Optional[float] = None
    trunk_angle_deg: Optional[float] = None
    smoothed_head_angle_deg: Optional[float] = None
    smoothed_trunk_angle_deg: Optional[float] = None
    head_warning_duration_sec: float = 0.0
    trunk_warning_duration_sec: float = 0.0
    combined_warning_duration_sec: float = 0.0
    confidence: float = 0.0
    detected: bool = False
    flags: Dict[str, bool] = field(default_factory=dict)
    message: str = ""


@dataclass
class AlertEvent:
    triggered: bool
    status: str
    message: str
    timestamp: float
    cooling_down: bool = False

