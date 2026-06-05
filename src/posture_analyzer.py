from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, Iterable, Optional, Tuple

from .models import PoseIndex, PoseResult, PostureAnalysis

Point = Tuple[float, float]


class PostureAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_visibility = float(config.get("min_visibility", 0.5))
        self.head_samples: Deque[Tuple[float, float]] = deque()
        self.trunk_samples: Deque[Tuple[float, float]] = deque()
        self.active_since: Dict[str, float] = {}

    def analyze(self, pose_result: PoseResult, timestamp: float) -> PostureAnalysis:
        if not pose_result.valid or len(pose_result.landmarks) < 25:
            self._trim_samples(timestamp)
            return self._invalid(timestamp, pose_result, pose_result.reason or "invalid pose")

        metrics = self._compute_metrics(pose_result)
        if metrics["head_angle"] is None and metrics["trunk_angle"] is None:
            self._trim_samples(timestamp)
            return self._invalid(timestamp, pose_result, "required keypoints are not reliable")

        if metrics["head_angle"] is not None:
            self.head_samples.append((timestamp, metrics["head_angle"]))
        if metrics["trunk_angle"] is not None:
            self.trunk_samples.append((timestamp, metrics["trunk_angle"]))

        self._trim_samples(timestamp)
        head_smoothed = self._average(value for _, value in self.head_samples)
        trunk_smoothed = self._average(value for _, value in self.trunk_samples)
        flags = self._build_flags(head_smoothed, trunk_smoothed)
        durations = self._update_durations(flags, timestamp)
        status = self._status_from_flags(flags, durations)

        return PostureAnalysis(
            timestamp=timestamp,
            valid=True,
            status=status,
            head_angle_deg=metrics["head_angle"],
            trunk_angle_deg=metrics["trunk_angle"],
            smoothed_head_angle_deg=head_smoothed,
            smoothed_trunk_angle_deg=trunk_smoothed,
            head_warning_duration_sec=durations["head_warning"],
            trunk_warning_duration_sec=durations["trunk_warning"],
            combined_warning_duration_sec=durations["combined_warning"],
            confidence=metrics["confidence"],
            detected=True,
            flags=flags,
            message=self._message_for_status(status),
        )

    def _compute_metrics(self, pose_result: PoseResult) -> Dict[str, Optional[float]]:
        landmarks = pose_result.landmarks
        left_shoulder = self._point_if_visible(landmarks, PoseIndex.LEFT_SHOULDER)
        right_shoulder = self._point_if_visible(landmarks, PoseIndex.RIGHT_SHOULDER)
        left_hip = self._point_if_visible(landmarks, PoseIndex.LEFT_HIP)
        right_hip = self._point_if_visible(landmarks, PoseIndex.RIGHT_HIP)
        head = self._head_point(landmarks)

        confidences = [
            self._visibility(landmarks, index)
            for index in (
                PoseIndex.NOSE,
                PoseIndex.LEFT_EAR,
                PoseIndex.RIGHT_EAR,
                PoseIndex.LEFT_SHOULDER,
                PoseIndex.RIGHT_SHOULDER,
                PoseIndex.LEFT_HIP,
                PoseIndex.RIGHT_HIP,
            )
            if index < len(landmarks)
        ]
        confidence = self._average(confidences) or 0.0

        if left_shoulder is None or right_shoulder is None or head is None:
            return {"head_angle": None, "trunk_angle": None, "confidence": confidence}

        shoulder_center = self._midpoint(left_shoulder, right_shoulder)
        shoulder_width = self._distance(left_shoulder, right_shoulder)
        vertical_axis = (0.0, -1.0)

        trunk_vector = None
        trunk_angle = None
        if left_hip is not None and right_hip is not None:
            hip_center = self._midpoint(left_hip, right_hip)
            trunk_vector = self._vector(hip_center, shoulder_center)
            trunk_angle = self._angle_between(trunk_vector, vertical_axis)

        reference_vector = trunk_vector or vertical_axis
        neck_vector = self._vector(shoulder_center, head)
        neck_deviation = self._angle_between(neck_vector, reference_vector)
        gap_angle = self._head_gap_angle(head, shoulder_center, shoulder_width)
        head_angle = max(neck_deviation, gap_angle)

        return {
            "head_angle": head_angle,
            "trunk_angle": trunk_angle,
            "confidence": confidence,
        }

    def _build_flags(
        self, head_angle: Optional[float], trunk_angle: Optional[float]
    ) -> Dict[str, bool]:
        head_mild = head_angle is not None and head_angle >= float(self.config["head_down_mild_deg"])
        head_warning = head_angle is not None and head_angle >= float(self.config["head_down_warning_deg"])
        head_severe = head_angle is not None and head_angle >= float(self.config["head_down_severe_deg"])
        trunk_mild = trunk_angle is not None and trunk_angle >= float(self.config["trunk_flex_mild_deg"])
        trunk_warning = trunk_angle is not None and trunk_angle >= float(self.config["trunk_flex_warning_deg"])
        trunk_severe = trunk_angle is not None and trunk_angle >= float(self.config["trunk_flex_severe_deg"])
        combined = head_warning and trunk_warning

        return {
            "head_mild": head_mild,
            "head_warning": head_warning,
            "head_severe": head_severe,
            "trunk_mild": trunk_mild,
            "trunk_warning": trunk_warning,
            "trunk_severe": trunk_severe,
            "combined_warning": combined,
        }

    def _update_durations(self, flags: Dict[str, bool], timestamp: float) -> Dict[str, float]:
        tracked = (
            "head_warning",
            "head_severe",
            "trunk_warning",
            "trunk_severe",
            "combined_warning",
        )
        durations: Dict[str, float] = {}
        for name in tracked:
            if flags[name]:
                self.active_since.setdefault(name, timestamp)
                durations[name] = timestamp - self.active_since[name]
            else:
                self.active_since.pop(name, None)
                durations[name] = 0.0
        return durations

    def _status_from_flags(self, flags: Dict[str, bool], durations: Dict[str, float]) -> str:
        warning_duration = float(self.config.get("warning_duration_sec", 3.0))
        severe_duration = float(self.config.get("severe_duration_sec", 2.0))
        combined_severe_duration = float(self.config.get("combined_severe_duration_sec", 5.0))

        severe = (
            flags["head_severe"]
            and durations["head_severe"] >= severe_duration
            or flags["trunk_severe"]
            and durations["trunk_severe"] >= severe_duration
            or flags["combined_warning"]
            and durations["combined_warning"] >= combined_severe_duration
        )
        if severe:
            return "severe_warning"
        if flags["combined_warning"] and durations["combined_warning"] >= warning_duration:
            return "combined_warning"
        if flags["head_warning"] and durations["head_warning"] >= warning_duration:
            return "head_down_warning"
        if flags["trunk_warning"] and durations["trunk_warning"] >= warning_duration:
            return "trunk_flex_warning"
        return "normal"

    def _trim_samples(self, timestamp: float) -> None:
        window = float(self.config.get("smoothing_window_sec", 1.0))
        while self.head_samples and timestamp - self.head_samples[0][0] > window:
            self.head_samples.popleft()
        while self.trunk_samples and timestamp - self.trunk_samples[0][0] > window:
            self.trunk_samples.popleft()

    def _invalid(self, timestamp: float, pose_result: PoseResult, reason: str) -> PostureAnalysis:
        self.active_since.clear()
        return PostureAnalysis(
            timestamp=timestamp,
            valid=False,
            status="invalid",
            confidence=pose_result.min_visibility,
            detected=pose_result.valid,
            message=reason,
        )

    def _head_point(self, landmarks: Any) -> Optional[Point]:
        nose = self._point_if_visible(landmarks, PoseIndex.NOSE)
        if nose is not None:
            return nose

        left_ear = self._point_if_visible(landmarks, PoseIndex.LEFT_EAR)
        right_ear = self._point_if_visible(landmarks, PoseIndex.RIGHT_EAR)
        if left_ear is not None and right_ear is not None:
            return self._midpoint(left_ear, right_ear)
        return None

    def _point_if_visible(self, landmarks: Any, index: int) -> Optional[Point]:
        if index >= len(landmarks) or self._visibility(landmarks, index) < self.min_visibility:
            return None
        item = landmarks[index]
        return float(item.x), float(item.y)

    @staticmethod
    def _visibility(landmarks: Any, index: int) -> float:
        item = landmarks[index]
        return max(float(getattr(item, "visibility", 0.0)), float(getattr(item, "presence", 0.0)))

    @staticmethod
    def _midpoint(first: Point, second: Point) -> Point:
        return (first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0

    @staticmethod
    def _vector(start: Point, end: Point) -> Point:
        return end[0] - start[0], end[1] - start[1]

    @staticmethod
    def _distance(first: Point, second: Point) -> float:
        return math.hypot(first[0] - second[0], first[1] - second[1])

    @staticmethod
    def _angle_between(first: Point, second: Point) -> float:
        first_len = math.hypot(first[0], first[1])
        second_len = math.hypot(second[0], second[1])
        if first_len <= 1e-9 or second_len <= 1e-9:
            return 0.0
        dot = first[0] * second[0] + first[1] * second[1]
        cos_value = max(-1.0, min(1.0, dot / (first_len * second_len)))
        return math.degrees(math.acos(cos_value))

    def _head_gap_angle(self, head: Point, shoulder_center: Point, shoulder_width: float) -> float:
        if shoulder_width <= 1e-9:
            return 0.0
        expected_ratio = float(self.config.get("head_expected_gap_ratio", 0.55))
        expected_gap = max(shoulder_width * expected_ratio, 1e-9)
        actual_gap = shoulder_center[1] - head[1]
        deficit = max(0.0, expected_gap - actual_gap)
        return math.degrees(math.atan2(deficit, expected_gap))

    @staticmethod
    def _average(values: Iterable[float]) -> Optional[float]:
        values = list(values)
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _message_for_status(status: str) -> str:
        messages = {
            "normal": "Posture normal",
            "head_down_warning": "Head-down posture detected",
            "trunk_flex_warning": "Trunk flex posture detected",
            "combined_warning": "Head-down and trunk flex posture detected",
            "severe_warning": "Severe posture risk detected",
            "invalid": "Invalid pose",
        }
        return messages.get(status, status)

