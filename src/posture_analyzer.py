from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from .models import PoseIndex, PoseResult, PostureAnalysis

Point = Tuple[float, float]
FeatureMap = Dict[str, Any]


class PostureAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_visibility = float(config.get("min_visibility", 0.5))
        self.trunk_min_visibility = float(config.get("trunk_min_visibility", self.min_visibility))
        self.trunk_mode = str(config.get("trunk_mode", "auto")).lower()
        self.head_samples: Deque[Tuple[float, float]] = deque()
        self.trunk_angle_samples: Deque[Tuple[float, float]] = deque()
        self.upper_body_samples: Deque[Tuple[float, float]] = deque()
        self.active_since: Dict[str, float] = {}

        self.calibration_started_at: Optional[float] = None
        self.calibration_samples: List[FeatureMap] = []
        self.upper_body_baseline: Optional[FeatureMap] = None

    @property
    def is_calibrating(self) -> bool:
        return self.calibration_started_at is not None

    @property
    def is_calibrated(self) -> bool:
        return self.upper_body_baseline is not None

    def start_calibration(self, timestamp: Optional[float] = None) -> None:
        self.calibration_started_at = timestamp if timestamp is not None else time.monotonic()
        self.calibration_samples = []
        self.upper_body_baseline = None
        self.head_samples.clear()
        self.trunk_angle_samples.clear()
        self.upper_body_samples.clear()
        self.active_since.clear()

    def analyze(self, pose_result: PoseResult, timestamp: float) -> PostureAnalysis:
        if not pose_result.valid or len(pose_result.landmarks) < 25:
            self._trim_samples(timestamp)
            return self._invalid(timestamp, pose_result, pose_result.reason or "invalid pose")

        metrics = self._compute_metrics(pose_result)
        if metrics["head_angle"] is None and metrics["trunk_angle"] is None and metrics["upper_features"] is None:
            self._trim_samples(timestamp)
            return self._invalid(timestamp, pose_result, "required keypoints are not reliable")

        if self.is_calibrating:
            return self._handle_calibration(metrics, pose_result, timestamp)

        upper_body_score = self._upper_body_score(metrics["upper_features"])
        trunk_value, trunk_signal = self._select_trunk_signal(metrics["trunk_angle"], upper_body_score)

        if metrics["head_angle"] is not None:
            self.head_samples.append((timestamp, metrics["head_angle"]))
        if metrics["trunk_angle"] is not None:
            self.trunk_angle_samples.append((timestamp, metrics["trunk_angle"]))
        if upper_body_score is not None:
            self.upper_body_samples.append((timestamp, upper_body_score))

        self._trim_samples(timestamp)
        head_smoothed = self._average(value for _, value in self.head_samples)
        trunk_angle_smoothed = self._average(value for _, value in self.trunk_angle_samples)
        upper_body_smoothed = self._average(value for _, value in self.upper_body_samples)

        if trunk_signal == "hip_angle":
            trunk_for_flags = trunk_angle_smoothed
        elif trunk_signal == "upper_body_score":
            trunk_for_flags = upper_body_smoothed
        else:
            trunk_for_flags = None

        flags = self._build_flags(head_smoothed, trunk_for_flags, trunk_signal)
        durations = self._update_durations(flags, timestamp)
        status = self._status_from_flags(flags, durations)

        return PostureAnalysis(
            timestamp=timestamp,
            valid=True,
            status=status,
            head_angle_deg=metrics["head_angle"],
            trunk_angle_deg=metrics["trunk_angle"],
            upper_body_score=upper_body_score,
            smoothed_head_angle_deg=head_smoothed,
            smoothed_trunk_angle_deg=trunk_angle_smoothed,
            smoothed_upper_body_score=upper_body_smoothed,
            head_warning_duration_sec=durations["head_warning"],
            trunk_warning_duration_sec=durations["trunk_warning"],
            combined_warning_duration_sec=durations["combined_warning"],
            confidence=metrics["confidence"],
            detected=True,
            calibrated=self.is_calibrated,
            calibration_sample_count=len(self.calibration_samples),
            trunk_signal=trunk_signal,
            flags=flags,
            message=self._message_for_status(status),
        )

    def _compute_metrics(self, pose_result: PoseResult) -> Dict[str, Any]:
        landmarks = pose_result.landmarks
        left_shoulder = self._point_if_visible(landmarks, PoseIndex.LEFT_SHOULDER)
        right_shoulder = self._point_if_visible(landmarks, PoseIndex.RIGHT_SHOULDER)
        left_hip = self._point_if_visible(landmarks, PoseIndex.LEFT_HIP, self.trunk_min_visibility)
        right_hip = self._point_if_visible(landmarks, PoseIndex.RIGHT_HIP, self.trunk_min_visibility)
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
            return {
                "head_angle": None,
                "trunk_angle": None,
                "upper_features": None,
                "confidence": confidence,
            }

        shoulder_center = self._midpoint(left_shoulder, right_shoulder)
        shoulder_width = self._distance(left_shoulder, right_shoulder)
        vertical_axis = (0.0, -1.0)

        trunk_vector = None
        trunk_angle = None
        trunk_points = self._trunk_reference_points(
            left_shoulder,
            right_shoulder,
            left_hip,
            right_hip,
        )
        if trunk_points is not None:
            trunk_start, trunk_end = trunk_points
            trunk_vector = self._vector(trunk_start, trunk_end)
            trunk_angle = self._angle_between(trunk_vector, vertical_axis)

        reference_vector = trunk_vector or vertical_axis
        neck_vector = self._vector(shoulder_center, head)
        neck_deviation = self._angle_between(neck_vector, reference_vector)
        gap_angle = self._head_gap_angle(head, shoulder_center, shoulder_width)
        head_angle = max(neck_deviation, gap_angle)
        upper_features = self._upper_body_features(
            landmarks,
            head,
            left_shoulder,
            right_shoulder,
            shoulder_center,
            shoulder_width,
        )

        return {
            "head_angle": head_angle,
            "trunk_angle": trunk_angle,
            "upper_features": upper_features,
            "confidence": confidence,
        }

    def _handle_calibration(
        self,
        metrics: Dict[str, Any],
        pose_result: PoseResult,
        timestamp: float,
    ) -> PostureAnalysis:
        assert self.calibration_started_at is not None
        features = metrics["upper_features"]
        if features is not None:
            self.calibration_samples.append(features)

        elapsed = timestamp - self.calibration_started_at
        calibration_sec = float(self.config.get("upper_body_calibration_sec", 8.0))
        min_samples = int(self.config.get("upper_body_min_samples", 20))
        progress = min(1.0, max(0.0, elapsed / max(calibration_sec, 0.1)))

        if elapsed >= calibration_sec and len(self.calibration_samples) >= min_samples:
            self.upper_body_baseline = self._average_features(self.calibration_samples)
            self.calibration_started_at = None
            self.active_since.clear()
            self.head_samples.clear()
            self.trunk_angle_samples.clear()
            self.upper_body_samples.clear()
            return PostureAnalysis(
                timestamp=timestamp,
                valid=True,
                status="normal",
                head_angle_deg=metrics["head_angle"],
                trunk_angle_deg=metrics["trunk_angle"],
                upper_body_score=0.0,
                smoothed_head_angle_deg=metrics["head_angle"],
                smoothed_trunk_angle_deg=metrics["trunk_angle"],
                smoothed_upper_body_score=0.0,
                confidence=metrics["confidence"],
                detected=True,
                calibrated=True,
                calibration_progress=1.0,
                calibration_sample_count=len(self.calibration_samples),
                trunk_signal="upper_body_score",
                message="Calibration complete",
            )

        if elapsed >= calibration_sec:
            progress = min(0.99, len(self.calibration_samples) / max(min_samples, 1))

        return PostureAnalysis(
            timestamp=timestamp,
            valid=True,
            status="calibrating",
            head_angle_deg=metrics["head_angle"],
            trunk_angle_deg=metrics["trunk_angle"],
            smoothed_head_angle_deg=metrics["head_angle"],
            smoothed_trunk_angle_deg=metrics["trunk_angle"],
            confidence=metrics["confidence"],
            detected=pose_result.valid,
            calibrated=False,
            calibrating=True,
            calibration_progress=progress,
            calibration_sample_count=len(self.calibration_samples),
            trunk_signal="upper_body_score",
            message="Hold normal sitting posture",
        )

    def _select_trunk_signal(
        self,
        trunk_angle: Optional[float],
        upper_body_score: Optional[float],
    ) -> Tuple[Optional[float], str]:
        if self.trunk_mode == "hip_based":
            return trunk_angle, "hip_angle" if trunk_angle is not None else "none"
        if self.trunk_mode == "upper_body_proxy":
            return upper_body_score, "upper_body_score" if upper_body_score is not None else "none"
        if trunk_angle is not None:
            return trunk_angle, "hip_angle"
        if upper_body_score is not None:
            return upper_body_score, "upper_body_score"
        return None, "none"

    def _build_flags(
        self,
        head_angle: Optional[float],
        trunk_value: Optional[float],
        trunk_signal: str,
    ) -> Dict[str, bool]:
        head_mild = head_angle is not None and head_angle >= float(self.config["head_down_mild_deg"])
        head_warning = head_angle is not None and head_angle >= float(self.config["head_down_warning_deg"])
        head_severe = head_angle is not None and head_angle >= float(self.config["head_down_severe_deg"])

        if trunk_signal == "upper_body_score":
            mild_threshold = float(self.config.get("upper_body_mild_score", 40))
            warning_threshold = float(self.config.get("upper_body_warning_score", 60))
            severe_threshold = float(self.config.get("upper_body_severe_score", 80))
        else:
            mild_threshold = float(self.config["trunk_flex_mild_deg"])
            warning_threshold = float(self.config["trunk_flex_warning_deg"])
            severe_threshold = float(self.config["trunk_flex_severe_deg"])

        trunk_mild = trunk_value is not None and trunk_value >= mild_threshold
        trunk_warning = trunk_value is not None and trunk_value >= warning_threshold
        trunk_severe = trunk_value is not None and trunk_value >= severe_threshold
        combined = head_warning and trunk_warning

        return {
            "head_mild": head_mild,
            "head_warning": head_warning,
            "head_severe": head_severe,
            "trunk_mild": trunk_mild,
            "trunk_warning": trunk_warning,
            "trunk_severe": trunk_severe,
            "combined_warning": combined,
            "upper_body_proxy": trunk_signal == "upper_body_score",
            "hip_based_trunk": trunk_signal == "hip_angle",
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
        while self.trunk_angle_samples and timestamp - self.trunk_angle_samples[0][0] > window:
            self.trunk_angle_samples.popleft()
        while self.upper_body_samples and timestamp - self.upper_body_samples[0][0] > window:
            self.upper_body_samples.popleft()

    def _invalid(self, timestamp: float, pose_result: PoseResult, reason: str) -> PostureAnalysis:
        self.active_since.clear()
        return PostureAnalysis(
            timestamp=timestamp,
            valid=False,
            status="invalid",
            confidence=pose_result.min_visibility,
            detected=pose_result.valid,
            calibrated=self.is_calibrated,
            calibrating=self.is_calibrating,
            calibration_sample_count=len(self.calibration_samples),
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

    def _point_if_visible(
        self,
        landmarks: Any,
        index: int,
        min_visibility: Optional[float] = None,
    ) -> Optional[Point]:
        threshold = self.min_visibility if min_visibility is None else min_visibility
        if index >= len(landmarks) or self._visibility(landmarks, index) < threshold:
            return None
        item = landmarks[index]
        return float(item.x), float(item.y)

    def _trunk_reference_points(
        self,
        left_shoulder: Point,
        right_shoulder: Point,
        left_hip: Optional[Point],
        right_hip: Optional[Point],
    ) -> Optional[Tuple[Point, Point]]:
        if left_hip is not None and right_hip is not None:
            return self._midpoint(left_hip, right_hip), self._midpoint(left_shoulder, right_shoulder)
        if left_hip is not None:
            return left_hip, left_shoulder
        if right_hip is not None:
            return right_hip, right_shoulder
        return None

    def _upper_body_features(
        self,
        landmarks: Any,
        head: Point,
        left_shoulder: Point,
        right_shoulder: Point,
        shoulder_center: Point,
        shoulder_width: float,
    ) -> Optional[FeatureMap]:
        if shoulder_width <= 1e-9:
            return None

        head_offset = (
            (head[0] - shoulder_center[0]) / shoulder_width,
            (head[1] - shoulder_center[1]) / shoulder_width,
        )
        features: FeatureMap = {
            "neck_vector": head_offset,
            "head_offset": head_offset,
            "gap_ratio": (shoulder_center[1] - head[1]) / shoulder_width,
            "shoulder_center": shoulder_center,
            "shoulder_width": shoulder_width,
            "arm_points": {},
        }

        arm_points: Dict[str, Point] = {}
        for name, index in (
            ("left_elbow", PoseIndex.LEFT_ELBOW),
            ("right_elbow", PoseIndex.RIGHT_ELBOW),
            ("left_wrist", PoseIndex.LEFT_WRIST),
            ("right_wrist", PoseIndex.RIGHT_WRIST),
        ):
            point = self._point_if_visible(landmarks, index)
            if point is not None:
                arm_points[name] = (
                    (point[0] - shoulder_center[0]) / shoulder_width,
                    (point[1] - shoulder_center[1]) / shoulder_width,
                )
        features["arm_points"] = arm_points
        return features

    def _upper_body_score(self, features: Optional[FeatureMap]) -> Optional[float]:
        if self.upper_body_baseline is None or features is None:
            return None

        baseline = self.upper_body_baseline
        weights = self.config.get("upper_body_weights", {})
        neck_weight = float(weights.get("neck_angle", 0.30))
        gap_weight = float(weights.get("head_shoulder_gap", 0.25))
        head_shift_weight = float(weights.get("head_shift", 0.20))
        shoulder_shift_weight = float(weights.get("shoulder_shift", 0.15))
        arm_weight = float(weights.get("arm_posture", 0.10))

        neck_delta = self._angle_between(features["neck_vector"], baseline["neck_vector"])
        neck_score = self._bounded_score((neck_delta - 8.0) / 22.0)

        gap_delta = float(baseline["gap_ratio"]) - float(features["gap_ratio"])
        gap_scale = max(0.18, abs(float(baseline["gap_ratio"])) * 0.35)
        gap_score = self._bounded_score(gap_delta / gap_scale)

        head_shift = self._distance(features["head_offset"], baseline["head_offset"])
        head_shift_score = self._bounded_score(head_shift / 0.35)

        shoulder_shift = self._distance(features["shoulder_center"], baseline["shoulder_center"])
        shoulder_scale = max(float(baseline["shoulder_width"]) * 0.25, 1e-9)
        shoulder_shift_score = self._bounded_score(shoulder_shift / shoulder_scale)

        arm_score = self._arm_posture_score(features["arm_points"], baseline["arm_points"])

        score = (
            neck_weight * neck_score
            + gap_weight * gap_score
            + head_shift_weight * head_shift_score
            + shoulder_shift_weight * shoulder_shift_score
            + arm_weight * arm_score
        )
        return max(0.0, min(100.0, score))

    def _arm_posture_score(self, current: Dict[str, Point], baseline: Dict[str, Point]) -> float:
        common = [name for name in current if name in baseline]
        if not common:
            return 0.0
        shifts = [self._distance(current[name], baseline[name]) for name in common]
        return self._bounded_score((self._average(shifts) or 0.0) / 0.35)

    def _average_features(self, samples: List[FeatureMap]) -> FeatureMap:
        keys = ("neck_vector", "head_offset", "shoulder_center")
        averaged: FeatureMap = {
            key: self._average_points(sample[key] for sample in samples) for key in keys
        }
        averaged["gap_ratio"] = self._average(float(sample["gap_ratio"]) for sample in samples) or 0.0
        averaged["shoulder_width"] = self._average(float(sample["shoulder_width"]) for sample in samples) or 1.0

        arm_names = sorted({name for sample in samples for name in sample["arm_points"]})
        arm_points: Dict[str, Point] = {}
        for name in arm_names:
            points = [sample["arm_points"][name] for sample in samples if name in sample["arm_points"]]
            if len(points) >= max(3, len(samples) // 4):
                arm_points[name] = self._average_points(points)
        averaged["arm_points"] = arm_points
        return averaged

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
    def _bounded_score(value: float) -> float:
        return max(0.0, min(100.0, value * 100.0))

    @staticmethod
    def _average(values: Iterable[float]) -> Optional[float]:
        values = list(values)
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _average_points(points: Iterable[Point]) -> Point:
        points = list(points)
        if not points:
            return 0.0, 0.0
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    @staticmethod
    def _message_for_status(status: str) -> str:
        messages = {
            "normal": "Posture normal",
            "calibrating": "Calibrating normal posture",
            "head_down_warning": "Head-down posture detected",
            "trunk_flex_warning": "Upper-body flex posture detected",
            "combined_warning": "Head-down and upper-body flex posture detected",
            "severe_warning": "Severe posture risk detected",
            "invalid": "Invalid pose",
        }
        return messages.get(status, status)

