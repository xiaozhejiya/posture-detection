from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .models import AlertEvent, POSE_CONNECTIONS, PoseResult, PostureAnalysis

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None


class Visualizer:
    def __init__(self, config: Dict[str, Any]):
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed. Install requirements.txt first.")
        self.config = config

    def draw(
        self,
        frame: Any,
        pose_result: PoseResult,
        analysis: PostureAnalysis,
        fps: float,
        alert: AlertEvent,
    ) -> Any:
        height, width = frame.shape[:2]
        if self.config.get("show_skeleton", True) and pose_result.valid:
            self._draw_skeleton(frame, pose_result, width, height)

        self._draw_panel(frame, analysis, fps, alert)
        return frame

    def _draw_skeleton(self, frame: Any, pose_result: PoseResult, width: int, height: int) -> None:
        color_line = (80, 220, 120)
        color_point = (0, 210, 255)
        min_visibility = 0.45

        for start, end in POSE_CONNECTIONS:
            if start >= len(pose_result.landmarks) or end >= len(pose_result.landmarks):
                continue
            first = pose_result.landmarks[start]
            second = pose_result.landmarks[end]
            if first.visibility < min_visibility or second.visibility < min_visibility:
                continue
            cv2.line(frame, first.as_pixel(width, height), second.as_pixel(width, height), color_line, 2)

        if not self.config.get("show_landmarks", True):
            return

        for landmark in pose_result.landmarks:
            if landmark.visibility < min_visibility:
                continue
            cv2.circle(frame, landmark.as_pixel(width, height), 3, color_point, -1)

    def _draw_panel(
        self,
        frame: Any,
        analysis: PostureAnalysis,
        fps: float,
        alert: AlertEvent,
    ) -> None:
        panel_x, panel_y = 16, 16
        line_height = 26
        trunk_text = self._angle_text(analysis.smoothed_trunk_angle_deg)
        upper_body_text = self._score_text(analysis.smoothed_upper_body_score)
        if analysis.trunk_signal == "upper_body_score" and upper_body_text != "N/A":
            trunk_text = "proxy"
        rows = [
            ("Status", analysis.status),
            ("Head angle", self._angle_text(analysis.smoothed_head_angle_deg)),
            ("Trunk angle", trunk_text),
            ("Upper score", upper_body_text),
            ("Head duration", f"{analysis.head_warning_duration_sec:.1f}s"),
            ("Trunk duration", f"{analysis.trunk_warning_duration_sec:.1f}s"),
            ("Calibration", self._calibration_text(analysis)),
            ("FPS", f"{fps:.1f}"),
        ]
        panel_width = 410
        panel_height = line_height * len(rows) + 22
        cv2.rectangle(
            frame,
            (panel_x - 8, panel_y - 8),
            (panel_x + panel_width, panel_y + panel_height),
            (20, 20, 20),
            -1,
        )

        status_color = self._status_color(analysis.status)
        for index, (label, value) in enumerate(rows):
            y = panel_y + index * line_height + 18
            color = status_color if label == "Status" else (235, 235, 235)
            cv2.putText(frame, f"{label}: {value}", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

        if alert.triggered or analysis.status in {"combined_warning", "severe_warning"}:
            self._draw_alert_banner(frame, analysis.message or analysis.status)

    def _draw_alert_banner(self, frame: Any, message: str) -> None:
        height, width = frame.shape[:2]
        text = f"ALERT: {message}"
        cv2.rectangle(frame, (0, height - 58), (width, height), (0, 0, 180), -1)
        cv2.putText(
            frame,
            text,
            (24, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2,
        )

    @staticmethod
    def _angle_text(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        return f"{value:.1f} deg"

    @staticmethod
    def _score_text(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        return f"{value:.1f}"

    @staticmethod
    def _calibration_text(analysis: PostureAnalysis) -> str:
        if analysis.calibrating:
            return f"{analysis.calibration_progress * 100:.0f}% ({analysis.calibration_sample_count})"
        if analysis.calibrated:
            return "ready"
        return "not ready"

    @staticmethod
    def _status_color(status: str) -> Tuple[int, int, int]:
        if status == "normal":
            return (80, 220, 120)
        if status == "invalid":
            return (180, 180, 180)
        if status == "calibrating":
            return (0, 210, 255)
        if status == "severe_warning":
            return (0, 0, 255)
        return (0, 165, 255)
