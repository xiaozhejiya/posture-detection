from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

from .models import AlertEvent, PoseResult, PostureAnalysis


class PostureLogger:
    def __init__(self, config: Dict[str, Any], video_config: Dict[str, Any]):
        self.config = config
        self.video_config = video_config
        self.enabled = bool(config.get("enable", True))
        self.format = str(config.get("format", "csv")).lower()
        self.write_each_frame = bool(config.get("write_each_frame", True))
        self.enable_event_log = bool(config.get("enable_event_log", True))
        self.file: Optional[TextIO] = None
        self.writer: Optional[csv.DictWriter] = None
        self.event_file: Optional[TextIO] = None
        self.event_writer: Optional[csv.DictWriter] = None

        if not self.enabled:
            return

        log_dir = Path(str(config.get("log_dir", "data/logs")))
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "jsonl" if self.format == "jsonl" else "csv"
        self.path = log_dir / f"posture_{stamp}.{suffix}"
        self.file = self.path.open("a", encoding="utf-8", newline="")

        if suffix == "csv":
            self.writer = csv.DictWriter(self.file, fieldnames=list(self._empty_row().keys()))
            self.writer.writeheader()
        if self.enable_event_log:
            event_prefix = str(config.get("event_log_prefix", "posture_events"))
            self.event_path = log_dir / f"{event_prefix}_{stamp}.{suffix}"
            self.event_file = self.event_path.open("a", encoding="utf-8", newline="")
            if suffix == "csv":
                self.event_writer = csv.DictWriter(
                    self.event_file,
                    fieldnames=list(self._empty_event_row().keys()),
                )
                self.event_writer.writeheader()

    def write(
        self,
        pose_result: PoseResult,
        analysis: PostureAnalysis,
        alert: AlertEvent,
        fps: float,
    ) -> None:
        if not self.enabled:
            return
        if self.file is not None and (self.write_each_frame or alert.triggered):
            row = self._row(pose_result, analysis, alert, fps)
            if self.writer is not None:
                self.writer.writerow(row)
            else:
                self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.file.flush()

        if self.event_file is not None and self._should_write_event(analysis, alert):
            event_row = self._event_row(analysis, alert, fps)
            if self.event_writer is not None:
                self.event_writer.writerow(event_row)
            else:
                self.event_file.write(json.dumps(event_row, ensure_ascii=False) + "\n")
            self.event_file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
        if self.event_file is not None:
            self.event_file.close()
            self.event_file = None

    def _empty_row(self) -> Dict[str, Any]:
        return {
            "time": "",
            "video_source_type": "",
            "detected": "",
            "valid": "",
            "head_angle_deg": "",
            "trunk_angle_deg": "",
            "upper_body_score": "",
            "smoothed_head_angle_deg": "",
            "smoothed_trunk_angle_deg": "",
            "smoothed_upper_body_score": "",
            "status": "",
            "alert_triggered": "",
            "alert_cooling_down": "",
            "trunk_signal": "",
            "calibrated": "",
            "calibrating": "",
            "calibration_progress": "",
            "calibration_sample_count": "",
            "confidence": "",
            "fps": "",
            "message": "",
        }

    def _empty_event_row(self) -> Dict[str, Any]:
        return {
            "time": "",
            "video_source_type": "",
            "event_type": "",
            "status": "",
            "message": "",
            "head_angle_deg": "",
            "trunk_angle_deg": "",
            "upper_body_score": "",
            "smoothed_head_angle_deg": "",
            "smoothed_trunk_angle_deg": "",
            "smoothed_upper_body_score": "",
            "head_warning_duration_sec": "",
            "trunk_warning_duration_sec": "",
            "combined_warning_duration_sec": "",
            "trunk_signal": "",
            "active_flags": "",
            "confidence": "",
            "fps": "",
        }

    def _row(
        self,
        pose_result: PoseResult,
        analysis: PostureAnalysis,
        alert: AlertEvent,
        fps: float,
    ) -> Dict[str, Any]:
        return {
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "video_source_type": self.video_config.get("type", ""),
            "detected": pose_result.valid,
            "valid": analysis.valid,
            "head_angle_deg": self._fmt(analysis.head_angle_deg),
            "trunk_angle_deg": self._fmt(analysis.trunk_angle_deg),
            "upper_body_score": self._fmt(analysis.upper_body_score),
            "smoothed_head_angle_deg": self._fmt(analysis.smoothed_head_angle_deg),
            "smoothed_trunk_angle_deg": self._fmt(analysis.smoothed_trunk_angle_deg),
            "smoothed_upper_body_score": self._fmt(analysis.smoothed_upper_body_score),
            "status": analysis.status,
            "alert_triggered": alert.triggered,
            "alert_cooling_down": alert.cooling_down,
            "trunk_signal": analysis.trunk_signal,
            "calibrated": analysis.calibrated,
            "calibrating": analysis.calibrating,
            "calibration_progress": self._fmt(analysis.calibration_progress),
            "calibration_sample_count": analysis.calibration_sample_count,
            "confidence": self._fmt(analysis.confidence),
            "fps": self._fmt(fps),
            "message": analysis.message,
        }

    def _event_row(
        self,
        analysis: PostureAnalysis,
        alert: AlertEvent,
        fps: float,
    ) -> Dict[str, Any]:
        return {
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "video_source_type": self.video_config.get("type", ""),
            "event_type": self._event_type(analysis),
            "status": alert.status,
            "message": alert.message,
            "head_angle_deg": self._fmt(analysis.head_angle_deg),
            "trunk_angle_deg": self._fmt(analysis.trunk_angle_deg),
            "upper_body_score": self._fmt(analysis.upper_body_score),
            "smoothed_head_angle_deg": self._fmt(analysis.smoothed_head_angle_deg),
            "smoothed_trunk_angle_deg": self._fmt(analysis.smoothed_trunk_angle_deg),
            "smoothed_upper_body_score": self._fmt(analysis.smoothed_upper_body_score),
            "head_warning_duration_sec": self._fmt(analysis.head_warning_duration_sec),
            "trunk_warning_duration_sec": self._fmt(analysis.trunk_warning_duration_sec),
            "combined_warning_duration_sec": self._fmt(analysis.combined_warning_duration_sec),
            "trunk_signal": analysis.trunk_signal,
            "active_flags": ",".join(sorted(name for name, active in analysis.flags.items() if active)),
            "confidence": self._fmt(analysis.confidence),
            "fps": self._fmt(fps),
        }

    @staticmethod
    def _should_write_event(analysis: PostureAnalysis, alert: AlertEvent) -> bool:
        return alert.triggered and analysis.status in {
            "head_down_warning",
            "trunk_flex_warning",
            "combined_warning",
            "severe_warning",
        }

    @staticmethod
    def _event_type(analysis: PostureAnalysis) -> str:
        if analysis.status == "head_down_warning":
            return "head_down"
        if analysis.status == "trunk_flex_warning":
            return "trunk_flex"
        if analysis.status == "combined_warning":
            return "combined"
        if analysis.flags.get("combined_warning"):
            return "severe_combined"
        if analysis.flags.get("head_severe") or analysis.flags.get("head_warning"):
            return "severe_head_down"
        if analysis.flags.get("trunk_severe") or analysis.flags.get("trunk_warning"):
            return "severe_trunk_flex"
        return "severe"

    @staticmethod
    def _fmt(value: Optional[float]) -> str:
        if value is None:
            return ""
        return f"{value:.2f}"
