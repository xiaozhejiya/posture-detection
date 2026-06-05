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
        self.file: Optional[TextIO] = None
        self.writer: Optional[csv.DictWriter] = None

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

    def write(
        self,
        pose_result: PoseResult,
        analysis: PostureAnalysis,
        alert: AlertEvent,
        fps: float,
    ) -> None:
        if not self.enabled or self.file is None:
            return
        if not self.write_each_frame and not alert.triggered:
            return

        row = self._row(pose_result, analysis, alert, fps)
        if self.writer is not None:
            self.writer.writerow(row)
        else:
            self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def _empty_row(self) -> Dict[str, Any]:
        return {
            "time": "",
            "video_source_type": "",
            "detected": "",
            "valid": "",
            "head_angle_deg": "",
            "trunk_angle_deg": "",
            "smoothed_head_angle_deg": "",
            "smoothed_trunk_angle_deg": "",
            "status": "",
            "alert_triggered": "",
            "alert_cooling_down": "",
            "confidence": "",
            "fps": "",
            "message": "",
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
            "smoothed_head_angle_deg": self._fmt(analysis.smoothed_head_angle_deg),
            "smoothed_trunk_angle_deg": self._fmt(analysis.smoothed_trunk_angle_deg),
            "status": analysis.status,
            "alert_triggered": alert.triggered,
            "alert_cooling_down": alert.cooling_down,
            "confidence": self._fmt(analysis.confidence),
            "fps": self._fmt(fps),
            "message": analysis.message,
        }

    @staticmethod
    def _fmt(value: Optional[float]) -> str:
        if value is None:
            return ""
        return f"{value:.2f}"

