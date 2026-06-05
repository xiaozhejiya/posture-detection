from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.logger import PostureLogger
from src.models import AlertEvent, PoseResult, PostureAnalysis


class PostureLoggerTest(unittest.TestCase):
    def test_triggered_posture_event_is_written_to_event_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = PostureLogger(
                {
                    "enable": True,
                    "log_dir": temp_dir,
                    "format": "csv",
                    "write_each_frame": False,
                    "enable_event_log": True,
                    "event_log_prefix": "posture_events",
                },
                {"type": "usb"},
            )

            normal = PostureAnalysis(
                timestamp=1.0,
                valid=True,
                status="normal",
                head_angle_deg=5.0,
                trunk_angle_deg=4.0,
                smoothed_head_angle_deg=5.0,
                smoothed_trunk_angle_deg=4.0,
                confidence=0.9,
                detected=True,
            )
            logger.write(PoseResult(valid=True), normal, AlertEvent(False, "normal", "", 1.0), fps=15.0)

            warning = PostureAnalysis(
                timestamp=4.5,
                valid=True,
                status="head_down_warning",
                head_angle_deg=31.2,
                trunk_angle_deg=8.0,
                smoothed_head_angle_deg=29.5,
                smoothed_trunk_angle_deg=8.5,
                head_warning_duration_sec=3.2,
                confidence=0.88,
                detected=True,
                flags={"head_warning": True, "trunk_warning": False},
                message="Head-down posture detected",
            )
            logger.write(
                PoseResult(valid=True),
                warning,
                AlertEvent(True, "head_down_warning", "Head-down posture detected", 4.5),
                fps=14.7,
            )
            logger.close()

            event_files = list(Path(temp_dir).glob("posture_events_*.csv"))
            self.assertEqual(len(event_files), 1)

            with event_files[0].open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_type"], "head_down")
            self.assertEqual(rows[0]["status"], "head_down_warning")
            self.assertEqual(rows[0]["head_warning_duration_sec"], "3.20")


if __name__ == "__main__":
    unittest.main()

