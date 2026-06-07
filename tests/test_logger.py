from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.alert_system import AlertSystem
from src.config_loader import DEFAULT_CONFIG
from src.logger import PostureLogger
from src.models import AlertEvent, Landmark, PoseIndex, PoseResult, PostureAnalysis
from src.posture_analyzer import PostureAnalyzer


def make_pose(
    nose=(0.70, 0.35),
    left_shoulder=(0.62, 0.46),
    right_shoulder=(0.78, 0.46),
    left_hip=(0.43, 0.70),
    right_hip=(0.57, 0.70),
) -> PoseResult:
    landmarks = [Landmark(0.0, 0.0, visibility=0.0) for _ in range(33)]
    for index, point in (
        (PoseIndex.NOSE, nose),
        (PoseIndex.LEFT_SHOULDER, left_shoulder),
        (PoseIndex.RIGHT_SHOULDER, right_shoulder),
        (PoseIndex.LEFT_HIP, left_hip),
        (PoseIndex.RIGHT_HIP, right_hip),
    ):
        landmarks[index] = Landmark(point[0], point[1], visibility=0.99)
    return PoseResult(landmarks=landmarks, valid=True, min_visibility=0.99)


def make_invalid_pose() -> PoseResult:
    return PoseResult(valid=False, reason="no person detected", min_visibility=0.0)


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

            event_dirs = list(Path(temp_dir).glob("posture_events_*"))
            self.assertEqual(len(event_dirs), 1)
            event_files = list(event_dirs[0].glob("posture_events_*.csv"))
            self.assertEqual(len(event_files), 1)

            with event_files[0].open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_type"], "head_down")
            self.assertEqual(rows[0]["status"], "head_down_warning")
            self.assertEqual(rows[0]["head_warning_duration_sec"], "3.20")

    def test_event_is_written_after_short_invalid_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rule_config = DEFAULT_CONFIG["posture_rule"].copy()
            rule_config["trunk_mode"] = "hip_based"
            rule_config["smoothing_window_sec"] = 0.1
            rule_config["missing_signal_grace_sec"] = 1.0
            rule_config["warning_duration_sec"] = 1.0
            rule_config["trunk_flex_severe_deg"] = 120
            rule_config["head_down_warning_deg"] = 90
            rule_config["head_down_severe_deg"] = 120
            analyzer = PostureAnalyzer(rule_config)
            alerts = AlertSystem(rule_config)
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

            for pose, timestamp in (
                (make_pose(), 1.0),
                (make_pose(), 1.2),
                (make_invalid_pose(), 1.4),
                (make_pose(), 2.0),
                (make_pose(), 2.6),
            ):
                analysis = analyzer.analyze(pose, timestamp=timestamp)
                alert = alerts.update(analysis)
                logger.write(pose, analysis, alert, fps=10.0)
            logger.close()

            event_dirs = list(Path(temp_dir).glob("posture_events_*"))
            self.assertEqual(len(event_dirs), 1)
            event_files = list(event_dirs[0].glob("posture_events_*.csv"))
            self.assertEqual(len(event_files), 1)

            with event_files[0].open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_type"], "trunk_flex")
            self.assertEqual(rows[0]["status"], "trunk_flex_warning")

    def test_logs_are_split_into_run_directories_by_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = PostureLogger(
                {
                    "enable": True,
                    "log_dir": temp_dir,
                    "format": "csv",
                    "write_each_frame": True,
                    "enable_event_log": True,
                    "event_log_prefix": "posture_events",
                    "max_file_size_mb": 0.001,
                },
                {"type": "usb"},
            )
            warning = PostureAnalysis(
                timestamp=4.5,
                valid=True,
                status="head_down_warning",
                head_angle_deg=31.2,
                smoothed_head_angle_deg=31.2,
                head_warning_duration_sec=3.2,
                confidence=0.88,
                detected=True,
                flags={"head_warning": True},
                message="Head-down posture detected",
            )

            for index in range(30):
                timestamp = 4.5 + index
                logger.write(
                    PoseResult(valid=True),
                    warning,
                    AlertEvent(True, "head_down_warning", "Head-down posture detected", timestamp),
                    fps=14.7,
                )
            logger.close()

            posture_dirs = [
                path
                for path in Path(temp_dir).glob("posture_*")
                if path.is_dir() and not path.name.startswith("posture_events_")
            ]
            event_dirs = list(Path(temp_dir).glob("posture_events_*"))
            self.assertEqual(len(posture_dirs), 1)
            self.assertEqual(len(event_dirs), 1)
            self.assertGreater(len(list(posture_dirs[0].glob("posture_*.csv"))), 1)
            self.assertGreater(len(list(event_dirs[0].glob("posture_events_*.csv"))), 1)


if __name__ == "__main__":
    unittest.main()
