from __future__ import annotations

import unittest

from src.config_loader import DEFAULT_CONFIG
from src.models import Landmark, PoseIndex, PoseResult
from src.posture_analyzer import PostureAnalyzer


def make_pose(
    nose=(0.50, 0.20),
    left_shoulder=(0.42, 0.40),
    right_shoulder=(0.58, 0.40),
    left_hip=(0.43, 0.70),
    right_hip=(0.57, 0.70),
    hip_visibility=0.99,
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
    landmarks[PoseIndex.LEFT_HIP] = Landmark(left_hip[0], left_hip[1], visibility=hip_visibility)
    landmarks[PoseIndex.RIGHT_HIP] = Landmark(right_hip[0], right_hip[1], visibility=hip_visibility)
    return PoseResult(landmarks=landmarks, valid=True, min_visibility=0.99)


class PostureAnalyzerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DEFAULT_CONFIG["posture_rule"].copy()

    def test_normal_pose_stays_normal(self) -> None:
        analyzer = PostureAnalyzer(self.config)
        result = analyzer.analyze(make_pose(), timestamp=1.0)
        self.assertTrue(result.valid)
        self.assertEqual(result.status, "normal")
        self.assertIsNotNone(result.smoothed_head_angle_deg)
        self.assertIsNotNone(result.smoothed_trunk_angle_deg)
        self.assertLess(result.smoothed_head_angle_deg, 25)
        self.assertLess(result.smoothed_trunk_angle_deg, 25)

    def test_head_down_requires_duration(self) -> None:
        analyzer = PostureAnalyzer(self.config)
        pose = make_pose(nose=(0.50, 0.36))
        early = analyzer.analyze(pose, timestamp=1.0)
        late = analyzer.analyze(pose, timestamp=4.2)
        self.assertEqual(early.status, "normal")
        self.assertEqual(late.status, "head_down_warning")

    def test_severe_trunk_warning_after_duration(self) -> None:
        analyzer = PostureAnalyzer(self.config)
        pose = make_pose(
            nose=(0.70, 0.35),
            left_shoulder=(0.62, 0.46),
            right_shoulder=(0.78, 0.46),
            left_hip=(0.43, 0.70),
            right_hip=(0.57, 0.70),
        )
        analyzer.analyze(pose, timestamp=1.0)
        result = analyzer.analyze(pose, timestamp=3.2)
        self.assertEqual(result.status, "severe_warning")

    def test_low_confidence_hips_can_still_trigger_trunk_warning(self) -> None:
        analyzer = PostureAnalyzer(self.config)
        pose = make_pose(
            nose=(0.74, 0.27),
            left_shoulder=(0.55, 0.46),
            right_shoulder=(0.71, 0.46),
            left_hip=(0.43, 0.70),
            right_hip=(0.57, 0.70),
            hip_visibility=0.25,
        )
        early = analyzer.analyze(pose, timestamp=1.0)
        late = analyzer.analyze(pose, timestamp=4.2)
        self.assertEqual(early.status, "normal")
        self.assertEqual(late.status, "trunk_flex_warning")
        self.assertIsNotNone(late.smoothed_trunk_angle_deg)

    def test_calibrated_upper_body_proxy_triggers_when_hips_are_missing(self) -> None:
        config = self.config.copy()
        config["trunk_mode"] = "upper_body_proxy"
        config["upper_body_calibration_sec"] = 1.0
        config["upper_body_min_samples"] = 2
        config["upper_body_severe_score"] = 101
        config["head_down_warning_deg"] = 90
        config["head_down_severe_deg"] = 120
        analyzer = PostureAnalyzer(config)
        analyzer.start_calibration(timestamp=0.0)

        normal_pose = make_pose(hip_visibility=0.0)
        calibrating = analyzer.analyze(normal_pose, timestamp=0.2)
        calibrated = analyzer.analyze(normal_pose, timestamp=1.2)
        self.assertEqual(calibrating.status, "calibrating")
        self.assertTrue(calibrated.calibrated)

        flex_pose = make_pose(
            nose=(0.65, 0.39),
            left_shoulder=(0.42, 0.45),
            right_shoulder=(0.58, 0.45),
            hip_visibility=0.0,
        )
        early = analyzer.analyze(flex_pose, timestamp=2.0)
        late = analyzer.analyze(flex_pose, timestamp=5.2)
        self.assertEqual(early.status, "normal")
        self.assertEqual(late.status, "trunk_flex_warning")
        self.assertEqual(late.trunk_signal, "upper_body_score")
        self.assertIsNotNone(late.smoothed_upper_body_score)

    def test_low_confidence_is_invalid(self) -> None:
        analyzer = PostureAnalyzer(self.config)
        pose = make_pose()
        pose.landmarks[PoseIndex.LEFT_SHOULDER] = Landmark(0.42, 0.40, visibility=0.1)
        result = analyzer.analyze(pose, timestamp=1.0)
        self.assertFalse(result.valid)
        self.assertEqual(result.status, "invalid")


if __name__ == "__main__":
    unittest.main()
