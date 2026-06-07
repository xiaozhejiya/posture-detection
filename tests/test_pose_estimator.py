from __future__ import annotations

import unittest

import numpy as np

from src.models import PoseIndex
from src.pose_estimator import MoveNetTFLiteEstimator


class MoveNetTFLiteEstimatorTest(unittest.TestCase):
    def test_movenet_keypoints_map_to_internal_pose_indices(self) -> None:
        estimator = object.__new__(MoveNetTFLiteEstimator)
        estimator.backend = "movenet_tflite"
        estimator.min_keypoint_score = 0.3
        estimator.input_width = 192
        estimator.input_height = 192
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[0] = [0.10, 0.20, 0.95]
        keypoints[5] = [0.40, 0.30, 0.90]
        keypoints[6] = [0.40, 0.70, 0.91]
        keypoints[11] = [0.80, 0.35, 0.85]
        keypoints[12] = [0.80, 0.65, 0.86]
        transform = {
            "scale": 1.0,
            "pad_left": 0.0,
            "pad_top": 0.0,
            "width": 192.0,
            "height": 192.0,
        }

        result = estimator._to_pose_result(keypoints, transform, timestamp=1.0)

        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.landmarks[PoseIndex.NOSE].x, 0.20, places=5)
        self.assertAlmostEqual(result.landmarks[PoseIndex.NOSE].y, 0.10, places=5)
        self.assertAlmostEqual(result.landmarks[PoseIndex.LEFT_SHOULDER].x, 0.30, places=5)
        self.assertAlmostEqual(result.landmarks[PoseIndex.RIGHT_SHOULDER].x, 0.70, places=5)
        self.assertAlmostEqual(result.landmarks[PoseIndex.LEFT_HIP].visibility, 0.85, places=5)
        self.assertAlmostEqual(result.landmarks[PoseIndex.RIGHT_HIP].visibility, 0.86, places=5)


if __name__ == "__main__":
    unittest.main()

