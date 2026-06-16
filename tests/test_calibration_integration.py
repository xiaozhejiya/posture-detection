from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from main import load_saved_calibration, save_current_calibration, should_start_calibration
from src.calibration_store import CalibrationStore
from src.config_loader import DEFAULT_CONFIG
from src.posture_analyzer import PostureAnalyzer


BASELINE = {
    "neck_vector": (0.0, -1.0),
    "head_offset": (0.0, -1.0),
    "shoulder_center": (0.5, 0.4),
    "gap_ratio": 0.55,
    "shoulder_width": 0.16,
    "arm_points": {},
}


class CalibrationIntegrationTest(unittest.TestCase):
    def test_saved_calibration_is_loaded_into_analyzer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = CalibrationStore(Path(temp_dir) / "default.json")
            store.save(BASELINE)
            analyzer = PostureAnalyzer(DEFAULT_CONFIG["posture_rule"])

            loaded = load_saved_calibration(analyzer, store)

            self.assertTrue(loaded)
            self.assertEqual(analyzer.upper_body_baseline, BASELINE)

    def test_current_calibration_is_saved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = CalibrationStore(Path(temp_dir) / "default.json")
            analyzer = PostureAnalyzer(DEFAULT_CONFIG["posture_rule"])
            analyzer.upper_body_baseline = BASELINE

            save_current_calibration(analyzer, store)

            self.assertEqual(store.load(), BASELINE)

    def test_saved_calibration_skips_requested_startup_calibration(self) -> None:
        self.assertFalse(
            should_start_calibration(
                saved_calibration_loaded=True,
                calibrate_requested=True,
                force_calibration=False,
                auto_start_if_missing=True,
            )
        )

    def test_force_calibration_overrides_saved_calibration(self) -> None:
        self.assertTrue(
            should_start_calibration(
                saved_calibration_loaded=True,
                calibrate_requested=False,
                force_calibration=True,
                auto_start_if_missing=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
