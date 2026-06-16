from __future__ import annotations

import unittest
from pathlib import Path

from src.config_loader import load_config


class ConfigLoaderTest(unittest.TestCase):
    def test_missing_config_defaults_to_movenet_thunder(self) -> None:
        config = load_config(Path("missing-config.yaml"))

        pose = config["pose"]
        self.assertEqual(pose["backend"], "movenet_tflite")
        self.assertEqual(pose["model_asset_path"], "models/movenet_thunder_int8.tflite")
        self.assertEqual(pose["model_variant"], "thunder")
        self.assertEqual(pose["input_size"], 256)

    def test_missing_config_includes_display_and_calibration_defaults(self) -> None:
        config = load_config(Path("missing-config.yaml"))

        visualization = config["visualization"]
        self.assertTrue(visualization["fullscreen"])
        self.assertEqual(visualization["window_width"], 800)
        self.assertEqual(visualization["window_height"], 480)

        calibration = config["calibration"]
        self.assertTrue(calibration["enable_persistence"])
        self.assertEqual(calibration["file_path"], "data/calibration/default.json")
        self.assertTrue(calibration["auto_start_if_missing"])


if __name__ == "__main__":
    unittest.main()
