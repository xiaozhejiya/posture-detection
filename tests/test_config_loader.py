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


if __name__ == "__main__":
    unittest.main()
