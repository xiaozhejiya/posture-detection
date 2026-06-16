from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.calibration_store import CalibrationStore


BASELINE = {
    "neck_vector": (0.0, -1.0),
    "head_offset": (0.0, -1.0),
    "shoulder_center": (0.5, 0.4),
    "gap_ratio": 0.55,
    "shoulder_width": 0.16,
    "arm_points": {
        "left_elbow": (-0.5, 0.8),
        "right_elbow": (0.5, 0.8),
    },
}


class CalibrationStoreTest(unittest.TestCase):
    def test_missing_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = CalibrationStore(Path(temp_dir) / "default.json")

            self.assertIsNone(store.load())

    def test_saved_baseline_can_be_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = CalibrationStore(Path(temp_dir) / "nested" / "default.json")

            store.save(BASELINE)
            loaded = store.load()

            self.assertEqual(loaded, BASELINE)

    def test_unsupported_schema_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "default.json"
            path.write_text(
                json.dumps({"schema_version": 999, "baseline": BASELINE}),
                encoding="utf-8",
            )
            store = CalibrationStore(path)

            self.assertIsNone(store.load())


if __name__ == "__main__":
    unittest.main()
