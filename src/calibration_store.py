from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

FeatureMap = Dict[str, Any]
Point = Tuple[float, float]


class CalibrationStore:
    SCHEMA_VERSION = 1
    REQUIRED_POINT_KEYS = ("neck_vector", "head_offset", "shoulder_center")

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> Optional[FeatureMap]:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if payload.get("schema_version") != self.SCHEMA_VERSION:
            return None

        baseline = payload.get("baseline")
        if not isinstance(baseline, dict):
            return None
        return self._normalize_baseline(baseline)

    def save(self, baseline: FeatureMap) -> None:
        normalized = self._normalize_baseline(baseline)
        if normalized is None:
            raise ValueError("Invalid calibration baseline.")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "baseline": self._to_jsonable(normalized),
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _normalize_baseline(self, baseline: FeatureMap) -> Optional[FeatureMap]:
        normalized: FeatureMap = {}
        for key in self.REQUIRED_POINT_KEYS:
            point = self._point(baseline.get(key))
            if point is None:
                return None
            normalized[key] = point

        try:
            gap_ratio = float(baseline["gap_ratio"])
            shoulder_width = float(baseline["shoulder_width"])
        except (KeyError, TypeError, ValueError):
            return None
        if shoulder_width <= 0.0:
            return None

        arm_points = baseline.get("arm_points", {})
        if not isinstance(arm_points, dict):
            return None

        normalized_arms: Dict[str, Point] = {}
        for name, point_value in arm_points.items():
            point = self._point(point_value)
            if point is None:
                return None
            normalized_arms[str(name)] = point

        normalized["gap_ratio"] = gap_ratio
        normalized["shoulder_width"] = shoulder_width
        normalized["arm_points"] = normalized_arms
        return normalized

    @staticmethod
    def _point(value: Any) -> Optional[Point]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_jsonable(baseline: FeatureMap) -> FeatureMap:
        return {
            "neck_vector": list(baseline["neck_vector"]),
            "head_offset": list(baseline["head_offset"]),
            "shoulder_center": list(baseline["shoulder_center"]),
            "gap_ratio": float(baseline["gap_ratio"]),
            "shoulder_width": float(baseline["shoulder_width"]),
            "arm_points": {
                name: list(point) for name, point in baseline.get("arm_points", {}).items()
            },
        }
