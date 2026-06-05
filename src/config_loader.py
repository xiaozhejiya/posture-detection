from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "app": {
        "name": "Posture Detection MVP",
        "debug": True,
        "show_window": True,
    },
    "video_source": {
        "type": "usb",
        "camera_id": 0,
        "rtsp_url": "",
        "video_file": "",
        "width": 1280,
        "height": 720,
        "target_fps": 15,
        "reconnect": True,
        "reconnect_interval_sec": 2.0,
        "show_raw": True,
    },
    "pose": {
        "backend": "mediapipe",
        "model_asset_path": "models/pose_landmarker_lite.task",
        "model_complexity": 1,
        "min_detection_confidence": 0.5,
        "min_pose_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "smooth_landmarks": True,
        "use_face_landmarker": False,
    },
    "posture_rule": {
        "min_visibility": 0.5,
        "head_expected_gap_ratio": 0.55,
        "head_down_mild_deg": 15,
        "head_down_warning_deg": 25,
        "head_down_severe_deg": 35,
        "trunk_flex_mild_deg": 15,
        "trunk_flex_warning_deg": 25,
        "trunk_flex_severe_deg": 35,
        "smoothing_window_sec": 1.0,
        "warning_duration_sec": 3.0,
        "severe_duration_sec": 2.0,
        "combined_severe_duration_sec": 5.0,
        "cooldown_sec": 10.0,
    },
    "visualization": {
        "show_skeleton": True,
        "show_landmarks": True,
        "show_angles": True,
        "window_name": "Posture Detection MVP",
    },
    "logging": {
        "enable": True,
        "save_raw_video": False,
        "log_dir": "data/logs",
        "format": "csv",
        "write_each_frame": True,
    },
}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return deepcopy(DEFAULT_CONFIG)

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return deep_update(DEFAULT_CONFIG, loaded)
