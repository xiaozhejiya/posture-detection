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
        "calibrate_on_start": False,
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
        "backend": "movenet_tflite",
        "model_asset_path": "models/movenet_thunder_int8.tflite",
        "model_variant": "thunder",
        "input_size": 256,
        "min_keypoint_score": 0.3,
        "num_threads": 4,
    },
    "posture_rule": {
        "trunk_mode": "auto",
        "min_visibility": 0.5,
        "trunk_min_visibility": 0.2,
        "head_expected_gap_ratio": 0.55,
        "head_down_mild_deg": 15,
        "head_down_warning_deg": 25,
        "head_down_severe_deg": 35,
        "trunk_flex_mild_deg": 12,
        "trunk_flex_warning_deg": 20,
        "trunk_flex_severe_deg": 32,
        "upper_body_calibration_sec": 8.0,
        "upper_body_min_samples": 20,
        "upper_body_mild_score": 40,
        "upper_body_warning_score": 60,
        "upper_body_severe_score": 80,
        "upper_body_weights": {
            "neck_angle": 0.30,
            "head_shoulder_gap": 0.25,
            "head_shift": 0.20,
            "shoulder_shift": 0.15,
            "arm_posture": 0.10,
        },
        "smoothing_window_sec": 1.0,
        "warning_duration_sec": 3.0,
        "severe_duration_sec": 2.0,
        "combined_severe_duration_sec": 5.0,
        "missing_signal_grace_sec": 1.0,
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
        "log_dir": "logs",
        "format": "csv",
        "max_file_size_mb": 10,
        "write_each_frame": True,
        "enable_event_log": True,
        "event_log_prefix": "posture_events",
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
