from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import Landmark, PoseResult

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - handled at runtime
    mp = None


class MediaPipePoseEstimator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = str(config.get("backend", "mediapipe")).lower()
        self.pose = None
        self.landmarker = None
        self._last_timestamp_ms = 0

        if self.backend != "mediapipe":
            raise ValueError("Only pose.backend='mediapipe' is implemented in this MVP.")
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed. Install requirements.txt first.")
        if mp is None:
            raise RuntimeError("mediapipe is not installed. Install requirements.txt first.")

        if hasattr(mp, "solutions"):
            self._init_solutions_pose(config)
        else:
            self._init_task_pose_landmarker(config)

    def estimate(self, frame_bgr: Any, timestamp: Optional[float] = None) -> PoseResult:
        if frame_bgr is None:
            return PoseResult(valid=False, reason="empty frame", timestamp=timestamp or time.monotonic())

        if self.pose is not None:
            return self._estimate_with_solutions(frame_bgr, timestamp)
        if self.landmarker is not None:
            return self._estimate_with_task(frame_bgr, timestamp)
        raise RuntimeError("Pose estimator is not initialized.")

    def close(self) -> None:
        if self.pose is not None:
            self.pose.close()
            self.pose = None
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def _init_solutions_pose(self, config: Dict[str, Any]) -> None:
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=int(config.get("model_complexity", 1)),
            smooth_landmarks=bool(config.get("smooth_landmarks", True)),
            enable_segmentation=False,
            min_detection_confidence=float(config.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(config.get("min_tracking_confidence", 0.5)),
        )

    def _init_task_pose_landmarker(self, config: Dict[str, Any]) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

        model_path = Path(str(config.get("model_asset_path", "models/pose_landmarker_lite.task")))
        if not model_path.exists():
            raise FileNotFoundError(
                "MediaPipe Pose Landmarker model file not found: "
                f"{model_path}. Download it to this path or change pose.model_asset_path in config.yaml."
            )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=float(config.get("min_detection_confidence", 0.5)),
            min_pose_presence_confidence=float(config.get("min_pose_presence_confidence", 0.5)),
            min_tracking_confidence=float(config.get("min_tracking_confidence", 0.5)),
            output_segmentation_masks=False,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def _estimate_with_solutions(self, frame_bgr: Any, timestamp: Optional[float]) -> PoseResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        ts = timestamp or time.monotonic()

        if not results.pose_landmarks:
            return PoseResult(valid=False, reason="no person detected", timestamp=ts, backend=self.backend)

        landmarks: List[Landmark] = []
        visibilities = []
        for item in results.pose_landmarks.landmark:
            visibility = float(getattr(item, "visibility", 0.0))
            presence = float(getattr(item, "presence", visibility))
            landmarks.append(
                Landmark(
                    x=float(item.x),
                    y=float(item.y),
                    z=float(item.z),
                    visibility=visibility,
                    presence=presence,
                )
            )
            visibilities.append(visibility)

        return PoseResult(
            landmarks=landmarks,
            valid=True,
            timestamp=ts,
            min_visibility=min(visibilities) if visibilities else 0.0,
            backend=self.backend,
        )

    def _estimate_with_task(self, frame_bgr: Any, timestamp: Optional[float]) -> PoseResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts = timestamp or time.monotonic()
        timestamp_ms = max(int(ts * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not results.pose_landmarks:
            return PoseResult(valid=False, reason="no person detected", timestamp=ts, backend=self.backend)

        landmarks: List[Landmark] = []
        visibilities = []
        for item in results.pose_landmarks[0]:
            visibility = float(getattr(item, "visibility", 0.0))
            presence = float(getattr(item, "presence", visibility))
            confidence = max(visibility, presence)
            landmarks.append(
                Landmark(
                    x=float(item.x),
                    y=float(item.y),
                    z=float(item.z),
                    visibility=confidence,
                    presence=presence,
                )
            )
            visibilities.append(confidence)

        return PoseResult(
            landmarks=landmarks,
            valid=True,
            timestamp=ts,
            min_visibility=min(visibilities) if visibilities else 0.0,
            backend=self.backend,
        )
