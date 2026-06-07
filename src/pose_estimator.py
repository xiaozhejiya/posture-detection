from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .models import Landmark, PoseIndex, PoseResult

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None


MOVENET_TO_POSE_INDEX = {
    0: PoseIndex.NOSE,
    1: PoseIndex.LEFT_EYE,
    2: PoseIndex.RIGHT_EYE,
    3: PoseIndex.LEFT_EAR,
    4: PoseIndex.RIGHT_EAR,
    5: PoseIndex.LEFT_SHOULDER,
    6: PoseIndex.RIGHT_SHOULDER,
    7: PoseIndex.LEFT_ELBOW,
    8: PoseIndex.RIGHT_ELBOW,
    9: PoseIndex.LEFT_WRIST,
    10: PoseIndex.RIGHT_WRIST,
    11: PoseIndex.LEFT_HIP,
    12: PoseIndex.RIGHT_HIP,
    13: PoseIndex.LEFT_KNEE,
    14: PoseIndex.RIGHT_KNEE,
    15: PoseIndex.LEFT_ANKLE,
    16: PoseIndex.RIGHT_ANKLE,
}


class MoveNetTFLiteEstimator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = str(config.get("backend", "movenet_tflite")).lower()
        if self.backend != "movenet_tflite":
            raise ValueError("Only pose.backend='movenet_tflite' is supported.")
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed. Install requirements.txt first.")

        model_path = Path(str(config.get("model_asset_path", "models/movenet_thunder_int8.tflite")))
        if not model_path.exists():
            raise FileNotFoundError(
                "MoveNet TFLite model file not found: "
                f"{model_path}. Download it or change pose.model_asset_path in config.yaml."
            )

        self.min_keypoint_score = float(config.get("min_keypoint_score", 0.3))
        self.interpreter = self._create_interpreter(model_path, int(config.get("num_threads", 4)))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_height, self.input_width = self._input_size()

    def estimate(self, frame_bgr: Any, timestamp: Optional[float] = None) -> PoseResult:
        ts = timestamp or time.monotonic()
        if frame_bgr is None:
            return PoseResult(valid=False, reason="empty frame", timestamp=ts, backend=self.backend)

        input_data, transform = self._preprocess(frame_bgr)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        keypoints = self._read_output()
        return self._to_pose_result(keypoints, transform, ts)

    def close(self) -> None:
        self.interpreter = None

    @staticmethod
    def _create_interpreter(model_path: Path, num_threads: int) -> Any:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError as exc:
                try:
                    from tensorflow.lite import Interpreter
                except ImportError as tensorflow_exc:
                    raise RuntimeError(
                        "TFLite runtime is not installed. Install tflite-runtime on Raspberry Pi, "
                        "ai-edge-litert on Windows, or tensorflow as a fallback."
                    ) from tensorflow_exc
        return Interpreter(model_path=str(model_path), num_threads=num_threads)

    def _input_size(self) -> Tuple[int, int]:
        shape = self.input_details[0]["shape"]
        if len(shape) != 4:
            raise RuntimeError(f"Unexpected MoveNet input shape: {shape}")
        return int(shape[1]), int(shape[2])

    def _preprocess(self, frame_bgr: Any) -> Tuple[np.ndarray, Dict[str, float]]:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        scale = min(self.input_width / width, self.input_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        pad_left = (self.input_width - resized_width) // 2
        pad_top = (self.input_height - resized_height) // 2
        padded[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized

        dtype = self.input_details[0]["dtype"]
        if dtype == np.float32:
            tensor = padded.astype(np.float32)
        elif dtype == np.int32:
            tensor = padded.astype(np.int32)
        else:
            tensor = padded.astype(dtype)

        return np.expand_dims(tensor, axis=0), {
            "scale": scale,
            "pad_left": float(pad_left),
            "pad_top": float(pad_top),
            "width": float(width),
            "height": float(height),
        }

    def _read_output(self) -> np.ndarray:
        output_detail = self.output_details[0]
        output = self.interpreter.get_tensor(output_detail["index"])
        output = np.asarray(output)
        if np.issubdtype(output.dtype, np.integer):
            scale, zero_point = output_detail.get("quantization", (0.0, 0))
            if scale:
                output = (output.astype(np.float32) - float(zero_point)) * float(scale)
        return output.reshape(-1, 17, 3)[0]

    def _to_pose_result(
        self,
        keypoints: np.ndarray,
        transform: Dict[str, float],
        timestamp: float,
    ) -> PoseResult:
        landmarks = [Landmark(0.0, 0.0, visibility=0.0, presence=0.0) for _ in range(33)]
        scores = []
        for movenet_index, pose_index in MOVENET_TO_POSE_INDEX.items():
            y_norm, x_norm, score = [float(value) for value in keypoints[movenet_index]]
            x, y = self._unletterbox(x_norm, y_norm, transform)
            confidence = max(0.0, min(1.0, score))
            landmarks[pose_index] = Landmark(
                x=x,
                y=y,
                z=0.0,
                visibility=confidence,
                presence=confidence,
            )
            scores.append(confidence)

        max_score = max(scores) if scores else 0.0
        if max_score < self.min_keypoint_score:
            return PoseResult(
                landmarks=landmarks,
                valid=False,
                timestamp=timestamp,
                reason="no person detected",
                min_visibility=max_score,
                backend=self.backend,
            )

        visible_scores = [score for score in scores if score >= self.min_keypoint_score]
        return PoseResult(
            landmarks=landmarks,
            valid=True,
            timestamp=timestamp,
            min_visibility=min(visible_scores) if visible_scores else max_score,
            backend=self.backend,
        )

    def _unletterbox(
        self,
        x_norm: float,
        y_norm: float,
        transform: Dict[str, float],
    ) -> Tuple[float, float]:
        x_model = x_norm * self.input_width
        y_model = y_norm * self.input_height
        x_original = (x_model - transform["pad_left"]) / transform["scale"]
        y_original = (y_model - transform["pad_top"]) / transform["scale"]
        x = max(0.0, min(1.0, x_original / transform["width"]))
        y = max(0.0, min(1.0, y_original / transform["height"]))
        return x, y
