from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None


class VideoSource:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.capture = None
        self.finished = False
        self._last_open_attempt = 0.0

    @property
    def source_type(self) -> str:
        return str(self.config.get("type", "usb")).lower()

    def open(self) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed. Install requirements.txt first.")

        self.release()
        source = self._resolve_source()
        self.capture = cv2.VideoCapture(source)

        if self.source_type == "usb":
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config.get("width", 1280)))
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.get("height", 720)))
            self.capture.set(cv2.CAP_PROP_FPS, int(self.config.get("target_fps", 15)))
        elif self.source_type == "rtsp":
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._last_open_attempt = time.monotonic()
        if not self.capture.isOpened():
            self.capture = None
            raise RuntimeError(f"Cannot open video source: {source!r}")

        self.finished = False

    def read(self) -> Tuple[Optional[Any], float]:
        timestamp = time.monotonic()
        if self.capture is None:
            self.open()

        ok, frame = self.capture.read()
        timestamp = time.monotonic()
        if ok:
            return frame, timestamp

        if self.source_type == "file":
            self.finished = True
            return None, timestamp

        if self.config.get("reconnect", True):
            self._try_reconnect(timestamp)

        return None, timestamp

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _resolve_source(self) -> Any:
        if self.source_type == "usb":
            return int(self.config.get("camera_id", 0))
        if self.source_type == "rtsp":
            url = str(self.config.get("rtsp_url", "")).strip()
            if not url:
                raise ValueError("video_source.rtsp_url is required when type is 'rtsp'.")
            return url
        if self.source_type == "file":
            video_file = str(self.config.get("video_file", "")).strip()
            if not video_file:
                raise ValueError("video_source.video_file is required when type is 'file'.")
            return video_file
        raise ValueError(f"Unsupported video_source.type: {self.source_type!r}")

    def _try_reconnect(self, now: float) -> None:
        interval = float(self.config.get("reconnect_interval_sec", 2.0))
        if now - self._last_open_attempt < interval:
            return

        try:
            self.open()
        except RuntimeError:
            self._last_open_attempt = now

