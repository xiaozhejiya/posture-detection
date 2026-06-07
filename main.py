from __future__ import annotations

import argparse
import time
from typing import Any, Dict, Optional

from src.alert_system import AlertSystem
from src.config_loader import load_config
from src.logger import PostureLogger
from src.pose_estimator import MoveNetTFLiteEstimator
from src.posture_analyzer import PostureAnalyzer
from src.video_source import VideoSource
from src.visualizer import Visualizer

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None


class FpsMeter:
    def __init__(self) -> None:
        self.last_time: Optional[float] = None
        self.fps = 0.0

    def tick(self, timestamp: float) -> float:
        if self.last_time is None:
            self.last_time = timestamp
            return self.fps
        delta = max(timestamp - self.last_time, 1e-6)
        instant = 1.0 / delta
        self.fps = instant if self.fps <= 0.0 else self.fps * 0.9 + instant * 0.1
        self.last_time = timestamp
        return self.fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local posture detection MVP.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--source-type", choices=["usb", "rtsp", "file"], help="Override video source type.")
    parser.add_argument("--camera-id", type=int, help="Override USB camera id.")
    parser.add_argument("--rtsp-url", help="Override RTSP URL.")
    parser.add_argument("--video-file", help="Override local video file.")
    parser.add_argument("--no-window", action="store_true", help="Disable OpenCV display window.")
    parser.add_argument("--calibrate-on-start", action="store_true", help="Start upper-body calibration on startup.")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames; useful for smoke tests.")
    return parser.parse_args()


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    video = config["video_source"]
    if args.source_type:
        video["type"] = args.source_type
    if args.camera_id is not None:
        video["camera_id"] = args.camera_id
    if args.rtsp_url:
        video["rtsp_url"] = args.rtsp_url
        video["type"] = "rtsp"
    if args.video_file:
        video["video_file"] = args.video_file
        video["type"] = "file"
    if args.no_window:
        config["app"]["show_window"] = False
    return config


def main() -> int:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    show_window = bool(config["app"].get("show_window", True))

    if show_window and cv2 is None:
        raise RuntimeError("opencv-python is not installed. Install requirements.txt first.")

    source = VideoSource(config["video_source"])
    estimator = MoveNetTFLiteEstimator(config["pose"])
    analyzer = PostureAnalyzer(config["posture_rule"])
    alerts = AlertSystem(config["posture_rule"])
    logger = PostureLogger(config["logging"], config["video_source"])
    visualizer = Visualizer(config["visualization"]) if show_window else None
    fps_meter = FpsMeter()

    frame_count = 0
    window_name = str(config["visualization"].get("window_name", config["app"].get("name", "Posture")))

    try:
        source.open()
        if args.calibrate_on_start or bool(config["app"].get("calibrate_on_start", False)):
            analyzer.start_calibration(time.monotonic())
            print("[INFO] Upper-body calibration started. Keep a normal sitting posture.")
        while True:
            frame, timestamp = source.read()
            if frame is None:
                if source.finished:
                    break
                time.sleep(0.02)
                continue

            pose_result = estimator.estimate(frame, timestamp)
            analysis = analyzer.analyze(pose_result, timestamp)
            alert = alerts.update(analysis)
            fps = fps_meter.tick(timestamp)

            if alert.triggered:
                print(f"[ALERT] {analysis.message} status={analysis.status}")

            logger.write(pose_result, analysis, alert, fps)

            if show_window and visualizer is not None:
                display = visualizer.draw(frame.copy(), pose_result, analysis, fps, alert)
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("c"):
                    analyzer.start_calibration(timestamp)
                    print("[INFO] Upper-body calibration started. Keep a normal sitting posture.")

            frame_count += 1
            if args.max_frames is not None and frame_count >= args.max_frames:
                break

    except KeyboardInterrupt:
        pass
    finally:
        logger.close()
        estimator.close()
        source.release()
        if show_window and cv2 is not None:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
