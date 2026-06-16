from __future__ import annotations

from typing import Any, Dict


def configure_display_window(cv2_module: Any, window_name: str, config: Dict[str, Any]) -> None:
    cv2_module.namedWindow(window_name, cv2_module.WINDOW_NORMAL)
    if bool(config.get("fullscreen", False)):
        cv2_module.setWindowProperty(
            window_name,
            cv2_module.WND_PROP_FULLSCREEN,
            cv2_module.WINDOW_FULLSCREEN,
        )
        return

    width = int(config.get("window_width", 0) or 0)
    height = int(config.get("window_height", 0) or 0)
    if width > 0 and height > 0:
        cv2_module.resizeWindow(window_name, width, height)
