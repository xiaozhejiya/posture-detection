from __future__ import annotations

import unittest

from src.display_window import configure_display_window


class FakeCv2:
    WINDOW_NORMAL = 0
    WND_PROP_FULLSCREEN = 1
    WINDOW_FULLSCREEN = 2

    def __init__(self) -> None:
        self.calls = []

    def namedWindow(self, name: str, flag: int) -> None:
        self.calls.append(("namedWindow", name, flag))

    def setWindowProperty(self, name: str, prop: int, value: int) -> None:
        self.calls.append(("setWindowProperty", name, prop, value))

    def resizeWindow(self, name: str, width: int, height: int) -> None:
        self.calls.append(("resizeWindow", name, width, height))


class DisplayWindowTest(unittest.TestCase):
    def test_fullscreen_window_is_configured(self) -> None:
        cv2 = FakeCv2()

        configure_display_window(cv2, "Posture", {"fullscreen": True})

        self.assertEqual(
            cv2.calls,
            [
                ("namedWindow", "Posture", FakeCv2.WINDOW_NORMAL),
                ("setWindowProperty", "Posture", FakeCv2.WND_PROP_FULLSCREEN, FakeCv2.WINDOW_FULLSCREEN),
            ],
        )

    def test_windowed_mode_uses_configured_size(self) -> None:
        cv2 = FakeCv2()

        configure_display_window(
            cv2,
            "Posture",
            {"fullscreen": False, "window_width": 640, "window_height": 480},
        )

        self.assertEqual(
            cv2.calls,
            [
                ("namedWindow", "Posture", FakeCv2.WINDOW_NORMAL),
                ("resizeWindow", "Posture", 640, 480),
            ],
        )


if __name__ == "__main__":
    unittest.main()
