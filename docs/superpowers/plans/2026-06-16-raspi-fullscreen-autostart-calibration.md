# Raspberry Pi Fullscreen Autostart And Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为树莓派桌面环境实现开机自启动、OpenCV 小屏全屏显示，以及固定学生/固定机位的一份默认校准数据持久化。

**Architecture:** 保持现有 `main.py` 为运行入口，新增小而独立的 `CalibrationStore` 负责 JSON 校准文件读写，新增窗口配置 helper 让 OpenCV 窗口生命周期继续由入口层管理。部署侧使用桌面 autostart 启动 shell 脚本，不引入 systemd 图形服务。

**Tech Stack:** Python 3.11, OpenCV, PyYAML, unittest, Raspberry Pi Desktop autostart.

---

## File Structure

- Create `src/calibration_store.py`: 保存、读取、校验一份默认校准 JSON。
- Create `src/display_window.py`: 用可测试的小函数封装 OpenCV 窗口创建、全屏和普通尺寸设置。
- Modify `src/config_loader.py`: 增加 `visualization.fullscreen/window_width/window_height` 和 `calibration` 默认配置。
- Modify `config.yaml`: 增加实际部署默认配置。
- Modify `main.py`: 接入显示窗口 helper、校准文件读取、校准完成后的保存、两个新增命令行参数。
- Create `tests/test_calibration_store.py`: 覆盖校准文件缺失、保存读取、schema 拒绝。
- Modify `tests/test_config_loader.py`: 覆盖新增默认配置。
- Create `tests/test_display_window.py`: 用 fake cv2 对象验证全屏和普通窗口行为。
- Create `scripts/raspi/start_posture_detection.sh`: 树莓派桌面 autostart 调用的启动脚本。
- Create `scripts/raspi/posture-detection.desktop`: autostart 模板。
- Create `docs/树莓派开机自启动部署.md`: 中文部署说明。
- Modify `.gitignore`: 忽略运行时生成的 `data/calibration/*.json`，保留目录占位文件。
- Create `data/calibration/.gitkeep`: 保留校准目录。

### Task 1: Config Defaults

**Files:**
- Modify: `tests/test_config_loader.py`
- Modify: `src/config_loader.py`
- Modify: `config.yaml`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_config_loader.py`:

```python
    def test_missing_config_includes_display_and_calibration_defaults(self) -> None:
        config = load_config(Path("missing-config.yaml"))

        visualization = config["visualization"]
        self.assertTrue(visualization["fullscreen"])
        self.assertEqual(visualization["window_width"], 800)
        self.assertEqual(visualization["window_height"], 480)

        calibration = config["calibration"]
        self.assertTrue(calibration["enable_persistence"])
        self.assertEqual(calibration["file_path"], "data/calibration/default.json")
        self.assertTrue(calibration["auto_start_if_missing"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_config_loader`

Expected: fail because `fullscreen`, `window_width`, `window_height`, or `calibration` defaults are missing.

- [ ] **Step 3: Write minimal implementation**

Update `DEFAULT_CONFIG` in `src/config_loader.py`:

```python
    "visualization": {
        "show_skeleton": True,
        "show_landmarks": True,
        "show_angles": True,
        "window_name": "Posture Detection MVP",
        "fullscreen": True,
        "window_width": 800,
        "window_height": 480,
    },
    "calibration": {
        "enable_persistence": True,
        "file_path": "data/calibration/default.json",
        "auto_start_if_missing": True,
    },
```

Update `config.yaml` with the same keys under `visualization` and add top-level `calibration`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_config_loader`

Expected: pass.

### Task 2: Calibration Store

**Files:**
- Create: `tests/test_calibration_store.py`
- Create: `src/calibration_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_calibration_store.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_calibration_store`

Expected: fail with `ModuleNotFoundError: No module named 'src.calibration_store'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/calibration_store.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_calibration_store`

Expected: pass.

### Task 3: Display Window Helper

**Files:**
- Create: `tests/test_display_window.py`
- Create: `src/display_window.py`
- Modify: `main.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_display_window.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_display_window`

Expected: fail with `ModuleNotFoundError: No module named 'src.display_window'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/display_window.py`:

```python
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
```

Modify `main.py` to import `configure_display_window` and call it once after `window_name` is computed and before entering the read loop.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_display_window`

Expected: pass.

### Task 4: Main Calibration Integration

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add startup and completion helpers**

Add these functions to `main.py` before `main()`:

```python
def load_saved_calibration(analyzer: PostureAnalyzer, store: Optional[CalibrationStore]) -> bool:
    if store is None:
        return False
    baseline = store.load()
    if baseline is None:
        return False
    analyzer.upper_body_baseline = baseline
    return True


def save_current_calibration(analyzer: PostureAnalyzer, store: Optional[CalibrationStore]) -> None:
    if store is None or analyzer.upper_body_baseline is None:
        return
    store.save(analyzer.upper_body_baseline)
```

Use these helpers from `main()`.

- [ ] **Step 2: Wire command-line arguments**

Add to `parse_args()`:

```python
    parser.add_argument("--force-calibration", action="store_true", help="Ignore saved calibration and calibrate on startup.")
    parser.add_argument("--no-calibration-persistence", action="store_true", help="Disable calibration load/save for this run.")
```

- [ ] **Step 3: Wire store creation and startup decision**

After `analyzer = PostureAnalyzer(...)`, create the store when enabled:

```python
    calibration_config = config.get("calibration", {})
    calibration_store = None
    if (
        not args.no_calibration_persistence
        and bool(calibration_config.get("enable_persistence", True))
    ):
        calibration_store = CalibrationStore(str(calibration_config.get("file_path", "data/calibration/default.json")))
```

Then load calibration unless `--force-calibration` is set. Start calibration if forced, requested, or missing and configured to auto-start.

- [ ] **Step 4: Save after calibration completes**

Track `was_calibrating = analyzer.is_calibrating` before each `analyze()` call. After analysis, when `was_calibrating and analysis.calibrated and not analyzer.is_calibrating`, call `save_current_calibration()`, catching `OSError` and `ValueError` and printing a warning.

- [ ] **Step 5: Run affected tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_calibration_store tests.test_config_loader tests.test_display_window`

Expected: pass.

### Task 5: Raspberry Pi Deployment Assets

**Files:**
- Modify: `.gitignore`
- Create: `data/calibration/.gitkeep`
- Create: `scripts/raspi/start_posture_detection.sh`
- Create: `scripts/raspi/posture-detection.desktop`
- Create: `docs/树莓派开机自启动部署.md`
- Modify: `README.md`

- [ ] **Step 1: Add ignored runtime calibration data**

Add to `.gitignore`:

```gitignore
data/calibration/*.json
!data/calibration/.gitkeep
```

Create `data/calibration/.gitkeep`.

- [ ] **Step 2: Add startup script**

Create `scripts/raspi/start_posture_detection.sh`:

```bash
#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/runtime.log"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}" || exit 1

{
  echo "[$(date --iso-8601=seconds)] Starting posture detection"
  if [ ! -f ".venv/bin/activate" ]; then
    echo "Virtual environment not found: ${PROJECT_ROOT}/.venv"
    exit 1
  fi
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  exec python main.py --source-type usb --camera-id "${POSTURE_CAMERA_ID:-0}" --calibrate-on-start "$@"
} >> "${LOG_FILE}" 2>&1
```

- [ ] **Step 3: Add autostart template**

Create `scripts/raspi/posture-detection.desktop`:

```ini
[Desktop Entry]
Type=Application
Name=Posture Detection
Comment=Start posture detection in fullscreen mode
Exec=/home/pi/posture-detection/scripts/raspi/start_posture_detection.sh
Terminal=false
X-GNOME-Autostart-enabled=true
```

- [ ] **Step 4: Add Chinese deployment docs**

Create `docs/树莓派开机自启动部署.md` with commands for dependency install, manual test, chmod, copying `.desktop`, checking `logs/runtime.log`, disabling autostart, and calibration behavior.

- [ ] **Step 5: Link docs from README**

Add a short deployment pointer under README's Raspberry Pi environment or running section:

```markdown
树莓派开机自启动、小屏全屏显示和校准持久化部署步骤见：[docs/树莓派开机自启动部署.md](docs/树莓派开机自启动部署.md)。
```

### Task 6: Full Verification

**Files:**
- No code files.

- [ ] **Step 1: Run all unit tests**

Run: `.\.venv\Scripts\python.exe -m unittest discover -s tests`

Expected: pass.

- [ ] **Step 2: Run compile check**

Run: `.\.venv\Scripts\python.exe -m compileall main.py src tests`

Expected: pass.

- [ ] **Step 3: Inspect git diff**

Run: `git diff --stat`

Expected: only files listed in this plan are changed, plus no changes under `LCD-show/`.

## Self-Review

- Spec coverage: full-screen display is covered by Tasks 1 and 3; persisted default calibration is covered by Tasks 2 and 4; desktop autostart and Raspberry Pi deployment docs are covered by Task 5; verification is covered by Task 6.
- Placeholder scan: this plan uses exact paths, concrete commands, and concrete snippets.
- Type consistency: `FeatureMap`, point tuples, `CalibrationStore.load()`, `CalibrationStore.save()`, `configure_display_window()`, and `main.py` helper names are consistent across tasks.
