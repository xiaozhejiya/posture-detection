# Raspberry Pi Fullscreen Autostart And Calibration Design

## Context

The project currently runs posture detection from `main.py`, reads configuration through `src/config_loader.py`, displays the live frame with OpenCV in `src/visualizer.py`, and supports startup calibration with `--calibrate-on-start`. Raspberry Pi 5 deployment will boot directly into the graphical desktop. A small screen must show the live detection view full screen after boot.

The first deployment is for one fixed student and one fixed camera position. It only needs one default calibration profile.

## Goals

- Start posture detection automatically after the Raspberry Pi desktop session is available.
- Display the OpenCV detection window full screen on the small screen.
- Avoid requiring an 8 second calibration on every reboot when a valid saved calibration exists.
- Keep manual recalibration available from the running window.
- Keep the implementation compatible with current Windows/local test workflows.

## Non-Goals

- Multi-student profile management.
- Camera-position detection or automatic invalidation of stale calibration.
- A new GUI toolkit, kiosk browser, or system-level display manager integration.
- A systemd-based service for the first version.

## Recommended Approach

Use Raspberry Pi desktop autostart plus OpenCV full-screen display.

The desktop session will launch a project-owned shell script through a `.desktop` file in `~/.config/autostart/`. The shell script enters the project directory, activates `.venv`, and starts `main.py`. This avoids the common display-environment problems of system-level services while matching the known boot flow: the device automatically enters the graphical desktop.

## Code Design

### Display Configuration

Add visualization settings to `config.yaml` and `DEFAULT_CONFIG`:

```yaml
visualization:
  fullscreen: true
  window_width: 800
  window_height: 480
```

`main.py` will create the OpenCV window before the loop when `show_window` is enabled. If `visualization.fullscreen` is true, it will set `cv2.WND_PROP_FULLSCREEN` to `cv2.WINDOW_FULLSCREEN`. If false, it will size the normal window using `window_width` and `window_height` when both are positive.

`Visualizer.draw()` stays responsible only for drawing overlays. Window lifecycle remains in `main.py`.

### Calibration Persistence

Create `src/calibration_store.py` with a focused JSON store:

- `CalibrationStore(path)` stores one default calibration file.
- `load()` returns a calibration feature map when the file exists, has a supported schema version, and contains the required fields.
- `save(baseline)` writes the current baseline atomically enough for local deployment by writing JSON to the configured path.

Add calibration settings:

```yaml
calibration:
  enable_persistence: true
  file_path: "data/calibration/default.json"
  auto_start_if_missing: true
```

The saved JSON contains:

```json
{
  "schema_version": 1,
  "created_at": "2026-06-16T00:00:00Z",
  "baseline": {
    "neck_vector": [0.0, -1.0],
    "head_offset": [0.0, -1.0],
    "shoulder_center": [0.5, 0.4],
    "gap_ratio": 0.55,
    "shoulder_width": 0.16,
    "arm_points": {
      "left_elbow": [-0.5, 0.8]
    }
  }
}
```

The feature names match `PostureAnalyzer.upper_body_baseline`, so the analyzer does not need a second calibration model.

### Startup Behavior

On startup:

1. Load configuration.
2. Create `PostureAnalyzer`.
3. If calibration persistence is enabled, try to load `calibration.file_path`.
4. If a valid baseline exists, inject it into the analyzer and skip startup calibration.
5. If no valid baseline exists and `--calibrate-on-start` or `calibration.auto_start_if_missing` is true, start calibration.
6. When calibration completes, save `analyzer.upper_body_baseline`.

Add command-line controls:

- `--force-calibration`: ignore saved calibration and run calibration on startup.
- `--no-calibration-persistence`: do not load or save calibration data for this run.

The existing `c` key starts recalibration. After recalibration completes, the new baseline overwrites `data/calibration/default.json`.

## Deployment Design

Create `scripts/raspi/start_posture_detection.sh`:

- Determine the project root.
- Create `logs/` if needed.
- Activate `.venv`.
- Run `python main.py --source-type usb --camera-id 0 --calibrate-on-start`.
- Append stdout and stderr to `logs/runtime.log`.

Create `scripts/raspi/posture-detection.desktop`:

- `Type=Application`
- `Name=Posture Detection`
- `Exec=/home/pi/posture-detection/scripts/raspi/start_posture_detection.sh`
- `Terminal=false`
- `X-GNOME-Autostart-enabled=true`

The documentation will instruct the user to edit the project path if the repository is not located at `/home/pi/posture-detection`, then copy the desktop file to `~/.config/autostart/`.

## Error Handling

- If calibration JSON is missing, startup continues and calibration begins.
- If calibration JSON is malformed or uses an unsupported schema version, startup prints a warning and calibration begins.
- If saving calibration fails, posture detection keeps running and prints a warning.
- If OpenCV cannot open the camera, the program keeps its current failure behavior.
- If the display session is not available, the desktop autostart entry will fail visibly through `logs/runtime.log`; this is documented as a deployment issue.

## Testing

Unit tests will cover:

- Calibration store returns `None` when the file does not exist.
- Calibration store saves and reloads the expected baseline structure.
- Calibration store rejects unsupported schema versions.
- Config loader includes default visualization and calibration persistence values.

Existing analyzer tests already cover calibration completion and upper-body proxy scoring. The new integration behavior in `main.py` will stay small and use the tested store API.

Manual Raspberry Pi validation:

- Run the startup script from a terminal and confirm the window goes full screen.
- Press `c`, complete calibration, and confirm `data/calibration/default.json` is written.
- Reboot and confirm startup skips the 8 second calibration when the saved file exists.
- Delete the calibration file, reboot, and confirm calibration starts automatically.

## Acceptance Criteria

- Raspberry Pi boots into desktop and automatically starts posture detection.
- The detection view is full screen on the small display.
- A valid saved calibration skips automatic startup calibration.
- Manual recalibration with `c` updates the saved calibration.
- Local unit tests and compile checks pass.
