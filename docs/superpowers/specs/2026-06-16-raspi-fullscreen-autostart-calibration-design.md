# 树莓派全屏自启动与校准持久化设计

## 背景

当前项目通过 `main.py` 运行姿态检测，通过 `src/config_loader.py` 读取配置，通过 `src/visualizer.py` 使用 OpenCV 显示实时画面，并且已经支持 `--calibrate-on-start` 启动时校准。树莓派 5 部署环境会在开机后自动进入图形化桌面，小屏幕需要全屏实时显示检测画面。

第一版部署场景是固定一个学生、固定一个摄像头机位，因此只需要一份默认校准档案。

## 目标

- 树莓派进入桌面会话后自动启动姿态检测程序。
- OpenCV 检测窗口在小屏幕上全屏显示。
- 有有效校准文件时，重启后不再强制等待 8 秒校准。
- 运行过程中仍然可以手动重新校准。
- 保持现有 Windows 和本地调试流程可用。

## 非目标

- 多学生档案管理。
- 多摄像头机位管理。
- 自动识别摄像头位置变化并让旧校准失效。
- 替换为新的 GUI 框架、浏览器 kiosk 或系统级显示管理方案。
- 第一版不采用 systemd 服务作为图形程序的启动方式。

## 推荐方案

采用“树莓派桌面 autostart + OpenCV 全屏窗口”。

树莓派进入图形化桌面后，由桌面会话通过 `~/.config/autostart/` 下的 `.desktop` 文件启动项目脚本。启动脚本进入项目目录、激活 `.venv`，再运行 `main.py`。这个方案与当前已确认的开机流程一致，也能避开 systemd 启动图形窗口时常见的 `DISPLAY`、`XDG_RUNTIME_DIR` 等环境变量问题。

## 代码设计

### 显示配置

在 `config.yaml` 和 `src/config_loader.py` 的 `DEFAULT_CONFIG` 中给 `visualization` 增加配置：

```yaml
visualization:
  fullscreen: true
  window_width: 800
  window_height: 480
```

`main.py` 在 `show_window` 启用时提前创建 OpenCV 窗口。如果 `visualization.fullscreen` 为 `true`，则通过 `cv2.WND_PROP_FULLSCREEN` 和 `cv2.WINDOW_FULLSCREEN` 设置全屏。如果为 `false`，则在 `window_width` 和 `window_height` 都为正数时按配置设置普通窗口大小。

`Visualizer.draw()` 继续只负责绘制骨架、状态面板和告警条；窗口生命周期仍放在 `main.py`，避免显示逻辑和绘制逻辑混在一起。

### 校准持久化

新增 `src/calibration_store.py`，负责读写一份默认 JSON 校准文件：

- `CalibrationStore(path)`：管理一个校准文件路径。
- `load()`：当文件存在、schema 版本支持、字段结构有效时，返回校准特征。
- `save(baseline)`：把当前校准基准写入 JSON 文件。

新增校准配置：

```yaml
calibration:
  enable_persistence: true
  file_path: "data/calibration/default.json"
  auto_start_if_missing: true
```

保存的 JSON 结构如下：

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

字段名与 `PostureAnalyzer.upper_body_baseline` 保持一致，因此不需要引入第二套校准模型。

### 启动行为

程序启动时按以下顺序处理：

1. 读取配置文件。
2. 创建 `PostureAnalyzer`。
3. 如果启用了校准持久化，尝试读取 `calibration.file_path`。
4. 如果读取到有效校准基准，把它注入 `PostureAnalyzer`，并跳过启动校准。
5. 如果没有有效校准基准，并且传入了 `--calibrate-on-start` 或 `calibration.auto_start_if_missing` 为 `true`，则自动开始校准。
6. 当校准完成后，把 `analyzer.upper_body_baseline` 保存到校准文件。

新增命令行参数：

- `--force-calibration`：忽略已有校准文件，启动后强制重新校准。
- `--no-calibration-persistence`：本次运行不读取也不保存校准文件。

现有 `c` 按键继续用于手动重新校准。手动校准完成后，新基准会覆盖 `data/calibration/default.json`。

## 部署设计

新增 `scripts/raspi/start_posture_detection.sh`：

- 自动定位项目根目录。
- 确保 `logs/` 目录存在。
- 激活 `.venv`。
- 运行 `python main.py --source-type usb --camera-id 0 --calibrate-on-start`。
- 将标准输出和错误输出追加到 `logs/runtime.log`，方便排查开机启动问题。

新增 `scripts/raspi/posture-detection.desktop` 模板：

```ini
Type=Application
Name=Posture Detection
Exec=/home/pi/posture-detection/scripts/raspi/start_posture_detection.sh
Terminal=false
X-GNOME-Autostart-enabled=true
```

部署文档会说明：如果项目不在 `/home/pi/posture-detection`，需要先修改 `.desktop` 里的 `Exec` 路径，然后复制到：

```bash
~/.config/autostart/posture-detection.desktop
```

## 错误处理

- 校准 JSON 不存在时，程序继续启动并进入校准。
- 校准 JSON 格式错误或 schema 版本不支持时，打印警告并进入校准。
- 保存校准失败时，姿态检测继续运行并打印警告。
- OpenCV 无法打开摄像头时，沿用当前程序的失败行为。
- 如果桌面显示会话不可用，autostart 启动失败信息会通过 `logs/runtime.log` 暴露，部署文档会把它作为排查项。

## 测试设计

单元测试覆盖：

- 校准文件不存在时，`CalibrationStore.load()` 返回 `None`。
- 校准基准保存后可以重新读取，并且结构符合预期。
- 不支持的 schema 版本会被拒绝。
- 配置加载器包含默认的全屏显示和校准持久化配置。

现有 `PostureAnalyzer` 测试已经覆盖校准完成和上半身代理分数判断。`main.py` 里的新增集成逻辑保持轻量，只依赖已测试的 `CalibrationStore` 接口。

树莓派人工验收：

- 在终端手动运行启动脚本，确认窗口全屏。
- 按 `c` 重新校准，确认完成后写入 `data/calibration/default.json`。
- 重启树莓派，确认存在有效校准文件时跳过 8 秒启动校准。
- 删除校准文件后重启，确认会自动进入校准。

## 验收标准

- 树莓派开机进入桌面后自动启动姿态检测。
- 检测画面在小屏幕上全屏显示。
- 存在有效校准文件时，重启后跳过自动校准。
- 按 `c` 手动重新校准后会更新校准文件。
- 本地单元测试和语法检查通过。
