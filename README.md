# 学生伏案写作业姿态检测 MVP

这是一个本地运行的学生学习姿态检测原型，支持 USB 摄像头、RTSP 视频流和本地视频文件输入。系统通过 MediaPipe 提取人体关键点，再基于几何角度和持续时间规则判断低头、弯腰/躯干前倾等姿态风险。

## 已实现功能

- 支持 `usb`、`rtsp`、`file` 三类视频源。
- 使用 MediaPipe Pose 提取人体 33 点关键点。
- 计算头部低头角度和躯干前倾角度。
- 使用滑动平滑窗口和连续持续时间过滤，避免单帧误报。
- 在 OpenCV 窗口中显示骨架、角度、姿态状态、提醒状态和 FPS。
- 支持 CSV 或 JSONL 日志，默认保存到 `data/logs`。
- 默认不保存原始视频，符合第一版隐私要求。

## 环境准备

推荐使用 Python 3.11。

如果本机安装了 `uv`：

```powershell
uv venv --python 3.11
uv pip install -r requirements.txt
```

当前项目目录已有 `.venv` 时，也可以直接使用虚拟环境中的 Python：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

当前环境中的 MediaPipe 使用新版 `tasks` API，需要 Pose Landmarker `.task` 模型文件。默认路径为：

```text
models/pose_landmarker_lite.task
```

如果该文件缺失，可手动下载：

```text
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

## 运行方式

USB 摄像头：

```powershell
.\.venv\Scripts\python.exe main.py --source-type usb --camera-id 0
```

RTSP 视频流：

```powershell
.\.venv\Scripts\python.exe main.py --rtsp-url "rtsp://user:password@host:554/stream"
```

本地视频文件：

```powershell
.\.venv\Scripts\python.exe main.py --video-file "data/videos/sample.mp4"
```

无窗口快速测试：

```powershell
.\.venv\Scripts\python.exe main.py --no-window --video-file "data/videos/sample.mp4" --max-frames 100
```

OpenCV 窗口中按 `q` 或 `Esc` 退出。

## 配置说明

主要参数在 `config.yaml` 中调整：

- `video_source`：视频源类型、摄像头编号、RTSP 地址、本地视频路径、分辨率、目标帧率和重连策略。
- `pose`：MediaPipe 置信度阈值、模型复杂度和关键点平滑开关。
- `posture_rule`：低头角度阈值、躯干前倾角度阈值、平滑窗口、告警持续时间、严重告警持续时间和冷却时间。
- `visualization`：是否显示骨架、关键点、角度和窗口名称。
- `logging`：是否启用日志、日志目录、日志格式和是否逐帧记录。

## 日志

日志默认写入 `data/logs`，字段包括：

- 时间
- 视频源类型
- 是否检测到人体
- 姿态检测是否有效
- 头部角度
- 躯干角度
- 平滑后的角度
- 当前姿态状态
- 是否触发提醒
- 关键点置信度
- FPS

系统默认只保存角度、状态和事件信息，不保存原始视频。

## 测试

运行规则单元测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

运行语法检查：

```powershell
.\.venv\Scripts\python.exe -m compileall main.py src tests
```

## 注意事项

本项目是学习姿态提醒工具，不是医学诊断系统。检测结果适合用于行为提醒和规则调试，实际部署前需要根据摄像头角度、桌椅高度、学生坐姿习惯和光照条件重新校准阈值。
