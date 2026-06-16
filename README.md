# 学生伏案写作业姿态检测 MVP

这是一个本地运行的学生学习姿态检测原型，支持 USB 摄像头、RTSP 视频流和本地视频文件输入。系统通过 MoveNet TFLite 提取人体关键点，再基于几何角度、正常坐姿校准和持续时间规则判断低头、伏案/躯干前倾等姿态风险。

## 已实现功能

- 支持 `usb`、`rtsp`、`file` 三类视频源。
- 使用 MoveNet SinglePose Thunder TFLite 提取人体 17 点关键点，并映射为项目内部姿态点结构。
- 计算头部低头角度和躯干前倾角度。
- 使用滑动平滑窗口和连续持续时间过滤，避免单帧误报。
- 在 OpenCV 窗口中显示骨架、角度、姿态状态、提醒状态和 FPS。
- 支持 CSV 或 JSONL 日志，默认保存到 `logs`，并按运行批次和文件大小自动分片。
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

树莓派 5 / Linux ARM64：

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

如果 OpenCV 的 pip wheel 安装失败，可改用系统包安装 OpenCV，再用 `uv pip install` 安装其余依赖。

树莓派开机自启动、小屏全屏显示和校准持久化部署步骤见：[docs/树莓派开机自启动部署.md](docs/树莓派开机自启动部署.md)。

当前姿态估计后端使用 MoveNet SinglePose Thunder INT8 TFLite 模型。默认路径为：

```text
models/movenet_thunder_int8.tflite
```

如果该文件缺失，可手动下载：

```text
https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite
```

Thunder 的关键点稳定性通常好于 Lightning，但推理更慢。树莓派 5 建议使用 `tflite-runtime` 执行 `.tflite` 模型。Windows 调试环境如果没有 `tflite-runtime` wheel，可安装 `ai-edge-litert`，代码会自动回退到 `ai_edge_litert.Interpreter`。

## 运行方式

USB 摄像头：

```powershell
.\.venv\Scripts\python.exe main.py --source-type usb --camera-id 0
```

启动后自动校准上半身伏案基准：

```powershell
.\.venv\Scripts\python.exe main.py --source-type usb --camera-id 0 --calibrate-on-start
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

OpenCV 窗口中按 `q` 或 `Esc` 退出，按 `c` 重新开始上半身校准。校准时请保持正常坐姿 8 秒左右。

## 配置说明

主要参数在 `config.yaml` 中调整：

- `video_source`：视频源类型、摄像头编号、RTSP 地址、本地视频路径、分辨率、目标帧率和重连策略。
- `pose`：MoveNet TFLite 模型路径、关键点置信度阈值和推理线程数。
- `posture_rule`：低头角度阈值、躯干前倾角度阈值、髋部关键点专用置信度阈值、上半身伏案代理分数、校准时长、平滑窗口、告警持续时间、严重告警持续时间和冷却时间。
- `visualization`：是否显示骨架、关键点、角度和窗口名称。
- `logging`：是否启用日志、日志目录、日志格式和是否逐帧记录。

## 日志

日志默认写入 `logs`，每次启动会创建独立目录。

逐帧日志文件名类似：

```text
logs/posture_20260605_093000/posture_0001.csv
logs/posture_20260605_093000/posture_0002.csv
```

字段包括：

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

低头、弯腰、联合风险、严重风险真正触发提醒时，会额外写入独立事件文件，文件名类似：

```text
logs/posture_events_20260605_093000/posture_events_0001.csv
logs/posture_events_20260605_093000/posture_events_0002.csv
```

事件文件只记录触发瞬间，不记录每一帧。字段包括事件类型、姿态状态、提示文本、头部角度、躯干角度、异常持续时间、激活的规则标记、置信度和 FPS。

单个日志文件达到 `logging.max_file_size_mb` 后会自动滚动到下一个分片，并重新写入 CSV 表头。

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

边缘部署和性能优化建议见：[docs/边缘计算优化建议.md](docs/边缘计算优化建议.md)。
