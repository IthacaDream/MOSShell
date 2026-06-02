---
created: 2026-05-28
depends: []
description: 将 Reachy Mini 机器人的基础运控、视觉能力集成到 ghoshell_moss_contrib， 作为 MOSS 开源项目开箱即用的能力。目标是在
  .moss_ws 中作为 app 直接运行。
milestone: null
priority: P2
status: completed
title: Reachy Mini Contrib — 基础运控/视觉集成
updated: '2026-05-28'
---

# Reachy Mini Contrib

## Motivation

将 Reachy Mini 机器人能力集成到 MOSS contrib，提供开箱即用的 AI 控制能力。前提：用户已连接 Reachy Mini control app。

## 目标

- **基础运控**：通过 Reachy Mini control app 控制机器人关节运动、位姿、舞蹈、表情动画
- **视觉**：通过 `mini.media.get_frame()` 直接获取相机画面，注入模型上下文
- **开箱即用**：`MossInReachyMini(mini, ws).as_channel()` 一行组装，无需手动注入依赖

## Design Index

- 代码位置: `src/ghoshell_moss_contrib/moss_in_reachy_mini/`
- 入口: `moss.py` — `MossInReachyMini(mini, ws).as_channel()`
- 当前文件结构 (16 files):

```
moss_in_reachy_mini/
├── moss.py              # 入口，组装 channel + states
├── audio/
│   └── player.py        # TTS 播放器 (ReachyMiniStreamPlayer)
├── components/
│   ├── antennas.py      # 天线控制
│   ├── body.py          # 舞蹈 + 表情 (从 workspace 加载 emotions)
│   ├── head.py          # 头部运动 + 呼吸动画
│   └── vision.py        # 相机视觉 (直接 get_frame)
├── moves/
│   └── head_move.py     # 头部运动轨迹
└── state/
    ├── abcd.py          # BaseReachyState
    ├── asleep.py        # 休眠状态
    ├── boring.py        # 无聊状态
    └── waken.py         # 唤醒状态 (dance/emotion/head_move/antennas)
```

## 已完成任务

### 1. Import 路径修正 ✅

`from moss_in_reachy_mini.xxx` → `from ghoshell_moss_contrib.moss_in_reachy_mini.xxx`
17 files, 51 occurrences. coding by deepseek-v4-pro

### 2. 入口自组装 ✅

`__init__(self, mini: ReachyMini, ws: Workspace, logger=None)` — 3 参数，内部自组装：

- Layer 1: `Head(mini)`, `Body(mini, ws)`, `Antennas(mini)`
- Layer 2: `Vision(mini)` — 直接 `mini.media.get_frame()`
- Layer 3: `WakenState`, `BoringState`, `AsleepState`

砍掉了原有的 CameraWorker / FaceRecognizer / HeadTracker / FrameHub 等相机管线，
改为直接使用 ReachyMini SDK 的 `media.get_frame()`。

### 3. pyproject.toml optional dependency ⏳

已添加 `[project.optional-dependencies]` 的 `reachy_mini` 组，依赖解析有冲突（eclipse-zenoh 版本），待解决。

### 4. Audio Mixer 移除 ✅

删除 `audio/mixer.py` (341 lines) + `audio/pcm_utils.py` (87 lines)。
原因：多源混音在此场景是伪需求，TTS 说话时应 duck 而非 sum。
`ReachyMiniStreamPlayer._audio_stream_write` 现直推 `mini.media.push_audio_sample()`。

### 5. Music 缺失 ✅

已移除 Music 引用（连同 WakenState 中的 5 个音乐命令）。

### 额外清理 ✅

- 删除 `components/sound.py` — 音量控制无实际生效路径，随 mixer 一同移除
- 删除 `state/enrolling.py` — 人脸注册功能暂时移除
- 删除 `camera/` 整个目录 (7 files) — 相机管线简化
- 删除 `components/head_tracker.py` — 人脸追踪移除
- 删除 `audio/file_player.py`, `audio/mic_hub.py` — 音频播放简化

### Body 表情自加载 ✅

`Body(mini, ws)` 通过 `_load_emotions(ws)` 从 workspace configs storage 加载表情数据。
若本地 `reachy_mini_emotions` 目录为空，自动从 HuggingFace 下载：
`huggingface_hub.snapshot_download("pollen-robotics/reachy-mini-emotions-library", repo_type="dataset")`

## 当前任务

### 6. 在 .moss_ws 中作为 app 运行

目标：`moss.py` 能在 workspace 中通过 manifest 声明为 app，由 MOSS Host 自动发现并启动。

需要：
- 在 workspace 的 `apps/` 下创建 app 声明
- 验证 Channel 在 Matrix 通讯总线中正常工作
- 验证 Vision context_messages 返回的图片能正确注入模型上下文
- 验证 state 切换 (waken → boring → asleep) 在 runtime 中正常流转

## Key Decisions

- **最小依赖原则**：相机/人脸/混音等复杂管线暂不引入，先跑通基础运控 + 视觉
- **直通优于抽象**：相机直接用 `mini.media.get_frame()`，不走 CameraWorker 线程模型
- **下载优于内置**：表情数据不在仓库内，运行时从 HuggingFace 拉取

## Implementation Notes

- Vision.context_messages() 返回 PIL Image 列表，需确认 MOSS 框架是否正确处理
- Antennas 只在 WakenState 中注册命令，Boring/Asleep 状态无天线控制
- Body 依赖 `reachy_mini_dances_library` 提供 `AVAILABLE_MOVES`