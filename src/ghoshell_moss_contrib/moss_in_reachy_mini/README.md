# Reachy Mini Body Channel

Reachy Mini 躯体控制器的 MOSS Channel 封装 —— 将机器人硬件能力反射为 AI 可调用的命令，通过状态机管理机器人生命周期。

## 架构

```
MossInReachyMini (root channel)
├── Body        — 表情/情绪、舞蹈、手势、身体姿态
├── Head        — 头部运动 & 空闲模式 (hold / breathing)
├── Antennas    — 天线物理控制
├── Vision      — 摄像头采集 (独立子 channel)
├── Audio       — 语音播放 (BaseAudioStreamPlayer)
├── WakenState  — 唤醒：电机使能，全交互
├── BoringState — 待机：电机使能，超时自动休眠
└── AsleepState — 休眠：电机断电，不响应交互
```

## 命令概览

### Waken 状态 (唤醒)

| 命令 | 组件 | 功能 |
|------|------|------|
| `angry` / `happy` / `sad` ... | Body | 播放预设表情动画 |
| `dance` | Body | 执行舞蹈动作 (来自 `reachy_mini_dances_library`) |
| `puppet_motion` / `recorded_move` | Body | 时间序列关节控制 |
| `head_pose` / `head_move` / `head_idle` | Head | 头部姿态与空闲模式切换 |
| `antennas_set_position` | Antennas | 天线位置控制 |
| `look` | Vision | 主动获取摄像头图像 |
| `say` | (speech) | TTS 语音输出 |

### Boring 状态 (待机)

电机保持使能，等待用户交互。长时间无交互自动切换到 Asleep 状态。

### Asleep 状态 (休眠)

电机断电，低头闭眼，不响应交互命令。通过 `switch_state` 唤醒。

## 目录结构

```
moss_in_reachy_mini/
├── moss.py           # 根 Channel 组装 & 生命周期
├── components/
│   ├── body.py       # 身体控制 (表情/舞蹈/姿态)
│   ├── head.py       # 头部控制 (IdleMode: hold/breathing)
│   ├── antennas.py   # 天线控制
│   └── vision.py     # 摄像头子 Channel
├── state/
│   ├── abcd.py       # BaseReachyState 基类
│   ├── waken.py      # 唤醒态
│   ├── boring.py     # 待机态
│   └── asleep.py     # 休眠态
├── moves/
│   └── head_move.py  # HeadMove / BreathingMove 定义
├── audio/
│   └── player.py     # 音频流播放器
└── README.md
```

## 作为 App 运行

在 workspace 的 `apps/body/` 下创建入口：

```python
from reachy_mini import ReachyMini
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss_contrib.moss_in_reachy_mini.moss import MossInReachyMini


async def main(_matrix: Matrix) -> None:
    ws = _matrix.workspace
    logger = _matrix.logger
    mini = ReachyMini()
    reachy = MossInReachyMini(mini=mini, ws=ws, logger=logger)
    await _matrix.provide_channel(reachy.as_channel())


if __name__ == "__main__":
    matrix = Matrix.discover()
    matrix.run(main)
```

## 依赖

- `reachy_mini` — Reachy Mini 硬件 SDK
- `reachy_mini_dances_library` — 预设舞蹈动作库
- `ghoshell_moss` — MOSS 运行时

## Emotions 资源

情绪动画文件存放在 `.moss_ws/configs/reachy_mini_emotions/`。若该目录不存在，首次运行时会自动从 HuggingFace 下载。
