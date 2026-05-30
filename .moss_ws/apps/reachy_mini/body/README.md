# Reachy Mini Body App

Reachy Mini 机器人躯体控制 MOSS App。通过 Matrix/Zenoh 通讯总线将机器人硬件能力反射为 AI 可调用的 Channel 命令。

## 架构

```
MOSS Host (Python 3.14)
  └─ Circus daemon
       └─ uv run main.py (独立 venv, Python 3.13)
            └─ ReachyMini() → MossInReachyMini → Matrix.provide_channel()
                                                       ↕ Zenoh
MOSS Host ── AppStoreChannel proxy ── apps.reachy_mini_body channel
```

## 依赖隔离

此 app 拥有独立的 `pyproject.toml` 和 `.venv`，`reachy-mini` 的重量级传递依赖（`pygobject`→`pycairo`、`gstreamer_python` 等）隔离在 app venv 中，不污染核心开发环境。

## 环境变量

复制 `.env.example` 为 `.env`，按需修改：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `REACHY_MEDIA_BACKEND` | 相机后端，macOS 上建议 `no_media` | `no_media` |

macOS 上 pip 安装的 `gstreamer_python` 存在 GObject 类型系统兼容问题，设置 `no_media` 跳过本地相机初始化，通过 daemon WebRTC 获取画面。

## 命令

当前 waken 状态可用命令：

- `dance(name)` — 预设舞蹈动作
- `emotion(emoji)` — 预设表情动画
- `head_move(...)` — 6-DOF 头部运动
- `head_reset(idle_mode)` — 复位，可选 `hold`/`breathing`
- `antennas_move(left, right, duration)` — 天线控制
- `antennas_reset(duration)` — 天线复位
- `switch_state(name)` — 切换状态（waken/boring/asleep）

## 开发

```bash
# 创建 venv 并安装依赖
cd .moss_ws/apps/reachy_mini/body
PKG_CONFIG=/opt/homebrew/bin/pkg-config uv sync --python 3.13

# 前台调试（需先启动 MOSS Host）
moss apps test reachy_mini/body

# 通过 MCP CTML 启动
<apps:start fullname="reachy_mini/body" timeout="30"/>
```
