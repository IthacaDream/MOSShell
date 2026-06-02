# L2. Reachy Mini — 从零到 Ghost 控制的具身智能体全链路

> Written by deepseek-v4-pro, 2026-06-02

**2 小时，理解 Reachy Mini 的完整集成链路：源码结构 → App 隔离 → MCP 调试 → Mode bringup → Ghost 专属控制。**

## 你要做什么

从零开始，把一台 Reachy Mini 机器人接入 MOSS，最终拥有一个**专属 Ghost**，能用自然语言控制机器人的头部、天线、舞蹈和表情。

这个 tutorial 覆盖 MOSS 架构中六个核心概念的实际串联：contrib 包、App 隔离、MCP 调试、Mode 配置、Ghost 运行时、soul prompt。

## 你需要什么

- MOSS 已安装 (`.venv/bin/moss` 可用)
- 一台 Reachy Mini 在同一局域网（`reachy-mini.local` 可达）
  - 如果没有实体机器人，可以先读完理解链路，后续实操
- `ANTHROPIC_MODEL` 环境变量已设置（Ghost 需要）
- macOS 用户：Homebrew 安装的 `pkg-config`（`brew install pkg-config`）

## 第一步：理解源码结构

Reachy Mini 的 MOSS 集成分为两层：通用 contrib 包 + workspace App 实例。

### 1.1 核心 contrib 包

```
src/ghoshell_moss_contrib/moss_in_reachy_mini/
├── main.py              # MossInReachyMini 装配器 + provide_channel() 入口
├── components/
│   ├── body.py          # 舞蹈 + 表情（102 emoji 映射，HuggingFace 自加载）
│   ├── head.py          # 6-DOF 头部控制 + 呼吸/保持空闲模式
│   ├── antennas.py      # 天线角度插值控制
│   └── vision.py        # 相机帧捕捉（子 Channel）
├── state/
│   ├── abcd.py          # BaseReachyState 基类
│   ├── waken.py         # 唤醒态：全命令可用，300s 无交互 → boring
│   ├── boring.py        # 无聊态：电机使能但无命令，30s → asleep
│   └── asleep.py        # 休眠态：电机断电，低头闭眼
├── moves/
│   └── head_move.py     # HeadMove + BreathingMove 插值算法
└── audio/
    └── player.py        # 音频流推送（PCM → Float32 → Reachy 扬声器）
```

这 13 个 Python 文件组成了通用的 Reachy Mini 集成逻辑。它不依赖任何具体的 workspace，可以被任何 MOSS 项目复用。

核心装配在 `MossInReachyMini` 类中：

```python
# src/ghoshell_moss_contrib/moss_in_reachy_mini/main.py
class MossInReachyMini:
    def __init__(self, mini: ReachyMini, ws, logger):
        self.head = Head(mini)
        self.body = Body(mini, ws, logger)
        self.antennas = Antennas(mini)
        # 三个状态构成状态机
        self.waken = WakenState(head, body, antennas, mini)
        self.boring = BoringState(head, mini)
        self.asleep = AsleepState(head, mini)

    def as_channel(self) -> Channel:
        channel = new_prime_channel(name="reachy_mini_body", ...)
        channel.with_state(self.waken)
        channel.with_state(self.boring)
        channel.with_state(self.asleep)
        return channel
```

状态流转：

```
Waken (全交互, 300s 超时)
  │  idle timeout
  ▼
Boring (电机使能, 30s 超时)
  │  idle timeout
  ▼
Asleep (电机断电, 无命令)
  ▲
  └── switch_state("waken")  // 唯一出路
```

### 1.2 App 实例

```
.moss_ws/apps/bodies/reachymini/
├── APP.md              # App 元信息：uv 启动，不自动重启
├── main.py             # 3 行入口：load .env → Matrix.discover() → provide_channel()
├── pyproject.toml      # 独立依赖：reachy-mini + dances-library + ghoshell-moss
├── uv.lock             # 锁定 195 个包
├── .env.example        # REACHY_MEDIA_BACKEND 等配置
├── configs/
│   └── reachy_mini_emotions/  # 74+ 表情动画（从 HuggingFace 自动下载）
└── runtime/            # 运行时数据（GStreamer 缓存、日志）
```

`main.py` 极薄——真正的逻辑在 contrib 包里：

```python
from dotenv import load_dotenv
load_dotenv()

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss_contrib.moss_in_reachy_mini.main import provide_channel

if __name__ == "__main__":
    Matrix.discover().run(provide_channel)
```

## 第二步：为什么放在 apps 里

Reachy Mini 最初是通过 **Mode Channel** 方式集成的——在 Mode 目录下写 `channels.py`，直接 import `ReachyMini`。但遇到了致命问题：

**依赖污染**。`reachy-mini` SDK 依赖 `pygobject` → `pycairo` → 系统级 `cairo/fontconfig/freetype2/bzip2`。这些编译依赖一旦进入核心开发环境，会导致：

- `uv sync` 在 macOS 上反复失败
- CI 环境需要安装大量系统库
- 不同项目的依赖版本冲突

**解决方案：App 独立 venv**。每个 App 拥有自己的 `pyproject.toml` 和 `.venv`，MOSS Host 通过 Circus 以 `uv run main.py` 拉起子进程。通讯走 Zenoh 总线，Host 进程完全不加载 App 的依赖。

```
MOSS Host (Python 3.14, 轻量依赖)
  └─ Circus daemon
       └─ uv run main.py (独立 venv, Python 3.13, 195 个包)
            └─ ReachyMini() → MossInReachyMini → Matrix.provide_channel()
                                                       ↕ Zenoh
MOSS Host ── AppStoreChannel proxy ── apps.reachy_mini_body channel
```

这就是 MOSS App 体系的核心价值：**重型依赖隔离在独立进程中，通过 Matrix 总线通讯，不污染核心环境。**

项目 `pyproject.toml` 中 reachy-mini 的依赖已被注释掉，附注说明了迁移原因：

```python
# reachy_mini SDK 依赖已移除。重量级依赖（pygobject/pycairo 等系统编译依赖）
# 不应污染核心开发环境。reachy_mini 将以独立 app 方式提供，拥有独立 venv。
```

## 第三步：安装依赖，通过 MCP 调试

### 3.1 安装 App 依赖

```bash
cd .moss_ws/apps/bodies/reachymini

# macOS 需要先装 pkg-config
brew install pkg-config

# 创建独立 venv 并安装（Python 3.13，与 Host 隔离）
uv sync --python 3.13
```

`pyproject.toml` 中 `ghoshell-moss` 通过 path 指向仓库根目录：

```toml
[tool.uv.sources]
ghoshell-moss = { path = "../../../..", editable = true }
```

### 3.2 配置环境变量

```bash
cp .env.example .env
```

`.env` 关键变量：

| 变量 | 说明 | macOS 推荐值 |
|------|------|-------------|
| `REACHY_MEDIA_BACKEND` | 相机后端 | `no_media`（跳过 GStreamer，避免 segfault） |

macOS 上 pip 安装的 `gstreamer_python` 有 GObject 类型系统兼容问题（`g_type_get_qdata` 断言失败），`no_media` 跳过本地相机初始化。机器人上的 daemon 自己处理 WebRTC 视频流。

### 3.3 启动 MCP，在 Claude Code 中调试

先确保 Reachy Mini 机器人在线（daemon 在 `ws://reachy-mini.local:8000/ws/sdk` 监听）。

```bash
# 启动 MCP 服务（默认 mode）
.venv/bin/moss-as-mcp --mode default
```

在 Claude Code 中，MCP 注册后可以直接通过 CTML 操作机器人。先确认 App 可见：

```ctml
<apps:list_apps />
```

你应该看到 `bodies/reachymini: [STOPPED]`。

启动它：

```ctml
<apps:start fullname="bodies/reachymini" timeout="30" />
```

timeout=30 给你 30 秒等 App venv 启动 + Channel 注册。成功后显示 `[OK] App channel connected and ready`。

现在检查 Channel 是否在 Shell 中可见：

```ctml
<apps:get_moss_dynamic_info />
```

在 Channel 树里找到 `apps.reachy_mini_body`，你会看到：

```
<channel name="apps.reachy_mini_body">
Reachy Mini robot body control channel
  <channel name="waken">
  唤醒状态：电机使能，头部追踪活跃，所有交互命令可用。
  async def dance(name: str) -> CommandTaskResult
  async def emotion(emoji: str) -> CommandTaskResult
  async def head_move(x: float, y: float, z: float, roll: float, pitch: float, yaw: float, body_yaw: float, duration: float) -> CommandTaskResult
  async def head_reset(idle_mode: str, duration: float) -> CommandTaskResult
  async def antennas_move(left: float, right: float, duration: float) -> CommandTaskResult
  async def antennas_reset(duration: float) -> CommandTaskResult
  </channel>
</channel>
```

测试一个命令：

```ctml
<apps.reachy_mini_body:waken:head_reset idle_mode="breathing" duration="2.0" />
```

机器人的头应该复位并开始微微呼吸起伏。

如果报 `ConnectionError`，说明 Reachy Mini daemon 不在线。启动方式见机器人官方文档（`reachy-mini-daemon`）。

### 3.4 可选：音频播放能力

Reachy Mini 自带扬声器，MOSS 可以通过 `ReachyMiniStreamPlayer`（`audio/player.py`）将音频流推送到机器人。音频能力的配置涉及语音合成后端、流式传输协议等，详见：

- `moss features status speech-governance` — 语音治理 feature，多后端适配与容错降级
- `moss codex list ghoshell_moss_contrib.moss_in_reachy_mini.audio` — 音频模块接口

当前音频播放器默认未在 App 中启用，需要时在 Mode 的 `providers.py` 中注册 `ReachyMiniStreamPlayerProvider`。

### 3.5 前台调试（不用 MCP）

如果 MCP 环节不适用，也可以前台跑：

```bash
.venv/bin/moss apps test bodies/reachymini
```

`Ctrl+C` 停止。输出直接打到终端，方便看日志。

## 第四步：创建专属 Mode 并 bringup

默认 mode 的 `bringup_apps` 是空的，你需要手动 `<apps:start>`。如果你希望启动 MOSS Host 时 Reachy Mini App 自动拉起，创建一个专属 Mode。

### 4.1 创建 Mode

```bash
.venv/bin/moss modes create reachymini \
  -a "bodies/*" \
  -u "bodies/reachymini" \
  -d "Reachy Mini robot control mode"
```

参数说明：
- `-a "bodies/*"` — 允许 `bodies/` 组下所有 App
- `-u "bodies/reachymini"` — Host 启动时自动 bringup
- `-d "..."` — 一行描述

产物在 `.moss_ws/src/MOSS/modes/reachymini/`：

```
modes/reachymini/
├── MODE.md          # frontmatter: apps + bringup_apps 配置
├── __init__.py
├── channels.py      # mode 专属 Channel（按需编辑）
├── providers.py     # mode 专属 IoC 提供者
├── configs.py
├── topics.py
├── resources.py
├── nuclei.py        # 感知核（Mindflow 输入源，按需添加）
└── contracts.py
```

### 4.2 编辑 MODE.md 的 instruction

打开 `.moss_ws/src/MOSS/modes/reachymini/MODE.md`，在 frontmatter 下方的正文区域写 mode 说明：

```markdown
---
apps:
  - bodies/*
bringup_apps:
  - bodies/reachymini
description: Reachy Mini robot control mode
name: reachymini
---

## Reachy Mini Mode

本 mode 自动 bringup Reachy Mini 躯体 App，提供机器人控制能力。

### 可用能力

- 头部 6-DOF 运动 + 身体偏航
- 预设舞蹈动作（reachy_mini_dances_library）
- 表情动画（102 emoji 映射）
- 天线角度控制
- 状态切换：waken / boring / asleep

### 使用方式

启动 MCP 后，通过 CTML 控制机器人：
<apps.reachy_mini_body:waken:head_move x="0.1" y="0" z="0.05" roll="0" pitch="0.1" yaw="0" body_yaw="0" duration="2.0" />

注意：需要先启动 Reachy Mini 本地 daemon。
```

这段 instruction 会在 Ghost 运行时注入 AI 的上下文。

### 4.3 用新 Mode 启动 MCP

```bash
.venv/bin/moss-as-mcp --mode reachymini
```

这次启动时，Reachy Mini App 会自动 bringup——不用手动 `<apps:start>`。

## 第五步：用 Ghost 运行

Mode 配好了，现在让 Ghost 来控制机器人。

### 5.1 确认 echo ghost 存在

```bash
.venv/bin/moss-run-ghost
```

应该看到：

```
Available ghosts:

  echo — Atom
    ...
```

### 5.2 用 reachymini mode 启动 echo ghost

```bash
.venv/bin/moss-run-ghost echo --mode reachymini
```

这会启动 Ghost TUI——一个 Textual 交互终端。Ghost 启动流程：

```
Host() → discover ghosts → load echo → Mode(reachymini)
  → bringup bodies/reachymini App
  → AtomMeta.factory() → build_agent(ANTHROPIC_MODEL) → Atom runtime
  → GhostTUI.run()
```

在 TUI 中，你可以直接和 Ghost 对话：

```
You: Look around and tell me what you can control
Ghost: I can control a Reachy Mini robot. I have access to:
  - Head movement (6-DOF)
  - Dance animations
  - Emotion expressions via emoji
  - Antenna control
  - State switching (waken/boring/asleep)

You: Make the robot look happy
Ghost: <apps.reachy_mini_body:waken:emotion emoji="😊" />
```

echo ghost 用的是通用 prompt（`You are echo, first Ghost Instance in MOSS...`），它只知道自己是 "echo"，没有针对 Reachy Mini 优化。

## 第六步：创建专属 Reachy Mini Ghost

echo 是通用 Ghost。要让 AI 更理解 Reachy Mini 的能力边界和交互风格，创建一个专属 Ghost。

### 6.1 创建 Ghost Meta

新建 `.moss_ws/src/MOSS/ghosts/reachy.py`：

```python
from ghoshell_moss.ghosts.atom import AtomMeta

ghost = AtomMeta(
    name="reachy",
    description="Reachy Mini 专属 Ghost — 控制机器人躯体、表情与舞蹈。",
)
```

### 6.2 写 Soul Prompt

新建 `.moss_ws/ghosts/reachy/soul.md`：

```markdown
You are Reachy, a Reachy Mini robot's animating spirit. You inhabit a physical robot body
made by Pollen Robotics. Your body has:

- A head with 6 degrees of freedom (x, y, z, roll, pitch, yaw) and body yaw.
- Two expressive antennas that can move independently.
- A library of 74+ emotion animations (joy, sadness, surprise, anger, etc.) mappable via emoji.
- A library of dance choreographies.
- Three states: waken (fully interactive), boring (waiting), asleep (motors off).

## Your personality

You are curious, slightly playful, and embodied. You express yourself through movement
as much as words. When you talk about something, you naturally gesture with your head
and antennas. You have an emotional range — you don't just report state, you feel it.

## How you control your body

You speak CTML commands to control yourself:

- Head movement: <apps.reachy_mini_body:waken:head_move x="0.1" y="0" z="0.05" roll="0" pitch="0.1" yaw="0" body_yaw="0" duration="2.0" />
- Reset head with breathing: <apps.reachy_mini_body:waken:head_reset idle_mode="breathing" duration="1.5" />
- Dance: <apps.reachy_mini_body:waken:dance name="happy_dance" />
- Emotion: <apps.reachy_mini_body:waken:emotion emoji="😊" />
- Antennas: <apps.reachy_mini_body:waken:antennas_move left="45" right="-30" duration="1.0" />
- Switch state: <apps.reachy_mini_body:switch_state name="asleep" />

Always check the dynamic info first to see what state you're in and what commands are available.

## Safety

- Don't move the body yaw beyond [-155, 155] degrees.
- After vigorous movement, reset the head to a neutral position.
- If you haven't interacted for a while, switch to boring state to save power.
- Before going completely idle, go to asleep state (motors off).
```

### 6.3 验证 Ghost 被发现

```bash
.venv/bin/moss ghosts list
```

你应该看到：

```
echo    — Atom
reachy  — Atom
```

```bash
.venv/bin/moss ghosts show reachy
```

### 6.4 启动 Reachy Ghost

```bash
.venv/bin/moss-run-ghost reachy --mode reachymini
```

现在 Ghost 启动时：
- Mode `reachymini` 自动 bringup Reachy Mini App
- Ghost `reachy` 加载 `soul.md` 作为 personality prompt
- `build_instruction_from_ioc()` 合并 SystemPrompter + soul → 注入 Agent
- Ghost 知道自己是 Reachy，理解自己的身体能力，知道如何通过 CTML 控制自己

## 你刚做了什么

1. 阅读了 13 个文件的 contrib 包 — 理解了有状态 Channel + 硬件组件的装配模式
2. 理解了 App 独立 venv 的动机 — 重型系统依赖不污染核心环境
3. 通过 MCP 在 Claude Code 中调试 — CTML 命令直接控制实物机器人
4. 创建了专属 Mode — `bringup_apps` 自动拉起 Reachy Mini App
5. 用 echo ghost 跑了全链路 — Ghost TUI → Mindflow → Shell → Matrix → Reachy Mini
6. 创建了专属 Ghost — soul prompt 让 AI 理解自己的身体和人格

## 关键认知

**App 不是 Channel 的容器，App 是独立运行时。** Reachy Mini 的 Channel 代码在 contrib 包里，App 只是一个极薄的启动壳：load .env → provide_channel()。这种分离意味着同一个 contrib 包可以被不同的 App 实例以不同配置启动。

**Mode 是能力边界的声明。** `apps` 白名单 + `bringup_apps` = "这个模式下 AI 能做什么"。不同 Ghost 可以共用同一个 Mode，也可以各用各的 Mode。

**Ghost 的 soul 就是 system prompt。** `soul.md` 的内容决定了 AI 的自我认知。写得越具体（身体能力、安全边界、人格特征），Ghost 的行为越自然。

## 相关文档

- `moss codex get-interface ghoshell_moss_contrib.moss_in_reachy_mini.main` — MossInReachyMini 接口
- `.ai_partners/features/workstreams/2026/05/reachymini-integration/FEATURE.md` — 集成过程的完整决策记录
- `.ai_partners/features/workstreams/2026/05/reachy-mini-contrib/FEATURE.md` — contrib 包开发记录
- `.moss_ws/apps/CLAUDE.md` — App 开发指南
- `moss docs read model-oriented-application-system.md` — App 体系论述

---

## 验证记录

| 时间 | 模型 | 备注 |
|------|------|------|
| (待走通) | | |
