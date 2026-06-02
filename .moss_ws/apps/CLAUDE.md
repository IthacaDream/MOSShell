# MOSS App 开发

App 是 MOSS 架构中 **AI 可在运行时创建、启动、调用、关闭的进程单元**。它是 Ghost 可插拔的器官——独立进程、独立依赖、Matrix 总线通讯。

## 核心概念

- **运行时自迭代**：AI 不是工具调用者，而是能力的创造者。完整闭环：`create` → 写码 → `start` → CTML 调用 → `stop`，全程在同一会话中完成，不需要重启 host
- **生命周期**：`STOPPED → STARTING → RUNNING → ERROR`，由 Circus 管理子进程
- **通讯总线**：Matrix (Zenoh) 动态注册，Shell 通过 ChannelProxy 发现和调用
- **Mode 控制**：App 的可见性和自动启动由 Mode 的 `apps` 白名单和 `bringup_apps` 控制
- **App 不一定是 Channel provider**：可以是 GUI 工具、自主感知 Agent、纯后台进程——生命周期由 AI 决定

## 目录约定

每个 App 是 `apps/<group>/<name>/` 下的目录，最小包含 `APP.md`：

```
apps/<group>/<name>/
├── APP.md           # frontmatter: executable, script, workers, respawn, max_age
├── main.py          # 入口脚本
├── CLAUDE.md        # AI 开发者上下文（本 App 专属）
├── pyproject.toml   # 可选：独立依赖环境
└── runtime/         # 运行时数据（自动创建）
    ├── assets/
    ├── configs/
    └── logs/
```

`APP.md` 的 frontmatter 即为 `AppWatcher` 全部配置：

```yaml
---
executable: uv        # 启动器，默认 uv
script: main.py       # 入口脚本
arguments: ""         # 启动参数
description: ""       # 描述
respawn: false        # 崩溃后自动重启
workers: 1            # 工作进程数
max_age: null         # 进程最大存活秒数
---
```

不写则用默认值。Group 名以 `_` 开头的不匹配 `*` 通配符，用于系统级 App。

## 创建 App

```bash
moss apps create <group/name> -d "what it does"
```

产物包含 `APP.md` + `main.py`（helloworld 模板）+ `CLAUDE.md`。创建后自动刷新 AppStore 发现缓存。

## App 入口模式

所有 App 共享同一个入口模式：

```python
from ghoshell_moss.core.blueprint.matrix import Matrix

async def main(matrix: Matrix):
    # App 逻辑

if __name__ == "__main__":
    Matrix.discover().run(main)
```

`Matrix.discover()` 通过环境变量自动发现当前进程的 Cell 身份，`run(main)` 管理 Matrix 生命周期。

### Channel App（最常见）

向 Matrix 注册 Channel，让 Shell/Ghost 通过 CTML 调用：

```python
from ghoshell_moss.core.blueprint.channel_builder import new_channel

channel = new_channel(name="my_tool", description="does something")

@channel.build.command()
async def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@channel.build.context_messages
async def context() -> list[str]:
    return ["current state"]

async def main(matrix: Matrix):
    await matrix.provide_channel(channel)

if __name__ == "__main__":
    Matrix.discover().run(main)
```

Channel 名称成为 CTML 调用路径：`apps.<group>_<name>:add`

### 纯进程 App

不参与 Matrix 通讯，AI 仅管理其启停：

```python
if __name__ == "__main__":
    print("hello world")
```

### GUI App

GUI 占据主线程，Matrix 在异步侧运行：

```python
async def gui_main(matrix: Matrix):
    # TUI 或 GUI 主循环
    ...

if __name__ == "__main__":
    Matrix.discover().run(gui_main)
```

### Signal 生产者 App

周期性向总线推送消息，适合感知模块：

```python
async def producer(matrix: Matrix):
    session = matrix.session_storage
    while True:
        msg = Message.new().with_content("periodic signal")
        session.output('log', msg)
        await asyncio.sleep(1)
```

## 开发与调试

```bash
# 发现与管理
moss apps list                     # 所有 App，含运行状态
moss apps show <group/name>        # 单个 App 详情
moss apps test <group/name>        # 前台运行，Ctrl+C 停止

# 运行时控制（MCP / CTML）
<apps:list_apps />
<apps:start fullname="group/name" timeout="3.0" />
<apps:stop fullname="group/name" />
```

`start` 的 timeout：
- `-1` (默认): 不等待，立即返回
- `0`: 无限等待直到 Channel connected
- `>0`: 等待 N 秒后超时返回 WARN

## 依赖隔离

App 默认以 `uv run` 启动，三层回退：

| 优先级 | 方式 | 适用场景 |
|--------|------|----------|
| 1 | `pyproject.toml`（独立 venv） | 需要特定依赖版本的复杂 App |
| 2 | PEP 723（`// script` 内联元数据） | 单文件 App，轻量依赖 |
| 3 | 共享运行时 | 快速原型，只依赖 `ghoshell_moss` |

无论哪种方式，App 进程内必须能 `import ghoshell_moss`。

## Mode 集成

在 Mode 的 `MODE.md` 中声明：

```yaml
apps:
  - '_system_tests/*'
  - 'perception/*'
bringup_apps:
  - 'perception/vision'
```

- `apps` — 该 Mode 下可见的 App（通配符匹配 `group/name`）
- `bringup_apps` — Mode 启动时自动拉起
- 不带 bringup 的 App 按需启动，不占上下文

## 深入路径

- App 体系完整论述：`moss docs read model-oriented-application-system.md`
- Channel 构建：`moss codex get-interface ghoshell_moss.core.blueprint.channel_builder`
- Matrix API：`moss codex get-interface ghoshell_moss.core.blueprint.matrix`
- App 抽象定义：`moss codex get-interface ghoshell_moss.core.blueprint.app`
- AppStoreChannel（AI 控制面）：`moss codex get-interface ghoshell_moss.channels.app_store_channel`
- States Channel（高阶有状态 Channel）：`moss codex list ghoshell_moss.core.blueprint.states_channel`
- 完整示例：项目根目录 `.moss_ws/apps/` 下的正式 App（如 `bodies/reachymini`）
