# Model-Oriented Application System

App 是 MOSS 架构中**面向模型的操作系统进程单元**。模型通过 AppStoreChannel 在运行时发现、启动、关闭 App——不依赖人类操作。

独立进程 + 独立依赖 + Matrix 总线通讯 = 每个 App 是 Ghost 的一个可插拔"器官"。

## 1. What App Is

App 是放在 `apps/<group>/<name>/` 下的目录，包含 `APP.md`（元信息声明）和入口脚本。MOSS 通过目录约定自动发现它，通过 Circus 管理其进程生命周期，通过 Matrix (Zenoh) 将它接入通讯总线。

**App 不一定是 Channel provider。** 一个 App 可以是：
- 一个给人用的 GUI（模型打开它，人操作它）
- 一个暴露能力的 Channel 树根（模型通过 CTML 调用其命令）
- 一个内嵌感知循环的独立 Agent（自主运行，通过 Signal 向 Ghost 汇报）

关键区别：App 的生命周期由**模型**决定。模型看到所有已发现的 App，按需 `start`/`stop`。这不同于微服务的"自主运行"——这里是 OS 进程管理模型，模型是操作者。

核心抽象：

```bash
moss codex get-interface ghoshell_moss.core.blueprint.app
```

- `AppInfo` — 环境中可发现的 App 描述
- `AppWatcher` — App 的启动配置（executable, script, workers, respawn 等）
- `AppStore` — App 生命周期管理抽象（list, init, start, stop）
- `AppState` — 状态机：stopped → starting → running → error

AI 控制面（运行时通过 CTML 调用）：

```bash
moss codex get-interface ghoshell_moss.host.app_store_channel
```

AppStoreChannel 暴露 `list_apps` / `start` / `stop` 三个命令，注册在 Shell 的 Channel 树中。

## 2. Minimal Path

最简创建一个 App 并让它运行：

```bash
moss apps init my_group/my_app -d "what it does"
```

产物：
```
apps/my_group/my_app/
├── APP.md       # frontmatter 元信息（executable, script, workers...）
├── main.py      # 入口脚本
└── CLAUDE.md    # AI 开发者上下文
```

`APP.md` 的 frontmatter 字段即 `AppWatcher` 的全部配置。不写则用默认值（`uv run main.py`）。

调试：

```bash
moss apps list                # 发现所有 App
moss apps show my_group/my_app  # 查看详情
moss apps test my_group/my_app  # 前台运行，Ctrl+C 停止
```

## 3. App 入口模式

所有 App 的入口是同一个模式：

```python
from ghoshell_moss.core.blueprint.matrix import Matrix

async def main(matrix: Matrix):
    # App 逻辑

if __name__ == "__main__":
    Matrix.discover().run(main)
```

`Matrix.discover()` 自动发现当前进程的 Cell 身份，`run(main)` 管理 Matrix 的 `__aenter__`/`__aexit__` 生命周期。

在此入口之上，根据用途分化为几种典型模式。

### 3.1 纯进程 App

不做任何 Matrix 交互。模型可以打开它、关闭它——仅此而已。

例：`_system_tests/helloworld`

```python
def main():
    print("hello world")

if __name__ == "__main__":
    main()
```

典型用途：GUI 输入表单、独立工具窗口、游戏——模型启动，人使用。

### 3.2 Channel App

向 Matrix 注册一个 Channel，让 Shell/Ghost 可以调用其命令、读取其上下文。

例：`_system_tests/provide_channel_case`

```python
from ghoshell_moss.core.blueprint.channel_builder import new_channel

channel = new_channel(name="my_tool", description="does something")

@channel.build.command()
async def add(a: int, b: int) -> int:
    return a + b

@channel.build.context_messages
async def context() -> list[str]:
    return ["current state info"]

async def main(matrix: Matrix):
    await matrix.provide_channel(channel)

if __name__ == "__main__":
    Matrix.discover().run(main)
```

Channel 构建的完整知识：

```bash
moss codex get-interface ghoshell_moss.core.blueprint.channel_builder
```

要点：`Builder.command()` 将 Python 函数反射为 Command；`Builder.context_messages()` 在每个思维关键帧注入动态上下文。一个 App 提供一个 Channel 树根，子节点通过 `import_channels()` 组织。

### 3.3 GUI App

GUI 占据主线程，Matrix 在异步侧运行。

例：`_system_tests/output_monitor`（基于 prompt_toolkit 的 TUI）

```python
async def monitor_main(matrix: Matrix):
    monitor = MossMonitor(matrix)  # 持有 matrix.session.output_buffer()
    await monitor.run()            # TUI 主循环

if __name__ == "__main__":
    Matrix.discover().run(monitor_main)
```

关键：`matrix.session.output_buffer()` 订阅总线输出，TUI 定时刷新。`matrix.create_task()` 将后台协程托管给 Matrix 生命周期。

### 3.4 Signal 生产者 App

周期性向总线推送消息。

例：`_system_tests/output_producer`

```python
async def producer_task(matrix: Matrix):
    session = matrix.session
    while True:
        msg = Message.new().with_content("periodic signal")
        session.output('log', msg)
        await asyncio.sleep(1)

if __name__ == "__main__":
    Matrix.discover().run(producer_task)
```

### 3.5 跨 App Channel 调用

一个 App 可以通过 Matrix 代理调用另一个 App 的 Channel。

例：`_system_tests/proxy_channel_case`

```python
async def main(matrix: Matrix):
    proxy = matrix.channel_proxy(
        "apps/_system_tests/provide_channel_case",
        name="remote_tool",
    )
    async with proxy.bootstrap(matrix.container) as runtime:
        await runtime.wait_connected()
        foo = runtime.get_own_command('foo')
        result = await foo(3, 5)  # → 8
```

## 4. 依赖隔离

App 默认以 `uv run` 启动，三层回退策略：

| 优先级 | 方式 | 适用场景 |
|---|---|---|
| 1 | `pyproject.toml`（独立 venv） | 需要特定依赖版本的复杂 App |
| 2 | PEP 723（`// script` 内联元数据） | 单文件 App，轻量依赖声明 |
| 3 | 共享运行时 | 快速原型，只依赖 `ghoshell_moss` |

独立依赖意味着视觉 App 可以装 `opencv`，GUI App 可以装 `PyQt6`，互不污染。

## 5. Mode 集成

App 的可见性和自动启动由 Mode 控制。在 Mode 的 `MODE.md` 中声明：

```yaml
apps:
  - '_system_tests/*'
  - 'perception/*'
bringup_apps:
  - 'perception/vision'
  - 'body/miku'
```

- `apps` — 该 Mode 下可见的 App（通配符过滤）
- `bringup_apps` — Mode 启动时自动拉起

不带 bringup 标记的 App 按需启动，不占上下文。

## 6. 与 Host 中其它概念的关系

- **Channel**：App 可以提供 Channel，但不是必须。Channel 是"能力"，App 是"进程容器"
- **Matrix**：App 通过 Matrix 与其它 Cell 通讯，Matrix 屏蔽通讯协议
- **Manifests**：App 的发现走目录约定（`apps/<group>/<name>/APP.md`），不是 manifests 体系
- **Primitive**：Shell 原语是内嵌在 Shell 进程中的 Python 函数；App 是独立进程
- **Ghost**：Ghost 通过 AppStoreChannel 管理 App，通过 Channel 使用 App 的能力

## 7. 深入理解

- 完整 Matrix API：`moss codex get-interface ghoshell_moss.core.blueprint.matrix`
- Session 与双工通讯：`moss codex get-interface ghoshell_moss.core.blueprint.session`
- Topic 发布订阅：`moss codex list ghoshell_moss.core.concepts.topic`
- Channel 开发哲学：`moss docs read channel-system`（待完善）
- 自迭代闭环：`moss docs read architecture-topology` §2.9，§3.1 "开发迭代环"

完整示例代码：`.moss_ws/apps/_system_tests/` 目录下 7 个 App。
