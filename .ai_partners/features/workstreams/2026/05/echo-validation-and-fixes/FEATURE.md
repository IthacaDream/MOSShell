---
title: Echo Ghost Validation & Fixes
status: in-progress
priority: P0
created: 2026-05-23
updated: 2026-05-25T23:00
depends: [first-ghost-prototype]
description: >-
  echo ghost human validation 中发现的 bug 修复。
---

# Echo Ghost Validation & Fixes

本 workstream 只记录在 echo 验证中**新发现**且不属于 first-ghost-prototype 已知遗留的问题。
first-ghost-prototype 已追踪的项（cleanup 死锁、hook 体系、echo soul、bootstrap warning、TUI 全链路验证、mindflow inspect）不在此重复。

## PROMPT 面板空渲染（2026-05-23, fixed）

`ghost_runtime.py:267` — `moment.reaction_instruction is not None` → `moment.reaction_instruction`。

`Signal.prompt` 默认 `""`，空字符串 `is not None`，导致 PROMPT 面板始终渲染但无内容。
`conversation.py:191` 对同一字段已用 truthiness check，此处漏了。

## Logger 集中化（2026-05-23, fixed）

### 问题

1. **无启动早期 logger**：Environment → Host → Matrix 在容器就绪前无法记录日志。
2. **GhostRuntimeImpl.__aenter__ 用 `steps` 列表攒日志**：因为 step 1 时 MossRuntime 未启动，拿不到 matrix.logger，只能把日志消息推到列表里，等 step 2 进入后再 flush。
3. **MatrixImpl.logger 三层回退**：explicit → IoC container → `logging.getLogger()`，过度设计且回退路径不一致。
4. **两处 WorkspaceLoggerProvider 竞争**：manifests 注册一个（无参），`_default_providers` 又注册一个（带 log_name），manifests 先到先得，逻辑正确但隐晦。
5. **GhostRuntimeImpl 中 4 种 logger 访问路径**：`self.moss.logger` / `self._moss_runtime.matrix.logger` / `matrix.logger` — 都解析到同一对象但不统一。

### 方案

**Environment 预置 console logger → workspace LoggerProvider 替换 → 全链路统一使用。**

```
Environment.__init__        → logger = getLogger('moss') + console handler (默认)
Host / Matrix.__init__      → 使用 env.logger (默认 console)
Matrix.__aenter__           → container.get(LoggerItf) → WorkspaceLoggerProvider
                            → env.set_logger(...)       (替换为 workspace 配置的 logger)
之后所有消费者                → env.logger = workspace 配置的 logger
```

### 改动

| 文件 | 改动 |
|------|------|
| `core/blueprint/environment.py` | 新增 `_create_default_logger()` (NullHandler)、`logger` 属性、`set_logger()` 方法 |
| `host/impl.py` | 移除 Host 的 `logger` 构造参数和 `LoggerItf`/`logging` import |
| `host/matrix.py` | `_logger` → `_logger_override`；删除 `container.set(LoggerItf, ...)` 覆盖；`logger` 属性简化为 `_logger_override or env.logger`；`__aenter__` 中早期解析 LoggerItf 并 `env.set_logger()` |
| `host/ghost_runtime.py` | 删除 `steps` 列表 workaround，统一所有 logger 访问为 `self.moss.logger` |
| `host/providers/logger_provider.py` | 修复 YAML 配置加载判断：忽略 `NullHandler` 实例 |

### 二次简化（2026-05-25）

初版实现引入了不必要的复杂度：Environment 持有 NullHandler logger 然后被 Matrix.__aenter__ 替换、WorkspaceLoggerProvider 同时管 YAML 加载/命名/默认 handler 三件事、env.set_logger() 形成双向依赖。

**简化方案**：

| 职责 | 归属 | 说明 |
|------|------|------|
| logging.yml 按约定加载 | `Environment.bootstrap()` | workspace/configs/logging.yml 存在则 `config_logger_from_yaml()` |
| 日志名约定 (`moss.<address>`) | `Environment.logger` property | cell address 替换 `/` → `.`，`logging.getLogger()` O(1) 无缓存 |
| 默认 file handler | `WorkspaceLoggerProvider` | 仅确保 moss root logger 有 handler，返回 `logging.getLogger('moss')` |

**Matrix.logger 优先级**：`_logger`（IoC 注入）→ `env.logger`（约定名）。

**删除**：`_create_default_logger()`、`NullHandler`、`Environment.set_logger()`、`WorkspaceLoggerProvider` 的 `logger_name` 参数和 YAML 加载逻辑。

## TUI 面板折叠/展开（2026-05-23, fixed）

### 问题

Ghost TUI 中 MOMENT 面板渲染内容极大（80+ 行），每次对话都铺满屏幕，严重干扰人的交互流。用户需要在折叠（默认）和展开之间快速切换。

### 方案

`ConsoleOutput.format_output()` 对 `role == 'moment'` 的面板默认渲染为单行摘要（`⊟ MOMENT (N messages, M lines) ctrl+o to expand`），其余角色默认展开。

`ctrl+o` 一键展开当前缓冲区中所有面板（`ConsoleOutput.replay_recent(force_expand=True)`），不改变持久状态——新输出仍默认折叠。重复按 ctrl+o 不重复渲染（`_recent_expanded` 防抖）。

### 改动

- `tui.py` — `ConsoleOutput`: 新增 `_recent_items` 缓冲（max 50）、`_recent_expanded` 防抖、`replay_recent()` 方法；`format_output()` 接受 `force_expand` 参数，`role == 'moment'` 默认折叠
- `tui.py` — `MossHostTUI.default_key_bindings()`: 新增 `c-o` 绑定
- `tui.py` — Welcome 快捷指南新增 `Expand Panels | ctrl+o`

### 位置

- `src/ghoshell_moss/host/tui.py`

---

## TUI logos 流式换行（基础设施已就绪，待集成，2026-05-23）

> 渲染基础设施由 [tui-stream-rendering](../tui-stream-rendering/FEATURE.md) 提供。
> 集成在 `ghost_ui.py:_consume_logos`，见该 feature 的 Integration Guide。

### 问题

`moss-run-ghost` TUI 中模型 logos 流式输出换行频率极高，文本按 token 粒度断裂：

```
Oops

，刚才 calculator

 可能还没完全启动我就

急着算了
```

### 根因分析

`ghost_ui.py:89` `_consume_logos` 对每个流式 delta 调用 `self.console.rprint(delta)`。`rprint()` → `_renderable_queue` → `_direct_print()` → `Console.print()` 每次调用自动换行。中文模型 token 粒度极细，每个 token 一个 delta，每个 delta 一行。

`ghost_runtime.py:273-274` 中 `session.pub_logos(delta)` 以 `articulate()` 的 yield 粒度发布 delta，不做任何聚合。

### 尝试过的方案

| 方案 | 效果 |
|------|------|
| `Console.print(text, end='', markup=False)` | Rich 渲染管线光标控制序列互相干扰，后一次覆盖前一次，同位置闪烁 |
| `Console.file.write(text) + flush()` | 同上，仍被覆盖 |

在当前 `rprint()` → render queue → `Console.print()` 渲染设施下，未找到干净的同行追加方式。需要更底层的终端控制或不同的渲染架构。

### 位置

- `tui_entries/ghost_ui.py` — `_consume_logos`
- `tui.py` — `ConsoleOutput.rprint()` / `_direct_print()`
- `ghost_runtime.py` — delta 发布点

## TUI 空 COMMAND-RESULT 块（2026-05-23, fixed）

### 问题

CTML 命令 `<apps.tools_calculator:add _args="[3, 7]"/>` 在 TUI 中渲染 5 个 `COMMAND-RESULT` Panel，其中 4 个为空：

```
╭─  COMMAND-RESULT  ──╮  ← 空（__content__）
╰──────────────────────╯
╭─  COMMAND-RESULT  ──╮  ← 空（__enter__）
╰──────────────────────╯
╭─  COMMAND-RESULT  ──╮  ← 10.0（add 实际结果）
╰──────────────────────╯
╭─  COMMAND-RESULT  ──╮  ← 空（__content__）
╰──────────────────────╯
╭─  COMMAND-RESULT  ──╮  ← 空（__exit__）
╰──────────────────────╯
```

### 根因

`ghost_runtime.py:374-375` — 每个 `CommandTask` 完成时无条件发 `session.output('command-result', ...)`，
即使 `msgs` 为空。Channel scope 生命周期产生 `__content__`、`__enter__`、`__exit__` 等内部任务，
结果均为空但照样渲染 Panel。

### 方案

TUI 消费侧过滤，不动 session 数据总线：`_on_session_output` 遇到空 messages 直接 return。

`ghost_ui.py:82` — `if not item.messages: return`

## MossHostTUI._get_runtime classmethod 约束不自然（2026-05-23, fixed）

`_get_runtime` 被 ABC 定义为 `@classmethod`，但四个实现无一使用 `cls`，调用点也是 `self._get_runtime(...)` 实例调用。去掉 `@classmethod` 改为普通实例方法，参数 `host` 也省掉（`self.host` 已就绪）。

修改 4 个文件 5 处：ABC 定义 + 调用点 + GhostTUI + MossRuntimeTUI + EchoCase。

## Container bootstrap warning — GhostREPLState 过早访问 session（2026-05-23, fixed）

### 问题

`moss-run-ghost echo` 启动时三条 `UserWarning: container have not bootstrapped before using`。

### 根因

`tui.run()` → `create_states()` → `GhostREPLState.__init__` 访问 `ghost_runtime.moss.session`
→ `MatrixImpl.session` → `container.force_fetch(Session)`，此时 `bootstrap()` 尚未调用。

`bootstrap()` 在 `_main_loop()` → `enter_async_context(runtime)` → `MatrixImpl.__aenter__` 才执行，
晚于 `create_states()`。

触发链：
1. `force_fetch(Session)` → `get(Session)` → warning #1 (containers.py:332)
2. `WorkspaceSessionProvider.factory()` → `con.get(LoggerItf)` → warning #2 (moss_session_provider.py:47)
3. factory 内 `con.force_fetch(TopicService)` → `ZenohTopicServiceProvider.factory()` → `con.get(LoggerItf)` → warning #3

### 方案

`GhostREPLState._session` 从 `__init__` 的 eager 赋值改为 `@property`，延迟到首次访问（`__aenter__` 之后，此时 Matrix 已启动、container 已 bootstrap）。

### 位置

- `tui_entries/ghost_ui.py:26` — `self._session = ghost_runtime.moss.session` → `@property`

## Zenoh session 修复（2026-05-23, fixed）

`core/session/zenoh_session.py` 中 4 处 TODO/已知问题。

### 1. `storage()` 竞态条件（L146）

**问题**：lazy init check-then-act，两线程同时首次访问 property 时 `make_session_level_storage` 执行两次，第一个 Storage 对象悬空。

**方案**：eager init — `__init__` 中直接调用 `make_session_level_storage()`，消除 lazy init 和检查窗口。`storage` property 简化为直接返回。

### 2. `add_signal()` 无防蠢限频（L160）

**判断**：非 bug，设计缺口。当前调用方（TUI 按键、脚本单次发送）频率天然受限，限频属于提前优化。

**方案**：不添加限频逻辑。删除 TODO，加 docstring 标注调用方负责控制频率，限频应在 Mindflow 的 signal ingestion 层实现。

### 3. 流式 key 解析静默丢弃（L262）

**问题**：running 状态下收到 key 不匹配 prefix 的 sample 时静默 return，掩盖系统 bug（zenoh key_expr 路由错误）。

**方案**：加 `logger.warning` 输出 unexpected key 和当前 prefix。非 running 状态仍静默（已在前面 `is_running()` 检查 return）。

### 4. 消费能力不足静默失败（L358）

**问题**：原使用 `sync_q.put()` 阻塞 zenoh 回调线程，满时阻塞影响全局通讯总线。注释将 `SyncQueueShutDown`（正常关闭）误描述为"消费能力不足"。

**方案**：
- `sync_q.put()` → `sync_q.put_nowait()`，避免阻塞 zenoh 内部线程
- `queue.Full` 时 `logging.getLogger(__name__).warning` 记录丢弃
- `SyncQueueShutDown`（正常关闭）静默标记 `_closed = True`
- 修正注释，区分两种异常路径

### 5. 判断总结

| # | 实际严重程度 | 处理方式 |
|---|-------------|---------|
| 1 | 低 — 实际不太可能触发 | eager init |
| 2 | 设计缺口，非 bug | docstring 约定 |
| 3 | 中 — 掩盖真实 bug | warning log |
| 4 | 低 — 默认无界队列不触发 | put_nowait + 丢弃 log |

---

## _stream_execute 返回 status_messages() 导致命令结果未进入 Reaction outcomes（2026-05-24, fixed）

### 问题

模型调用命令后，命令的实际返回值在后续对话轮次中不可见，导致模型无法基于工具结果进行推理，可能反复调用同一命令形成循环。

### 根因

`GhostRuntimeImpl._stream_execute`（`ghost_runtime.py:407,417`）结尾处：

```python
status = interpretation.status_messages()
session.output('system', *status)
return status, interpretation.observe
```

`status_messages()` 只包含 `tag='shell'` 的元信息（如 `compiled commands: N / done: M`），描述命令被执行了，但不含返回值。

命令的真实返回值在 `interpretation.executed_messages()` 中——`CommandTaskResult.as_messages()` 产出的 `tag='command'` 消息，内含序列化后的 result 数据和业务层附加的 messages。

`_stream_execute` 的返回值最终进入 `action.outcome()` → `Reaction.outcomes` → `Moment.previous_reaction_messages()` → 模型上下文。因为 outcomes 里只有 shell 状态，模型在所有后续轮次中都看不到命令的实际输出。

### 方案

返回值从 `status_messages()` 换为 `as_messages()`（已存在的 `Interpretation.as_messages()` = `status_messages()` + `executed_messages()`）：

```python
messages = interpretation.as_messages()          # status + executed
session.output('system', *interpretation.status_messages())  # 显示仍只输出状态
return messages, interpretation.observe
```

Session 输出保持只显示状态（给人看的），返回值承载全部执行结果（给模型看的）。语义上也正确——outcomes 本就该承载全部执行结果。

### 影响面

所有基于 `CommandTaskResult` 的命令（`PyChannel.build.command()` 注册的 channel 命令）。任何 `observe=True` 的命令，其返回值此前对模型不可见。

### 位置

- `src/ghoshell_moss/host/ghost_runtime.py:339-417` — `_stream_execute`

---

## channels() key 约定不一致导致 main channel 未被发现（2026-05-25, fixed）

### 问题

`moss-run-ghost echo` 无法发现 `.moss_ws/src/MOSS/manifests/channels.py` 中注册的 channel。Shell 启动后使用的是 fallback 空白默认 channel，注册的 AppStoreChannel 和原语（sleep/noop/observe/interrupt）全部丢失。

### 根因

`channel-discovery-rework` 重构后，`search_channels_from_package` 用 **Python 变量名**作为 `channels()` 字典的 key（注释写明"以 attr name 作为唯一键"）。但 `MossRuntimeImpl.__init__` 用 `channels().get("__main__")` 去查找——`"__main__"` 是 `Channel.name()` 的值，不是变量名。

`.moss_ws/src/MOSS/manifests/channels.py` 中变量名为 `main`，`Channel.name()` 为 `"__main__"`。`channels()` 返回 `{"main": PyChannel}`，`.get("__main__")` 返回 `None` → fallback 空白 channel。

### 方案

`MossRuntimeImpl.__init__` 中把 key lookup 改为遍历 values 匹配 `ch.name() == "__main__"`：

```python
# before
manifests_main = self._matrix.manifests.channels().get("__main__")

# after
manifests_main = next(
    (ch for ch in self._matrix.manifests.channels().values() if ch.name() == "__main__"),
    None,
)
```

不改 `search_channels_from_package` 的 key 语义，因为注释已明确约定"以 attr name 作为唯一键"，且调用方（`manifests_cli.py`、`inspector_manifests.py`）均按变量名 key 使用。

### 位置

- `src/ghoshell_moss/host/moss_runtime.py:65-68`

## Python 3.10 兼容性降级（2026-05-25, fixed）

### 问题

项目原本在 Python 3.11+ 上开发（`typing.Self`、`enum.StrEnum`、f-string 内 `\n` 转义等 3.11+ 语法特性）。降级到 Python 3.10 后编译/导入失败。

### 语法兼容

| 问题 | 方案 | 涉及文件 |
|---|---|---|
| `typing.Self`（3.11+） | → `typing_extensions.Self` | `input_signal_nucleus.py`, `base_attention.py`, `buffer_nucleus.py`, `zenoh_session.py`, `echo_case.py` |
| `typing.TypedDict`（Pydantic 要求 3.12 以下用 `typing_extensions`） | → `typing_extensions.TypedDict` | `host.py`, `repl_registrar.py`, `speech.py`, `abcd.py` |
| `enum.StrEnum`（3.11+） | → `(str, Enum)` | `matrix.py`, `app.py` |
| f-string 内 `\n` 转义（3.12 前不允许） | 提取到外部变量 | `slide_studio.py`（2 处） |

### 测试适配

| 问题 | 方案 |
|---|---|
| `ExceptionGroup`（3.11+ 内置） | 统一 `from exceptiongroup import ExceptionGroup`（anyio 的传递依赖，全版本可用） |
| `inspect.isclass(dict[str, int])` 在 3.10 返回 `True` | 改用 `type(a) is types.GenericAlias`（版本无关的精确断言） |
| `test_topic_keep_latest` 消费者/主循环竞态 | 消费者读前加 `await service.wait_sent()`，消除竞态窗口 |
| `test_primitive_cannot_be_used_in_non_root_channel` 解析错误被 `raise_exception()` 直接抛出 | 改为 `pytest.raises(InterpretError)`，匹配实际错误传播路径 |

### 位置

- `src/ghoshell_moss/core/mindflow/` — Self → typing_extensions
- `src/ghoshell_moss/core/session/zenoh_session.py` — Self → typing_extensions
- `src/ghoshell_moss/host/repl/echo_case.py` — Self → typing_extensions
- `src/ghoshell_moss/core/blueprint/` — TypedDict → typing_extensions, StrEnum → (str, Enum)
- `src/ghoshell_moss/contracts/speech.py` — TypedDict → typing_extensions
- `src/ghoshell_moss/host/repl/repl_registrar.py` — TypedDict → typing_extensions
- `src/ghoshell_moss/message/contents/abcd.py` — TypedDict → typing_extensions
- `src/ghoshell_moss_contrib/channels/slide_studio.py` — f-string 修复
- `tests/` — 上述 4 个测试文件

## 复苏指引

新 AI 实例进入此 workstream 时：
- 读本文件完整内容
- 读 `first-ghost-prototype/FEATURE.md` 的"当前状态"和"已知遗留问题"章节 — 本 workstream 不重复追踪那些项
- 读 `.memory/daily/2026-05/22.md` — deepseek-v4-pro 的 session 记录
- logos 流式换行基础设施已由 tui-stream-rendering 实现，需完成 `ghost_ui.py:_consume_logos` 集成