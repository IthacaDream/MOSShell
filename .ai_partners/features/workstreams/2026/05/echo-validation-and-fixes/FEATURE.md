---
title: Echo Ghost Validation & Fixes
status: completed
priority: P0
created: 2026-05-23
updated: 2026-05-28T17:30
depends: [first-ghost-prototype]
description: >-
  echo ghost human validation 中发现的 bug 修复。
  包含跨进程 context_messages 图片序列化丢失的完整修复。
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

## TUI logos 流式换行（2026-05-25, fixed）

> 渲染基础设施由 [tui-stream-rendering](../tui-stream-rendering/FEATURE.md) 提供。

### 问题

`moss-run-ghost` TUI 中模型 logos 流式输出换行频率极高，文本按 token 粒度断裂。

### 方案

`_consume_logos` 使用 `LiveStreamSink` 按 utterance 聚合渲染。`LiveStreamSink` 跨 asyncio/sync 边界：asyncio 侧 `send(delta)` 通过 janus 队列传递 delta，渲染线程在 `render(console)` 中阻塞消费。`ghost_runtime._articulate_loop` 在每个 articulation 结束后 `pub_logos("\n\n")` 作为 utterance 边界标记，触发 `sink.commit()`。

渲染终版使用 **ANSI 原地替换 + Rich Panel**：`\033[{N}F` 上移光标 + `\033[J` 清屏 → `console.capture()` 渲染 Panel → `console.file.write()` 直写终端。模型回复以独立的 RESPONSE panel 在终端原位流式更新。

迭代经历：
- `console.print(text, end='')` → Rich 与 prompt_toolkit 光标冲突，吞字符
- `console.file.write(text)` → 无 panel，不符合用户期望
- `ANSI + Panel` → 终版，独立 panel 原地流式更新

### 位置

- `tui_entries/ghost_ui.py` — `_consume_logos`：按 utterance 管理 `LiveStreamSink` 生命周期
- `tui.py` — `LiveStreamSink.render()`：ANSI 原地替换 + Panel 渲染（tui-stream-rendering 基础设施）

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

## 跨进程 context_messages 图片序列化丢失（2026-05-27, fixed）

### 问题

`.moss_ws/apps` 下的 channel 通过 `context_messages` 返回 PIL Image，经过 proxy/provider (zenoh) 协议跨进程传输后，proxy 侧收到的 context 中 image content 为空。

### 根因

四个 bug 叠加导致图片在序列化链路中消失：

**Bug 1 — `Message.wrap_content()` 漏调 `.to_content()`**（`message.py:359-360`）

`Base64Image.from_pil_image(item)` 返回的是 Pydantic BaseModel 实例，不是 `Content` TypedDict。直接放入 `contents` list 后，序列化时 `model_dump_json()` 无法正确识别 TypedDict 结构，`type` 和 `source` 键丢失。

**Bug 2 — `Message.content_as_string()` 假设 `type` 键一定存在**（`message.py:487-493`）

`content.get('type')` 未做防御，图片 Content 序列化失败后 `type` 缺失，导致 TUI 渲染时崩溃。

**Bug 3 — `PyChannelBuilder._wrap_messages()` 丢弃累积的非 Message 对象**（`py_channel.py:131`）

静态方法中 `result.append(msg)` 应为 `result.append(last)`。连续的非 Message raw items（如 PIL Image + str）被累积到 `last` Message 中，但遇到下一个 Message 时错误地 append 了当前 Message 而非累积的 `last`，导致前面累积的内容被丢弃。

**Bug 4（主因）— `ChannelEventModel.to_channel_event()` 的 `exclude_unset=True`**（`protocol.py:62`）

Pydantic 通过 `model_fields_set` 追踪字段是否被"显式设置"。`Message.with_content()` 内部使用 `self.contents.append()`（原地修改），不触发 `__setattr__`，Pydantic 认为 `contents` 字段 "unset"。`exclude_unset=True` 导致 `model_dump_json()` 静默丢弃 `contents`——这是跨进程传输后图片丢失的**根本原因**。

单独测试 `Message.model_dump_json()` 能保留 contents 是因为没有 `exclude_unset`；但 `ChannelMetaUpdateEvent` → `ChannelMeta` → `Message` 的嵌套序列化经过 `to_channel_event()` 时 `exclude_unset=True` 被递归应用，contents 全部丢失。

最小复现：
```python
from pydantic import BaseModel, Field
class Demo(BaseModel):
    items: list[str] = Field(default_factory=list)
d = Demo()
d.items.append('hello')
d.model_dump_json(exclude_unset=True)  # → {} — items 丢失
```

### 方案

| # | 文件 | 修复 |
|---|------|------|
| 1 | `message.py:360` | `Base64Image.from_pil_image(item)` → `.to_content()` |
| 2 | `message.py:488-497` | `content['type']` → `content.get('type', 'unknown')`，图片类型展示 `media_type` 和 `base64_size` |
| 3 | `py_channel.py:131` | `result.append(msg)` → `result.append(last)` |
| 4 | `protocol.py:62` | 删除 `exclude_unset=True`；整个代码库中 `default_factory=list` + 原地 `.append()` 的模式太普遍，不应逐个修改 |

### 单测覆盖

新增测试文件 `tests/ghoshell_moss/core/ctml/shell/test_shell_image.py`（10 个用例）：

| 测试 | 覆盖场景 |
|------|---------|
| `test_shell_image_baseline` | shell 命令直接返回 PIL Image |
| `test_shell_image_return_bytes` | shell 命令返回图片 bytes |
| `test_shell_image_in_sub_channel` | 子 channel 中的 image 命令 |
| `test_shell_image_with_args` | 参数化图片生成 |
| `test_context_messages_with_pil_image` | channel context_messages 返回 PIL Image（本地） |
| `test_shell_context_messages_with_image` | shell 级 context_messages + refresh_metas |
| `test_context_messages_with_image_and_text` | 混合 image + text 的 context_messages |
| `test_context_messages_image_survives_serialization` | Message → JSON → dict → Message 往返 |
| `test_image_content_survives_message_roundtrip` | wrap_content → Message → JSON → from_content 全链路 |
| `test_context_messages_image_through_bridge` | **关键**：通过 thread bridge（同 `to_channel_event`/`from_channel_event` 序列化路径）的全链路集成测试 |

### 为什么之前的单测没发现

本地 PyChannel 测试不跨进程序列化，`ChannelMeta` 在进程内以 Python 对象传递，不经过 `to_channel_event()` → `model_dump_json(exclude_unset=True)` 路径。只有 app 部署通过 zenoh bridge 跨进程时才触发。

Thread bridge（`create_thread_bridge`）虽然用 in-process Queue，但事件数据经过相同的 `model_dump_json()` / `json.loads()` 序列化/反序列化——因此 `test_context_messages_image_through_bridge` 能复现真实 zenoh 场景的 bug。

### 位置

- `src/ghoshell_moss/message/message.py` — wrap_content, content_as_string
- `src/ghoshell_moss/core/py_channel.py` — _wrap_messages static method
- `src/ghoshell_moss/core/duplex/protocol.py` — to_channel_event
- `tests/ghoshell_moss/core/ctml/shell/test_shell_image.py` — 新增

## 多 app 并行 bringup 失败 — circusd arbiter 全局锁冲突（2026-05-28, fixed）

### 问题

`.moss_ws` 的 default mode 中 `bringup_apps` 设置两个 app（如 `tools/calculator` 和 `tools/ping_pong`），启动时第二个 app 报错：

```
CommandError: VALUE_ERROR: failed to start tools/ping_pong
```

circusd 返回 `{'status': 'error', 'reason': 'arbiter is already running arbiter_start_watchers command', 'errno': 5}`。

只设置一个 app 时正常。

### 根因

原 `start_app` 逻辑分两步：`add` watcher → `start` watcher。`start` 命令走 `arbiter.start_watchers()`，该方法有 `@synchronized("arbiter_start_watchers")` 全局锁。`asyncio.gather` 并发启动两个 app 时，两个 `start` 命令间隔仅 ~3ms，circusd 的 arbiter 还没处理完第一个 `arbiter_start_watchers` 就收到了第二个，触发 `ConflictError`。

### 方案

利用 circusd `add` 命令的 `start: true` 选项。`add` 内部调用 `watcher.start()` 直接启动进程，该方法的 `@synchronized("watcher_start")` 是 **per-watcher 实例锁**，不同 watcher 之间不互斥。合并 add+start 为一次 circus 调用，绕过全局锁。

同时修复 `r1['status']` 方括号取值在成功时 KeyError 的问题（`add` + `start: true` 成功返回 `{'started': [...], 'kept': []}`，无 `status` 键），改为 `r1.get('status')`。

### 改动

| 文件 | 改动 |
|------|------|
| `host/app_store.py:262-291` | `start_app`: add watcher 时传 `params['start'] = True`；`r1['status']` → `r1.get('status')`；add 已启动则跳过单独 start |
| `host/app_store.py:67` | 新增 `_call_lock = asyncio.Lock()`，串行化 ZMQ socket 访问防止线程竞争 |
| `host/app_store.py:326-329` | `_call_circus`: async with `_call_lock` 保护 ZMQ 调用 |
| `tests/ghoshell_moss/host/test_app_store_start.py` | 新增 5 个单测覆盖 error 路径和成功路径 |

### 位置

- `src/ghoshell_moss/host/app_store.py`
- `tests/ghoshell_moss/host/test_app_store_start.py`

## 复苏指引

新 AI 实例进入此 workstream 时：
- 读本文件完整内容
- 读 `first-ghost-prototype/FEATURE.md` 的"当前状态"和"已知遗留问题"章节 — 本 workstream 不重复追踪那些项
- 读 `.memory/daily/2026-05/22.md` — deepseek-v4-pro 的 session 记录
- logos 流式换行基础设施已由 tui-stream-rendering 实现，需完成 `ghost_ui.py:_consume_logos` 集成