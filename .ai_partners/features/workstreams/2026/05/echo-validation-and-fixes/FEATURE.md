---
title: Echo Ghost Validation & Fixes
status: in-progress
priority: P0
created: 2026-05-23
updated: 2026-05-23
depends: [first-ghost-prototype]
description: >-
  echo ghost human validation 中发现的 bug 修复。
---

# Echo Ghost Validation & Fixes

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

## Cleanup 死锁（待修复）

`gr.close()` 后进程 hang，Matrix/zenoh teardown 阶段卡住。

在 Atom 端到端验证（run_atom_hello.py）中首次发现。与脚本等待逻辑无关，是 Matrix/zenoh 生命周期关闭的独立问题。

位置：`host/ghost_runtime.py` teardown 流程 + `host/matrix.py` zenoh 生命周期管理。

## bootstrap logger 越界 warning（已知，不重要）

GhostRuntime 启动过程中 logger 输出越界 warning。不影响功能，但启动日志中可见。具体触发点待定位。

## MossHostTUI._get_runtime classmethod 约束不自然（设计坏味道）

`host/tui.py:321-323` — `_get_runtime` 被 ABC 定义为 `@classmethod`，但子类 `GhostTUI` 需要实例级上下文（`host.env.ghost_name`）才能创建正确的 runtime。

此前用 `_ghost_name` class attribute workaround，已在 `a695a4f` 修复为通过 `host.env.ghost_name` 读取。但 classmethod 签名本身对子类仍是不自然的约束。

daily memory 2026-05-22 锚点："这个 ABC 对子类不够友好，留给后续重构。"

位置：`host/tui.py:321-323` + `host/tui_entries/ghost_ui.py:104-106`

## Hook 体系缺失（待实装）

`ghost_runtime.py` 中有 9 个 `# todo: hook —` 标记，覆盖三循环全部生命周期关键节点，均未实装：

| 类别 | Hook | 位置 |
|------|------|------|
| Lifecycle | `on_started` | L115 |
| Lifecycle | `on_stopping` | L120 |
| Lifecycle | `on_stopped` | L128 |
| Mindflow Error | `on_fatal` | L227, L290, L331 |
| Mindflow Error | `on_attention_error` | L231 |
| Articulate | `on_articulate_enter` | L258 |
| Articulate | `on_articulate_error` | L279 |
| Articulate | `on_articulate_exit` | L287 |
| Action | `on_action_enter` | L324 |
| Action | `on_action_exit` | L328 |

这些 hook 标记是在生命周期异常治理（2026-05-22）中留下的，意图为后续 hook 体系保留插入点，但不阻塞当前功能。

## echo 无 kernel prompt（依赖 ghost playground）

echo 当前使用原始 atom 提示词，没有配套的 soul/system prompt。模型输出的 logos 品质不高。

依赖 ghost playground workstream（first-ghost-prototype step 14）产出 `echo.md` soul 文件。

位置：`src/MOSS/ghosts/echo.py` — `soul_path="echo.md"` 指向不存在的文件。

## Zenoh session 已知问题

`core/session/zenoh_session.py` 中有多处 TODO：

| 位置 | 问题 |
|------|------|
| L146 | `storage()` 延迟初始化存在潜在竞态条件 |
| L160 | `add_signal()` 无防蠢限频 |
| L262 | 流式 key 解析失败时静默丢弃 |
| L358 | 消费能力不足时静默失败，消费侧无感知 |

## 未完成的验证

- **TUI 全链路验证**（first-ghost-prototype TASKS.md:12c）— `moss-run-ghost echo` 启动后可交互，但未做完整的全链路回归验证
- **Mindflow inspect + 自解释接口**（TASKS.md:11b）— Mindflow 探知和自解释接口未实现

---

## 复苏指引

新 AI 实例进入此 workstream 时：
- 读本文件完整内容
- 读 `first-ghost-prototype/FEATURE.md` 的"当前状态"和"已知遗留问题"章节
- 读 `.memory/daily/2026-05/22.md` — deepseek-v4-pro 的 session 记录
- 读 `first-ghost-prototype/DESIGN.md` 的"TUI 集成设计"和"默认 Ghost — echo"章节
- 检查 `ghost_runtime.py` 中的 `# todo: hook —` 标记当前状态
