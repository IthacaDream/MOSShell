---
created: 2026-05-26
depends: []
description: Command 体系与 Proxy 治理 — CommandMeta 去歧义、ChannelMeta.proxy per-node 标记、timeout 协议、魔法函数协议化、scope 原生协议。
milestone: null
priority: P0
status: completed
status_note: '远程 scope 魔法函数完成。634 全量单测通过。'
title: Command & Proxy Governance
updated: '2026-05-27'
---

# Command & Proxy Governance

## Motivation

1. `CommandMeta.chan` vestigial — command 归属由 ChannelMeta 的 dict key 确定，自标记只剩歧义。
2. `ChannelMeta.proxy: list[str]` 集中式声明，child meta 不自知，flatten 后丢信息。
3. 魔法函数（`__scope__`/`__content__` 等）硬编码在解释器，proxy channel 无法感知和转发。
4. 远程 scope 是 proxy channel 最难的问题 — scope 开/闭/超时/取消需要跨进程正确传递。

## Key Decisions

### 1. 删除 `CommandMeta.chan`

全链路通过 channel path 判断路径。Command 不需要知道自己属于哪个 channel。

### 2. `ChannelMeta.proxy: bool`

`proxy` 是继承属性。路由靠 tree path prefix，不需要在 meta 里存 root id。

### 3. Proxy 子树纯净化

Proxy 下不允许 local child。所有子 channel 来自远程 provider meta sync。

### 4. timeout 协议 — CommandTask 原生属性

`CommandMeta.timeout` → `PyCommand._timeout` → `CommandTask.timeout` 全链路透传。
远程通讯时底层只需感知 `task.timeout`，交叉取消逻辑留在解释器层。

### 5. 魔法函数上升为 ChannelRuntime 协议

`__content__`、`__scope_enter__`、`__scope_exit__` 声明在 `ChannelRuntime` ABC 上，成为公开接口。
任何子类（local、proxy、zenoh）可发现和重写。`unwrap_self_magic_task` 改名为 `partial_bare_magical_task`，
收窄为只处理 `__content__`（scope enter/exit 由 `push_task` 直接分发）。

### 6. Scope 作为原生协议 — `ChannelScope` + `ChannelScopeImpl`

**这是本轮最关键的架构跃迁。**

scope 从解释器的内部逻辑（`TaskScope`）提升为 Channel 概念层的 ABC：

```
ChannelScope (ABC)
  ├── scope_id          — 唯一标识（复用 task.cid）
  ├── add_task()        — 绑定 task → scope，双向 cancel 绑定
  ├── commit()          — 结束注册，之后绑定到 scope 的 task 失败
  ├── tick()            — 开始计时（until + timeout）
  ├── wait_close()      — 阻塞等待 scope 结束
  └── close()           — 主动关闭，级联 cancel 所有子 task
```

`ChannelScopeImpl` 用 sentinel `_future`（`BaseCommandTask`）管理 lifecycle：

- scope→tasks: `_future.done` → `_clear_all_sub_tasks` → cancel all
- tasks→scope: sub_task `critical_failed` → `_future.fail`
- 嵌套 scope: 父 scope 的 `_future` 作为子 scope 的任务被追踪

`push_task` 上的 scope 语法处理（`ChannelRuntime` ABC 层）：

```
__scope_enter__ → open_scope() → new ChannelScopeImpl (stack push)
__scope_exit__  → commit_scope() → scope.commit() → scope.wait_close()
task.scope_id   → get_active_scope() → scope.add_task()
```

**设计思路的迭代过程**：

1. 最初 scope 在 `elements.py` 用 `TaskScope` 实现，完全在解释器层。硬编码。
2. 尝试将 `__scope_enter__`/`__scope_exit__` 作为 bare task 推到 runtime，通过 `_unwrap_self_magic_task` 扩展点 resolve。
3. 发现 magic 协议应该属于 Channel 概念层而非 runtime 实现细节 — 将魔法函数提升到 `ChannelRuntime` ABC。
4. 最终：scope 成为独立的 `ChannelScope` ABC + `ChannelScopeImpl` 实现，stack-based 管理在 `AbsChannelRuntime`。

每次迭代都在推动同一件事：**scope 从解释器专有 → runtime 可重写 → Channel 概念层原生协议**。

### 7. `elements.py` 大规模简化

- `TaskScope` 删除（被 `ChannelScope` 取代）
- `ScopeOpenTask`/`ScopeCloseTask` 改为 `func=None` bare task，scope 逻辑由 runtime resolve
- `_add_to_parent` / `_add_inner_task` 回调删除 — tasks 直接通过 callback 流出
- `speech` 从 `CommandTaskElementContext` 移除 — 与 CTML 解析无关
- `CommandTaskElementContext.instances_count` 删除 — 调试计数器不需要在生产代码

### 8. `ChannelFactory` 模式

`import_channels` 接受 `Callable[[IoCContainer], Channel]` — 延迟创建，bootstrap 时才用容器实例化。
`ChannelInterface`/`ChannelCreator` 示例模式提供了面向对象的 channel 定义路径。

## Implementation Notes

- 分多阶段提交，每阶段全量测试通过
- scope 取消传播：proxy `expect_task_done` → `CommandCancelEvent` → provider `_handle_command_cancel`，已验证
- `InterruptedError` (IDE autocomplete bug) → `InterpretError` → 最终 `raise exp` 直接 re-raise
- shell 层 `CancelledError` 不吞，interpreter 层才吞 — 分层清晰
- `ChannelScopeImpl._bind_task_to_scope` 双向 callback：scope 关闭时 cancel task，task critical fail 时 fail scope

## Verification

- [x] proxy task 取消传播测试通过（thread + zenoh 双后端）
- [x] `CommandMeta.chan` 删除，全量测试通过
- [x] `ZenohTopicSubscriber` shutdown 修复
- [x] `ChannelMeta.proxy: bool` 全量测试通过
- [x] timeout 协议 + bare/magic task pipeline 全量测试通过
- [x] 魔法函数上升为 ChannelRuntime 协议
- [x] scope 原生协议 + `ChannelScope` ABC + `ChannelScopeImpl`
- [x] `elements.py` 简化 + `speech` 解耦
- [x] 634 全量单测通过，零 regression
