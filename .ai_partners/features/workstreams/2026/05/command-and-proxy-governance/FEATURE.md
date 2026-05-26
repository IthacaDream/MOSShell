---
created: 2026-05-26
depends: []
description: CommandMeta 去 chan 歧义 + ChannelMeta.proxy 从 list[str] 改为 per-node str|None
  标记，降低 proxy 心智成本。
milestone: null
priority: P0
status: in-progress
status_note: 'Phase 3 done: timeout 协议 + bare/magic task pipeline。proxy bool 落地。下一波：远程 scope 魔法函数实验。'
title: Command & Proxy Governance
updated: '2026-05-26'
---

# Command & Proxy Governance

## Motivation

1. `CommandMeta.chan` 是 vestigial organ — 历史上 command 自己标记归属 channel，但现在 command 总是通过 `ChannelMeta.commands` 持有，channel path 由外层 dict key 确定。`chan` 字段只剩歧义。
2. `ChannelMeta.proxy: list[str]` 是集中式声明（root meta 列出所有 proxy 子路径），child meta 不自知。flatten 成 `dict[str, ChannelMeta]` 后无法从自身字段判断 proxy 归属。
3. 规则收窄：**proxy 子树不再允许混合 local channel**。proxy 之下全部是远程镜像。心智模型干净，proxy 标记可以安全继承。

## Key Decisions

### 1. 删除 `CommandMeta.chan`

全链路通过 channel path 判断路径。Command 不需要知道自己属于哪个 channel。

- 唯一生产代码写入: `PyCommand._generate_meta()` → `meta.chan = self._chan`
- 唯一生产代码读取: 无（`commands_to_dict` 是测试辅助）
- `BaseCommandTask.chan` 不受影响 — 那是 task 的路由信息，不同概念

### 2. `ChannelMeta.proxy: bool`

最终落地为 `bool` 而非 `str | None`。原因是 proxy 的子孙不需要标记父节点 — proxy-ness 是继承属性，路由靠 tree path prefix 匹配决定了。`str` 存储 root id 在链路中暂无用武之地。

- `False` → 本地 channel
- `True` → 远程 proxy channel，所有子孙自动继承
- Proxy meta sync 时统一设 `virtual=True, proxy=True`
- 路由逻辑删除 `proxy: list[str]` 集中式检查，tree dispatch 自然处理

### 3. Proxy 子树纯净化

Proxy channel 下不再允许挂载 local child。所有子 channel 均来自远程 provider 的 meta sync。

### 4. timeout 协议 — CommandTask 原生属性

`CommandMeta.timeout` → `PyCommand._timeout` → `CommandTask.timeout` 全链路透传。
`dry_run_with_timeout()` 封装 `asyncio.wait_for(dry_run(), timeout)`，runtime 层统一调用。

- `timeout=None` → 无限等待
- `timeout=0` → 收敛为 None（不合法输入在 PyCommand/CommandTask 层抛 ValueError）
- `fail()` 同时处理 `asyncio.TimeoutError` 和 `TimeoutError` → errcode TIMEOUT
- `execute_command(name, timeout=...)` 支持单次调用覆盖

### 5. bare/magic task pipeline

魔法函数（`__xxx__`）的 bare task（func=None）不再在解释器层硬编码处理，而是通过 channel runtime 的扩展点 resolve：

- `_unwrap_self_bare_task`: 检查自身是否有注册命令 → set_command，否则走 magic unwrap
- `_unwrap_self_magic_task`: 扩展点，默认 no-op。scope 魔法函数在此实现
- `_check_new_compiled_task_scope`: 扩展点，scope 管理在此介入
- 显式注册过的命令不触发魔法函数逻辑（`set_command` 后直接 return）
- Provider 端：魔法命令不要求 command 已存在，创建 bare task 走 runtime resolve

## Implementation Notes

- 分三阶段提交
- Phase 1: 删 `CommandMeta.chan` + Zenoh 关闭修复 + 取消传播测试
- Phase 2: `proxy: list[str]` → `bool` + bare/magic task pipeline 地基
- Phase 3: timeout 协议透传 + `__await__` 重写 + `task_context` 传递
- `__await__` 从 `ThreadSafeFuture` + callback 改为 `async def _wait_done()`，语义等价但更简洁
- `wait()` 移除 observe 中断 — observe 是上层关注点，不打断 wait 流
- `CommandUtil` 新增 `is_task_done()` / `get_task_context()` / `create_task()`
- 推荐永远用 async def 定义 command — 同步函数无法真正被 cancel

## Verification

- [x] proxy task 取消传播测试通过（thread + zenoh 双后端）
- [x] `CommandMeta.chan` 删除，全量测试通过
- [x] `ZenohTopicSubscriber` shutdown 修复，bridge suite 28 全通过
- [x] `ChannelMeta.proxy: bool` 全量测试通过
- [x] timeout 协议 + bare/magic task pipeline 全量测试通过
- [ ] 远程 scope 魔法函数实验