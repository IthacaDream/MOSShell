---
created: 2026-05-26
depends: []
description: CommandMeta 去 chan 歧义 + ChannelMeta.proxy 从 list[str] 改为 per-node str|None
  标记，降低 proxy 心智成本。
milestone: null
priority: P0
status: in-progress
status_note: 'Phase 1 done: 删 CommandMeta.chan + Zenoh 订阅者 shutdown 修复。Phase 2: ChannelMeta.proxy 重构。'
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

### 2. `ChannelMeta.proxy: str | None`

| 值 | 语义 |
|----|------|
| `None` | 本地 channel，非 proxy |
| `"proxy_root_path"` | 是 proxy，入口在 `proxy_root_path` |

- Proxy-ness 继承：proxy 节点下的所有子节点自动是 proxy，`proxy` 字段值指向同一 root
- 嵌套 proxy（proxy 下再挂 proxy）各自标记自己的 root
- 路由逻辑从 `path in self_meta.proxy` 改为 `child_meta.proxy is not None`
- 每个 meta 自描述，flatten 后不丢失信息

### 3. Proxy 子树纯净化

Proxy channel 下不再允许挂载 local child。所有子 channel 均来自远程 provider 的 meta sync。

## Implementation Notes

- 分两阶段提交：先删 `CommandMeta.chan`（独立 PR，影响面极小），再改 `proxy` 字段
- `CommandMeta.chan` 删除后 `commands_to_dict` 需要改为从外层 dict key 获取 chan
- `proxy` 字段变更时 `_tree_channel_runtime.py:232` 的路由判断和 `proxy.py:538` 的赋值逻辑同步修改
- 取消传播已验证通过（`test_proxy_task_cancel_propagates_to_provider`），可以作为本次改动的回归保护
- `ZenohTopicSubscriber`: 新增 `_watch_stopped()` 后台 task，service 关闭时 `immediate=True` 强制 shutdown queue。解决 `poll()` 在 service 关闭后无限阻塞的问题。三种关闭路径（`__aexit__` / `_listening_loop` finally / `_watch_stopped`）完整覆盖。

## Verification

- [x] proxy task 取消传播测试通过（thread + zenoh 双后端）
- [x] `CommandMeta.chan` 删除，全量测试通过
- [x] `ZenohTopicSubscriber` shutdown 修复，bridge suite 28 全通过
- [ ] `ChannelMeta.proxy` 改为 `str | None`，全量测试通过
- [ ] proxy 子树 pure 约束验证