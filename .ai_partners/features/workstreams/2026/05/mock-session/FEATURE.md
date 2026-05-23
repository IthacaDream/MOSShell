---
created: 2026-05-23
depends: []
description: 纯内存无副作用 Mock Session，不依赖 zenoh，用于测试 Session 消费者（Ghost/Mindflow/Channel）。
milestone: null
priority: P1
status: completed
status_note: MockSession + channel_builder CommandUtil 单测 + Observe 逆向还原, 25 tests
  pass
title: Mock Session
updated: '2026-05-23'
---

# Mock Session

> `MockSession` 是 `Session` ABC 的纯内存实现。signal 路由、output 广播、stream pub/sub
> 全部在进程内同步完成，无外部依赖，无副作用。

## 动机

当前唯一实现 `MossSessionWithZenoh` 依赖 zenoh。所有消费 Session 的组件测试
（AppStoreChannel、Mindflow、GhostRuntime 等）都需要真实 zenoh 连接：
启动慢、资源可能跨测试泄漏、MagicMock 无法模拟有状态的内部逻辑（回调链、stream 队列）。

需要一个行为完整的 mock，跑通 signal → callback → output 的真实链路。

## 核心决策

### 1. 同步交付，不引入线程

`add_signal()` → 直接同步调用所有 `on_signal` 回调。
`output()` → 直接同步调用所有 `on_output` 回调。

**Why**: "发 signal → 验 output" 是最常见的测试模式，同步交付让断言直接。
`MossSessionWithZenoh` 在 zenoh 回调线程中也是同步执行的，行为等价。

### 2. stream 用 dict[str, list[bytes]] 存储

`pub_stream_delta(key, payload)` → `_streams[key].append(payload)` 并触发 sub 回调。
`get_stream(key)` → 返回 `MockStreamSubscriber`，从列表中迭代消费。

**Why**: 覆盖 pub/阻塞消费/回调消费三种模式的最简实现，无需 janus.Queue 或线程。

### 3. storage 直接用 LocalStorage + tmpdir

不接受复杂的 `session_root_storage` 构造。MockSession 构造时用 `tempfile.mkdtemp()` 创建
`LocalStorage`，实现 `make_session_level_storage` 的逻辑直接照搬 zenoh 版。

**Why**: `LocalStorage` 本身无副作用，不需要 mock。tmpdir 用完即弃，测试无需清理。

### 4. 无副作用的对象直接复用 zenoh 版

- `SimpleOutputBuffer` — 纯内存，直接复用
- `Session.add_input_signal()` / `Session.pub_logos()` / `Session.get_logos()` — ABC 中已实现，
  只需实现底层抽象方法即可自动继承
- `make_session_level_storage()` — 逻辑照搬 `MossSessionWithZenoh`

**Why**: 不要重新发明轮子。MockSession 只替换 zenoh 通讯层，其余逻辑与真实 Session 一致。

### 5. TopicService 用 QueueBasedTopicService 配对

构造时注入 `QueueBasedTopicService`（`ghoshell_moss.core.topic.queue_based`），
纯 Python + janus.Queue 实现，无 zenoh 依赖，与 MockSession 的无副作用理念一致。

使用前需要 `await topic_service.start()`，生命周期由调用方管理（与 zenoh 版一致，
Session 不管理 TopicService 的启停）。

**Why**: 已有现成的队列实现，不需要再造 mock。Session 只负责持有引用。

### 6. 放在 `src/ghoshell_moss/core/session/`，与 zenoh 版并列

对标 mock-ghost（`src/ghoshell_moss/ghosts/mock/`），作为 Session 接口的参考实现。
第三方开发可通过 IoC 注入 MockSession 测试自己的组件。

**Why**: MockSession 不仅是测试工具——它是 Session 接口的自解释参考实现。
与 `MossSessionWithZenoh` 并列，帮助理解接口契约和 IoC 替换模式。

### 7. 抓取模式：历史记录

额外暴露：
- `signals: list[Signal]` — 所有 `add_signal()` 的历史
- `outputs: list[OutputItem]` — 所有 `output()` 的历史
- `stream_pubs: dict[str, list[bytes]]` — 按 key 索引的 stream pub 记录

**Why**: "发 X → 验 Y" 是最高频断言模式，历史记录比注册回调更简洁。

## 实现清单

- [x] `MockSession` — 实现 `Session` 全部抽象方法，暴露历史记录属性
- [x] `_MockStreamSubscriber` — 实现 `StreamSubscriber` protocol，基于 `asyncio.Queue`
- [x] 冒烟验证：跑通 signal/output/stream 三条路径，15 tests pass
- [x] channel_builder CommandUtil 单测，10 tests pass，覆盖 new_command / send_signal / send_input_signal / observe / raise_observe / get_contract / logger
- [x] Observe 逆向还原：`CommandTaskResult.to_observe()` + `task.result()` 自动重建 Observe 对象

## 文件

- `src/ghoshell_moss/core/session/mock_session.py` — MockSession + _MockStreamSubscriber
- `tests/ghoshell_moss/core/blueprint/test_channel_builder_util.py` — CommandUtil 蓝图级单测

## 实现笔记

- `SimpleOutputBuffer` 不依赖 zenoh，`output_buffer()` 方法直接照搬 zenoh 版的实现。
- **线程/协程转换**: zenoh 版用 `janus.Queue` 桥接 zenoh 回调线程 → async consumer。
  MockSession 无跨线程场景，`_MockStreamSubscriber` 用 `asyncio.Queue` + `put_nowait` (sync 侧)
  → `await get` (async 侧)。`put_nowait` 只做 deque append，单线程下安全无竞态。
- `ABC` 中 `add_input_signal()` / `pub_logos()` / `get_logos()` 已实现，MockSession 只需实现
  底层 `add_signal` / `pub_stream_delta` / `get_stream` / `sub_stream`。
- `storage` 属性用 `make_session_level_storage(LocalStorage(tmpdir))` 懒初始化，与 zenoh 版逻辑一致。
- `QueueBasedTopicService` 是推荐配对: `topics = QueueBasedTopicService(); await topics.start()`。