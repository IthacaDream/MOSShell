---
title: Matrix Channel Hub — Zenoh 原生 Channel 发现与管理
status: draft
priority: P0
created: 2026-05-24
updated: 2026-05-24
depends: []
milestone:
description: >-
  通过 ZenohChannelHub + Matrix Channel 统一 Channel 发现与管理，
  吸收 cell-discovery-refactor，最终替换 manifests 中的 AppStoreChannel 注册。
---

# Matrix Channel Hub

## Motivation

当前 Channel 发现链路存在两层断裂：

1. **Matrix.provide_channel() 是统一的提供入口**，但发现靠外部知识（AppStore 枚举 → 逐个 proxy），网络层没有自反射能力。
2. **AppStoreChannel 通过 manifests 注册**，但 Matrix 才是通讯拓扑的单一事实源。这导致两个"发现"概念（cell liveness + channel proxy）各自维护。

这次改造的核心判断：**Matrix 上既有 provide 接口，就不该是 apps 单独发现 channel。应该有一个 Matrix Channel，在 bootstrap 时获取 Matrix 实例，用 states_channel 的 virtual channel 模式自动反射网络中所有可用的 Channel。**

同时吸收 `cell-discovery-refactor`：用 zenoh queryable/wildcard 替代 N 个 liveness subscriber，把 cell 存活和 channel 发现统一到同一个 key 体系。

## Design Index

- `BridgeExpr` 重构：`src/ghoshell_moss/bridges/zenoh_bridge/_utils.py`
- `ZenohChannelHub`：新建，参考 `src/ghoshell_moss/bridges/zmq_channel/zmq_hub.py`
- `MatrixChannel` + `MatrixChannelState`：新建，基于 `states_channel.py` 模式
- `MatrixImpl.provide_channel()` / `channel_proxy()`：`src/ghoshell_moss/host/matrix.py`
- 最终移除 manifests 中的 AppStoreChannel 注册

## Key Decisions

### D1: BridgeExpr 可配置 prefix，NodeChannelBridgeExpr 作为默认值

```python
class BridgeExpr:
    def __init__(self, session_scope, *, prefix_parts=None):
        # prefix_parts=None → 默认 "MOSS/{scope}/node"（向后兼容）
        # prefix_parts=["MOSS", scope, "matrix"] → hub 模式
```

原 `NodeChannelBridgeExpr` 保持行为不变。ZenohChannelProvider / ZenohProxyChannel 接收可选 BridgeExpr，不传用默认。

### D2: ZenohChannelHub 用 zenoh liveliness 替代心跳超时

ZMQ Hub 需要自定义 heartbeat + cleanup_loop。Zenoh liveliness token 原生提供 PUT/DELETE 语义——provider 断开时 key 自动 DELETE，无需手动超时清理。

### D3: Matrix Channel 用 ChannelState.get_virtual_children() 做动态发现

bootstrap 时注入 Matrix 实例。`get_virtual_children()` 通过 hub 的 wildcard query 获取所有已注册 provider → 为每个动态生成 proxy 作为 virtual child。与 ZMQHubChannelState 同构。

### D4: cell-discovery-refactor 吸收合并

cell liveness key 和 channel discovery key 统一在 hub prefix 下：
- `MOSS/{scope}/matrix/cells/{address}` — cell 存活
- `MOSS/{scope}/matrix/channels/{address}` — channel 注册

一个 wildcard query 覆盖两者。不再需要 per-cell liveness subscriber。

### D5: AppStoreChannel 最终从 manifests 移除

Matrix Channel 管理 app 的 open/close，运行中的 app 自动出现在 virtual children 里。AppStoreChannel 的概念合并进 Matrix Channel。

## Implementation Plan (4 步迭代)

1. **ZenohChannelHub** — 参考 ZMQ Hub，zenoh liveliness 原生支持。无痛。
2. **BridgeExpr 重构** — 向前兼容，加 prefix 可配置。
3. **Matrix Channel** — 独立验证，新建 ChannelState，用 hub 做动态发现。
4. **替换 manifests 注册** — 移除 AppStoreChannel，简单链路回归。

## Blast Radius

| 改动 | 影响 |
|------|------|
| BridgeExpr 加 prefix_parts | NodeChannelBridgeExpr 不变，ZenohProviderConnection/ZenohProxyConnection 可选传入 |
| 新建 ZenohChannelHub | 零影响现有代码 |
| 新建 MatrixChannel + MatrixChannelState | 零影响现有代码 |
| MatrixImpl 改用 hub key | provide_channel/channel_proxy 内部调整，接口不变 |
| 移除 AppStoreChannel manifests 注册 | 需回归验证 Shell 树中 apps channel 仍可用 |

## Absorbed Features

- `cell-discovery-refactor` (completed 2026-05-24, absorbed here): liveness subscriber N→1, is_alive→reported_at, queryable 替代 pub/sub
