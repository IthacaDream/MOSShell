---
title: Ghost Playground — 多级隔离文件空间
status: completed
priority: P0
created: 2026-05-22
updated: 2026-05-22
step: implemented
depends:
  - first-ghost-prototype
milestone:
description: >-
  为 Ghost 提供统一的多级隔离 Storage 入口 (home/session/workspace)，解决子功能散落拼路径的防污染问题。
---

# Ghost Playground

> Ghost 的文件空间集合。类比 MossSystemPrompter 的 tree model —— 系统约定的 scope slots，
> 命名访问器是对 slot 的薄封装。无反向注册 API，常量和 scopes() 的 flat dict 结构自然提供扩展。

## Motivation

系统中 Workspace、Session.storage 等 Storage 已经存在，但它们是散落的 — IoC provider
要读写文件时，不知道应该往哪个 storage 写。各处随意 `container.fetch(Workspace).root()`
自己拼路径，session 数据可能写到 workspace，ghost 的记忆散落在 runtime 角落。

GhostPlayground 把已有的 storage 按 scope 组织成单一入口，让子功能 (memory, personality,
scratchpad, logs...) 明确选择 "我写到哪个 scope"。本质是**防污染**，不是权限控制。

## Design Index

- 讨论记录: `discuss/01-design-decisions.md`
- ABC 定义位置: `ghoshell_moss/core/blueprint/host.py` (与 MossSystemPrompter, GhostRuntime 同簇)
- 默认实现位置: `ghoshell_moss/host/ghost_playground.py`

## Key Decisions

### 1. 三个系统约定 scope，无 cwd

| scope | 来源 | 生命周期 |
|-------|------|----------|
| `home` | workspace 下按 ghost name 约定子目录 | 跨 session 持久 |
| `session` | Session.storage | session 结束即清理 |
| `workspace` | Workspace.root() | 持久，最大权限 |

cwd 被拒绝 — workspace 创建本身就是授权动作，moss 在 workspace 内读写合理。
cwd 是用户当前目录，moss 无理由静默获得这个权限。真需要时通过 provider 自己挂。

### 2. 对位 MossSystemPrompter 模式

| MossSystemPrompter | GhostPlayground |
|---|---|
| CTML_SLOT / PROJECT_SLOT ... | HOME_SCOPE / SESSION_SCOPE / WORKSPACE_SCOPE |
| ctml_instruction() 命名访问器 | home() / session() / workspace() |
| flatten() 自解释 | scopes() 自解释 |
| default_instruction() 推荐组装 | default_scope() → home |
| 无反向注册 API | 无反向注册 API |

### 3. 反向注册由 flat dict 结构自然提供

不提供 with_scope()。子类 override scopes() 即可追加:
```python
def scopes(self) -> dict[str, Storage]:
    return super().scopes() | {"remote": self._remote_storage}
```
常量 (HOME_SCOPE 等) 标记系统保证存在的 scope，consumer 可通过常量获取稳定引用，
AI 做 flatten 式自解释时也能区分 "系统约定" 和 "运行时扩展"。

### 4. 无 allow flag — ghost meta 已有 provider 体系

GhostMeta 本身提供 provider，具备几乎无限的能力。权限控制不在 playground 层重复。
workspace scope 直接可用 — 创建 workspace 本身就是授权。

### 5. 注入时机: Matrix 启动前

GhostRuntimeImpl.__aenter__ step 1 预注入阶段，通过 `container.bound(GhostPlayground)`
检查 manifests 是否已声明。未声明则补充默认实现。必须在 MossRuntime.__aenter__
(step 2, 启动 Matrix) 之前完成，否则依赖方可能拿到 None 或找不到而崩溃。

### 6. ABC 在 blueprint/host.py，实现在 host/

遵循 MossSystemPrompter 的放置模式: 通用 contracts (Storage/Workspace) 已在
contracts/workspace.py，MOSS 特定约定放 blueprint/host.py。

## Implementation Notes

- GhostRuntimeImpl 构造 playground 时: home 路径 = `workspace.root().sub_storage(f"ghosts/{ghost.name()}")`
- session scope = `session.storage` (Session 上已有)
- workspace scope = `workspace.root()`
- 注入后 Ghost 及其子功能通过 `container.fetch(GhostPlayground)` 获取
