---
title: Storage & Scope Governance — Matrix 统一存储声明，GhostWorkspace 缩并，Scope 体系完善
status: completed
priority: P1
created: 2026-06-02
updated: 2026-06-02
depends:
  - ghost-playground
milestone:
description: >-
  Matrix 接管全部环境存储声明（ghost_home/mode_home/ghosts_storage/modes_storage），
  GhostPlayground 缩并为 GhostWorkspace dataclass，完善 scopes/get_scoped_storage 体系。
  人类实现后由 deepseek-v4-pro 整体 review，修复 3 个 bug，清理 SessionStorages 死代码。
---

# Storage & Scope Governance

> 人类工程师实现，deepseek-v4-pro 整体 review。

## Motivation

系统中存储声明散落在三处：`Workspace`（workspace 级别）、`Session`（会话级别）、`GhostPlayground`
（ghost 级别）。三者各自拼路径，没有统一的存储地图。Matrix 作为通讯拓扑和资源管理的单一事实源，
应该接管所有存储声明。同时 GhostPlayground 的多级 scope 抽象过于厚重 — ghost 只需要知道自己
的 home 目录和源码位置，不需要在 ghost 层重复 Matrix 已经提供的存储树。

## 治理目标

1. **Matrix 成为存储声明单一事实源** — 所有持久化路径约定通过 Matrix 的属性和 `storages()` 字典暴露
2. **GhostPlayground 缩并为 GhostWorkspace** — 从三层 scope 抽象（home/session/workspace）缩为
   两个字段（home + source），ghost 只需 Path 不需 Storage 抽象
3. **Ghost/Mode 发现容错** — scan 非严格模式，错误收集后通过 CLI 暴露，不阻断发现结果
4. **Workspace stubs 对齐** — `stubs/workspace/` 目录结构与存储树约定一致

## Key Decisions

### 1. Matrix 统一存储声明

Matrix ABC 新增完整的存储属性树：

```
matrix.workspace.root()     → workspace/
matrix.ghosts_storage       → workspace/ghosts/
matrix.get_ghost_storage(n) → workspace/ghosts/{name}/
matrix.ghost_home           → workspace/ghosts/{ghost_name}/
matrix.modes_storage        → workspace/modes/
matrix.get_modes_storage(n) → workspace/modes/{name}/
matrix.mode_home            → workspace/modes/{mode_name}/
matrix.storages()           → 上述所有 + session/cell/tmp 的平铺字典
```

`storages()` 是自解释入口 — AI 遍历一次即可理解全局存储地图。命名属性（`ghost_home` 等）是便捷访问器。

### 2. GhostPlayground → GhostWorkspace

原 `GhostPlayground` ABC 提供 `home()` / `session()` / `workspace()` 三层 scope 访问器，
定义在 `blueprint/host.py`，实现在 `host/ghost_playground.py`。由于 Matrix 已经提供
`ghost_home`，再加一个 GhostPlayground 中间层只是无意义的转发。

缩并为 `GhostWorkspace` dataclass（在 `blueprint/ghost.py`）：

```python
@dataclass(frozen=True)
class GhostWorkspace:
    home: pathlib.Path              # ghost 持久化存储目录
    source: Optional[pathlib.Path]  # ghost 源码所在目录
```

关键设计选择：`home` 类型为 `Path` 而非 `Storage`。GhostWorkspace 是值对象（"在哪里"），
不是服务（"怎么读写"）。ghost 需要读写文件时直接从 IoC 拿 `Matrix.ghost_home`（`Storage`），
不需要通过 GhostWorkspace 绕一道。

`GhostWorkspaceProvider` 在边界上做转换：从 `Matrix.ghost_home.abspath()` 取 Path，注入 IoC。

### 3. Session storage 位置不变

Session storage 树保持独立，由 `HostSessionProvider` 注入到 `MossSessionWithZenoh`：

```
runtime/sessions/                                          ← sessions_root_storage
  scope-{scope}/                                           ← scope_storage
    session-{id}/                                          ← storage
runtime/sessions-tmp/                                      ← sessions_tmp_root_storage
  {scope}-{id}/                                            ← tmp_storage
```

Matrix 通过 `storages()` 代理 session storage（`'session'`、`'tmp'` key），保持单入口。

### 4. Scope 体系

`ScopesKey = Literal['mode', 'session_scope', 'session_id', 'ghost', 'cell']`

- `scopes()` — 返回五个维度的当前值，用于构建隔离路径
- `get_scoped_url(*scopes, **kwargs)` — 基于 scope 生成唯一资源 ID（如记忆 key）
- `get_scoped_storage(scope, *scopes)` — 基于 scope 层级获取子存储路径

### 5. Ghost/Mode 发现容错

`host/impl.py` 移除 `strict_scan` 参数。`all_modes()` 和 `all_ghosts()` 始终返回已发现的结果，
扫描错误收集到 `_scan_ghost_errors` 和 `_scan_manifest_errors`。CLI（`ghosts_cli`、`modes_cli`）
在结果后展示 scan errors，不阻断输出。

## 最终的存储树全貌

```
workspace/
  configs/                    ← matrix.workspace.configs()
  assets/                     ← matrix.workspace.assets()
  runtime/
    sessions/                 ← session.sessions_root_storage
      scope-{scope}/
        session-{id}/         ← session.storage
    sessions-tmp/             ← session.sessions_tmp_root_storage
      {scope}-{id}/           ← session.tmp_storage
    locks/                    ← workspace.lock()
    logs/                     ← workspace.logs()
    cells/{address}/          ← matrix.cell_workspace (非 host cell)
  ghosts/                     ← matrix.ghosts_storage
    {name}/                   ← matrix.ghost_home
    None/                     ← 无 ghost 运行时
  modes/                      ← matrix.modes_storage
    {name}/                   ← matrix.mode_home
```

## AI Review 记录

deepseek-v4-pro 对人类实现做了整体 review，发现并修复：

| # | 问题 | 修复 |
|---|------|------|
| 1 | `scopes()` 中 session_id / session_scope 的值互换 | 互换两行 |
| 2 | `get_scoped_storage('mode')` 调了 `get_ghost_storage` 而非 `get_modes_storage` | 改为正确方法 |
| 3 | `GhostWorkspaceProvider` 将 `Storage` 直接传给 `Path` 类型字段 | Provider 边界调用 `.abspath()` 转换 |
| 4 | `SessionStorages` 类无人使用，且目录结构与实际不一致 | 从三处删除 |

同时验证了 `stubs/workspace/` 目录与存储树约定一致。
