---
title: Docs Quality Polish — 已有文档修正 + 缺失文档补齐，docs 体系一次性完结
status: in-progress
priority: P1
created: 2026-06-02
updated: 2026-06-02
depends: []
milestone:
description: >-
  逐篇审查已有 6 篇 docs，修正过时引用、统一标题风格；补齐 docs-howtos-governance 路线图中标记为 P0 待写的关键文档。目标是 docs 体系本次闭环。
---

# Docs Quality Polish

> Use `moss features set-status docs-quality-polish <status> -m "note"` to update state.

## Motivation

`moss docs list` 已有 6 篇 AI 参考文档，但 `docs-howtos-governance` (P0, in-progress) 路线图中标记了 5 篇 P0 待写/待修订。加上已有文档自身存在过时引用和风格不一致的问题，这两件事应该合并——一次把 docs 体系推到可用闭环。

## 现状全景

### 已有 docs (6 篇) — 需要修

| 文件 | 标题 | 问题 |
|------|------|------|
| `architecture-topology.md` | MOSS 架构拓扑 | 无明显问题 |
| `channel-system.md` | MOSS Channel 体系 | 无明显问题 |
| `ctml.md` | CTML — 流式控制语法 | 无明显问题 |
| `model-oriented-application-system.md` | Model-Oriented Application System | **标题英文，文件名过长** |
| `script.md` | MOSS Script | **定位模糊，描述需强化** |
| `workspace-and-mode.md` | MOSS Workspace 与 Mode | **4 处过时引用** |

### 缺失 docs (5 篇 P0) — 需要写

来自 `docs-howtos-governance` 路线图：

| # | 文档 | 优先级 | 说明 |
|---|---|---|---|
| 01 | MOSS 是什么 (`what-is-moss.md`) | P0 | AI 进入项目的第一篇。一句话说清解决什么问题、能做什么、不能做什么 |
| 03 | Channel 与 Command 体系 | P0 | `channel-system.md` 侧重构建/集成/发现全链路。这篇聚焦 AI 写代码时的核心 API 知识——Command 定义、Builder 模式、签名反射、occupy 语义 |
| 04 | Matrix 通讯与集成体系 | P0 | 待根据 zenoh fractal、cell discovery refactor 增补。当前架构拓扑 2.5 节只有一段概述 |
| 05 | Ghost 开发与集成 | P0 | ghost prototype 验证后写。描述 Ghost 适配层的最小协议——生命周期对接、感知接入、行动输出、反身性控制 |
| 07 | MOSS 开发工作流 | P0 | AI 进入项目后 "怎么干活"。目前散落在 CLAUDE.md、features spec 里，新 AI 拼凑成本高 |

### 缺失 docs (3 篇 P1) — 视情况

| # | 文档 | 说明 |
|---|---|---|
| 09 | 迭代管理机制 | features + how-tos + docs 自身如何协同演进 |
| 10 | CTML/Logos 语言参考 | `ctml.md` 已有设计哲学和测试地图，缺正式的语言语义参考 |
| 11 | glossary 迁移 | 从 `how_tos/get-moss-design/glossary.md` 迁入 docs |

## Key Decisions

### 1. 标题统一用中文

`model-oriented-application-system.md` 标题改为 "MOSS App 体系"，文件名改为 `app-system.md`。

### 2. 过时引用全量修正 (workspace-and-mode.md)

- 2.2 节用户故事表：`primitives` 从关键工具列删除
- 4.3 节 "深入路径" 表：`get-moss-design/` 旧路径替换为当前有效引用
- 8 节知识探索路径：删除 `moss manifests primitives`，补充 nuclei
- 9 节深入概念：`get-moss-design/` 旧路径替换

### 3. script.md 保留在 docs

体量小但概念独立性够——`Matrix.discover()` 入口是 App、GUI、Signal 生产者等多种模式的共享基础设施。在描述中强化 "轻量参考" 定位。

### 4. 新文档优先级

P0 全部写。P1 视时间和人类工程师意愿决定。但写不完的标记清楚状态，不假装完成了。

### 5. 与 docs-howtos-governance 的关系

本 feature 接手路线图中 "写新文档" 的部分。how-tos 清淤（删除 get-moss-design/、展平 for-moss-app-developer/、glossary 迁移）仍属于 docs-howtos-governance。

## Implementation Notes

### Phase 1: 已有文档修正 ✅ (2026-06-02)

- [x] `workspace-and-mode.md` — 修正 4 处过时引用 (2.2 primitives, 4.3 深入路径拆分 contracts+Provider, 8 nuclei, 9 get-moss-design 替换)
- [x] `model-oriented-application-system.md` → `app-system.md` — 标题中文化 ("MOSS App 体系")，更新 channel-system.md 和 apps CLAUDE.md 引用
- [x] `script.md` → `moss-script.md` — 重命名 + 标题/描述强化 "轻量参考" 定位
- [x] `README.md` — 概念化重写，不耦合具体文档名，末尾注明 "文档清单由 moss docs list 动态生成"

### Phase 2: Howtos 清淤 ✅ (2026-06-02)

实际从 docs-howtos-governance 手中接手了 howtos 清淤工作：

- [x] 删除 `get-moss-design/how-ioc-container-work-in-moss.md` — 薄文，docs 已覆盖
- [x] 删除 `get-moss-design/how-matrix-discovers-capabilities.md` — 过时，docs 已覆盖
- [x] `glossary.md` 迁入 docs/ (#11) — 质量高，更新自引用路径
- [x] `for-moss-app-developer/` → `host-dev/` — 目录改名，短而准确
- [x] 6 篇文件名去 `how-to-` 前缀，标题去 "How to " 前缀
- [x] 全量交叉引用更新 (manifests.py, stubs, workspace-and-mode.md, CLAUDE.md, howtos README)
- [x] 删除空 `get-moss-design/` 目录

### Phase 3: P0 新文档撰写

- [ ] `what-is-moss.md` (#01) — 最高优，AI 入口第一站
- [ ] `channel-and-command.md` (#03) — 聚焦 API 知识，与 channel-system.md 互补
- [ ] `matrix-system.md` (#04) — Matrix 通讯与集成
- [ ] `ghost-development.md` (#05) — Ghost 适配层最小协议
- [ ] `development-workflow.md` (#07) — AI 开发工作流

### Phase 4: P1 (视情况)

- [ ] CTML/Logos 语言参考 (#10)
- [ ] 迭代管理机制 (#09)

### CLI 侧适配

文件重命名后需确认 `moss docs list` 扫描逻辑——是否依赖文件名约定。

### 不做的事

- how-tos 清淤（属于 docs-howtos-governance，不在本 feature 范围内）
- docs README 路线图更新（同属于 docs-howtos-governance）
