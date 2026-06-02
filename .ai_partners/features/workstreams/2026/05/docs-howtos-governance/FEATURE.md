---
title: Docs & How-Tos Governance — 知识库清淤与文档路线图
status: completed
priority: P0
created: 2026-05-25
updated: 2026-06-02
depends: []
milestone:
description: >-
  清淤 how-tos 遗留综述文档，建立 docs 体系完整路线图。docs 和 how-tos 边界已明确，但 how-tos 仍残留 docs 体系建立前的 get-moss-design/ 目录。
---

# Docs & How-Tos Governance

> Use `moss features set-status docs-howtos-governance <status> -m "note"` to update state.

## Motivation

docs 体系已在 `ai-docs-topology` (P2) 和 `docs-cli-polish` (P2) 两个 feature 中落地——四篇 AI 参考文档 + `moss docs list/read` CLI。docs vs how-tos 的边界已明确：docs = 低频系统化知识（"为什么这么设计"），how-tos = 日常任务导向（"怎么做 X"）。

但 how-tos 里还残留 docs 体系建立前的 `get-moss-design/` 子目录（4 篇），其中 glossary 是参考文档，what-is-workspace 与 docs 已有文档重叠，IoC 和 matrix-discovers 是薄文可删或改造。同时 docs 缺少几篇关键综述文档，需要补全路线图。

**这次治理的目标**：清淤 how-tos，建立 docs 文档路线图，让下一个 AI 化身能直接进入实现。

## Key Decisions

### 1. how-tos 四个文档的命运

| 文档 | 判断 | 理由 |
|---|---|---|
| `get-moss-design/what-is-workspace.md` | **删除** | 内容已被 docs 的 `workspace-and-mode.md` 全覆盖，后者写得更好 |
| `get-moss-design/how-ioc-container-work-in-moss.md` | **删除** | 73 行薄文，核心价值（声明即注入、AI 无副作用注册）已在 workspace-and-mode 和 glossary 中体现。无独特内容需要保留 |
| `get-moss-design/how-matrix-discovers-capabilities.md` | **改造为 slim how-to** | 唯一独特价值是 "声明是声明，实现是实现" 那段洞察。收敛为 `for-moss-app-developer/` 下的操作指南 |
| `get-moss-design/glossary.md` | **迁入 docs** | 术语表是参考文档，天然属于 docs |
| `how-to-make-how-to.md` | 保留，展平 | meta 文档，留在 how-tos |
| `for-moss-app-developer/*` (2篇) | 保留，展平 | 操作步骤类，留在 how-tos |

### 2. how-tos 目录展平

去掉所有子目录。`for-moss-app-developer/` 和 `get-moss-design/` 都不保留。文件名自解释。未来再做目录治理。

### 3. docs 文档路线图（含编号方案）

编号反映推荐阅读顺序，也反映依赖关系。

| # | 文档 | 优先级 | 状态 | 说明 |
|---|---|---|---|---|
| 01 | MOSS 是什么 | P0 | **待写** | 一句话说清 MOSS 解决什么问题、能做什么、不能做什么。是未来 README 的材料，也是 AI 进入项目的第一篇 |
| 02 | 架构拓扑 | P0 | 已有 | `architecture-topology.md`，很好 |
| 03 | Channel 与 Command 体系 | P0 | **待写** | AI 写 MOSS 代码的核心 API 知识。需要人类详细过 |
| 04 | Matrix 通讯与集成体系 | P0 | **待修订** | 根据最新开发（zenoh fractal、cell discovery refactor）增补 |
| 05 | Ghost 开发与集成 | P0 | **待写** | ghost prototype 任务中完成 |
| 06 | App 系统 | P1 | 已有 | `model-oriented-application-system.md`，编号靠后 |
| 07 | MOSS 开发工作流 | P0 | **待写** | AI 进入项目后 "怎么干活" —— CLI 工具使用、环境探查、测试改动、dev loop。目前这些知识散落在 CLAUDE.md、features spec 中，新 AI 要拼凑很久 |
| 08 | workspace + mode | P1 | 已有 | `workspace-and-mode.md` |
| 09 | 迭代管理机制 | P1 | **待写** | features + how-tos + docs 自身如何协同演进 |
| 10 | CTML/Logos 语言参考 | P1 | **待写** | 命令语言的语义参考。目前 CTML 知识只有原始 prompt 文件，缺少正式的语言参考文档 |
| 11 | glossary | P1 | 从 how-tos 迁入 | 术语表 |

### 4. 为什么 "MOSS 开发工作流" 和 "Channel/Command" 是最有价值的投资

"MOSS 是什么" 决定 AI 能不能建立正确的心智模型。但 "开发工作流" 决定 AI 多快能上手干活，"Channel/Command" 决定能干得多好。这两篇是 AI 生产力的最大杠杆。

### 5. 文档编号不是文件名前缀

编号只存在于路线图（本文档和 docs README）。文件名保持语义化（如 `what-is-moss.md`），不写 `01-what-is-moss.md`。编号是阅读顺序建议，不是命名约束。这样增删文档不触发重命名。

## Implementation Notes

### 本次会话已完成

- 创建了本 feature workstream
- 与人类对齐了完整的文档路线图和 how-tos 清淤方案

### 2026-06-02: Howtos 清淤完成 (docs-quality-polish 执行)

执行由 docs-quality-polish feature (P1, 2026/06) 完成：

1. **how-tos 治理** ✅：
   - 删除 `get-moss-design/what-is-workspace.md` (ai-docs-workspace-mode 中完成)
   - 删除 `get-moss-design/how-ioc-container-work-in-moss.md`
   - 删除 `get-moss-design/how-matrix-discovers-capabilities.md` (原计划改造，审阅后判定过时无救)
   - `for-moss-app-developer/` → `host-dev/` (目录改名，非展平)
   - 6 篇文件名去 `how-to-` 前缀，标题去 "How to " 前缀
   - 删除空 `get-moss-design/` 目录

2. **glossary 迁移** ✅：`how_tos/get-moss-design/glossary.md` → `docs/glossary.md`

3. **docs 新文档撰写** → 移交 docs-quality-polish feature Phase 3-4

### 不做的事

- 不在此 feature 中写新文档正文（每个文档是独立工作，可能跨越多个 feature）
- 不在此 feature 中做 how-tos 的二级目录治理（等文档数量上去后再议）
- 不改变 docs/hows-tos 的 CLI 行为（只改内容，不改工具）
