---
created: 2026-06-02
depends: []
description: 将核心知识体系随包分发：CLI 自导引命令（moss start）+ Channels 模块正式化（docstring 约定、codex
  channeltypes、observe 治理）。
milestone: null
priority: P1
status: completed
title: Moss Start — CLI 认知入口与 Channels 正式化
updated: '2026-06-02'
---

# Moss Start — CLI 认知入口

## Motivation

当前 `moss` CLI 的介绍体系完全依赖源码仓库中的 `CLAUDE.md`。通过 PyPI 安装后，`moss --help` 只有 Click 自动生成的干瘪命令树，核心讯息（AI 使用指南、命令体系导览、核心概念速览、快速开始路径）对用户不可见。

CLI 工具作为产品，应该自包含导引能力。`moss start` 就是每个 MOSS 会话的认知入口——像 CLAUDE.md 为 AI 加载项目上下文一样，`moss start` 为使用者加载 MOSS 的认知地图。

## Design Index

- 源码位置: `src/ghoshell_moss/cli/` (新增 `start_cli.py`)
- 文档内容源: `src/ghoshell_moss/cli/start.md` (随包分发)
- CLAUDE.md 集成: 通过 `@` 引用语法指向 start 文档，保持 CLAUDE.md 精简

## Key Decisions

### K1: 命令命名为 `moss start`（原名 `moss guide`）

最初命名为 `moss guide`（2026-06-02），实施中发现 `guide` 在 AI 模型中被污染——每个实例都将其理解为"打开一份文档目录"，而非 CLI flow 的第一个 kickoff。

2026-06-02 讨论后改为 `moss start`：
- `start` 是 CLI 生态中最强的"入口"约定（`npm start`、`docker start`）
- 自带动作感——不是打开手册，是"从这里跑起来"
- 在 `moss --help` 中无需上下文即可理解其作为入口的角色
- 短，单一单词，行业约定清晰

### K2: 文档随包分发，作为 package_data

start 的 markdown 内容放在 `src/ghoshell_moss/cli/start.md`，通过 `pyproject.toml` 的 `package-data` 随包安装。CLI 命令读取自身包内文档，不依赖外部文件。

### K2.1: 单一文档，不走 MarkdownKnowledgeBase

start 是 CLI 的认知入口，一个文件承载完整的引导叙事流。与 docs/how-tos 不同：
- docs/how-tos 是多文档知识库，需要 list/read/recall，适合 mkb
- start 是单次阅读的行动流，直接读文件渲染，零依赖

`moss start` 无参数时显示文档全文。后续可扩展 `moss start <topic>`。

### K3: 面向双读者：人类 + AI

输出默认带 rich 排版给人类读；支持 `--ai` flag 剥离排版给 AI 读。与现有 moss CLI 约定一致。

### K4: 内容从 CLAUDE.md 迁移，做结构化重写

不是简单搬运，而是按 start 的定位重新组织：
- 命令体系导览（何时用什么，不只是列表）
- AI 使用指南（`--ai` flag、`codex get-interface` vs 读源码的决策逻辑）
- 核心概念速览（channel, matrix, ghost 是什么）
- 快速开始路径

不包括：Git 提交规范、角色定位、哲学讨论（属于项目协作层面，非 CLI 产品文档）。

### K5: CLAUDE.md 用 `@` 引用保持单一事实源

CLAUDE.md 中不再重复 start 内容，用 `@src/ghoshell_moss/cli/start.md` 引入。Claude Code 支持此语法，最多 4 层嵌套。

## Channels 正式化 (2026-06-02)

K6-K8 已在另一会话完成 (deepseek-v4-pro)，当前工作区有未提交 diff。本会话聚焦 moss start。
K9 待完成。

### K6: 模块级 docstring 约定 ✅ (另一会话)

```python
"""一句话功能描述 | 功能类型 | 状态
"""
```

- 第一行，一行。正则或 split 可解析。
- 由 `ast.get_docstring` 读取，直接对接 codex 的 `_get_item_description`。
- 需要时 docstring 后续段落可追加 Example 段——只给一种推荐集成方式，不执行。

### K7: `moss codex channeltypes` 命令

- 复用 `_show_package_module` 模式，包路径 `ghoshell_moss.channels`
- 无参数列出所有 channel（Module | Description 表格，shortdoc 含 type + status）
- 带 module_name 参数时反射该 channel 完整接口（复用现有 reflect 机制）
- 定位对标 `codex concepts` 而非 `manifests channels`——开发时能力目录，非运行时能力树

### K8: observe 显式标注

- 不改功能，只显式标记 `observe=True/False`
- 规则：结果是"信息"（模型需基于内容推理）→ True；结果是"确认"（只需知成败）→ False
- 当前全部依赖默认值 False，大量命令语义错误

### K9: Type 体系在 channels/CLAUDE.md 正式维护

六种类型（草拟，随 channel 增多演进）：

| Type | 含义 |
|------|------|
| `系统管理` | MOSS 架构级组件生命周期管理 |
| `通讯桥接` | 跨进程/跨运行时通讯连接与路由 |
| `交互能力` | 向外部世界的输出或感知 |
| `集成` | 将已有外部能力封装为 Channel |
| `系统控制` | 操作系统级别控制 |
| `认知模块` | 对文件系统等资源的动态结构化认知 |

Status 三态：`alpha`（原型/无测试/接口随意改）→ `beta`（可用但接口可能变动）→ `active`（正式维护/测试覆盖/跟随项目 semver）。

## Implementation Notes

- `start_cli.py` 已完成 (2026-06-02)，直接读 `cli/start.md` 渲染，不依赖 mkb
- start.md 由人类工程师写骨架，AI 用已有的 CLI 知识补完内容
- 后续可扩展 `moss start <topic>` 子命令

## Delivery Flow

1. ✅ AI 实现 `start_cli.py` 命令，注册到 `main.py`
2. 🔧 人类写 `start.md` 骨架 → AI 补完内容
3. AI review 最终内容
4. CLAUDE.md 用 `@` 引用指向 `start.md`
5. 修改定稿后整体交付，status → completed