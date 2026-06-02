---
created: 2026-06-02
depends: []
description: 将核心知识体系随包分发：CLI 自导引命令（moss guide）+ Channels 模块正式化（docstring 约定、codex
  channeltypes、observe 治理）。
milestone: null
priority: P1
status: completed
title: Moss Guide — CLI 自导引与 Channels 正式化
updated: '2026-06-02'
---

# Moss Guide — CLI 自导引系统

## Motivation

当前 `moss` CLI 的介绍体系完全依赖源码仓库中的 `CLAUDE.md`。通过 PyPI 安装后，`moss --help` 只有 Click 自动生成的干瘪命令树，核心讯息（AI 使用指南、命令体系导览、核心概念速览、快速开始路径）对用户不可见。

CLI 工具作为产品，应该自包含导引能力。`moss guide` 就是所有命令使用的开端。

## Design Index

- 源码位置: `src/ghoshell_moss/cli/` (新增 `guide_cli.py`)
- 文档内容源: `src/ghoshell_moss/cli/docs/ai/` (随包分发的 AI 可读文档)
- CLAUDE.md 集成: 通过 `@` 引用语法指向 guide 文档，保持 CLAUDE.md 精简

## Key Decisions

### K1: 命令命名为 `moss guide`

候选: `intro`, `tour`, `start`, `about`。选择 `guide` 因为：
- 暗示"跟着走就能了解全局"，是行动导向而非一次性介绍
- 生态常见 (`npm guide` 等)，用户预期明确
- 可扩展子命令 (`moss guide commands`, `moss guide concepts`)

### K2: 文档随包分发，作为 package_data

guide 的 markdown 内容放在 `src/ghoshell_moss/cli/docs/ai/` 下，通过 `pyproject.toml` 的 `package-data` 随包安装。CLI 命令读取自身包内文档，不依赖外部文件。

### K3: 面向双读者：人类 + AI

输出默认带 rich 排版给人类读；支持 `--ai` flag 剥离排版给 AI 读。与现有 moss CLI 约定一致。

### K4: 内容从 CLAUDE.md 迁移，做结构化重写

不是简单搬运，而是按 guide 的定位重新组织：
- 命令体系导览（何时用什么，不只是列表）
- AI 使用指南（`--ai` flag、`codex get-interface` vs 读源码的决策逻辑）
- 核心概念速览（channel, matrix, ghost 是什么）
- 快速开始路径

不包括：Git 提交规范、角色定位、哲学讨论（属于项目协作层面，非 CLI 产品文档）。

### K5: CLAUDE.md 用 `@` 引用保持单一事实源

CLAUDE.md 中不再重复 guide 内容，用 `@src/ghoshell_moss/cli/docs/ai/guide.md` 引入。Claude Code 支持此语法，最多 4 层嵌套。

## Channels 正式化 (2026-06-02)

`ghoshell_moss.channels` 目录从原型阶段进入正式模块。四个平行任务：

### K6: 模块级 docstring 约定

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

- 实现参考现有的 `docs_cli.py` 或 `howto_cli.py` 的 read 模式
- guide 内容第一版由 AI 起草，人类二稿，AI review，两轮后交付
- `moss guide` 无参数时显示总览索引；后续可扩展 `moss guide <topic>` 子命令

## Delivery Flow

1. AI 实现 `guide_cli.py` 命令
2. AI 起草 guide 文档一稿
3. 人类二稿修改
4. AI review
5. 修改定稿后整体交付，status → completed