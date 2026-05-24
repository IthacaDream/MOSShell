---
title: Mode Init Logic
status: draft
priority: P1
created: 2026-05-25
updated: 2026-05-25
depends: []
milestone:
description: >-
  完善 mode 初始化模板、CLI hints、CLAUDE.md 上下文和 how-tos，让 mode 创建和发现路径更自解释。
---

# Mode Init Logic

> Use `moss features set-status mode-init-logic <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

Mode 是 MOSS 最核心的隔离和复用机制——它是叠加在全局 manifests 之上的能力视图。但当前 mode 的初始化路径有几个缺口：

1. **Stub 是空壳**：`ghoshell_moss.host.stubs.mode` 只有一个 `__init__.py`，`new_mode()` 复制空目录后再手动写 MODE.md。开发者创建新 mode 后不知道可以添加哪些 manifest 文件。
2. **default mode 太简略**：workspace stub 的 `default` mode 只有一个空 `MODE.md` + `__init__.py`，没有展示 mode 的完整能力结构。而 `system_test` mode 却有 `primitives.py` 和 `contracts.py`。两个 mode 不一致，`system_test` 反而是更好的参考。
3. **没有 mode 级别的开发上下文**：AI 进入 mode 目录时没有 CLAUDE.md 引导。
4. **CLI hints 贫乏**：`modes create` 和 `modes show` 的输出缺少下一步引导。

## Key Decisions

### 1. Package stub 作为唯一模板源，default mode 镜像它

`ghoshell_moss.host.stubs.mode`（`MODE_STUB_PACKAGE`）是 `new_mode()` 的复制源。它被充实为完整的 mode 模板，包含所有可能的 manifest 文件，每个文件带极简 "code as prompt" 注释。

Workspace stub 的 `default` mode 与此模板保持一致。两者同源，但 package stub 是不可变的干净模板（随包发布），workspace 内的 `default` 是运行时实例。

选择 package stub 而非 workspace default 作为 `new_mode()` 复制源的理由：用户可能修改 workspace default，不应让修改污染后续新建的 mode。

### 2. CLAUDE.md 放在 modes 目录级别（被动上下文）

`CLAUDE.md` 放在 `src/MOSS/modes/` 下，介绍 mode 开发思路、manifest 文件约定、如何配置 apps/bringup。

它不会被 Claude Code 自动加载（仅在 cwd 向上查找），但作为"被动提示"存在：当 AI 或人类因任何原因进入该目录时可以看到，同时被根 CLAUDE.md 或未来的 `moss dev-startups` 引用。

### 3. CLI hints 增强

- `modes create` 成功后列出创建的文件清单 + 下一步操作提示（编辑 MODE.md、添加 primitives 等）
- `modes show <name>` 增加当前 mode 的 manifest 文件清单（哪些文件存在），并提示 `moss --mode <name> manifests explain` 查看完整能力视图
- 提示保持简洁，不重复 `manifests explain` 的详细内容

### 4. How-tos 做 step-by-step，不重复 docs

`workspace-and-mode.md` 已经有完整的架构论述。How-to 只做操作指南：如何创建一个 mode、如何配置它的 manifests、如何验证。定位是 skills/step-by-step，面向零背景知识的模型。

### 5. 实施前重新探索 manifests

manifests 机制在近期有改动，动手前需要先 `moss manifests explain` + 阅读源码确认当前 manifest 类型和发现约定，确保模板文件与当前实现一致。

## Design Index

- 架构论述：`moss docs read workspace-and-mode.md`
- Mode 核心模型：`ghoshell_moss.core.blueprint.matrix:Mode`
- Mode 发现逻辑：`src/ghoshell_moss/host/modes.py`
- Manifests 自解释：`moss manifests explain`
- 关键源码：
  - `ghoshell_moss.core.blueprint.environment` — MODE_STUB_PACKAGE 定义
  - `ghoshell_moss.host.modes` — `new_mode()`, `list_modes_from_root_package()`, `find_mode_from_package()`
  - `ghoshell_moss.cli.modes_cli` — CLI 命令
  - `ghoshell_moss.host.stubs.mode` — 当前空 stub（改造目标）
  - `ghoshell_moss.host.stubs.workspace/src/MOSS/modes/` — workspace stub 中的 mode 实例

## Implementation Notes

### 模板文件结构（package stub 改造后）

```
ghoshell_moss/host/stubs/mode/
  CLAUDE.md           # mode 开发上下文（被动提示）
  __init__.py
  MODE.md             # 模板 MODE.md，带 apps/bringup 语法说明
  primitives.py       # Shell 原语声明
  contracts.py        # mode 专属契约
```

每个 `.py` 文件带一行注释说明用途（"code as prompt"），MODE.md 的 frontmatter 和正文都写清楚各项含义。

### MODE.md 模板要点

- frontmatter 中 `apps` 默认 `['*/*']`，`bringup_apps` 默认 `[]`
- 正文（instruction）写清楚：apps 白名单语法（`group/*`、`*/*`、`_` 前缀语义）、bringup 含义、mode 叠加语义（覆盖全局 manifests 的同类别项）
- 引导开发者先编辑 MODE.md 配置权限边界，再加 manifest 文件

### modes show 增强

在现有输出基础上追加一段 manifest 文件清单：

```
Manifest files for this mode:
  primitives.py  [present]
  contracts.py   [not found]
Tip: moss --mode <name> manifests explain
```

### 与 system_test mode 的关系

`system_test` 已接近可以删除或用正式测试用例替代的阶段。执行时根据当前代码状态判断——如果尚需保留，模板改造后其内容可以精简；如果已无用，直接清理。
