---
created: 2026-05-25
depends: []
description: 最终对齐 CLI 命令树结构，为正式文档做准备。合并 concepts 到 codex，ws 改名为 workspace，apps init
  改名为 create。
milestone: null
priority: P1
status: completed
status_note: '2026-05-25: ws→workspace, concepts merged into codex (core→concepts),
  apps init→create'
title: CLI Command Tree Governance
updated: '2026-05-25'
---

# CLI Command Tree Governance

> 2026-05-25, 人类工程师 + Claude Code 讨论定稿。
> 旧设计 `src/ghoshell_moss/cli/.design/2026-05-17-cli_command_tree_redesign.md` 已过期，以此为准。

## Motivation

`moss` CLI 经过多轮迭代，积累了一些命名不一致和结构偏移。在准备正式文档之前，做最后一次命令树对齐。改动量小，只做必要调整。

## Key Decisions

1. **`ws` → `workspace`**。全拼更清晰，与 `manifests`/`modes`/`apps` 等全拼命名风格统一。

2. **`concepts` 组移除，并入 `codex`**。concepts 底层就是 codex 反射工具生成的，技术同源。`core` 改名为 `concepts`（`codex concepts` 语义 = "展示核心概念"），`blueprint` 和 `contracts` 平迁为 codex 子命令。

3. **`apps init` → `apps create`**。`create` 比 `init` 更准确表达 "创建新应用" 的语义。

4. **不将其他命令收敛到 workspace**。所有命令都是 MOSS devops 体系的一部分，按功能分组做顶级入口比强行按 "是否依赖 workspace" 分层更直观。

5. **how-tos 保持独立顶级组**，不并入 codex。用途不同：how-tos 是任务导向知识库，codex 是代码反射。

## Final Command Tree

```
workspace/              # 环境地基
  where, init, override, copy-env

codex/                  # 代码反射 + 全局知识索引
  get-interface, get-source, where, list, eval
  concepts, blueprint, contracts

manifests/              # 环境发现 (10 命令)
  providers, topics, configs, channels, primitives
  contracts, ctml-versions, resources, nuclei, explain

ctml/                   # CTML 协议
  list, read

modes/                  # 模式管理
  list, show, create

apps/                   # 应用管理
  list, show, create, test

how-tos/                # How-To 知识库
  list, read

features/               # 特性追踪
  specification, list, status, create, set-status, init

ghosts/                 # Ghost 发现
  list, show

script/                 # 开发期脚本
  list, run, init

docs/                   # 架构参考文档
  list, read

(root)
  help, all-commands
```

## Implementation Plan

1. `main.py`: `ws` → `workspace` 注册，移除 `concepts` 注册，将 concepts 三个命令注册到 codex
2. `concepts_cli.py`: 改 `core` → `concepts`（命令名），保持模块逻辑不变
3. `apps_cli.py`: `init` → `create`
4. 验证: `moss --ai all-commands` 确认树结构正确
5. 更新相关 help text 中的命令引用

## Post-Implementation Note

旧命令名 (`moss ws`, `moss concepts core/blueprint/contracts`, `moss apps init`) 可能在以下位置仍有残留引用，后续随看随修：
- how-tos 目录下的 markdown 文档
- docs 目录下的架构参考文档
- `.discuss/` 下的历史讨论记录
- workspace stub 中的模板文件