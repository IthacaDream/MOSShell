---
title: Reachy Mini Integration — 完整 workspace 集成与测试体系
status: in-progress
priority: P1
created: 2026-05-29
updated: 2026-05-29
depends: []
milestone:
description: >-
  将 moss_in_reachy_mini 作为 app 集成到 .moss_ws，建立完整的端到端测试体系。
---

# Reachy Mini Integration

## Motivation

`reachy-mini-contrib` 完成了基础运控/视觉的 contrib 包封装。本 workstream 承接它，将 Reachy Mini 正式集成到 `.moss_ws` 作为可被 Host 自动发现和启动的 app，并建立完整的测试体系，让 MOSS 具备第一个真实的机器人控制能力。

## 第一步：Workspace 入库

`.moss_ws/` 正式纳入 git 管理。这是一个长期手动维护的 workspace，不通过 stub 模板生成。

### Key Decisions

- **手动维护而非模板生成**：`.moss_ws/` 是项目开发的主 workspace，内容由人类工程师手动管理。stubs 目录下的模板服务于 `moss workspace create` 命令，面向外部用户。
- **根 `.gitignore` 移除 `.moss_ws/`**：此 workspace 将成为仓库的一部分。
