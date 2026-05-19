---
created: 2026-05-19
depends: []
description: moss docs 补齐 list/read 子命令，人类和 AI 都能直觉使用。
milestone: null
priority: P2
status: completed
status_note: list/read + three-set hint everywhere, how-tos recall removed
title: Docs CLI Polish
updated: '2026-05-19'
---

# Docs CLI Polish

> Use `moss features set-status docs-cli-polish <status> -m "note"` to update state.

## Motivation

`moss docs` 只有一个光杆命令打目录树——没有 `list`、没有 `read`。没有文档集（ai/en/zh）的切换提示。AI 和人类都不知道有哪些 doc set 可用、怎么进入。

## Key Decisions

1. **三层 doc set (ai/en/zh) 切换提示** — 所有命令（默认/list/read）都打印当前所在 set 及如何切换到其他 set。这是最重要的 hint。
2. **默认行为** — AI 模式显示树 + doc set 提示；人类模式显示全局介绍（README + doc set 列表 + 命令示例）。
3. **list 只列当前 set** — 不跨 set 合并。打印 set 级 README（如果存在）作为上下文。
4. **read 路径相对当前 set 根** — AI 模式直接在 ai/ 下找，人类模式在 en/ 下找。`--path` 可覆盖。
5. **照 how-tos 模式** — `list` + `read`，不发明新范式。

## Implementation Notes

- `docs_cli.py` 重写为 typer app with callback + list + read
- `_available_doc_sets()` 自动发现 DOCS_ROOT 下的子目录
- `_print_doc_sets_hint()` 统一打印 doc set 切换提示
- `main.py` 注册方式从 `app.command` 改为 `app.add_typer`
- `howto_cli.py` 同步删除 `recall` 子命令