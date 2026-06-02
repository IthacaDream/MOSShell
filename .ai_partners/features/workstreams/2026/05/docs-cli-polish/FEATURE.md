---
created: 2026-05-19
depends: []
description: moss docs 补齐 list/read 子命令，人类和 AI 都能直觉使用。
milestone: null
priority: P2
status: completed
status_note: list/read with MarkdownKnowledgeBase, three-set removed, how-tos hints everywhere
title: Docs CLI Polish
updated: '2026-05-23'
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

## 2026-05-23 Refactor Round

### Key Decisions

1. **人类/AI 文档分离** — 删除 en/zh 空目录。人类文档放仓库根 `docs/`，不归 CLI 管。`moss docs` 只服务 AI 参考文档。
2. **展平 ai/ 目录** — 只剩 AI 文档后，`ai/` 子目录冗余，四篇文档直接放在 `docs/` 下。
3. **复用 MarkdownKnowledgeBase** — docs_cli.py 从 419 行手写扫描器重写为 136 行，对标 howto_cli.py 模式。删除所有手写函数、tree view、doc set 切换、`--lang`/`--path` 参数。
4. **加 frontmatter 元数据** — 四篇文档各加 `title` + `description`（触发条件写入 description）。克制元数据字段，仅两个。
5. **docs vs how-tos 边界** — docs = 低频系统化知识（"为什么这么设计"），how-tos = 日常任务导向（"怎么做 X"）。所有 CLI 输出末尾有 how-tos hint。
6. **CLI 界面全英文** — help、hint、错误提示均为英文。文档正文语言不动。

### 遗留

- `docs/README.md` 正文是中文，`moss docs` 默认输出读的就是它。未来考虑英文化或中英双语，但非紧急。