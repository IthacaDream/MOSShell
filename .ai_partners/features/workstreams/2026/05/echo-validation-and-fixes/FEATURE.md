---
title: Echo Ghost Validation & Fixes
status: in-progress
priority: P0
created: 2026-05-23
updated: 2026-05-23
depends: [first-ghost-prototype]
description: >-
  echo ghost human validation 中发现的 bug 修复。
---

# Echo Ghost Validation & Fixes

## PROMPT 面板空渲染（2026-05-23, fixed）

`ghost_runtime.py:267` — `moment.reaction_instruction is not None` → `moment.reaction_instruction`。

`Signal.prompt` 默认 `""`，空字符串 `is not None`，导致 PROMPT 面板始终渲染但无内容。
`conversation.py:191` 对同一字段已用 truthiness check，此处漏了。
