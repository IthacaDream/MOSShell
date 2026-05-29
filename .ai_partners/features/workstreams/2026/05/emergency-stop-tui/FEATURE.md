---
title: Emergency Stop — TUI 快捷键全局急停
status: draft
priority: P0
created: 2026-05-29
updated: 2026-05-29
depends:
  - first-ghost-prototype
description: >-
  TUI 快捷键触发全局急停，级联中断所有有状态双工节点（mindflow → shell → interpreter → speech），
  暂停 ghost 三循环但保持 TUI 存活可恢复。
---

# Emergency Stop — TUI 快捷键全局急停

## Motivation

当前 TUI 的 Escape 只做 REPL 级 interrupt（取消 `_operation_task`），ghost 三循环继续运行。
需要一个真正的急停——一键暂停所有双工节点，模型停止思考、命令停止执行、音频静音，
但 TUI 保持存活，可以恢复。

## 双工节点急停现状分析（2026-05-29 讨论）

链路拓扑：

```
input signal → mindflow → articulate_loop → action_loop
                  │            │                │
             attention   ghost.articulate   interpreter → command tasks
             impulse     (model API)         shell
             nuclei                          speech (TTS/audio)
```

逐节点 pause 覆盖情况：

| 节点 | pause 方法 | 做了什么 | 缺口 |
|------|-----------|---------|------|
| **Mindflow** | `BaseMindflow.pause()` | abort attention, 清空 signal 队列, nuclei.clear() | ✅ 已完备 |
| **Shell** | `CTMLShell.pause()` → `clear()` | `speech.clear()` + `main_runtime.tree.clear()` | ❌ 不调 `stop_interpretation()`，in-flight interpreter 继续跑 |
| **Interpreter** | 无独立 pause | 只能通过 `Shell.stop_interpretation(cancel_executing=True)` 中断 | Shell.clear() 漏调 |
| **Speech** | `Speech.clear()` | 清空活跃 TTS/音频输出流 | ✅ 被 Shell.clear() 调到 |
| **articulate_loop** | 无 pause | GhostRuntimeImpl 的 janus queue consumer | ❌ 模型 API 调用无法中断 |
| **action_loop** | 无 pause | GhostRuntimeImpl 的 janus queue consumer | ❌ 已在跑的 action 无法中断 |

## Key Decisions

### 1. 补齐 Shell.clear() 的 interpreter 中断

在 `CTMLShell._clear()` 中补 `stop_interpretation()`，使 `shell.pause()` 能中断 in-flight interpreter。

**改动点**: `ctml_shell.py:_clear()` — 在 `asyncio.gather` 中加 `self.stop_interpretation()`

### 2. GhostRuntime.pause() 统一急停入口

新增 `GhostRuntime.pause(toggle)` 方法，级联调用：
1. `mindflow.pause(toggle)` — 停分发、清信号
2. `shell.pause(toggle)` — 清 speech + channel tasks + interpreter（补齐后）
3. articulate_loop 的 in-flight model call 如何中断（待定）

pause(False) 逆序恢复。

### 3. TUI 快捷键绑定

在 `MossHostTUI.default_key_bindings()` 中新增全局急停快捷键（如 `Ctrl+\`），
直接调用 `GhostRuntime.pause(True/False)` 切换。

因为是全局快捷键，无论当前在哪个 state 都能触发。
需要在 TUI 状态栏显示是否处于急停状态。

## Open Questions

1. articulate_loop 中 `ghost.articulate(articulator)` 正在跑时的中断方案：
   - 方案 A：cancel 托管 articulate_loop 的 asyncio task（但会杀整个 loop，恢复时要重建）
   - 方案 B：在 Ghost/Articulator 上设 cancel flag，`ghost.articulate()` 内部检查
   - 方案 C：accept 当前 API 调用继续到结束，mindflow.pause 已经阻止了新 attention 入队

2. 快捷键选 `Ctrl+\` 还是其他？需避免与终端 SIGQUIT 冲突。

3. pause/resume 的 TUI 状态指示——是否需要改 prompt 样式（如变红）来提示用户当前急停中？

## Implementation Notes

- mindflow 的 pause 已完备，本次主要补 shell 侧缺口
- `Shell.stop_interpretation()` 已存在且调 `old.close(cancel_executing=True)`，直接复用
- REPL 已有 `/ghost.pause()` 和 `/ghost.resume()`（GhostInspector），快捷键是对它的 TUI 级补充
