---
title: TUI Stream Rendering
status: completed
priority: P0
created: 2026-05-23
updated: 2026-05-25
depends: []
description: >-
  跨 asyncio/sync 边界的流式渲染基础设施 — LiveStreamSink + duck-type render(console) 模式。
---

# TUI Stream Rendering

为 TUI 提供跨 asyncio/sync 线程边界的流式渲染能力，解决 logos 逐 token 换行问题并建立通用的"asyncio 创建 → 卸载到渲染线程"模式。

## Motivation

当前 `rprint()` → render queue → `_direct_print()` → `Console.print()` 管线每个 delta 自动换行。中文模型 token 级 delta 导致输出在 token 粒度断裂。

此问题在 echo-validation-and-fixes 中记录为 "TUI logos 流式换行"。本 feature 提供基础设施，集成由 echo-validation-and-fixes 后续完成。

## Key Decisions

### 1. `render(console)` duck-type 协议

- `__rich_console__` 是无状态被动协议，Rich 决定调用时机，不适合有时序性的流式输出
- `render(console)` 是主动控制：对象自己决定何时、如何 print
- 不用 ABC，duck-type (`hasattr(obj, 'render')`) 足够——Rich 自身的 `RichCast` 等也是 duck-type
- 未来动画组件（Status、进度条等）也通过同一模式：asyncio loop 创建，卸载到渲染线程

### 2. `LiveStreamSink` — janus Queue + Segment buffer + render count

- janus.Queue 双面队列：async_q 给 asyncio 侧写入，sync_q 给渲染线程阻塞消费
- 首次 render: live 模式，实时消费同时攒 Segment buffer
- `_render_count > 0`: 直接 `Text.assemble(*buffer)` 回放
- 粘字符串聚合: 队列非空时攒 pending deltas，队列空时 flush 并阻塞等待下一个 delta，减少 `console.print()` 调用次数

### 3. 生命周期: send / commit / close

- `send(delta)` / `send_nowait(delta)`: committed 后静默丢弃；shutdown 后 catch 异常
- `commit()`: 标记结束 + 发送 None sentinel 通知渲染端
- `close()`: 先确保 committed + sentinel 入队，再 `shutdown(immediate=False)` 排空已入队数据
- 支持 async context manager（`__aenter__`/`__aexit__`），异常路径自动 close

### 4. 渲染线程退出改用 closing_event

- None sentinel 在 render queue 中只做 "clear" 语义（clear_console 发出，drain loop 跳过）
- 渲染线程退出仅靠 `while not closing_event` + timeout get
- 避免 clear 和 exit 两种 None sentinel 冲突

### 5. _direct_print 异常保护

- 渲染异常不应杀死渲染线程。try/except + `console.print_exception()` 兜底

## Design Discussion (2026-05-23, human + claude-opus-4.7)

- 初始方案: `ConsoleRenderer` ABC + `ConsoleStreamRenderer`。讨论后认定 ABC 画蛇添足，`render(console)` 保留为 duck-type。
- 曾考虑用 `__rich_console__` 魔术方法，但迭代器返回约定与流式消费的同步阻塞语义冲突，放弃。
- 统一模式: asyncio state 创建动画/流式对象 → janus queue → 渲染线程 `render(console)` 消费。未来 Status、进度条等复用同一基础设施。

## Implementation Notes

- `LiveStreamSink.render()` 阻塞消费 janus sync_q，期间渲染线程被独占（符合流式输出预期，此时不应有其他输出竞争）
- per-state `ConsoleOutput.clear_func` 暂时 no-op，后续接入 state buffer + re-render on switch
- `clear_console()` 的 None sentinel 在 drain loop 中被 continue 跳过

## Integration Guide

echo-validation-and-fixes 在 `ghost_ui.py:_consume_logos` 中集成:

```python
sink = LiveStreamSink()
async with sink:
    async for delta in articulate():
        await sink.send(delta)
    sink.commit()
self.console.rprint(sink)
```

不使用 context manager 时，异常路径需手动 `await sink.close()`。

## 集成完成（2026-05-25）

### 集成方式

`ghost_ui.py:_consume_logos` 按 utterance 使用 `LiveStreamSink`：
- `"\n\n"`（ghost_runtime._articulate_loop finally 块发布的 articulation 边界标记）触发 `sink.commit()`
- 每个 utterance 创建新 sink → `rprint(sink)` → 渲染线程进入 `render()` 独占模式
- asyncio 侧 `await sink.send(delta)` → janus 队列桥接到渲染线程

### render() 终版方案

经过多次迭代，最终使用 **ANSI 原地替换 + Rich Panel**：

1. `console.file.write("\033[{N}F")` + `"\033[J"` — 上移光标并清屏，擦除上一版 panel
2. `console.capture()` + `console.print(Panel(...))` — 用 Rich 渲染 panel 但不输出
3. `console.file.write(captured_output)` — 将 rendered panel 直写终端

迭代历史：
- 初版 `console.print(text, end='')` → Rich 与 prompt_toolkit 光标控制冲突，吞字符
- 二版 `console.file.write(text)` → 无 panel，与用户期望不符
- 终版 `ANSI + Panel` → 独立的 RESPONSE panel，原地流式更新

### 涉及文件

- `tui.py` — `LiveStreamSink.render()`：ANSI 原地替换 + Panel 渲染
- `ghost_ui.py` — `_consume_logos`：按 utterance 管理 sink 生命周期
