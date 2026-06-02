---
created: 2026-05-23
depends: []
description: Observe return 和 ObserveError raise 当前在 interpreter 层语义相同（均中断）。 需区分为：Observe
  = 非中断观察标记，ObserveError = 紧急中断。
milestone: null
priority: P0
status: completed
status_note: Observe 非中断 + ObserveError 中断语义区分已实现，含单测和 prompt 更新
title: Observe/ObserveError 语义区分 — 非中断标记 vs 紧急中断
updated: '2026-05-23'
---

# Observe/ObserveError 语义区分

> `moss features set-status observe-semantics <status> -m "note"`

## Motivation

CTML 的并行执行模型要求 command 能区分两种观察语义：

1. **非中断观察**（`return Observe`）：命令产生了值得关注的结果，但不应该 kill 并行任务。
   模型在下一关键帧查看，与其它并行任务的结果一起决策。
2. **紧急中断**（`raise ObserveError`）：出事了，立刻停掉一切，现在就重新思考。

当前实现将两者汇合为同一路径——`BaseCommandTask.fail()` 把 `ObserveError` 转为 `Observe` 再 resolve，
`_task_done_callback` 检测到任何 `observe=True` 就 `_stopped_event.set()`。`CommandUtil.observe()` 和
`CommandUtil.raise_observe()` 的命名差异构成误导。

### 更深层的必要性

CTML 作为独立交互范式，必须有自包含的 react 锚点。四种 react 动因的完整拓扑：

| 动因 | 机制 | 触发者 | 语义 |
|------|------|--------|------|
| 模型自决 | `Observe` return | Command 开发者 | "结果值得关注，但不中断并行任务" |
| 外部异常 | `ObserveError` raise | 现实世界反馈 | "出事了，立刻停掉一切" |
| 语法驱动 | Interpreter error | CTML 解析器 | "输出不合法，修一下" |
| 外部输入 | Signal → Impulse | 环境/感知模块 | "有新信息，挑战注意力" |

Scope 边界 (`<_>`) 是第五种 — 模型规划的"分号"，属于 CTML 语法自然属性，不需要额外机制。

## Design Index

- `src/ghoshell_moss/core/concepts/command.py` — `Observe`, `ObserveError`, `BaseCommandTask.fail()`, `BaseCommandTask.wait()`
- `src/ghoshell_moss/core/concepts/errors.py` — `CommandErrorCode.OBSERVE = 400`, `is_critical()`
- `src/ghoshell_moss/core/concepts/interpreter.py` — `Interpretation.on_done_task()`, `Interpretation.observe`
- `src/ghoshell_moss/core/ctml/interpreter.py` — `CTMLInterpreter._task_done_callback()`, `_set_interpreter_error()`
- `src/ghoshell_moss/core/blueprint/channel_builder.py` — `CommandUtil.observe()`, `CommandUtil.raise_observe()`
- `src/ghoshell_moss/core/ctml/prompts/v1_0_0.zh.md` — CTML 语法规范中的 observe 描述

## Key Decisions

### K1: Observe 非中断，ObserveError 中断

**决策**: `return Observe` → 累积到 `Interpretation.messages`，不设 `_stopped_event`。`raise ObserveError` → 设 `_stopped_event`，取消所有未完成任务。

**理由**: CTML 并行流中，一个感知命令返回了值得注意的东西不应 kill 旁边正在跑的 motor command。具身场景:
```ctml
<_>
  <robot:move_to target="kitchen"/>
  <vision:scan objects="true"/>     <!-- 返回 Observe -->
  <audio:listen duration="2.0"/>
</_>
```
`vision:scan` 发现异常物体标记了 `Observe`。但 `move_to` 还在走路。非中断 `Observe` 让模型在下个关键帧看到三者结果一起决策。

### K2: ObserveError 改用 errcode 400 而非转为 Observe

**决策**: `BaseCommandTask.fail()` 遇到 `ObserveError` 时，设 `errcode = CommandErrorCode.OBSERVE` (400)，不再调用 `self.resolve(error.as_observe())`。
`is_critical(400)` 自然命中，`on_done_task()` 中 `result.observe or is_critical(task.errcode)` 两条路径均可触发中断。

**理由**: 利用已有的 error code 分级体系（<400 不中断，>=400 中断），不引入新 flag。`ObserveError` 本身就是 400 语义。

### K3: Interpretation 增加区分字段

**决策**: 保持 `observe: bool` 为累积标记。新增或复用现有机制区分"仅累积"和"中断"。`_task_done_callback` 中通过 `is_critical(task.errcode)` 判断是否设 `_stopped_event`。

**被拒绝**: 在 `CommandTaskResult` 中新增 `interrupt: bool` 字段。多余——errcode 已经表达了中断语义。

## Implementation Notes

### 改动链路 (3 + 1 处)

**1. `BaseCommandTask.fail()` (command.py:1382)**
```python
# Before:
if isinstance(error, ObserveError):
    self.resolve(error.as_observe())
    return

# After:
if isinstance(error, ObserveError):
    self._set_result(None, "failed", CommandErrorCode.OBSERVE, error.message)
    return
```
不要再转成 Observe → resolve。让 errcode = 400 携带中断语义。

**2. `Interpretation.on_done_task()` (interpreter.py:219)**
当前已有 `if result.observe or CommandErrorCode.is_critical(task.errcode): self.observe = True`
— 这条逻辑不变。但需确认：`Observe` return 路径（errcode=0, observe=True）只设 `self.observe`，不中断。
`ObserveError` 路径（errcode=400, is_critical=True）既设 `self.observe` 又作为中断信号。

**3. `CTMLInterpreter._task_done_callback()` (interpreter.py:262)**
```python
# Before:
if self._interpretation.observe:
    # cancel tasks + stop

# After:
# 仅对 critical 错误中断。单纯的 observe 只累积消息。
if self._interpretation.observe and any_critical_task_failed():
    # cancel tasks + stop
```
具体做法：检查当前 task 的 errcode 是否 is_critical，而非检查 `self._interpretation.observe`。

**4. `BaseCommandTask.wait(throw=True)` (command.py:1477)**
当前 `observe=True` 时抛出 `CommandErrorCode.OBSERVE.error("need observe")`。确认此行为对非中断 Observe 是否合理 — 
如果 caller 用 `throw=True` wait 一个返回 Observe 的 task，抛异常是合理的（caller 需要知道这是个 observe result）。

### observe 仍是 interpretation 级别

`Interpretation.observe` 字段保持为 True 当任一 command 返回 Observe 或触发 ObserveError。
区别在于 `_stopped_event` 是否被设置。这对 `Interpretation.as_messages()` 的输出无影响 —
模型总能在下一轮看到所有 observe 消息。区别只是：并行任务是否被提前 cancel。

### 测试修复范围

预计受影响的测试：
- `tests/ghoshell_moss/core/ctml/v1_0/test_ctml_v1.py` — ObserveError 相关测试
- `tests/core/command/test_command_task.py` — task result observe 测试
- CTML prompt `v1_0_0.zh.md` — "运行中断机制" 章节需区分两种语义

### 考古记录

- `0ceeeda` (2026-03-01): `ObserveError` 引入，构造器接收 `Observe` 对象。`fail()` 第一天起就转成了 `Observe`
- `fe049f0` (2026-03-26): "remove ghoshell_moss.types.Observe, stupid sugar" — 合并到 command.py
- `fd76a26` (2026-05-05): "prepare ghost design" — ObserveError 改为接收 `message: str`

两个符号从诞生起就语义相同。区分意图写在了命名和 docstring 里，但从未在代码中实现。