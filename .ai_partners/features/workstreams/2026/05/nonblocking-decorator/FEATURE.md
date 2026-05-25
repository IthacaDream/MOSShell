---
title: CTML 命令装饰器 — @nonblocking
status: in-progress
priority: P2
created: 2026-05-26
updated: 2026-05-26
depends: []
milestone:
description: >-
  CTML interface 中 # not blocking 注释改为 @nonblocking 装饰器, 提示词新增命令装饰器节, priority 暂不暴露给模型.
---

# CTML 命令装饰器 — @nonblocking

> Use `moss features set-status nonblocking-decorator <status> -m "note"` to update state.

## Motivation

当前 `# not blocking` 是 Python 注释式标记, 模型训练数据中注释权重天然低于代码. 改为 `@nonblocking` 装饰器: 肯定句式、Python 原生语义、信息权重更高. 同时 CTML 提示词中缺乏对非阻塞语义的说明, 需要插入一节但不重复已有 FIFO occupy 知识.

## Design Index

- CTML 提示词: `src/ghoshell_moss/core/ctml/prompts/v1_0_0.zh.md`
- 生成逻辑: `src/ghoshell_moss/core/ctml/v1_0/prompts.py`

## Key Decisions

### 为什么不用 async/non-async 区分阻塞语义

Python `async def` 语义 (可暂停等待) 与 `blocking` (占据通道) 语义错轴——前者是"让出", 后者是"占据". 且 `blocking=False` 仍阻塞 interpreter, async/non-async 无法表达这个二层. 改用装饰器, 不污染 Python 语法, 未来全双工到来时底层行为可独立演进.

### priority 暂不暴露

`priority` 只在同通道排队时生效, 场景窄. 当前保留在源码字段不动, 但不生成到 interface. 未来需要时加 `@priority(N)` 装饰器即可.

### 提示词密度

新节"命令装饰器"只有 4 行, 比"命令参数传递"短一半以上. 只描述存在的装饰器 (`@nonblocking`), 不展开不存在的情况. 为未来装饰器留白.

## Implementation Notes

- `prompts.py`: `# not blocking` → `@nonblocking`, 移除 priority 生成行
- `v1_0_0.zh.md`: 在"命令参数传递"和"开标记规则"之间插入"命令装饰器"节
- Sidecar: `features_cli.py` create 命令移除 `-d` 短选项 (与 `--dir` 冲突, `-d` 被误读为 description), hint 嵌入完整路径
