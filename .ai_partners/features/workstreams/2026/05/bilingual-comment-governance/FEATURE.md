---
title: Bilingual Comment Governance — contracts/blueprint/concepts 中文注释双语化 + 过期内容修正
status: draft
priority: P1
created: 2026-05-29
updated: 2026-05-29
depends: []
milestone: beta-release
description: >-
  将 ghoshell_moss.contracts、ghoshell_moss.core.blueprint、ghoshell_moss.core.concepts
  三个模块中所有中文注释翻译为英文，保留双语，同时修正过期术语和描述。
---

# Bilingual Comment Governance

> 为 Beta 正式发布做准备：核心抽象层的中文注释全部双语化，过期内容清理。

## Motivation

这三个模块是 MOSS 架构的**抽象定义层**——contracts 是最小基础依赖，blueprint 是构建蓝图，concepts 是核心概念。它们共同构成 `moss codex` 对外暴露的 "自解释接口"。目前这些文件存在两个问题：

1. **大量中文注释未被翻译**。Code as prompt 的原则要求 AI 和人类都能无歧义地理解接口契约，而当前注释只在中文语境下可读。
2. **部分描述可能已过期**。项目经过多轮重构（Channel 发现机制、Cell 通讯模型、Session 体系等），某些注释中的术语或描述可能不再准确。

## Scope

三个 package，共 24 个 `.py` 文件：

| Package | Path | Files |
|---------|------|-------|
| contracts | `src/ghoshell_moss/contracts/` | 7 (`__init__`, configs, logger, resource, speech, system_prompter, workspace) |
| blueprint | `src/ghoshell_moss/core/blueprint/` | 11 (`__init__`, app, channel_builder, conversation, environment, ghost, host, manifests, matrix, mindflow, session, states_channel) |
| concepts | `src/ghoshell_moss/core/concepts/` | 7 (`__init__`, channel, command, errors, interpreter, shell, tools, topic) |

中文注释密度体感（基于抽样）：

- **重度** (几乎全中文): `contracts/resource.py`, `concepts/channel.py`, `concepts/shell.py`, 若干 `blueprint/*.py`
- **中度** (类和方法 docstring 有中文): `blueprint/environment.py`, `concepts/command.py`, `concepts/interpreter.py` 等
- **轻度** (少量中文行内注释): `contracts/configs.py`, `blueprint/__init__.py` 等

## Key Decisions

### 双语保留策略

每条中文注释翻译为英文后，**保留中文原文在上、英文翻译在下**。格式：

```python
"""
中文原文。
English translation.
"""
```

单行注释同理：

```python
# 中文注释
# English translation
```

**为什么保留双语而不是纯英文**：项目当前开发者和主要 AI 协作者都习惯中文语境。纯英文化会提高后续开发的理解成本。双语是过渡态，未来可以再讨论是否去掉中文。

### 过期内容判定标准

以下情况视为 "过期"，需要修正而非直译：

1. **术语已变更**: 例如早期用 "ZMQ" 现在用 "Zenoh"，早期用 "Client/Server" 现在用 "Provider/Proxy" 等。翻译时使用当前术语，同时检查描述是否仍然准确。
2. **已废弃的抽象**: 如果注释描述的功能已被移除或重构，标记并在翻译中反映当前实际状态。
3. **TODO/FIXME**: 检查是否仍然有效，无效的删除。
4. **与代码行为不一致**: 翻译过程中实际阅读代码逻辑，发现 docstring 与实现不符的，修正 docstring。

不确定是否过期的内容：保留原文，翻译中加 `[NOTE: verify if still accurate]` 标记。

### 优先级

contracts > concepts > blueprint。contracts 是基础依赖，被最多模块引用，应该最先完成。

## Implementation Notes

- 纯机械翻译为主，不要重构代码或修改接口签名。
- 发现明显的 bug 或设计问题可以顺手修，但不要在这个 workstream 里做架构改动。
- 英文翻译风格：简洁、技术准确，不追求文学性。用项目已有的英文注释（如 Channel 类中部分英文 docstring）作为风格参照。
- `__init__.py` 文件如果只有 re-export 没有实质注释，可以跳过。
- 翻译完一个文件后，跑一下 `moss codex get-interface <模块路径>` 确认自解释输出正常。
