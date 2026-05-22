---
title: Channel Module — 生命周期感知的模块化 Channel 集成机制
status: in-progress
priority: P1
created: 2026-05-22
updated: 2026-05-23
depends: []
milestone:
description: >-
  Channel 树体系缺失的能力层：通过 ChannelModule Protocol + BaseStateChannel.with_module()
  实现累积叠加，与 with_state（排他）共享 ChannelState 基础设施但语义正交。
---

# Channel Composer

> Exploring 阶段完成。本文档是 knowledge index，面向下一组 human + AI pair 的 implementation handoff。

## Motivation

### 直接触发

`contracts/speech.py` 与 `channels/speech_channel.py` 的边界越界：

- `contracts/speech.py` 是契约层，却包含 `make_content_command_from_speech()` 和 `TTSSpeech.commands()` — 在抽象接口里生成 `Command` 对象。契约层不应知道 Command。
- `ctml_shell.py:_speech_context_manager` 用 `isinstance(self._speech, TTSSpeech)` 分支注册 TTS 高阶命令到主 channel — 硬编码、不可扩展。
- `SpeechChannel` 已存在但被绕过 — shell 直接在主 channel 上注册 speech 命令。

深层原因：**main channel 集成 speech 是基础设计，拿掉不合理。但 speech 的高阶功能（voice 控制、tone 切换）需要模块化的集成方式，而不是 isinstance 分支。**

### 架构缺口

Channel 五层管理体系中**第 4 层（Composer）的缺失**：

| 层级 | 机制 | 语义 | 现状 |
|------|------|------|------|
| 1. Tree Channels | `import_channels` | 纵向层级（mount 树节点） | 已有 |
| 2. States | `ChannelState` + `switch_state` | 横向排他切换（生命周期级） | 已有，测试不足 |
| 3. Virtual Channels | `add_virtual_channel` | 动态纵向隔离（生命周期绑定） | 已有 |
| **4. Composer** | **`Builder.compose()`** | **累积叠加（生命周期感知、可探测）** | **缺失** |
| 5. Skill/Focus | instruction + meta filter | 软注意力隔离 | 未开始 |

### 行业参照

树形命名空间 + 模块化能力组合的系统都在同一位置撞墙：

- **Plan 9** — `mount`（创建树节点）之外引入 `bind`（union mount，合并到已有目录）。同一文件服务器，两种集成模式。
- **COM** — aggregation。内部组件暴露的接口，外部组件选择直接暴露或隐藏。调用者不知道内部组件存在。
- **VS Code** — contribution point。Extension 向已有 UI 表面贡献命令、视图、菜单项，不创建新窗口。

共同主题：**能力单元如何集成到宿主，取决于集成模式（创建边界 vs 融入边界），而非能力单元自身的性质。**

## Design Index

- Channel Builder API: `src/ghoshell_moss/core/blueprint/channel_builder.py`
- StatefulChannel 实现: `src/ghoshell_moss/core/blueprint/states_channel.py`
- PyChannel 运行时: `src/ghoshell_moss/core/py_channel.py` — `StateChannelRuntime` 是关键实现锚点
- Shell speech 集成: `src/ghoshell_moss/core/ctml/shell/ctml_shell.py:_speech_context_manager`
- Speech 契约层: `src/ghoshell_moss/contracts/speech.py`
- Speech 实现层: `src/ghoshell_moss/core/speech/`
- Speech Channel: `src/ghoshell_moss/channels/speech_channel.py`
- CTML 语法规范: `src/ghoshell_moss/core/ctml/prompts/v1_0_0.zh.md`

## Key Decisions

### K1: ChannelState 就是 ChannelModule 的基础实现

**决策**: 不新建 ABC。用 `ChannelModule` Protocol 定义能力模块的接口契约，`ChannelState`（和 `PyChannelBuilder`）自动满足。

**理由**: 人类工程师指出 ChannelState 设计之初就已中立化，自身就是可横切的能力单元。
- `ChannelModule` 是 Protocol — 结构子类型，零继承负担
- `PyChannelBuilder` 不需改一行代码就已满足协议
- 类型系统参与认知分离：`with_state(state: ChannelState)` vs `with_module(module: ChannelModule)` 签名本身就表达语义差异

**被拒绝**: 新建 `ChannelComposer` ABC。多余的抽象，ChannelState 已经拥有了 Composer 需要的全部要素。
实现时发现 `compose(builder)` 方法本质上在重做 Builder 已有的注册逻辑 — 纯声明式 Protocol 更干净。

### K2: Composer 不是 Builder 语法糖

**决策**: Composer 必须有可探测性，否则退化为 `builder.command()` + `builder.startup()` 的糖。

可探测性的具体含义：
- Channel 知道装配了哪些 composer → `channel.composers()` 可列
- 可按组移除 → `builder.decompose("tts")` 整组移除
- Meta 暴露给 AI → "当前可用能力模块：基础播报、音色控制"
- 模型看到的不是平铺命令，而是 "这些命令来自某个可识别、可描述的能力模块"

**被拒绝**: 仅提供 `com.bind(channel_builder)` 无状态注册。无法回答 "谁注册的"、"怎么整组移除"、"AI 怎么理解模块结构"。

### K3: Composer 生命周期是响应式，不是主动式

**决策**: Composer 的生命周期钩子响应宿主事件，不控制宿主。

对比：
- ChannelState（current_state）有独立的 `on_running()` task，被 `switch_state()` 创建和取消
- Composer 没有独立 task，只有钩子：`on_startup()` / `on_close()`，可选的 `get_instruction()` / `get_context_messages()`

**被拒绝**: Composer 拥有独立运行 task。寄生语义不需要独立并发。

### K4: "按约定安全地被动集成" 是根本逻辑

**决策**: Speech 模块声明 "我能提供什么"，集成方决定 "放在哪里"（compose 到 main channel，或 import 为独立 tree node，或两者）。

当前 `TTSSpeech.commands()` 和 `SpeechChannel.bootstrap()` 把 "生成命令" 和 "注册到哪里" 耦合在一起，需要解耦。Speech 模块不知道也不关心被集成到哪里。

## Implementation Notes

### 三个单词，三个目的（认知模型约束）

Channel 的组合方式归约为三个问题：

| 方法 | 回答的问题 | 语义 | 基数 |
|------|-----------|------|------|
| `import_channels` | 我是**什么结构**？ | 树节点 mount | N — 纵向层级 |
| `with_state` | 我**处于什么模式**？ | 排他切换 | 1 — 横向身份 |
| `with_module` | 我**拥有什么能力**？ | 累积叠加 | N — 横向能力 |

Virtual channels 归入 tree 层 — 它们是动态 tree 管理，不是独立维度。

### API

```python
# channel_builder.py — Protocol，不是 ABC
@runtime_checkable
class ChannelModule(Protocol):
    def name(self) -> str: ...
    def own_commands(self) -> dict[str, Command]: ...
    # 以下全可选
    async def on_startup(self) -> None: ...
    async def on_close(self) -> None: ...
    async def get_instruction(self) -> str: ...
    async def get_context_messages(self) -> list[Message]: ...

# BaseStateChannel / PyChannel — 两个新方法
class BaseStateChannel(StatefulChannel):
    def with_module(self, module: ChannelModule) -> Self:
        """注册为永久能力模块。所有 module 同时激活，累积叠加。"""
    def modules(self) -> dict[str, ChannelModule]: ...
```

PyChannelBuilder 自动满足 ChannelModule Protocol — 不需改一行代码。

### StateChannelRuntime 集成锚点

五处集成（`_modules` 遍历与 `_states` 遍历模式一致）：

- `_own_commands()` — main → **modules** → current_state 优先级叠加（空 modules 时 `if len()` 快速路径）
- `on_startup()` — main.on_startup → **modules*.on_startup** → auto-switch
- `on_close()` — **modules*.on_close** → current_state.on_close → main.on_close
- `_get_context_messages()` — main + modules* + current 合并
- `_generate_own_metas()` — meta.composers 记录 module name 列表（for debug only）

### ChannelMeta 扩展

```python
composers: list[str] = Field(default_factory=list)
# 仅 name 列表，for debug。不做语义化（那是 state 的事）。
```

## Exploration Path

1. 初始判断：speech channel 边界越界 → 把 contracts 工厂方法移走
2. 用户纠偏：真正问题不是 contracts 不纯，是缺少模块化集成机制
3. 行业参照：Plan 9 bind、COM aggregation、VS Code contribution points
4. 收敛：区分 bind（无状态）、compose（累积叠加、生命周期感知）、state（排他切换）
5. 非糖证明：可探测性是 composer 的存在理由
6. 正交性确认：composer 与 state 是两种不同的集成模式，不是趋同
