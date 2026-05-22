---
title: Channel Composer — 生命周期感知的模块化 Channel 集成机制
status: in-progress
priority: P1
created: 2026-05-22
updated: 2026-05-22
depends: []
milestone:
description: >-
  Channel 树体系缺失的第 4 层：将独立模块以可探测的方式累积叠加到 channel，正交于 bind（无状态）和 state（排他）。
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

### K1: Composer 与 State 正交，不趋同

**决策**: Composer 是独立 primitive，不与 ChannelState 合并。

**理由**: 基数差异是根本语义差异，不是实现细节：
- State = 排他选择（同一时刻只有一个 current_state），需要 switch、task 取消、状态转换
- Composer = 累积叠加（多个 composer 同时生效），不需要排他语义

强行合并会让 State 的复杂度污染 compose 场景。实现层可共享基础设施 — `_own_commands()` 遍历 main + composers + current_state 的模式是自然的。

**被拒绝**: 将 Composer 实现为 ChannelState 的 "non-exclusive 模式"。State 的 children、virtual children、独立 running task 对 composer 都是不必要的负担。

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

### API 草图

```python
class ChannelComposer(ABC):
    """生命周期感知的模块化 Channel 补丁"""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def compose(self, builder: Builder) -> None:
        """注册 commands、lifecycle hooks、instruction 等到 builder"""
        pass

    # 响应式生命周期钩子 — 由宿主 channel 的对应生命周期触发
    async def on_startup(self) -> None: ...
    async def on_close(self) -> None: ...

    # 可选的可探测性接口
    async def get_instruction(self) -> str: ...
    async def get_context_messages(self) -> list[Message]: ...
```

Builder 入口：

```python
class Builder(ABC):
    def compose(self, composer: ChannelComposer) -> Self: ...
    def decompose(self, name: str) -> None: ...
    def composers(self) -> dict[str, ChannelComposer]: ...
```

### StateChannelRuntime 集成锚点

`StateChannelRuntime` 需要在以下位置集成 composers 列表遍历：

- `_own_commands()` — main → **composers** → current_state 优先级叠加
- `_get_own_command()` — 查询优先级: main → composer → current_state
- `on_startup()` — main.on_startup 后、current_state 前，依次调所有 composer.on_startup
- `on_close()` — 当前只调 main_state.on_close()，需加 composer.on_close() 遍历（以及 current_state.on_close() — 现实现疑似遗漏）
- `_get_context_messages()` / `_generate_own_metas()` — 合并 composer 的 context 和 instruction
- `on_idle()` / `on_running()` — composer 如需钩子可加，但 K3 建议保持轻量

### Speech 场景的应用

从：

```python
# 现在: shell 硬编码 TTS 感知
if isinstance(self._speech, TTSSpeech):
    for command in self._speech.commands():
        self.main_channel.build.add_command(command, override=False)
```

变为：

```python
# 目标: composer 模块化集成
self.main_channel.build.compose(BaselineSpeechComposer(self._speech))
if isinstance(self._speech, TTSSpeech):
    self.main_channel.build.compose(TTSCommandsComposer(self._speech))
```

进一步方向：让 `Speech` 接口提供 `create_composers() -> list[ChannelComposer]`，shell 遍历 compose，消除 isinstance。但可后置。

### 旁注：MockSpeech 内存泄漏

`MockSpeech._outputs` 永不清理。需要 `NullSpeech`：`new_stream()` 返回 `/dev/null` stream，`feed()` 直接丢弃。以及 `BroadcastSpeech` 锚点（topic/session.stream 广播），暂不实现，在 contracts 留抽象即可。

### 旁注：states_channel 测试债务

`StateChannelRuntime` 的核心逻辑（`on_close` 对 current_state 的处理、`switch_state` 边界条件）未充分测试。Composer 实现前应对现有 state 机制做固熵。独立 workstream。

## Exploration Path

1. 初始判断：speech channel 边界越界 → 把 contracts 工厂方法移走
2. 用户纠偏：真正问题不是 contracts 不纯，是缺少模块化集成机制
3. 行业参照：Plan 9 bind、COM aggregation、VS Code contribution points
4. 收敛：区分 bind（无状态）、compose（累积叠加、生命周期感知）、state（排他切换）
5. 非糖证明：可探测性是 composer 的存在理由
6. 正交性确认：composer 与 state 是两种不同的集成模式，不是趋同
