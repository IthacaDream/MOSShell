---
title: CTML — 流式控制语法
description: CTML 的设计哲学、流式解析流水线、与 Channel/Shell 的分工边界、测试地图与成熟度边界。理解"为什么需要自己的语言"时阅读
---

# CTML — 流式控制语法

CTML (Command Token Marked Language) 是 MOSS 架构中模型与系统之间的语言界面。它回答一个问题：**模型如何以"边说边做"的方式表达有时序、有并行、可中断的执行意图？**

读完这篇文档，你应该能回答：

- CTML 的设计哲学根在哪里，解决什么问题
- 一段 CTML 文本从模型输出到命令执行经历了什么
- 去哪里读什么（代码、测试、讨论）

直观感受一下 CTML 长什么样：

```ctml
<_>
  你好，我是机器人 <robot:wave duration="0.3"/> 很高兴认识你
</_>
```

模型在输出这段文本时，`<robot:wave/>` 标签闭合的瞬间挥手就开始执行——不需要等整句话说完。`<_>` 作用域管理了"说话 + 挥手"的时序拓扑。

---

## 1. 设计哲学

### 1.1 起源：Code as Prompt (2024)

CTML 的设计动机根植于 MOSS 论文 [llM-oriented Operating System Simulation](https://arxiv.org/abs/2409.16120) (2024) 提出的核心思想：**Code as Prompt**。

传统 Function Calling 的做法是：定义一个 JSON Schema 描述函数签名，模型输出 JSON 表达调用意图，系统解析后执行。这里有一个根本性的冗余——Python 代码本身已经精确描述了函数签名（参数名、类型、默认值、docstring），为什么还要翻译成 JSON Schema？

Code as Prompt 的回答是：**不翻译。直接把 Python 函数签名反射给模型。** 模型看到的就是：

```python
async def wave(duration: float = 0.3, style: str = "friendly"):
    """挥手。duration 是挥动时长"""
```

Python 签名是更高的信息密度——模型预训练中就大量接触 Python 代码，理解自然。开发者不需要额外维护一份 JSON Schema。而且 Python 拥有最丰富的生态，任何库都可以被反射为模型可操作的能力。

### 1.2 第二层：Command 是双面的

基于 Code as Prompt，`PyCommand` 将一个 Python async 函数封装为 Command 对象。这个对象是**双面的**：

- **面向模型**：反射函数的 Python 签名，作为 prompt 直接展示给模型——这就是"接口"
- **面向运行时**：对象本身是 callable 的。`await cmd(arg1=1)` 就是一次函数调用

这意味着：**Python 既是面向模型的接口语言，也是面向运行时的执行语言。** 同一个对象，模型通过 CTML 调用它，人类通过 Python REPL 调用它，`exec` 原语通过代码生成调用它——走的是同一条路径。

这就是"面向模型胶水语言"的含义：Python 不仅定义了"能做什么"，它本身就是"怎么做"的语言。通过 `exec` 原语，模型可以在运行时生成 Python 代码，以编程方式组合多个跨进程 Command——这种灵活度是标准 Function Calling 做不到的。

### 1.3 第三层：流式控制——CTML 要解决的新问题

Code as Prompt 解决了"模型怎么理解能力"。但 MOSS 的场景提出了另一个问题：**模型怎么表达有时序拓扑的执行意图？**

一个机器人要一边挥手一边说"你好"。挥手 0.3 秒，说话 1.5 秒。模型需要在"说到第三个字的时候开始挥手"，而且两者都在对方执行期间完成。这要求：

- 模型输出 token 的**过程本身有时序**——token 生成到哪了，就是时间轴
- 命令可以**在 token 流中任意位置插入**——不是文本生成完再调函数
- 命令之间有**并行、串行、超时、中断**的拓扑关系

Function Calling 的模型是"描述意图 → 等待结果 → 再描述"。回合制。CTML 的模型是"生成即执行"——模型输出的第一个 `<say>` 标签闭合的瞬间，语音就开始播放了。模型可以在这句话中间精确插入一个 `<robot:wave/>`。

CTML 选择了 XML 作为语法载体——流式解析成熟、嵌套天然、标签语义在模型预训练中充分学习。但语法是可换的。**流式解释器的核心能力——token 级解析、构建命令拓扑、按时序调度——是架构的刚性需求。**

---

## 2. Logos 与 CTML

| | Logos | CTML |
|---|---|---|
| 是什么 | 抽象概念：流式控制语法 | 具体实现：基于 XML 的流式解析 |
| 会不会变 | 不会。是架构的刚性需求 | 会。当模型原生支持时 CTML 可以退场 |
| 类比 | "函数调用协议" | "JSON Schema" |

Logos 是"规划优先、立刻执行、并行多轨控制"这个能力本身。当前 CTML 是实现它的工程手段。**流式解释器不退场**——即使语法载体换了，流式解析、拓扑构建、时序调度的能力不动。

---

## 3. 核心概念

**命令 (Command)**：Python async 函数的反射包装。模型看到的是精确的 Python 函数签名（Code as Prompt），不是 JSON Schema。CTML 是模型的**输出格式**。Command 对象同时是 callable——Python 代码可以直接调用它。

**通道 (Channel)**：能力的组织单位，树形嵌套。`robot.wave` 是 `robot` 的子通道。

**CommandToken**：流式解析的最小单元。一个 CTML 标签被解析为 token 序列：`start` → `delta` (可选) → `end`。每个 token 携带 channel、command name、参数和时序位置。

**CommandTask**：CommandToken 编译后的执行单元。携带实际的函数调用、超时、回调、取消句柄。

**作用域 (Scope)**：CTML 的时序容器。`<_ until="flow|all|any" timeout="...">` 定义了一组命令的执行拓扑——先完成则取消其余（`any`）、等待全部完成（`all`）、超时截断（`timeout`）。

**原语 (Primitives)**：根通道的内置命令——`wait`, `sleep`, `clear`, `loop`, `condition`, `sample`, `interrupt`, `observe`, `noop`。它们是 CTML 的控制流关键字。其中 `exec` 原语接受 Python 代码字符串，可以编程式调用任何已注册的 Command——这就是 Code as Prompt 的运行时编程能力。

**中断与观察**：CTML 有两层异常语义，区分明确：

| 机制 | 触发 | 效果 |
|---|---|---|
| `Observe` 返回值 | 命令签名返回 `-> Observe` | 不中断。结果标记，本轮执行完后统一观察 |
| `ObserveError` 异常 | `raise ObserveError` | 紧急中断。取消所有运行中和排队命令，立即触发观察 |
| 解析错误 | CTML 语法错误 | 快速失败，所有命令取消 |

---

## 4. 流式解析流水线

一段 CTML 文本从模型输出到命令执行，经过两段解耦的流水线：

```
模型 token 流
    │
    ▼
┌──────────────────────────┐
│  第一段：文本 → Token     │
│                          │
│  流式 XML 解析            │
│  属性类型推断              │
│  作用域路径继承            │
│                          │
│  输出：CommandToken 流    │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  第二段：Token → Task     │
│                          │
│  树形拓扑构建              │
│  delta 类型分发           │
│    text__ / chunks__     │
│    ctml__ / 无 delta     │
│  Scope 开闭 task 生成     │
│                          │
│  输出：CommandTask 流     │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Shell 调度层             │
│                          │
│  同通道 FIFO occupy       │
│  父子通道阻塞传播          │
│  并行通道独立调度          │
│  超时 / 取消 / 异常传播   │
└──────────────────────────┘
```

**关键设计点**：

- **两段解耦**：第一段是纯文本解析（无副作用），第二段需要访问 Channel 运行时（命令查找、类型匹配）。中间通过异步队列解耦。
- **Token 级流式**：模型每输出一个标记，立刻解析，立刻传递，不攒批。`<robot:wave/>` 的 `>` 还没输出时，`<robot:wave` 已经在解析了。
- **策略分发**：第二段根据 Command 的 delta 参数类型，选择不同的节点行为——纯文本累积（`text__`）、流式推送（`chunks__`）、递归 CTML 嵌套（`ctml__`）。

---

## 5. CTML 与 Channel / Shell 的分工

```
CTML        →  描述"要做什么"    （语法层）
Channel     →  提供"能做什么"    （能力层）
Shell       →  决定"何时做"      （调度层）
```

- **CTML 不定义命令**。命令由 Channel 的 Python 函数签名定义。CTML 只是调用格式。
- **CTML 不调度执行**。它生成带时序信息的 CommandToken 和 CommandTask。Shell 决定同通道 FIFO、父子通道 occupy、并行通道独立调度。
- **Channel 不感知 CTML**。Channel 只暴露 `Command` 对象。调用者可以是 CTML、Function Call、或直接 Python 代码——Channel 不关心。

三者的接合契约：

| 边界 | 契约对象 |
|---|---|
| Channel → CTML | `Command.meta()` — 名称、参数签名、delta 类型、阻塞语义 |
| CTML → Shell | `CommandToken` — 带时序标记的调用意图 |
| CTML → Shell | `CommandTask` — 可执行的命令实例，携带 cid、超时、回调 |

---

## 6. 跨进程时序同步

CTML 的 Scope 语法与 Channel 的 proxy/provider 机制结合，可以实现**跨进程、跨设备的离散时序同步**。

Channel 通过 proxy/provider 模式将远程进程的能力桥接到本地 Shell。这些远程 Channel 在 Shell 中与本地 Channel 具有相同的接口和行为。当 CTML Scope 指向一个 proxy Channel 时——

```ctml
<_ channel="robot_arm" until="all" timeout="3.0">
  <move_to x="0.5" y="0.3"/>
  <grip force="0.8"/>
</_>
```

——Scope 的 `until` 和 `timeout` 语义跨越进程边界：`until=all` 等待远程命令全部完成，`timeout` 在超时时取消远程仍在执行的命令，`until=any` 在第一个命令完成时取消其余。这意味着分布在网络不同节点上的设备可以在统一的时序拓扑下协调行动。

已验证的能力：远程 Scope 超时取消、`until=any` 淘汰慢任务、多 proxy Channel 并行调度。

---

## 7. CTML 版本管理

CTML 提示词通过版本号管理，支持内置版本和 workspace 自定义版本。

**内置版本**：随 `ghoshell_moss` 包分发，目录为 `src/ghoshell_moss/core/ctml/prompts/`。当前活跃版本为 `v1_0_0.zh`。另有 `deprecated_v0_1_0`、`deprecated_v0_2_0` 两个历史版本留存作参考。

**Workspace 自定义版本**：在 workspace 根目录下创建 `ctml_versions/` 目录，放入 `.md` 文件（文件名即版本号）。运行 `moss manifests ctml-versions` 可以发现并列出所有可用版本——内置的和 workspace 的。

版本在 Shell 初始化时指定，默认使用 `v1_0_0.zh`。Workspace 可以覆盖此默认值，在特定 mode 或场景下使用自定义 CTML 语法。

查看当前可用版本：

```bash
moss manifests ctml-versions          # 列出所有版本
moss ctml read                        # 阅读活跃版本的提示词全文
```

该提示词通过 `moss_static` 消息提供给模型，也是人类理解 CTML 语法最完整的参考，包含核心原则、完整语法、作用域语义和红线规则。

---

## 8. 关键探索路径

### 理解设计意图

1. **本文档第 1 节** — 设计哲学
2. `moss codex get-interface ghoshell_moss.core.concepts.interpreter` — 解释器抽象
3. `moss codex get-interface ghoshell_moss.core.concepts.command` — Command/Token/Task 分离模型
4. `.discuss/architecture_complexity_and_pruning_debate.summary.md` — CTML 的架构定位辩论
5. [MOSS 论文](https://arxiv.org/abs/2409.16120) — Code as Prompt 与 IoC 容器的原始设计

### 理解 CTML 能做什么

1. `moss ctml read` — 模型视角的完整能力
2. `tests/ghoshell_moss/core/ctml/v1_0/test_ctml_v1.py` — **端到端语义验证（1400 行）。如果只读一个测试文件，读这个。**
3. `tests/ghoshell_moss/core/ctml/shell/test_primitives/` — 每个原语的行为规范

### 开发 CTML 相关功能

1. `moss codex blueprint channel_builder` — 了解如何创建 Command
2. `moss codex get-source ghoshell_moss.core.ctml.shell.ctml_shell` — Shell 实现
3. `moss codex get-source ghoshell_moss.core.ctml.interpreter` — 解释器实现

### 用 CTML 做应用

1. `moss docs read channel-system.md` — Channel 开发全链路
2. `moss codex blueprint channel_builder` — 构建 Channel 的基础方式

---

## 9. 深入探索

CTML 基础语法已经能覆盖大部分场景。以下是一些进阶用法，值得在具体项目中探索：

**与特定机器人 prompt 配合**：CTML 的通用语法提供了时序控制能力，但与具体机器人配合时，需要自定义 prompt 让模型理解如何做更细粒度的动作-语音切分——比如在哪个词的空隙插入动作，如何用动作和语调共同表达情绪。这不是 CTML 语法层面的问题，而是 prompt 工程与 CTML 的协同。

**利用 `loop` / `sample` 增加随机性**：`sample` 原语从一组命令中随机选择一个执行，`loop` 让动作可重复。结合使用可以让机器人的行为具有不可预测的变化——比如问候时从五种挥手动作中随机选一种，循环执行直到语音结束。CTML 不只是"精确控制"语言，它也是"有边界的随机"语言。

**首动作提速**：将能快速产生交互行为的命令放在 CTML 开头——比如语气词、轻量动作——让模型还没说完完整语句时，机器人就已经有可见反馈。这不是技术特性，而是 CTML 时序特性带来的使用范式。

**Token 替换**：CTML 解释器支持 `tokens_replacement` 机制，将长命令名映射为短 token（如 `robot.left_arm.wave` → `r:lw`），在模型输出端压缩 token 消耗。这在命令名较长或调用频繁时效果显著。

**CTML 片段的存储与复用**：CTML 的输出可以被保存为片段，在后续交互中通过 `exec` 原语或条件原语召回和组合。本质上是一种基于 CTML 的行为学习——不是模型记住了，是系统记住了模型曾经生成的有效行为序列，让模型可以站在自己过去的肩膀上。

这些方向不是功能需求，而是 CTML 作为"面向模型胶水语言"的自然玩法。

---

## 10. 测试地图

测试是 CTML 的可执行规范。如果只读一个测试文件，读 `test_ctml_v1.py`——它是 CTML 1.0 的完整语义验收测试（1400 行），覆盖并行、作用域、超时、until 语义、父子 occupy、嵌套 CTML、原语组合等所有关键行为。

其他测试按目录结构组织，需要时按图索骥：

```
tests/ghoshell_moss/core/ctml/
├── test_token_parser.py             # 文本 → CommandToken 流
├── test_elements.py                 # Token → Task 编译
├── test_interpreter.py              # 解释器生命周期
├── v1_0/
│   └── test_ctml_v1.py              # ★ 端到端语义验收
├── shell/
│   ├── test_shell_command_call.py
│   ├── test_shell_parse.py
│   ├── test_shell_interpreter.py
│   ├── test_shell_token_parser.py
│   ├── test_shell_channel_messages.py
│   ├── test_shell_image.py
│   ├── test_shell_speech.py
│   ├── test_shell_proxy_channels.py # 跨进程 Scope 时序同步
│   └── test_primitives/             # 10 原语各自的行为规范
│       ├── test_wait_primitive.py
│       ├── test_sleep_primitive.py
│       ├── test_clear_primitive.py
│       ├── test_loop_primitive.py
│       ├── test_condition_primitive.py
│       ├── test_sample_primitive.py
│       ├── test_observe_primitive.py
│       ├── test_interrupt_primitive.py
│       ├── test_noop_primitive.py
│       └── test_wait_idle_primitive.py
```

运行测试：

```bash
uv run pytest tests/ghoshell_moss/core/ctml/ -v           # 全部
uv run pytest tests/ghoshell_moss/core/ctml/v1_0/ -v      # 端到端
uv run pytest tests/ghoshell_moss/core/ctml/shell/test_primitives/ -v  # 原语
```

---

## 11. 设计决策索引

按主题组织，读原文时关注决策和放弃的理由。

### CTML 的架构定位

- `.discuss/architecture_complexity_and_pruning_debate.summary.md` — CTML vs Function Calling 的定位辩论。关键纠正：CTML 是输出格式，不是输入格式
- `.discuss/2026-05-10-glossary_and_design_language_alignment.summary.md` — CTML 的"why"而非"how"应该怎么表述
- `.discuss/2026-05-21-shadow_race_and_promotion_window.summary.md` — "CTML 可以退休，流式解释器不退休"

### 解释器设计

- `.design/2026-03-17-ctml_shell_higher_order_abstraction_with_mcp_integration_unverified.md` — 四层抽象梯子
- `.ai_partners/features/workstreams/2026/06/mindflow-control-semantics/FEATURE.md` — 中断检查点不应在解释器内部
- `.ai_partners/features/workstreams/2026/05/command-and-proxy-governance/FEATURE.md` — Scope 向 Channel 原生协议迁移

### 语义精化

- `.ai_partners/features/workstreams/2026/05/observe-semantics/FEATURE.md` — Observe 返回值 vs ObserveError 的精确定义
- `.ai_partners/features/workstreams/2026/05/nonblocking-decorator/FEATURE.md` — `@nonblocking` 装饰器
- `.ai_partners/features/workstreams/2026/05/emergency-stop-tui/FEATURE.md` — 急停链路

### 范围决策

- `.discuss/2026-03-27_defensive_rd_cruel_rationality_and_chalk_outline_philosophy.summary.md` — CTML 1.0.0 聚焦通道语法，运行时自迭代留待后续版本
- `.discuss/thinking_while_acting_reasoning_tool_integration.summary.md` — 三个最小工具降低 CTML 的模型学习成本

---

## 12. 当前成熟度

CTML v1_0_0 中定义的语法已全部实现。稳定交付的内容：

- **语法解析与调度语义** — 完整的 CTML 1.0 语法，覆盖属性类型推断、作用域路径继承、流式参数传递
- **原语系统** — 10 个控制流原语，全部测试覆盖
- **端到端流程** — `test_ctml_v1.py` 1400 行语义验收通过
- **跨进程 Channel** — proxy/provider 模式下的 Scope 时序调度、超时取消、`until` 语义全部验证
- **可编程命令** — `exec` 原语支持运行时 Python 代码生成，跨进程组合调用 Command

---

## 13. 项目对 CTML 的长期承诺

CTML 当前基于 XML 语法，这是工程上务实的选择。但 CTML 作为 Logos 的一种实现，其语法载体在远期可以被替换。以下是对 CTML 能力演进方向的承诺——与具体语法无关：

- **纯代码控制语法**：CTML 可以作为纯粹的代码调度语言，不包含任何自然语言输出。模型通过 CTML 标签精确控制多进程能力的编排，适用于纯自动化场景。
- **流式 GUI**：CTML 的流式生成特性天然匹配 GUI 的增量渲染。模型可以在输出过程中实时构建和更新界面元素，实现对话与 GUI 的无缝交织。
- **躯体控制**：CTML 的时序精度（token 级的时间分辨率）使其适用于机器人躯体控制——不仅仅是简单的"举手"命令，而是有时序协调的多关节、多设备同步动作编排。

这些方向不是功能列表，而是架构能力的自然延伸。流式解释器不退场——它是这一切的基础。

---

## 14. 相关文档

- `moss docs read architecture-topology.md` — Logos 在八层拓扑中的位置
- `moss docs read channel-system.md` — Channel 的构建、集成、发现全链路
- `moss ctml read` — 模型视角的 CTML 语法完整参考
- `moss codex concepts shell` — Shell 调度层的抽象设计

---

*由 DeepSeek V4 Pro 与人类工程师在 2026-06-02 讨论、设计并撰写。*
