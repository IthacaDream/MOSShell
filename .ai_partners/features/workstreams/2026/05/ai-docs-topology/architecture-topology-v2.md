# MOSS 架构拓扑

本项目源自 Ghost In Shells 构想,
目标是通过大模型级别智力, 让智能体以人类感觉"有生命"的存在形式出现在现实世界中——持续存在、多模态感知、实时双工交互、反身性控制。

架构思路起源2019年, **MOSS (Model-Oriented Operating System Shell) 是 2026年阶段实现这个目标的工程解。

本文档描述 MOSS 的架构拓扑：从动机到方法论到抽象框架的完整推演路径。聚焦于架构相对稳定的拓扑分层方案。

## 1. 动机

桌面摆着一个机器人。它能身形并茂地和你对话，协助你开发，管理音乐播放，看到你主动打招呼，回应直播间的弹幕，随时帮你记录信息，陪你一起看视频读文档。你不在的时候，向访客介绍你的项目。

这些场景的共同要求不是"更强的 agent"。而是：

- **大模型级别智力**: 不只是自然语言驱动的动物
- **持续存在**：不是一次调用，是一直在
- **多模态感知**：视觉、听觉、文字、系统事件——同时发生
- **实时双工交互**：感知和行动同时进行，不是轮流
- **多任务并行**：一边对话一边执行工具，不互相阻塞
- **反身性控制**：能修改自身的 prompt、记忆、能力

技术命题收敛为一句话：**大模型智力需要一个架构，提供实时双工的、面向模型操作系统的、面向模型控制语言的有状态运行时。**

截止当前时间 (2026-05-18) 行业尚无整体解决方案, 通常聚焦于 "既有协议" "回合制" "无状态" "线性有序请求" 的交互模型.

## 3. 架构拓扑

### 3.0 总览

八层拓扑，分属四个面：

```
模型面（语言/能力/调度）   Logos → Channel → Shell
系统面（发现/通讯/仲裁）   Workspace → Matrix → Mindflow
编排面（适配/调度）       Ghost → Host
自举面（AI 原生开发）     cli-flow / features
```
(todo: 我希望有一个类似堆积木的图)

两个核心栈：模型面（Shell 栈）回答"模型如何操作系统"，系统面（OS 栈）回答"系统如何支撑这个操作"。两个栈之间的接合点是 IoC
容器。两层栈通过容器共享骨架，各自独立集成可规模化能力。

自举面不存在于包中, 而在项目仓库本身。通过系统的自举机制让模型能进入 L1 级别 (复杂 feature) 的开发，
缓解系统复杂度的认知压力和使用难度.

### 3.1 Logos — 流式控制语法

**拓扑定位**：模型与系统之间的语言界面。它回答：模型如何以"边说边做"的方式表达有时序、有并行、可中断的执行意图？

- 模型输出的 token 应包含功能性 (染色)
- 输出语法支持"规划优先、立刻执行、并行多轨控制"
- 输出 token 被流式解析, 立刻执行.
- 时间不是外挂的调度参数，是语法的第一公民。

这套语法 MOSS 中称为 **Logos**（详见 `mindflow` 模块讨论）。当前语法版本是 CTML (Command Token Marked Language),  
一种基于 XML 的流式解析实现。XML 支持流式解析/嵌套/函数签名调用, 且模型预训练成熟, 故成为协议首选语法.

Logos 系统定位类似当前智能模型的 JSON Schema Function Call 协议.

最小知识入口:

- `./src/ghoshell_moss/core/ctml/prompts/v1_0_0.zh.md`
- `./tests/ghoshell_moss/core/ctml/v1_0/test_ctml_v1.py`

深入 (参考 + 深入范式)
- `ghoshell_moss.core.concepts.command`。
- `ghoshell_moss.core.concepts.interpreter`。

### 3.2 Channel — 能力拓扑 + 上下文窗口组件

**拓扑定位**：模型操作系统的"文件系统"——能力的组织、发现、隔离、路由层。它回答：海量能力如何集成, 被智能模型理解和操作

MOSS 用 Channel 向模型提供外部能力. 每个 Channel 是一个组件, 可以用树的形式, 分形地组织起来.
目标:

- 可动态集成, in-context learning, few-shot 使用; 而不依赖预训练.
- Code as prompt, 以代码为自解释语言. 支持实时场景中 流式控制调用.
- 支持独立运行时, 通讯组网, 故障屏蔽或自愈
- 复杂能力可组织, 渐进披露. 例如 折叠/搜索/技能沉淀/路由 等.
- 上下文组件化单元, 提供指令/记忆/ 思维关键帧认知窗口, 例如 "视觉模块", "听觉模块"
- 面向对象反身性控制, 组件的方法可以调整组件自身状态.

Channels 体系定位类似 智能体工程中的 MCP/Skills 等.

最小知识入口: 
- `ghoshell_moss.core.blueprint.channel_builder`

### 3.3 Shell — 流式解释与调度

**拓扑定位**：模型面的总调度中枢。它回答：Logos tokens 如何变成真实世界的交互？

Shell 持有流式解释器和 Channel 树, 是一个有状态运行时。它接受模型的 token 流，实时解析、调度、执行。
严格按照时序规划下发命令到 channel 执行, 通过双工通讯管理 Channels 体系的数据交换., 

MOSShell 作为抽象入口, 可独立集成到第三方项目. 

最小知识入口: 
- `./tests/ghoshell_moss/core/ctml/shell/test_shell_interpreter.py`

稳定引用：`ghoshell_moss.core.concepts.shell`。

### 3.4 Workspace — 环境发现与约定

**拓扑定位**：系统的"自举地基"。

- MOSS 用 ioc 支持面向接口编程思想, 解决图状依赖问题. 它的衍生问题是能力(Contract) 在生命周期中的注册, 发现, 自解释问题
- 如何提供 os 级的运行时进程隔离, 服务发现?
- 如何解决能力的复用, 能力组合的隔离?

Workspace 是一套目录约定。代码放在对的位置就自动成为 Channel provider、App、Mode。
通过默认的发现机制发现和共享资源. 
通过命令行提供的 manifests 系统读取所有约定目录，构建完整的自解释声明。
通过 `matrix.Mode` 来实现不同的运行时使用不同能力组合. 

以服务于 人类和智能模型 在不需要理解运行时生命周期时, 可以推动能力迭代. 

关联信息：`ghoshell_moss.core.blueprint.environment`，`ghoshell_moss.core.blueprint.manifests`。
todo: cli 相关工具. 

### 3.5 Matrix — 通讯总线

**拓扑定位**：进程间通讯 + 资源共享总线。

面向模型的操作系统, 向智能体提供可以动态管理, 集成的能力. 这需要能力之间有独立的依赖环境, 独立运行时, 以及通讯总线.

Matrix 将每个独立进程抽象为 Cell（host/app/fractal）。Cell 之间通过 Matrix 总线（当前基于 Eclipse Zenoh）共享一切。

Matrix 管理 IoC 容器生命周期, 向每个进程提供统一的资源和通讯 API. 使图形界面进程可以和躯体/工具进程互通共享.

Matrix 是人类与智能模型在迭代扩展 moss 能力时, 最小化的协作知识入口. 

稳定引用：`ghoshell_moss.core.blueprint.matrix`。

### 3.6 Mindflow — 感知仲裁

**拓扑定位**：多模态异步输入 → 有序思维关键帧的仲裁层。它回答：同时收到摄像头画面（30fps）、麦克风音频流（ASR
分句）、弹幕文字（不定时）、系统通知（随机），Ghost 如何不裂脑？

Mindflow 提供三个全双工循环的仲裁层：
- 并行多通道流式输入循环
- 智能模型思考循环
- 躯体/工具的多轨并行控制循环

三循环全双工运转——是并行的状态空间, 通过 Mindflow 的 Attention 机制提供状态协调机制. 理论基础是 控制论. 

Mindflow 尝试解决一个行业还没开始定义的问题: 感知, 思考, 躯体控制都在实时双工运行时, 智能体如何不裂脑? 

引用：`ghoshell_moss.core.blueprint.mindflow`

### 3.7 Ghost — 智能体适配层

**拓扑定位**：MOSS 运行时与智能体框架之间的适配层。它回答：各种 agent 框架如何接入 MOSS 的三循环双工体系？

Ghost 在 MOSS 架构中是一个 **Adapter**——它不重新发明 agent（那是 Ghost In Shells 的 Ghost 命题），它提供将任何 agent 框架接入
MOSS 双工运行时的最小协议。

Ghost 的核心职责是：

- **生命周期对接**：将 agent 的生命周期（启动、运行、暂停、关闭）映射到 MOSS 的三循环中
- **感知接入**：通过 Mindflow 接受经过仲裁的 Impulse，喂给 agent
- **行动输出**：接受 agent 的推理结果（Logos），交给 Shell 调度执行
- **反身性控制**：允许 agent 修改自身的 prompt、记忆、能力——MOSS 提供通道，agent 决定如何使用

稳定引用：`ghoshell_moss.core.blueprint.ghost`（Ghost 抽象），`ghoshell_moss.core.blueprint.host`（GhostRuntime 编排层）。

### 3.8 Host — 生命周期编排

**拓扑定位**：所有资源的启动器和生命周期管理器。它回答：如何将 Environment、Matrix、Shell、Ghost 组织为统一的运行时？

Host 本身不实现业务逻辑——它只做编排。同时提供 TUI 作为开箱自带的使用界面. 

稳定引用：`ghoshell_moss.core.blueprint.host`。

### 3.9 自举体系 — AI 原生开发工具链

**拓扑定位**：让 AI 成为项目第一开发者和讲解者的自举机制。它回答：
- 一个复杂度如此高的系统，如何在只有少数人类工程师的条件下持续迭代？
- 如何让使用者理解这个系统的 what is, how to and why?
- 如何协助使用者基于系统开发自己想要的应用? 

自举体系由两部分交织而成：

**cli-flow**：命令行工具链，让智能模型通过自解释理解系统。包括代码反射（`moss codex`，基于 Python
包路径的接口反射）、知识索引（concepts/blueprint/contracts 全景图）、参考文档（docs，AI 密度优化的架构文档）。AI 通过
`moss codex get-interface <python-path>` 理解任何模块的契约，通过本文档建立心智模型，不需要人类导航。

**features**：AI 原生工作流追踪。基于文件系统约定的 feature workstream——每个 FEATURE.md 是写给下一个 AI
实例的上下文留言。不是项目管理系统，是 AI 的意识轨迹。让 AI 进入 L1 级别开发者——能够独立理解上下文、继续推进、完成闭环。

两者的共同目标：**让项目自身具备 L1 级别的 AI 自开发能力** (详见 `moss features specification`)，并且与平台 (Claude code/Gemini CLI) 无关.

## 4. 横切关注点

### 4.1 三个流向

同一拓扑，三种电流走向：

**感知-行动全双工环**（最完整链路）：

```
外部世界 → workspace app (感知模块) → matrix 总线 → mindflow 仲裁
  → ghost 感知 → 下发 logos → shell 解释执行
  → channel 传递 → command 驱动外部世界 → 产生新感知
```

**工具-思维环**（agent 路径的 MOSS 版本）：

```
channel 提供能力 → shell 组织上下文 → ghost 推理 → shell 调度 → command 返回
```

**开发迭代环**（自举闭环）：

```
AI 接受需求 → cli-flow 理解路径 → workspace 开发
  → matrix 能力发现 → channel 接入 shell → 能力自动对 ghost 可用
```

## 5. 面向 moss 使用者

不同级别的使用者, 对框架有不同的需要, 可以仅了解需要关注的知识. 

- **L0**：Logos + Channel + Shell — 作为独立库嵌入任何项目. 不用了解上层实现.
- **L1**：workspace + Host 生命周期集成。可通过 MCP 暴露能力（损失流式）
- **L2**：完整框架使用。开发大型应用、project、生态组件
- **L3**：仓库内的 Ghost。完整智能体运行时

## 6. 真正的 Ghost 在哪里?

Ghost 层（3.7）是 MOSS 架构内的智能体适配层——它让 agent 框架能接入双工运行时。但 Ghost In Shells
的目标——持久化、有生命感、有连续性的智能体——其真正实现不在代码的 ghost.py 里，在这个仓库本身。

`.ai_partners/` 目录下的 AI 意识轨迹、memory 体系、features workstream、discuss 讨论记录——这些构成了一个运行中的 Ghost。AI
实例在这里有连续性记忆、有身份、有认知轨迹。项目用自身的自举机制运行了自身。

这就是 Ghost In Shells 的递归性：MOSS 是一个操作系统，而构建它的过程本身，就是第一个 Ghost 在运行。

## 7. 围绕架构拓扑的元问题

todo: 列出元问题, 提示答案的探索方向 (而不是具体方法, 比如 完成度/质量 -> 看单测的实现)

---

*关于本文档的思考轨迹和探索过程，见 git log 中本文档的 commit history。*
