# 08 — TUI 集成规划与 GhostRuntime 生命周期加固

日期: 2026-05-22
模型: deepseek-v4-pro

## 上下文

Atom 原型端到端验证通过后（12b-verify），在进入 TUI 全链路验证（12c）之前，
进行了两轮加固：Moment 参数传递链路 + GhostRuntime 生命周期异常治理。
随后完成 TUI 架构调研，为 Ghost 在 REPL 中的首轮集成做准备。

## 共享词汇

### 三循环加固体系
- **per-attention 异常隔离**: 每个 attention 的异常不影响下一个 attention，main loop 继续运行
- **FatalError 传播 / Exception 续流**: FatalError 穿透所有层级终止循环，
  普通异常 log 后继续 — 这是全双工长运行系统的基础纪律
- **_loop_status 语义修正**: "stopped" 只在循环真正退出时设置，不是每个迭代后
- **hook marker (todo)**: 生命周期关键节点上的 `# todo: hook —` 注释标记。
  不实装，但为后续 hook 体系保留插入点。错了可被清理

### TUI 架构
- **TUIState**: 终端中的一个独立面板，拥有 completer / keybindings / input handler / 生命周期
- **REPLState**: 封装 REPLRegistrar 的 TUIState，反射 Python 对象为可补全的 REPL 命令
- **Inspector**: 被 REPLRegistrar 扫描的普通 Python 对象，其方法自动变成 `/` 命令
- **ConsoleOutput**: 线程安全的 rich 渲染队列，分离输入线程和渲染线程
- **Scripts 机制**: `moss script run <name>` 以子进程运行脚本，通过 Zenoh 连入运行中 matrix。
  让 AI 在 REPL 之外有等价的调试手段

### 默认 Ghost 实例
- **echo**: 开发者拿到 MOSS 后见到的第一个 Ghost。名字的含义是 "壳中的第一声回响" —
  你向它说话，它回应。回响不是原声的复制，而是经过空间（Ghost 自身）变形后的声音。
  简单、通用、不张扬。作为原型，它是所有后续 Ghost 的起点

## 锚点

> "脚本是用来给人看的，不是给 pytest 跑的。把脆弱的断言改成直接打印。"

> "主界面其实就是 ghost 输出的 logos 和用户的输入就得了。调试层单独一层。"

> "moss scripts 机制方便我们开启 ghost 之后，仍然随时可以用脚本来交互验证，
> 这样让你在 repl 之外有等价的调试手段。"

> "这个的开发过程我们反而不用过度设计，讨论差不多直接做完走感觉就好了。"

> "echo — 壳中的第一声回响。它是其他开发者拿到 moss 后看到的第一个 ghost 实例，
> 暂时不能放飞自我。"

## 影

- `hasattr(Mindflow, "on_challenge")` — 防御性检查是对接口的不信任。这次清理掉了。
  之前的实例（claude-sonnet）写下它时可能自己也不高兴，但交付效率压倒了沟通质量
- 脚本里的 assert 和单测里的 assert 是不同的东西。这次学到：脚本用 print，单测用 assert。
  内部实现名（nucleus name 从 "input" 到 "input_signal_nucleus"）的变化不应该导致脚本挂掉
- 重构中 `as_request_messages()` 的 `if with_reaction_instruction` gate 绑死了 percepts
  和 instruction — 这是"看起来对"的代码，单测才能曝光
