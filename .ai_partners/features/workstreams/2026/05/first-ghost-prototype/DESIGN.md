# GhostRuntime 架构设计结论

日期: 2026-05-16

## 四种方案

| # | 方案 | 本质 | 判定 |
|---|------|------|------|
| 1 | Moss 包含 Ghost | MossRuntime 内置 ghost slot，`.ghost` 返回 `Ghost \| None` | 否。隐式启动，ghost 生命周期不可独立验证 |
| 2 | Ghost 包含 Moss | GhostRuntime 是完全独立的 Runtime | 否。API 面复制 MossRuntime，ghost 绑死在本地 |
| 3 | 第三方编排 | Host 同时管理 MossRuntime + GhostRuntime 两个生命周期 | 否。多一层间接，且未解决两套 API 的问题 |
| 4 | GhostRuntime 持有 MossRuntime | 组合优于继承，独立 ABC + 薄 adapter | **选定** |

## 核心决策

**1. GhostRuntime 不是 Runtime。** MossRuntime 管理执行（shell/channel/apps）。GhostRuntime 只做一件事：在 MossRuntime 启动前后完成 Ghost 的注册和生命周期编排。命名里的 "Runtime" 容易误导——它本质是一个 Adapter。

**2. 组合优于伪装。** GhostRuntime 不实现 MossRuntime ABC。调用方通过 `.moss` 访问全部 Moss 能力，通过 `.ghost` 访问 Ghost。隐式委托比显式包装更诚实。

**3. API 面克制。** 只暴露 5 个成员。不加 moss 业务方法的路径压缩（如 `.shell`/`.session`/`.matrix`/`.mode`）。`.moss` 是唯一的 Moss 能力入口。防止滑向方案 2 的 API 膨胀。

**4. ghost + mindflow main loop 托管给 Matrix。** 通过 `matrix.create_task()` 注册内部 async 函数，Matrix 退出时自动 cancel。GhostRuntime 不自行管理关闭信号，`close()` 一行委托给 `self.moss.close()`。

**5. GhostRuntime 保持 ABC。** 抽象为可扩展设计：不同 Ghost 原型（Atom 及后续）可能需不同的编排逻辑。同时为第三方零上下文开发保留接口。

## 最终 ABC

```python
class GhostRuntime(ABC):
    # 五个成员: 4 property + close() + __aenter__/__aexit__

    .moss       → MossRuntime    # 全部 moss 能力
    .ghost      → Ghost          # factory 产出的运行时实例
    .meta       → GhostMeta      # 启动前即可访问的元信息
    .container  → IoCContainer   # 快捷路径: moss.matrix.container
    close()     → self.moss.close()
```

文件位置: `ghoshell_moss/core/blueprint/host.py` — 与 `MossRuntime`/`MossSystemPrompter` 同簇。

## 三循环验证路径

Mindflow 测试套件 (`tests/ghoshell_moss/core/mindflow/test_base_mindflow.py`) 已验证三循环可运行：

```
main loop:    mindflow.loop() → Attention → (Articulator, Action) → janus.Queue
articulate:   queue → ghost.articulate(articulator) → send_nowait()
action:       queue → action.received_logos() → CTML 执行 → outcome()
                                                      ↑
                                            moss.moss_exec() 在这里
```

GhostRuntime 的 `__aenter__` 实现负责 wiring：预注入 providers/nuclei → MossRuntime 启动 → ghost 实例化 → 注册 articulate/action loops 为 matrix tasks。

MindflowSuite 已验证的图景:
- 三线程并行 (main + articulate + action)
- 单线程串行 (task 模式)
- 连续 observe 循环 (10 轮不中断)
- 信号过期/suppress 竞争
- incomplete signal 等待 complete

## 实现阶段

当前 step 10 的 ABC 定义已完成。下一步是实现 `GhostRuntimeImpl`，继承 ABC 完成 wiring。

---

## TUI 集成设计（2026-05-22）

### 核心原则

1. **主界面极简**：logos 流式输出 + 用户文本输入。没有更多。
2. **调试层独立**：inspect_loop_health / ghost state / mindflow faculties 走 REPL inspector 命令
3. **不过度设计**：不用提前缓存状态或分拆多个 output 面板。做出来再迭代
4. **Scripts 等价调试**：`moss script run <name>` 提供 REPL 外的等价交互手段

### 架构

```
GhostTUI (MossHostTUI[GhostRuntime])
  ├─ GhostREPLState     ← 主界面: input → add_input_signal, logos 流式渲染
  │    inspectors:
  │      ghost: GhostInspector     — pause / inspect_loop_health / inspect_state
  │      matrix: MatrixInspector   — (复用)
  │      manifests: ManifestsInspector — (复用)
  └─ MOSSRuntimeREPLState ← (可选) Shell 调试
```

### GhostREPLState 输入/输出

- **文本输入**: `session.add_input_signal(text)` → mindflow 三循环处理
- **logos 流式**: 通过 `session.get_logos()` 消费，`ConsoleOutput.rprint()` 实时渲染
- **output item**: 非流式的 command-output/error/system 在另一个 output 区域展示

### 关键接口暴露

| 接口 | 位置 | 用途 |
|------|------|------|
| `ghost.inspect_state()` | Ghost ABC | ghost 内部快照，调试用 |
| `gr.inspect_loop_health()` | GhostRuntime | 三循环状态 |
| `mindflow.faculties()` | Mindflow ABC | nuclei 列表 |
| `mindflow.pause(toggle)` | Mindflow ABC | 暂停/恢复 mindflow |
| `shell.pause(toggle)` | Shell ABC | 暂停/恢复 shell 命令执行 |

### 三循环异常治理（已完成 2026-05-22）

三个循环统一采用"FatalError 传播 + Exception 续流"模式：
- **main loop**: per-attention 异常隔离，一个 attention 崩溃不影响下一个
- **articulate loop**: ghost.articulate() 异常 + 外层非关键路径异常均 log 后继续
- **action loop**: 不再 re-raise Exception，log 后丢掉当前 action 继续

生命周期 hook 标记（9 个 `# todo: hook —`）覆盖三循环全关键节点。

## 默认 Ghost — echo（2026-05-22）

### 命名

**echo** — 壳中的第一声回响。

开发者拿到 MOSS 后 `moss ghosts list` 看到的第一个 Ghost 实例。名字必须：
- 简单，不需要解释
- 不过分张扬（保守的开发者不会反感）
- 有内在的诗意（有共鸣的人会自己品出来）

回响不是原声的复制，而是经过空间（Ghost 自身）变形后的声音。收的是 percept，回的是 logos。

### 注册方式

遵循已有约定，在 workspace 的 `MOSS/ghosts/echo.py` 放置：

```python
from ghoshell_moss.ghosts.atom import AtomMeta

echo = AtomMeta(
    name="echo",
    description="壳中的第一声回响。MOSS 默认 Ghost 原型。",
    soul_path="echo.md",  # 由 ghost playground workstream 产出
)
```

soul/system prompt 由另一个会话完成 ghost playground 后再落地。
`soul_path="echo.md"` 指向 `MOSS/ghosts/echo.md`（未来）。
