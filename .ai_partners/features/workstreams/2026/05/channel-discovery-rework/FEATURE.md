---
title: Channel Discovery Rework — __main__ 单一发现 + 移除 primitives manifest 类型
status: completed
priority: P0
created: 2026-05-25
updated: 2026-05-25
depends: []
milestone:
description: >-
  Channel 发现改为 FastAPI-like 的单一 __main__ channel 发现，primitives 作为独立 manifest 类型移除，
  改为在 channels.py 中直接注册到 main channel。零隐式逻辑。
---

# Channel Discovery Rework

## Motivation

### 当前问题

1. **发现和组装分离**：`PackageManifests.channels()` 扫描 `MOSS.manifests.channels` 找所有 Channel 实例 →
   `MossRuntime._bootstrap_after_matrix()` 逐个 `import_channels` 到 main。发现出来的 Channel 不知道自己会被挂到哪里。

2. **只覆盖一种组合模式**：channel-composer 已经确立了三种对 main 的加工维度 — `import_channels`（树挂载）、
   `with_state`（排他切换）、`with_module`（累积叠加）。当前发现机制只支持第一种。

3. **`create_ctml_main_chan` 含隐式逻辑**：硬编码 default_primitives、experimental flag、auto-add commands。
   最初设计是为了降低第三方接入的认知成本，但环境发现体系建立后，隐式逻辑变成有害的——它在 manifest 机制之外
   偷偷给 main 塞东西。

### 目标

- manifests 发现直接返回 main channel（如果定义了），而非返回 channel 列表再由 MossRuntime 接线
- FastAPI-like 风格：在工作空间里 `main = new_main_channel()` 然后直接在它上面组合
- 零隐式逻辑：primitives 也从 manifests 来，不走 `create_ctml_main_chan` 的 auto-discovery

## Design Index

- 当前 manifests 实现: `src/ghoshell_moss/host/manifests/__init__.py` — `PackageManifests.channels()`
- 当前 channel 扫描: `src/ghoshell_moss/host/manifests/channels.py` — `search_channels_from_package()`
- 当前 main channel 构造: `src/ghoshell_moss/core/ctml/shell/ctml_main.py` — `create_ctml_main_chan()`
- MossRuntime 集成点: `src/ghoshell_moss/host/moss_runtime.py:209` — `import_channels` 调用处
- Shell 构造: `src/ghoshell_moss/core/ctml/shell/ctml_shell.py:542` — `new_ctml_shell()`
- Facade: `src/ghoshell_moss/__init__.py:44` — 暴露 `create_ctml_main_chan`
- Workspace stub: `src/ghoshell_moss/host/stubs/workspace/src/MOSS/manifests/channels.py`
- Manifests CLI: `src/ghoshell_moss/cli/manifests_cli.py:308`

## Key Decisions

### K1: 发现语义 = 替代而非叠加

**决策**: manifests 里有 `__main__` → 完全用它。没有 → fallback 到默认构造。不存在 "叠加"（在默认基础上加工）。

**Why**: 叠加就是隐式逻辑。用户不知道默认 main 里有什么，也不知道自己的加工会不会和默认冲突。
FastAPI 的 `app = FastAPI()` 不会有人在背后偷偷给你塞 middleware。

**被拒绝**: 叠加模式（manifest 的 main 是对默认 main 的附加操作）。

### K2: 新构造函数名 `new_main_channel()`

**决策**: 在 facade 层 (`ghoshell_moss/__init__.py`) 新增 `new_main_channel()`，与现有
`new_channel()`、`new_prime_channel()` 命名一致。废弃 `create_ctml_main_chan`。

**被拒绝**: `new_root_channel()` — root 是通用的树结构术语，`__main__` 才是 MOSS 的特定命名。

### K3: 构造函数极简，零隐式逻辑

**决策**: `new_main_channel()` 只创建 `PyChannel(name="__main__", blocking=True)`。
不自动添加 primitives。primitives 通过 manifest 的 primitives 机制进入。

**Why**: MossRuntime 已经在从 `manifests.primitives()` 获取 primitives 并传给 shell。
main channel 的构造和 primitives 的注册是两个独立的关注点。

### K4: manifests 文件名保留 `channels.py`

**决策**: workspace stub 下的 `MOSS/manifests/channels.py` 不重命名。它仍是 channels 的 manifest 入口。

**Why**: 只是发现的内容从 "所有 Channel 实例" 变成了 "命名为 `__main__` 的那个 Channel"。
入口点语义没变。

### K5: Mode 的 `__main__` 完全覆盖全局

**决策**: 当 mode manifests 中存在 `__main__` 时，它完全替换全局 manifests 的 `__main__`（而非合并子 channel）。

**Why**: `MergedManifests` 的 `self._channels.update()` 已经天然实现右侧覆盖的语义。
但需要显式声明：这不是 "在全局 main 上叠加 mode channel"，而是 "mode 定义了自己的完整 main"。
这给了 mode 完全控制自己 shell 的能力结构的能力。

### K6: workspace stub 的 `channels.py` 加注释说明

**决策**: stub 中的 `channels.py` 改为 FastAPI-like 风格后，需要在文件顶部加简短注释，
解释这种用法模式，让用户知道可以在这里定义 main channel 并做 import_channels / with_state / with_module。

## 实际变更

### 文件变更清单

| 文件 | 变更 |
|---|---|
| `core/blueprint/states_channel.py` | 新增 `new_main_channel()` (L278-289)，加入 `__all__` |
| `core/blueprint/manifests.py` | ABC `explain()` 中 channels 行改为描述 `__main__` 发现 |
| `ghoshell_moss/__init__.py` | facade 导出 `new_main_channel` |
| `host/moss_runtime.py` | `__init__` 中从 manifests 取 `__main__` 或 fallback；`_bootstrap_after_matrix` 去掉 import_channels 循环 |
| `host/manifests/__init__.py` | `PackageManifests.explain()` 改为描述目录约定和发现规则；`MergedManifests.explain()` 改为描述合并规则 |
| `host/stubs/workspace/.../channels.py` | FastAPI-like 风格重写，包含 primitives 注册 |
| `host/stubs/workspace/.../primitives.py` | **删除** |
| `host/stubs/workspace/.../modes/system_test/channels.py` | **新建** |
| `host/stubs/workspace/.../modes/system_test/primitives.py` | **删除** |
| `host/manifests/primitives.py` | **删除** |
| `core/blueprint/host.py` | ABC `run()` 移除 `with_primitives` 参数 |
| `host/impl.py` | `run()` 移除 `with_primitives` 传递 |
| `cli/manifests_cli.py` | 移除 `primitives` 命令 |

### 关键实现细节

- `new_main_channel()` 定义在 `states_channel.py`，与 `new_prime_channel` 并列。只创建 `PyChannel(name="__main__")`，零隐式逻辑
- `manifests.channels()` API 不变，仍返回 `dict[str, Channel]`。MossRuntime 从 dict 中取 `"__main__"` key
- `create_ctml_main_chan` 未删除，CTMLShell 仍保留其 fallback 路径，但 MossRuntime 始终传 main_channel 所以不再命中
- fractal_hub 的 `import_channels` 保留不动 — 它是运行时动态组件，不属于 manifests 发现
- explain 机制重构为"描述规则而非自述类身份"，后续文档治理时进一步完善
- **Primitives 移除**：从 manifest 类型中彻底移除，ABC 保留默认 `return {}`。原语改为在 `channels.py` 中通过 `main.build.add_command()` 注册。Mode 的 `__main__` 完全覆盖全局，mode 需显式声明自己需要的全部原语。

- `new_main_channel()` 定义在 `states_channel.py`，与 `new_prime_channel` 并列。只创建 `PyChannel(name="__main__")`，零隐式逻辑
- `manifests.channels()` API 不变，仍返回 `dict[str, Channel]`。MossRuntime 从 dict 中取 `"__main__"` key
- `create_ctml_main_chan` 未删除，CTMLShell 仍保留其 fallback 路径，但 MossRuntime 始终传 main_channel 所以不再命中
- fractal_hub 的 `import_channels` 保留不动 — 它是运行时动态组件，不属于 manifests 发现
- explain 机制重构为"描述规则而非自述类身份"，后续文档治理时进一步完善
