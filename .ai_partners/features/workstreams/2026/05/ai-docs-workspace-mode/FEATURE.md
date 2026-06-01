---
created: 2026-05-21
depends: []
description: 为 docs/ 撰写 workspace 与 moss mode 的系统论述文档，并在过程中修复实现不一致。是架构拓扑文档后最重要的一篇
  docs。
milestone: null
priority: P0
status: completed
status_note: '2026-06-02: 文档修正、stub注释、howto全部完成。开放问题留给未来查阅。'
title: Ai Docs Workspace & Mode
updated: '2026-06-02'
---

# Ai Docs Workspace & Mode

> Use `moss features set-status ai-docs-workspace-mode <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

`moss docs ai/` 目前有三篇，拓扑文档中 workspace（2.4 节）仅一段概述，mode 没有独立章节。
`moss how-tos get-moss-design/what-is-workspace.md` 是从设计文档降级为 how-to 的过渡产物，
需要正式拆分为：架构论述进 docs/，操作步骤留在 how-tos（后续重写）。

这篇文档是 MOSS 项目中最重要的架构文档之一——workspace 和 mode 是智能模型进入项目的第一个接触面。

## Key Decisions

### 1. docs 与 how-tos 的边界

- **docs**：系统论述。面向需要理解架构设计的智能模型，回答 what-is / why / 知识探索路径 / 工具与意义。
- **how-tos**：零知识引导。面向没有背景知识的模型（没人预训练 MOSS），带盲从心态 step by step 干活。

`what-is-workspace.md` 在本 feature 完成后删除，内容拆分到两边。

### 2. 智能模型是第一读者

文档围绕"智能模型作为 MOSS 环境的第一开发者和讲解者"来写。三个核心能力诉求：

1. **最小知识理解运行时**：配合工具（manifests / modes / apps），不需要调研复杂生命周期就能建立认知，可以向人类讲解
2. **方便修改**：文件级关注点分离——不同文件治理不同东西。package-module 等发现机制让修改路径可预测。stub 文件需要加头部注释做自解释
3. **开发时/运行时自迭代**：必要的隔离（尤其修改隔离）+ 通过工具反查必要知识

### 3. 贯穿五个用户故事

1. 了解 MOSS 已集成的能力 → 模型通过 manifests / modes / apps 快速理解
2. 开发新功能 → 模型知道如何独立集成（目录拓扑介绍是关键）
3. 知道启动什么 → 模型通过 modes 等命令看一眼就明白
4. 两条开发路径：
   - app → mode 集成（依赖/运行时隔离）
   - src → manifests 集成（依赖/运行时复用）
5. 从最佳实践上手，到必要知识扩展——不全貌到执行，先做事再理解

### 4. 文档结构共识

what-is → why → 用户故事 → 核心机制（自举层/声明层/视图层） → 启动链路 → 知识探索路径 → 相邻概念关联

contracts、providers、nucleus、ghost 等概念各有一句话 + 深入探索路径，不展开但不断链。

### 5. 需要对齐的代码改动

- stub 文件添加头部注释做自解释（本轮任务一部分）
- 文档写作过程中发现的不一致，在 feature 内记录并修复，或新建子任务
- 这是一个长 feature，可以反复对齐

### 6. CLI 重组不影响本文档

CLI 命令归类重组与文档写作并行进行，各自独立。文档引用 Python import path 和概念为主，CLI 命令路径改动后更新"知识入口"部分即可。

## Design Index

- 上游参考：`moss docs read architecture-topology.md` 2.4 节
- 待删除：`moss how-tos get-moss-design/what-is-workspace.md`
- 关键源码：
  - `ghoshell_moss.core.blueprint.environment` — Environment 发现与 MossMeta
  - `ghoshell_moss.core.blueprint.manifests` — Manifests 声明体系
  - `ghoshell_moss.host.manifests` — Host 层 manifests 实现
  - `ghoshell_moss.host.impl:Host` — Host 启动链路
  - `ghoshell_moss.host.stubs.workspace` — workspace stub 模板
  - `ghoshell_moss.host.moss_runtime:MossRuntimeImpl` — Runtime 执行层

## 2026-06-02 全面调研结果

通过 CLI 命令 + 源码交叉验证，对照文档发现以下不一致和待修项。

### 1. Primitives 已从 manifest 体系移除，文档未同步

**CLI 验证**：`moss manifests primitives` → `Error: No such command 'primitives'`

**源码验证**：`moss manifests explain` 明确输出 "Primitives 已移除。原语不再作为独立 manifest 类型。改为在 channels.py 中直接通过 main.build.add_command() 注册到 __main__ channel。"

**文档影响**：
- `workspace-and-mode.md` section 3.3 目录拓扑树中仍列着 `primitives.py`（应删除）
- section 4.2 仍写"八个领域"（实为七个）

### 2. 当前 7 大 manifest 声明类型（CLI 交叉验证）

| 类型 | 路径 | isinstance 检测 | 键语义 | CLI 命令 |
|------|------|----------------|--------|---------|
| **providers** | `MOSS.manifests.providers` | `Provider` | `contract()` 的 import path | `moss manifests providers` |
| **channels** | `MOSS.manifests.channels` | `Channel` + `name == "__main__"` | `"__main__"` | `moss manifests channels` |
| **configs** | `MOSS.manifests.configs` | `ConfigType` | `ConfigType.conf_name()` | `moss manifests configs` |
| **topics** | `MOSS.manifests.topics` | `issubclass(obj, TopicModel)` 或 `isinstance(obj, TopicSchema)` | `topic_name` | `moss manifests topics` |
| **resources** | `MOSS.manifests.resources` | `ResourceStorageMeta` | `{scheme}:{host}` | `moss manifests resources` |
| **nuclei** | `MOSS.manifests.nuclei` | `NucleusMeta` | `NucleusMeta.name()` | `moss manifests nuclei` |
| **CTML versions** | `ctml_versions/` 目录 | 扫描 `.md` 文件 | filename as version | `moss manifests ctml-versions` |

所有类型共享发现模式：`scan_package(约定路径, max_depth=2)` → `isinstance` 过滤 → 按类型特定键聚合。

`contracts` 不是独立的 manifest 扫描类型 —— `moss manifests contracts` 通过查询 IoC 容器的已绑定列表生成，
不扫描任何 `contracts.py` 文件。Mode 目录下的 `contracts.py` 是 mode 专属 Provider 的存放位置（语义上是
providers 的扩展），但当前 manifests 扫描只走 `providers` 子包。

### 3. Manifest 类型按性质分为三类（2026-06-02 澄清）

**工厂类型**（生产运行时实例，有副作用）：
- **providers** — `Provider.factory(container) → service`，注入 IoC 容器
- **nuclei** — `NucleusMeta.factory(container) → Nucleus`，生产感知核实例
- **ghosts** — `GhostMeta.factory(container) → Ghost`，生产 Ghost 实例（不在 manifests 体系中，在 `MOSS.ghosts` 独立发现）

**混合类型**（schema + 持久化 + 内存覆盖）：
- **configs** — 两种独立机制：
  1. `ConfigType` 子类 = schema + 文件路径绑定。`ConfigStore.get_or_create()` 优先从 `workspace/configs/{conf_name}.yml` 读文件，不存在才用传入实例做默认值。这是"文件优先"的持久化配置。
  2. `ConfigStore.set_config(conf, override=False)` = 纯内存覆盖，不写磁盘。mode 用此机制创建 mode 专属配置。
- **resources** — `ResourceStorageMeta` 声明资源存储的 scheme/host，`factory()` 生产 `ResourceStorage`

**协议类型**（被动声明，无副作用）：
- **topics** — `TopicModel` 子类（类本身即协议声明）或 `TopicSchema` 实例。定义"这个消息类型在总线上合法"
- **channels** — `Channel` 实例，声明能力树的结构

> 协议类型当前无运行时 enforcement。声明了也不会被系统强制执行——除非开发者在通信层加约束。
> 即使 `scan_package` 做子类发现，也无法在静态时期发现所有 TopicModel 子类（可能定义在任意模块中）。

### 4. topics 和 signals 的边界（2026-06-02 确认）

- `InputSignal` 属于 `SignalMeta` 继承链，不属于 `TopicModel`。`topics.py` 中 import `InputSignal` 是 bug。
- Signal 声明（`SignalMeta` 子类）和 Topic 声明（`TopicModel` 子类）是独立的协议类型，不应混合扫描。
- Signal 当前没有独立的 manifest 扫描入口——只在 `nuclei.py` 中作为 `NucleusMeta.signals()` 的返回值被引用。

### 5. 合并机制确认（源码验证）

`MergedManifests.__init__`（`host/manifests/impl.py:205-222`）：

```
configs    → dict.update (右边覆盖)
providers  → list.extend (追加)
topics     → dict.update (右边覆盖)
channels   → mode 的 __main__ 完全替换全局 (K5)
resources  → dict.update (右边覆盖)
nuclei     → dict.update (右边覆盖)
CTML       → dict.update (右边覆盖)
```

### 4. Ghosts 与 Modes 发现机制结构一致

**Modes** (`host/modes.py`):
```
scan_package(MOSS.modes, max_depth=1)  → 找到子包
  find_mode_from_package():
    1. module 属性 mode / __mode__  (Python Mode 实例)
    2. MODE.md 解析 YAML frontmatter
    3. 都没有 → 自动生成最小 Mode (约定优于配置)
  结果: mode.with_manifest(PackageManifests(...))
```

**Ghosts** (`host/ghosts.py`):
```
scan_package(MOSS.ghosts, max_depth=1)  → 找到子包
  find_ghost_from_package():
    1. module 属性 ghost / __ghost__  (GhostMeta 实例)
    2. 兜底: 第一个 isinstance(obj, GhostMeta)
  结果: {ghost_meta.name(): (ghost_meta, module_manifest)}
```

两者结构高度一致：`scan_package → find_X_from_package(约定属性 → 兜底)`。Modes
多一层 `MODE.md` 解析和 auto-generate 回退。文档 section 9 仅提到 ghost
但没有描述发现机制，应补充一致性的说明。

### 5. Stub 文件注释状态

**全局 manifests stubs** (`stubs/workspace/src/MOSS/manifests/`)：

| 文件 | 注释状态 |
|------|---------|
| `channels.py` | ✅ 完整中文自解释 |
| `providers.py` | ❌ 无注释（仅 imports + 实例） |
| `configs.py` | ❌ 无注释 |
| `topics.py` | ❌ 无注释 |
| `resources.py` | ❌ 无注释 |
| `nuclei.py` | ⚠️ 英文 docstring，非中文自解释注释 |

**Mode 创建模板** (`stubs/mode/`)：所有文件已有基本中文注释，状态良好。

**Mode default stub** (`stubs/workspace/src/MOSS/modes/default/`)：所有文件已有基本中文注释。

### 6. topics.py stub 内容问题

`.moss_ws` 和 stub 中的 `topics.py` 只有 `from ghoshell_moss.core.blueprint.mindflow import InputSignal`，
没有实例化任何 `TopicModel` 子类或 `TopicSchema` 实例。扫描结果：`moss manifests topics` 返回 0 条。
这个 stub 没有示范作用——应该有一个最小示例 TopicModel 子类定义。

### 7. 路径变更

文档从 `cli/docs/ai/workspace-and-mode.md` → `cli/docs/workspace-and-mode.md`（CLI 重构中 flatten: ec81171）。
FEATURE.md 的 Implementation Notes 中路径仍是旧的。

## Discovered Issues (历史，已处理)

### 1. ~~`search_channels_from_package` key 不一致~~ (已过时)

原 `search_channels_from_package` 以 Python 变量名为键，与 primitives 的 `Command.name()` 不一致。
该函数已 deprecated，统一使用 `search_main_channel_from_manifest` 按 `name == "__main__"` 过滤。
Primitives 已完全移除，此问题不再相关。

### 2. Manifests 自解释体系（已实现）

- `Manifests.explain()` — 基类通用模板 + PackageManifests/MergedManifests 各自组装
- `moss manifests explain` — CLI 唯一真相入口，接受 `--mode`
- 文档 4.2 节引用 CLI 而非硬编码表格

### 3. providers vs contracts 关系（已写入文档 4.2 节）

- providers 服务于 contracts — 添加 Provider 声明 contract 的绑定方式
- `moss manifests contracts` 看"可以拿到什么"（IoC 已绑定列表）
- `moss manifests providers` 看"由谁生产"（工厂声明）

## TODO (不随 commit 带入)

### 文档修正 (2026-06-02 ✅)
- [x] section 3.3 目录拓扑树中删除 `primitives.py`
- [x] section 4.2 "八个" → "七个"，按工厂/混合/协议三类重新分类
- [x] configs 修正为"两种独立机制"（文件持久化 + 内存覆盖）
- [x] nuclei 从"声明"修正为"工厂"
- [x] section 5.3 补充 ghost 发现机制与 mode 的结构一致性
- [x] section 9 ghost 补充工厂性质和发现一致性

### Stub 注释 (2026-06-02 ✅)
- [x] 全部 5 个文件补齐中文自解释注释 + `.moss_ws` 同步

### How-to (2026-06-02 ✅)
- [x] 删除 `get-moss-design/what-is-workspace.md`
- [x] 创建 `for-moss-app-developer/how-to-register-manifests.md`
- [x] 创建 `for-moss-app-developer/how-to-discover-environment.md`

### 待人类审阅
- `workspace-and-mode.md` 措辞、节奏、详略
- 两个新 howto 的结构和内容

### 开放问题（需要人类决策，不在本轮处理）
1. 协议类型（topics）是否需要运行时 enforcement？
2. Signal 是否需要独立 manifest 扫描入口？
3. NucleusMeta 是否应支持裸类注册（issubclass）？

## Implementation Notes

- 文档路径：`src/ghoshell_moss/cli/docs/workspace-and-mode.md`
- 初稿已完成，人类审阅对齐中
- 文档内容直接面向 AI 读者，写中文
- CLI `moss manifests explain` 是 manifests 体系的权威真相入口，文档引用它而非硬编码表格