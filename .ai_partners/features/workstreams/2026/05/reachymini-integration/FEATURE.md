---
title: Reachy Mini Integration — 完整 workspace 集成与测试体系
status: completed
priority: P1
created: 2026-05-29
updated: 2026-06-02
depends: []
milestone:
description: >-
  将 moss_in_reachy_mini 作为 mode channel 集成到 .moss_ws，MCP 端到端验证通过。
  后续：修复音频同步/中断问题，建立 app 级别隔离。
---

# Reachy Mini Integration

## Motivation

`reachy-mini-contrib` 完成了基础运控/视觉的 contrib 包封装。本 workstream 将 Reachy Mini 通过 mode channels 集成到 workspace，通过 MCP 端到端验证控制链路，记录发现的问题，最终迁移到 app 级别隔离。

## 2026-05-29 Session — Mode 集成 + MCP 端到端验证

### 完成项

- **Workspace 入库**：`.moss_ws/` 正式 git 跟踪，根 `.gitignore` 移除排除规则。手动维护。
- **依赖治理**：砍掉 `insightface`/`onnxruntime`/`gstreamer` 三个未引用包；清掉 gstreamer 自定义 PyPI 索引和无效 `[tool.uv] override-dependencies`；`huggingface-hub` 版本冲突用 `override-dependencies` 临时解决（reachy-mini 所有版本 pin `==1.3.0`，pydantic-ai 要 `>=1.3.4`）。
- **reachymini mode 创建**：`src/MOSS/modes/reachymini/`，`channels.py` 用 `main.import_channels(ReachyMiniChannelCreator(...).factory)` 延迟实例化。
- **MCP 全链路验证通过**：`moss-as-mcp --mode reachymini` → CTML Shell → reachy_mini channel → Reachy Mini SDK → robot。28 命令复杂 CTML 表演流畅执行。
- **Stubs gitignore 补齐**：`app/`、`scripts/`、`_system_tests/`、`ghosts/` 的 `.gitignore` 和占位文件。

### 发现的问题

#### 1. Audio Player 无阻塞/不同步（当前焦点）

语音播放器 (`ReachyMiniStreamPlayer`) 不阻塞 channel，不同步于 CTML 调度。表现：
- `<say>` 命令立刻返回，不等音频播完
- 语音和其他动作没有时序协调
- 语音不能被 interrupt

可能的根因：`__content__` → `SpeechChannelModule` → `ReachyMiniStreamPlayer` 的链路中缺少 blocking 语义。需要检查 Speech 模块的调度集成。

#### 2. ReachyMini() 构造即连接

`ReachyMiniChannelCreator.factory()` 里 `ReachyMini()` 构造时立刻 HTTP 连接 daemon。如果 daemon 未启动，`ConnectionError` 直接抛在 channel runtime 初始化阶段。应该把硬件连接延迟到 `bootstrap()` 生命周期。

#### 3. Matrix 启动失败时错误传播不完整

channel factory 失败 → `StatefulChannelRuntime.__init__()` 异常 → `CTMLShell.__aenter__` 失败。但 `MatrixImpl.__aexit__` 只 log 不 re-raise，且 `_start_and_close_ctx` 的 finally 块 `except Exception` 静默吞错。MCP/repl 在 channel 启动失败后仍保持连接不退出。

#### 4. Vision Context 图片注入非多模态模型 — 疑似框架 bug

`Vision.context_messages()` 返回 PIL Image，在 `moss_dynamic` 中序列化为 base64。deepseek-v4 不支持图片输入，但这些 base64 数据仍然出现在 context 中（534k 字符纯文本）。**这不是 reachymini 的问题** — 框架层 `moss_dynamic` 组装时没有按模型能力过滤 context 内容类型。应该在上游（Shell/Matrix 组装 context 处）根据目标模型的多模态能力决定是否注入图片数据。

#### 5. Mode import 失败时缺乏诊断

当 `reachy-mini` SDK 未安装时，`channels.py` import 失败，mode 静默降级到全局 main channel。`moss modes list` 应该在 mode 发现阶段做 import 检查并 report 错误，而不是让运行时静默失败。

#### 6. macOS pycairo 编译依赖

`reachy-mini` 传递依赖 `pygobject` → `pycairo`（meson 编译），需要系统 cairo + fontconfig + freetype2 + bzip2。macOS 上 `/usr/local/bin/pkg-config`（Intel 残留）不认 homebrew 路径。解决：`PKG_CONFIG=/opt/homebrew/bin/pkg-config`。

#### 7. 依赖体系需要分层治理

`host` extras 包含 `pydantic-ai`（AI 调用），`reachy_mini` extras 包含 `reachy-mini`（硬件 SDK）。两者通过 `huggingface-hub` 版本冲突死锁。且 `reachy-mini` 传递依赖 `pygobject` → `pycairo`（meson 编译），在 macOS 上需要 `PKG_CONFIG` 环境变量才能编译，直接破坏了 pre-commit hook（features-check 重建 venv 时必挂）。

**已执行**：`reachy_mini` extras 从 pyproject.toml 移除（注释保留），`[tool.uv] override-dependencies` 随之移除。Mode `channels.py` 的 import 链不触发 SDK import，仅在 runtime 调用 factory 时才需要。长远方案：
- reachy_mini 作为独立 app（独立 venv），不污染核心开发环境
- `pygobject`/`pycairo` 等系统级编译依赖不应在核心环境中出现
- host 应瘦身，重量级依赖下沉到 app 级别

#### 8. ReachyMini 运行时数据路径不规范

`Body.__init__` 通过 workspace 加载表情数据，自动从 HuggingFace 下载到本地目录。但当前路径未纳入 `workspace.runtime/` 约定。下载的表情库、dance 数据等应落在 `workspace.runtime/` 下并按约定管理，而非任意本地路径。

### Key Decisions

- **Mode 先于 App**：先用 mode channels 直接集成做快速验证，验证通过后再拆成独立 app 隔离重量级依赖。
- **`import_channels(factory)` 模式**：利用 `ChannelFactory = Callable[[IoCContainer], Channel]` 签名，将 `ReachyMiniChannelCreator.factory` 直接注册到 main channel builder，延迟到 runtime 启动时实例化。
- **Vision 暂时关闭**：视觉数据对非多模态模型的 context 污染问题需要框架层解决，不是 reachymini 特有问题。

### 下一步

1. **Scope 语法异常** — `<_>` scope 在 app channel 上大量 cancelled，不加 scope 正常。scope 开关未正确传递
2. 音频播放器阻塞/同步/中断
3. ReachyMini 硬件连接延迟到 bootstrap
4. Matrix 错误传播加固
5. 框架层 context 内容类型过滤
6. `host` extras 瘦身：`pydantic-ai` 迁到 `ghost` extras
7. 运行时数据路径纳入 `workspace.runtime/` 约定
8. Mode import 诊断

## 2026-05-30 Session — App 独立 venv 迁移 + 端到端验证

### 完成项

- **App 独立 venv 落地**：`.moss_ws/apps/reachy_mini/body/` 作为独立 app 运行，拥有自己的 `pyproject.toml` + `.venv` (Python 3.13, 195 packages)
- **依赖隔离**：`reachy-mini` SDK 及重量级传递依赖 (`pygobject`→`pycairo`, `gstreamer_python` 等) 隔离在 app venv 中，核心开发环境干净
- **全链路验证通过**：MOSS Host → Circus `uv run` → Matrix → Zenoh channel proxy → CTML 命令执行，端到端通畅
- **`_app_to_circus_params` bug 修复**：`args_list` 被 `[self._get_app_script(app)]` 覆盖，导致 `uv main.py` 而不是 `uv run main.py`。改回使用 `_get_app_executable()` 返回的完整 `args_list`
- **`ReachyMiniChannelCreator` 清理**：去 Matrix/Workspace 双路径，直接 `force_fetch(Matrix)`
- **`provide_channel()` 可复用入口**：放到 `ghoshell_moss_contrib.moss_in_reachy_mini.main`，app `main.py` 薄到 3 行
- **`.env` + `dotenv` 模式**：app 目录下 `.env` 管理环境变量，`main.py` 顶部 `load_dotenv()` 在任何 import 前加载
- **macOS GStreamer 兼容**：pip 安装的 `gstreamer_python` 在 uv venv 中 GObject 类型系统初始化失败（`g_type_get_qdata: assertion 'node != NULL'`），最终 segfault。默认使用 `media_backend='no_media'` 跳过本地相机，通过 `REACHY_MEDIA_BACKEND` 环境变量可配置
- **NumPy macOS 14 兼容**：NumPy 2.4.x 引用了 macOS 15 的 ILP64 Accelerate 符号，pin `numpy<2.3` 解决

### 发现的新问题

- **Scope 语法异常**：不加 `<_>` scope 时所有命令正常执行（11 done, 0 cancelled），加 scope 后大量 cancelled。不是音频同步问题，scope 开关本身未正确传递到 app channel
- **`pydantic-ai` 在 `[host]` extras 中**：app 不需要它（属于 ghost 依赖），当前 app 带上了不必要的 `pydantic-ai` → `host` extras 待瘦身

### Key Decisions

- **App 模式优于 Mode Channel**：通过 Zenoh proxy 通讯，app 进程独立管理生命周期，依赖隔离，可单独重启
- **`no_media` 为默认**：macOS 上 GStreamer 兼容问题暂不解决，等 pip 包修复或换 conda env
- **`.env` 在 app 目录**：app 自行管理环境变量，不依赖 host 传递

## 第一步：Workspace 入库

`.moss_ws/` 正式纳入 git 管理。这是一个长期手动维护的 workspace，不通过 stub 模板生成。

根 `.gitignore` 移除 `.moss_ws/` 排除规则。

## 2026-06-02 Session — 收尾审查 (review by deepseek-v4)

逐项验证 05-29 列出的 8 条"下一步"，确认哪些已通过后续提交修复、哪些已弱化、哪些应拆为独立 feature。

### 已解决

| # | 条目 | 解决提交 |
|---|------|----------|
| 1 | Scope 语法异常 | 79349e3 (scope command 传递) + 35e299e (duplex ordering + cancel sync) |
| 2 | 音频播放器阻塞/同步 | 35e299e (`__content__` 显式 `blocking=True`)；`say` 默认 `blocking=True`；`SpeechStream.say()` 内 `await wait_played()` 本质阻塞；32e901a (MiniAudioStreamPlayer.clear 中断) |
| 6 | host extras 瘦身 | 5d36f66 (`pydantic-ai` + `anthropic` → `[ghost]` extras；speech/zenoh 从 core 迁 host) |
| 7 | 运行时数据路径 | 79349e3 + 3e3e8c8 (`matrix.cell_workspace` 提供 cell 级隔离，落 `workspace.runtime()/cells/`) |

### 已弱化

| # | 条目 | 原因 |
|---|------|------|
| 8 | Mode import 诊断 | reachymini 已从 mode channel 迁移到 app 模式，不再依赖 mode 发现 |

### 仍残留（minor）

| # | 条目 | 说明 |
|---|------|------|
| 3 | ReachyMini() 构造即连接 | daemon 未启动时才暴露，不影响正常链路。后续改进：连接延迟到 `bootstrap()` |

### 拆为独立 feature（框架层）

| # | 条目 |
|---|------|
| 4 | Matrix 错误传播加固 |
| 5 | 框架层 context 内容类型过滤（`allow_vision=False` 临时方案已在 32e901a 加入） |

### 结论

核心目标（workspace 集成 + app 独立 venv 隔离 + MCP 端到端验证）全部达成。残留的 #3 是 minor improvement，不影响收尾。#4/#5 属于框架层问题，应由各自 feature 跟踪。
