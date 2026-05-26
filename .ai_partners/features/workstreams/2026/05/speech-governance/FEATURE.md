---
title: Speech Governance — 解耦、多后端、容错降级
status: in-progress
priority: P2
created: 2026-05-25
updated: 2026-05-26
depends: []
milestone:
description: >-
  Speech 体系治理：解耦 commands 权责泄漏，player 多后端轻量化，TTS 国际化多 provider，
  自动容错降级，session logos stream 可选的跨进程流式。
---

# Speech Governance

## Motivation

当前 MOSS 的 speech 体系有三个结构性问题，且彼此交织：

1. **权责泄漏**：`contracts/speech.py` 中的 `make_content_command_from_speech()` 和 `TTSSpeech.commands()` 直接生成 `Command` 对象。`CTMLShell._speech_context_manager()` (ctml_shell.py:185-203) 将 speech 以 command 形式注入 main channel。Speech 知道了 shell 的调度模型——这是反向依赖。

2. **Player 单点且老旧**：默认 audio player 只有 PyAudio（`depend_pyaudio()` in depends.py）。PyAudio 依赖 PortAudio C 库，安装经常失败，且 API 设计停留在 2000 年代。PulseAudio 实现只覆盖 Linux。没有一个跨平台、零系统依赖的轻量默认。

3. **TTS 单 provider + 国内锁定**：`TTSServiceProvider` 的 `use` 字段是 `Literal['volcengine_stream_tts_model']`，硬编码只有一个选项。`.env.example` 只提供火山引擎的凭证模板。国际用户无法使用。

4. **无容错降级**：如果 TTS API 挂了或 player 打不开，整个 speech 链路直接崩溃，没有 fallback 到 mock/文本输出的路径。

**时机**：zenoh fractal 改造已完成（`zenoh-fractal` feature），channel 可以通过 manifests 体系被动发现和集成。这为 speech channel module 化提供了基础设施——speech 的各种组合方式可以被 manifests 自动发现并集成到 shell，不需要 CTMLShell 硬编码 `_speech_context_manager`。

## Design Index

- 基准抽象: `src/ghoshell_moss/contracts/speech.py` (Speech, SpeechStream, StreamAudioPlayer, TTS, TTSBatch, TTSSpeech)
- 当前实现: `src/ghoshell_moss/core/speech/` (mock, stream_tts_speech, player/*, volcengine_tts/*)
- Shell 耦合点: `src/ghoshell_moss/core/ctml/shell/ctml_shell.py:_speech_context_manager`
- TTS provider: `src/ghoshell_moss/host/providers/tts_service_provider.py`
- 依赖声明: `src/ghoshell_moss/depends.py`
- 环境配置: `src/ghoshell_moss/host/stubs/workspace/.env.example`
- Session stream: `src/ghoshell_moss/core/blueprint/session.py` (logos stream protocol)
- Channel 分形参考: `zenoh-fractal` feature (Hub/Provider 分离 + manifests 自动发现)
- pyproject.toml: `src/ghoshell_moss/pyproject.toml` (optional dependency groups)

## 实施进度

### Phase 2: Player 轻量化 (P1) — DONE (2026-05-26)

**完成内容**:

1. `miniaudio>=0.67` 加入核心依赖 (`pyproject.toml` dependencies)，零系统依赖，跨平台一致
2. 新增 `MiniAudioStreamPlayer` (`core/speech/player/miniaudio_player.py`)，实现 `BaseAudioStreamPlayer` 的三个抽象方法
3. `AudioPlayerProvider` 替代 `PyAudioPlayerProvider`，默认 backend 为 `miniaudio`，可通过 `AudioPlayerConfig.backend: "pyaudio"` 切换
4. PyAudio 仍可通过 `pip install ghoshell_moss[audio]` 安装，惰性导入，切换 backend 即可使用
5. 10 个单元测试覆盖：生命周期、格式转换、重采样、流式播放、clear、幂等性

**miniaudio 1.x 适配要点**:
- miniaudio 1.71 使用 generator 模式：`PlaybackDevice.start(gen)` 接受 callback generator
- 内部线程通过 `gen.send(frame_count)` 请求精确帧数，generator 必须返回恰好 `frame_count * channels * 2` 字节
- 实现内部 buffer `_buf` 机制：队列数据先入 buffer，按需切分 yield，多余留在 buffer 供下次请求
- 数据不足时用静音补齐，避免 underrun 噪声

**变更文件**:
| 文件 | 变更 |
|------|------|
| `pyproject.toml` | 核心依赖新增 `miniaudio>=0.67` |
| `core/speech/player/miniaudio_player.py` | **新文件** — MiniAudioStreamPlayer |
| `core/speech/player/__init__.py` | 导出 MiniAudioStreamPlayer |
| `core/speech/__init__.py` | `make_baseline_tts_speech()` 默认改用 MiniAudioStreamPlayer |
| `host/providers/audio_player_provider.py` | AudioPlayerProvider (miniaudio 默认 + pyaudio 可切换) |
| `host/stubs/workspace/src/MOSS/manifests/providers.py` | 引用 AudioPlayerProvider |
| `host/stubs/workspace/src/MOSS/manifests/configs.py` | 引用 AudioPlayerConfig |
| `tests/ghoshell_moss/speech/test_miniaudio_player.py` | **新文件** — 10 个单元测试 |

**Player 选型结论**:
- **miniaudio**: 零系统依赖（libvorbis/libopus/libmp3 内置），纯 wheel 安装，跨平台一致，解码+播放一体
- **PyAudio**: 降级为可选，通过 `audio` extras 安装，config 切换 backend 即可使用
- **PulseAudio**: Linux only，保持现状

### 剩余 Phase

```
Phase 2 ✅ → D6 (docstring 示例) → Phase 1 (解耦) → Phase 3 (多 provider) → Phase 4 (降级)
```

## 调研结论 (2026-05-25)

### 问题 1: Speech → Command 的权责泄漏

**现状**：

`CTMLShell._speech_context_manager()` 的逻辑：

```python
# 1. 从 IoC 容器获取 Speech 实例
speech = self._container.get(Speech)

# 2. 如果是 TTSSpeech，注册其 commands 到 main channel
if isinstance(self._speech, TTSSpeech):
    for command in self._speech.commands():
        self.main_channel.build.add_command(command, override=False)

# 3. 生成 __content__ command
default_content_command = make_content_command_from_speech(self._speech)
self.main_channel.build.add_command(default_content_command, override=False)
```

问题链：
- `TTSSpeech.commands()` 返回 `list[Command]` — Speech 知道了 shell 调度模型
- `make_content_command_from_speech()` 在 contracts 层定义，同样耦合 Command
- CTMLShell 假设 speech 必然以 command 方式集成，无法支持非 command 的 speech 交互

**解耦方向**：Speech 不再直接生成 Command。Speech channel module 以独立 channel 形态存在，通过 manifests 被发现并注册到 shell。Channel 的 `@channel.build.command()` 装饰器自然完成 "方法 → Command" 的转换，这是 channel 层的职责，不是 speech 层。

参考 `zenoh-fractal` 中 `FractalHub.as_channel()` 的模式——能力提供方只暴露自身抽象，由 channel builder 完成到 shell 的桥接。

### 问题 2: Player 选型 — RESOLVED (Phase 2 完成)

**当前状态**：
- `BaseAudioStreamPlayer` — 基于线程 + queue 的抽象基类，含 scipy resample。设计 OK。
- `MiniAudioStreamPlayer` — **新默认**，基于 miniaudio，零系统依赖。
- `PyAudioStreamPlayer` — 降级为可选，通过 `ghoshell_moss[audio]` 安装。
- `PulseAudioStreamPlayer` — Linux only，保持可选。

### 问题 3: 国际 TTS API 集成

**当前状态**：
- `VolcengineTTS` — 火山引擎双向流式 TTS，WebSocket 协议，自身实现了一套 binary protocol (`protocol.py`)
- `TTSServiceProvider` — 工厂模式，但 `use` 字段是 literal，无法扩展

**候选 provider**：

| Provider | 协议 | 流式 | 中文 | 备注 |
|----------|------|------|------|------|
| Volcengine (当前) | WebSocket binary | Yes | 原生 | 国内首选 |
| **OpenAI TTS** | HTTP SSE / streaming | Yes | 尚可 | 国际最通用 |
| **ElevenLabs** | WebSocket | Yes | 一般 | 音色最丰富 |
| **Azure Speech** | WebSocket | Yes | 好 | 企业级 |
| **edge-tts** (免费) | WebSocket | Yes | 好 | 无需 API key |

**集成策略**：
- 每个 provider 实现 `TTS` 抽象，放在 `core/speech/` 下独立模块
- `TTSServiceProvider.use` 从 `Literal` 改为 `str`，支持运行时选择
- 依赖声明：每个 provider 的依赖通过 `depends.py` 中的新函数声明（如 `depend_openai_tts()`），对应 pyproject.toml 的 optional dependency group
- 环境变量：`.env.example` 增加各 provider 的凭证模板

**不需要做的**：不实现统一的 "TTS provider protocol" 抽象层。`TTS` ABC 已经是统一接口，每个 provider 直接实现它即可。provider 之间的差异（鉴权方式、音频格式、音色体系）由各自的 Config 和 `TTSInfo` 承载。

### 问题 4: 自动容错降级

**降级链路设计**：

```
TTS API (网络) → 失败 → 本地离线 TTS (如 edge-tts) → 失败 → MockSpeech (纯文本 buffer)
                                ↓
Player (miniaudio) → 失败 → 系统默认播放器 → 失败 → MockSpeech (无音频输出)
```

**实现要点**：
- `TTSServiceProvider` 支持配置 provider 优先级列表（非单个 use）
- Speech 启动时做 health check（如 1s 内能拿到音频即认为可用）
- 降级是自动且静默的，只在 logger 中记录
- MockSpeech 不需要改动——它已经是完美的最终兜底

### 问题 5: Session Logos Stream 与 Speech 的集成

**现状**：`Session` 提供 `pub_logos()` / `get_logos()` 跨进程流式协议。logos 的语义是 "模型生产的流式思想"（CTML），但底层机制是通用的有序字节流 pub/sub。

**与 speech 的关系**：speech channel module 在初始化时可以：
1. 订阅 logos stream 获取模型输出文本
2. 在自己的进程中执行 TTS + 播放
3. 将 speech 状态（播放中/完成/错误）通过 output protocol 反馈

这样 speech 可以作为独立 app cell 运行在单独进程中，通过 zenoh session 通讯。但这**不是第一版的 scope**——第一版先完成 speech 的解耦和 player/provider 治理，跨进程 speech 作为未来迭代方向。

### 问题 6: AI 模型构造 JSON 参数的认知冲突 (2026-05-25)

**现象**：模型在调用 `say` 命令时，`voice: dict` 参数需要传递 JSON 字符串，但模型总是用 CTML 的原生 key-value 属性语法来写，导致参数解析失败。

**根因**：CTML 的属性语法天然是**扁平的 key-value**（如 `<say speed="1.0" pitch="high">`），而 `voice: dict` 要求模型在属性值内部嵌套 JSON 字符串（如 `<say voice:dict="{'speed': 1.0}">`）。模型已经在用 XML 属性表达结构了，突然要求它在属性值内部再塞一个序列化结构，它在两种语法之间切换时容易出错。CTML 的 `:dict` 类型后缀机制虽然存在，但模型不容易自然地想到"先把 dict 序列化成 JSON 字符串，再塞进属性值"。

**本质**：这不是模型能力问题，而是 **"flat syntax for nested data" 的认知 dissonance**。CTML interface 设计原则应该是：凡是模型要通过 CTML 调用的命令，参数应尽量是标量类型（str/int/float/bool）或流式参数（chunks__/text__/ctml__），避免 dict/list 这类需要二次序列化的复杂类型。当无法避免 dict 参数时，必须在 docstring 中提供显式的 CTML 调用示例。

**当前缓解策略（方案 C）**：强化 command docstring 中的 CTML 示例。在 `say_doc()` 和类似涉及 dict 参数的命令中，显式展示 JSON 字符串的正确写法，让模型有明确的 copy-paste 模板。远期考虑拆分命令（方案 A：`set_voice` + `say` 分离），从根本上消除嵌套 dict 参数。

## Key Decisions

### D2: miniaudio 作为默认 Player (P1) — DONE

**决策**: 新增 `MiniAudioStreamPlayer(BaseAudioStreamPlayer)` 作为默认 player 实现。PyAudio 和 PulseAudio 降级为可选（通过 extras 安装）。

**实施**:
- `miniaudio>=0.67` 加入核心依赖，零系统依赖
- `AudioPlayerProvider` 支持 `backend: Literal["miniaudio", "pyaudio"]` 配置切换
- PyAudio 惰性导入，只在 backend="pyaudio" 且已安装 `ghoshell_moss[audio]` 时才加载
- miniaudio 1.x 使用 generator 模式，通过内部 buffer 实现帧精确 yield

**Why**:
- 零系统依赖，wheel 安装即用
- 跨平台一致
- 与 `BaseAudioStreamPlayer` 的三个抽象方法完全兼容
- PyAudio 历史上安装失败是 MOSS 试用者的第一道门槛

### D6: 强化 CTML docstring 中的 JSON 参数示例 (P1)

**决策**: 对涉及 `dict`/`list` 等复杂类型参数的命令，在 docstring 中显式提供 CTML 调用示例，展示 JSON 字符串的正确构造方式。不改变接口签名，不改变 CTML 解析规则。

**具体措施**:
- `say_doc()` 中加入 CTML 示例：`<say voice:dict=\"{'speed': 1.0, 'pitch': 'high'}\">你好</say>`
- 样本应覆盖：单层 dict、嵌套 dict、带默认值的省略写法
- 示例中展示 `:dict` 类型后缀的使用，利用 CTML parser 已有的 `AttrWithTypeSuffixParser` 机制
- 此模式作为约定推广到其它有 dict/list 参数的命令

**Why**: 改动最小，立即生效。模型对 docstring 中的示例有很强的跟随能力，好的示例可以显著降低出错率。长期看，如果某个 dict 参数频繁出错，再考虑拆分命令（方案 A）。

**非目标**: 不改 CTML 语法，不加新的 parser 约定。不强制所有命令都避免 dict 参数。

### D1: Speech 不再生成 Command，转为 Channel Module (P0)

**决策**: 删除 `contracts/speech.py` 中的 `make_content_command_from_speech()`，删除 `TTSSpeech.commands()`。Speech channel module 以独立 channel 形态存在，通过 `@channel.build.command()` 装饰器暴露能力给 shell。

**受影响文件**:
- `contracts/speech.py` — 删除 `make_content_command_from_speech` 和 `TTSSpeech.commands()`
- `core/ctml/shell/ctml_shell.py` — `_speech_context_manager` 不再注入 commands
- `core/speech/stream_tts_speech.py` — `BaseTTSSpeech` 不再继承 `TTSSpeech`（改为直接继承 `Speech`），或 `TTSSpeech` ABC 瘦身

**不动**:
- `Speech`, `SpeechStream`, `StreamAudioPlayer`, `TTS`, `TTSBatch` 抽象——这些是纯粹的 speech 域抽象，不含 command 耦合
- `MockSpeech` — 测试用，不需要 command 能力

**Why**: zenoh fractal 改造后，channel 可以通过 manifests 被自动发现。Speech 的能力（say, set_voice, use_tone 等）通过 channel 的 command 装饰器暴露，责任在 channel builder 而不是 speech 自身。

### D3: TTS provider 用 str + registry 替代 Literal (P1)

**决策**: `TTSServiceProvider.use` 改为 `str` 类型，支持多个 provider 名。provider 通过模块级注册表发现，而非 if/elif 硬编码。

```python
# 注册机制（每个 provider 模块自行注册）
TTS_PROVIDERS: dict[str, Callable[[IoCContainer, dict], TTS]] = {}

# TTSServiceProvider.factory() 改为
def factory(self, con):
    name = manager_conf.use
    if name not in TTS_PROVIDERS:
        raise LookupError(f"Unknown TTS: {name}. Available: {list(TTS_PROVIDERS)}")
    return TTS_PROVIDERS[name](con, manager_conf.provider_configs.get(name, {}))
```

**Why**: 新增 provider 不应修改 `TTSServiceProvider` 源码。每个 provider 在自己的模块中 `TTS_PROVIDERS["openai"] = factory_openai_tts`。

### D4: 降级链在 Speech 层实现 (P2)

**决策**: 新增 `FallbackSpeech` wrapper，接受 `[Speech, Speech, ...]` 优先级列表。启动时依次尝试 `speech.start()`，第一个成功者作为 active speech。当 active speech 失败时，自动切换到下一个。

```python
class FallbackSpeech(Speech):
    def __init__(self, *candidates: Speech):
        self._candidates = candidates
        self._active: Speech | None = None

    async def start(self):
        for candidate in self._candidates:
            try:
                await candidate.start()
                self._active = candidate
                return
            except Exception:
                logger.warning("Speech fallback: %s failed, trying next", candidate)
        self._active = MockSpeech()  # 最终兜底
        await self._active.start()
```

**Why**: 容错逻辑集中在 wrapper 中，不污染每个具体实现。MockSpeech 是永远可用的最终兜底。

### D5: 暂不做跨进程 speech (Out of Scope)

**决策**: Session logos stream 用于跨进程 speech 流式传输是合理方向，但本轮不实现。

**Why**:
- 第一版先完成解耦 + player/provider 治理，变更面已经很大
- 跨进程 speech 需要定义额外的 stream 协议（音频帧格式、时序同步、背压），是一个独立设计问题
- 当前优先级：让 speech 在单进程中正确、可靠、可扩展地工作

## Implementation Plan

### Phase 1: 解耦 (P0)

| # | 任务 | 影响文件 |
|---|------|----------|
| 1.1 | 删除 `make_content_command_from_speech()` | `contracts/speech.py` |
| 1.2 | 删除 `TTSSpeech.commands()` 方法 | `contracts/speech.py` |
| 1.3 | 删除 `TTSSpeech` ABC？还是瘦身？TBD | `contracts/speech.py` |
| 1.4 | 重构 `CTMLShell._speech_context_manager()` — 不再注入 commands | `core/ctml/shell/ctml_shell.py` |
| 1.5 | 创建 speech channel module（`core/speech/speech_channel.py` 或在 host manifests 中定义）| 新文件 |
| 1.6 | 确保 `make_content_command_from_speech` 的等价功能通过 channel command 提供 | speech channel module |

### Phase 2: Player 轻量化 (P1) — ✅ DONE (2026-05-26)

| # | 任务 | 影响文件 | 状态 |
|---|------|----------|------|
| 2.1 | 新增 `MiniAudioStreamPlayer` 实现 | `core/speech/player/miniaudio_player.py` | ✅ |
| 2.2 | miniaudio 加入核心依赖 | `pyproject.toml` | ✅ |
| 2.3 | `AudioPlayerProvider` 替代 `PyAudioPlayerProvider`，backend 可切换 | `host/providers/audio_player_provider.py` | ✅ |
| 2.4 | PyAudio 改惰性导入，按需加载 | `audio_player_provider.py` | ✅ |
| 2.5 | 更新 workspace manifests/stubs | stubs + .moss_ws | ✅ |
| 2.6 | 10 个单元测试 | `tests/ghoshell_moss/speech/test_miniaudio_player.py` | ✅ |

### Phase 3: TTS 多 provider (P1)

| # | 任务 | 影响文件 |
|---|------|----------|
| 3.1 | `TTSServiceProvider.use` 从 Literal 改为 str | `host/providers/tts_service_provider.py` |
| 3.2 | 实现 TTS provider registry 机制 | `core/speech/tts_registry.py` |
| 3.3 | VolcengineTTS 迁移到 registry 模式 | `core/speech/volcengine_tts/` |
| 3.4 | 新增 OpenAI TTS provider | `core/speech/openai_tts/` |
| 3.5 | 新增 edge-tts provider (免费离线兜底) | `core/speech/edge_tts/` |
| 3.6 | `.env.example` 增加多 provider 凭证模板 | `host/stubs/workspace/.env.example` |
| 3.7 | `depends.py` + `pyproject.toml` 各 provider 可选依赖声明 | |

### Phase 4: 容错降级 (P2)

| # | 任务 | 影响文件 |
|---|------|----------|
| 4.1 | 实现 `FallbackSpeech` wrapper | `core/speech/fallback.py` |
| 4.2 | `TTSServiceProvider` 支持 provider 优先级列表 | `host/providers/tts_service_provider.py` |
| 4.3 | Speech 启动时 health check 机制 | `core/speech/` |

### Phase 5 (Out of Scope, Future)

- Session logos stream 跨进程 speech
- Speech 作为独立 app cell 运行
- 音频格式协商（TTS 输出格式 × Player 接受格式的自动转换矩阵）

## Blast Radius Summary

| 改动 | 影响范围 |
|------|----------|
| ~~PyAudio → miniaudio 默认~~ | ✅ 已实施。miniaudio 核心依赖，PyAudio audio extras 可选。`AudioPlayerConfig.backend` 可切换 |
| 删除 `make_content_command_from_speech()` | 仅 `ctml_shell.py` 中 `_speech_context_manager` 调用方，测试 |
| 删除 `TTSSpeech.commands()` | 同上 |
| `CTMLShell._speech_context_manager` 重构 | 核心启动路径，需要全量回归 |
| 新增 `MiniAudioStreamPlayer` | ✅ 纯新增，不影响现有 player |
| `TTSServiceProvider.use` 类型变更 | 配置文件格式小幅调整 |
| 新增 TTS providers | 纯新增 |
| `FallbackSpeech` | 纯新增 |

**不动**: `Speech`, `SpeechStream`, `StreamAudioPlayer`, `TTS`, `TTSBatch` 抽象。`MockSpeech`。`BaseAudioStreamPlayer`。VolcengineTTS 核心逻辑。

## Test Plan

### T1: 单元测试
- ✅ `MiniAudioStreamPlayer` start/stop/write 基本流程
- ✅ `MiniAudioStreamPlayer` 格式转换 (PCM_S16LE, PCM_F32LE)
- ✅ `MiniAudioStreamPlayer` 重采样
- ✅ `MiniAudioStreamPlayer` 流式多次 add
- ✅ `MiniAudioStreamPlayer` clear / 幂等 start / close 后 add
- `FallbackSpeech` 降级链：第1个成功 → 不尝试第2个；第1个失败 → 自动切换到第2个；全部失败 → 兜底 MockSpeech
- TTS provider registry 注册/查找/异常
- Speech channel module commands 注册正确性

### T2: Shell 集成回归
- `ctml_shell_test()` 端到端：CTML 文本 → speech 输出
- MockSpeech 作为默认 speech 时 shell 启动正常
- TTSSpeech + miniaudio player 组合启动正常
- Speech clear 行为不变

### T3: 多 provider 切换
- VolcengineTTS provider 正常启停
- OpenAI TTS provider 正常启停
- edge-tts provider 正常启停
- 配置切换 provider 后重启 shell 正常

### T4: 降级行为
- TTS API 不可用时自动降级到下一级
- Player 不可用时自动降级到 MockSpeech
- 降级过程不抛异常，logger 中有 warning

## Related Features
- `zenoh-fractal` — Channel 分形改造，为本 feature 的 speech channel module 化提供基础设施
- `cell-discovery-refactor` — Cell 发现重构，影响 app cell 的启动/发现模式
