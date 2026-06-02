# Architecture Topology — Code Exploration Notes

Exploration targets: verify how the 7-layer topology (+ meta layer) maps to actual code.
Goal: understand real state so the topology doc reflects what IS, not just what's PLANNED.

## Layer Status Summary

| Layer | Status | Key Files |
|---|---|---|
| CTML | **working** | `core/ctml/interpreter.py`, `core/ctml/elements.py`, `core/ctml/shell.py` |
| Channel | **working** | `core/concepts/channel.py`, `core/blueprint/channel_builder.py`, `core/blueprint/states_channel.py` |
| Shell | **working** | `core/concepts/shell.py` (MOSShell ABC), `core/ctml/shell.py` (CTMLShell) |
| Workspace | **working** | `core/blueprint/environment.py`, `host/stubs/workspace/` (full convention tree) |
| Matrix | **working** | `core/blueprint/matrix.py`, `host/matrix.py` (MatrixImpl w/ zenoh) |
| Mindflow | **scaffold** | `core/blueprint/mindflow.py` — full interface design, no impl found in host |
| Ghost | **scaffold** | `core/blueprint/ghost.py` — GhostMeta/Ghost ABC, `host/ghost_runtime.py` exists |
| Host | **working** | `host/impl.py` (Host), `host/moss_runtime.py` (MossRuntimeImpl) |
| Meta | **partial** | CLI/features working, docs scaffolded, how-tos exist |

## Detailed Findings

### 1. CTML — Solid

- **Parser tree**: `BaseCommandTokenParserElement` is a recursive tree parser. Each element handles START/DELTA/END tokens. Elements form a tree matching CTML tag nesting.
- **Element types**: `CommandWithoutDeltaArgElement`, `DeltaIsTextElement`, `DeltaIsTextChunkElement`, `DeltaIsCommandTokensElement` — differentiated by `CommandDeltaArgType` (TEXT, TEXT_CHUNKS_STREAM, COMMAND_TOKEN_STREAM).
- **Scope**: `TaskScope` with `until="flow|all|any"` + `timeout`. Scopes collect child tasks and manage completion semantics.
- **Error handling**: `InterpretError` triggers global interrupt. `Observe` return type triggers observation.
- **Prompts**: `v1_0_0.zh.md` in `core/ctml/prompts/`. Also supports workspace-level ctml versions.
- **Root channel**: `create_ctml_main_chan()` creates `PrimeChannel` with primitives. Root commands omit path prefix.

### 2. Channel — Solid

- **Identity**: `ChannelFullPath` = `a.b.c` Python-module-style pathing. `ChannelName` validated by regex `^[a-zA-Z_][a-zA-Z0-9_]*$`.
- **Meta**: `ChannelMeta` carries name, description, failure, commands list, states, instructions, context messages, memory messages. This is the self-describing capability blob.
- **Runtime**: `ChannelRuntime` with `ChannelCtx` (contextvars-based). Commands can access their runtime via `ChannelCtx.runtime()`.
- **Tree**: `ChannelTree` — parent-child hierarchy. Parents can proxy to children. Stateful channels switch states.
- **Lifecycle**: `ChannelProvider` interface for providing channels. Channels can be installed/uninstalled/opened/closed.
- **Duplex**: `core/duplex/` — bidirectional communication primitives for channels.
- **Bridges**: `bridges/zenoh_bridge.py` — `ZenohChannelProvider`, `ZenohProxyChannel` connect channels across process boundaries.

### 3. Shell — Solid

- **MOSShell** (ABC): Generic over MAIN_CHANNEL. Manages interpreter lifecycle, channel metas, moss_static/moss_dynamic messages. Designed for "全双工交互" (full-duplex interaction).
- **CTMLShell**: Concrete implementation. Created via `new_ctml_shell()`. Wraps CTML interpreter + channel tree.
- **Key methods**: `interpreter(kind)`, `static_messages()`, `dynamic_messages()`, `refresh_metas()`, `clear()`.
- **Primitives**: Injected at shell creation. `__main__` channel commands run without path prefix.
- **Interpreter kinds**: `clear` (new interpretation), `append` (continue existing), `dry_run`.

### 4. Workspace — Solid

- **Environment discovery**: `Environment.discover()` searches: env var → cwd → parent dirs → home dir. Finds `.moss_ws/` or `MOSS.md`.
- **Workspace stub structure** (the full convention):
  ```
  .moss_ws/
  ├── MOSS.md              ← project-level meta config + system prompt
  ├── .env / .env.example  ← env vars
  ├── src/MOSS/
  │   ├── manifests/       ← channels.py, configs.py, primitives.py, providers.py, topics.py, resources.py, nuclei.py
  │   ├── modes/           ← default/, system_test/ — MODE.md + contracts.py + primitives.py
  │   └── ghosts/          ← (new, untracked)
  ├── apps/                ← _system_tests/{helloworld, matrix_exam, ...} — APP.md + main.py
  ├── configs/             ← zenoh_config_*.json5, circus.ini, logging.yml
  ├── runtime/             ← conversations/, sessions/, logs/, model_contexts/, locks/
  ├── souls/               ← (new, untracked)
  └── ctml_versions/       ← workspace-level ctml prompt overrides
  ```
- **Mode**: `src/MOSS/modes/<name>/` — each mode has MODE.md, optional contracts.py, primitives.py. Mode filters which apps are active.
- **App**: `apps/<group>/<name>/` — APP.md + main.py. Each app is an independent runtime discovered by AppStore.

### 5. Matrix — Solid (zenoh-based)

- **Matrix** (ABC): `cells()`, `container`, `manifests`, `session()`, `provide_channel()`.
- **MatrixImpl**: Creates Cells from AppStore. `HostMainCell` (type='host') + `AppCell` (type='app') per discovered app. Supports `fractal` type for cross-Matrix communication.
- **IPC**: Zenoh-based. `ZenohChannelProvider` bridges channels across process boundaries. `ZenohProxyChannel` represents remote channels locally.
- **IoC**: `Matrix.container` is an `IoCContainer`. Providers registered from manifests. This container is passed to CTML shell as `parent_container`.
- **Session**: Scoped communication. Session ID from environment. Zenoh key expressions use session scope.
- **Lifecycle**: `MatrixLifecycleObject` — objects bound to matrix lifecycle. Matrix manages start/stop.

### 6. Mindflow — Interface Only

- Extensive interface design: Signal, Impulse, Nucleus, Attention, Articulator, Action, Logos.
- Three-loop model: perception loop, thinking loop, execution loop — full-duplex state management.
- Priority system: DEBUG → INFO → NOTICE → WARNING → ERROR.
- Nucleus: per-modality signal processor. Multiple Nuclei feed into one Attention.
- Attention: arbitration center. Receives Impulses from Nuclei, decides what to observe.
- Articulator: converts model output (Logos) into Actions.
- **No implementation found in host layer.** The design is complete but unimplemented.

### 7. Ghost — Scaffold

- **GhostMeta** (ABC): Bootstrapper — file-as-config, discoverable via manifests. Carries name, nuclei_metas, contracts, providers. `factory(container)` produces Ghost runtime.
- **Ghost** (ABC): The runtime. `speak()`, `react()`, `execute()` methods. Manages lifecycle.
- **GhostRuntime** (host blueprint): Wraps Ghost + Shell together. Manages the full agent lifecycle.
- `host/ghost_runtime.py` exists as a module but wasn't explored in depth.
- **The real Ghost**: The project's `.ai_partners/` system (memory, features, discuss, dialogs) IS a working Ghost implementation — just not in the code layers the topology describes.

### 8. Host — Solid

- **Host** (`host/impl.py`): Bootstrap orchestration.
  1. `Environment.discover()` → workspace path
  2. `PackageManifests.from_environment()` → discover all manifests
  3. `list_modes_from_root_package()` → discover modes
  4. Select mode (from arg, env, or default)
  5. `MergedManifests([env_manifest, mode.manifest])` → merge
  6. `HostAppStore(...)` → discover apps
  7. `MatrixImpl(mode, env, manifest, app_store, workspace)` → create matrix
  8. `Host.run()` → `MossRuntimeImpl(env, workspace, mode, matrix)` → create runtime
- **MossRuntimeImpl**: Wraps CTML shell + Matrix.
  - Creates CTML shell with `parent_container=self.matrix.container`
  - Provides `moss_exec(logos)`, `moss_observe()`, `moss_interrupt()`, `moss_dynamic_messages()`
  - Manages shell lifecycle (start/stop/pause)

### 9. Meta Layer

- **CLI**: Full command tree. `moss codex` (reflection), `moss concepts` (architecture), `moss ctml` (prompts), `moss ws` (workspace), `moss manifests` (discovery), `moss modes`, `moss apps`, `moss how-tos`, `moss features`, `moss docs`.
- **features**: Working. AI-native workstream tracking. This very feature is using it.
- **docs**: Scaffold only. CLI works, no content.
- **how-tos**: Working. Procedural knowledge base with AI recall.

## Cross-Layer Connections

### Shell ← Channel
- `new_ctml_shell(main_channel=..., primitives=[...])` — channels injected at shell creation.
- `parent_container` — IoC container from Matrix, shared with Shell. Channel providers registered in container.
- `moss_static` / `moss_dynamic` — Shell collects ChannelMeta from all channels and formats as model-visible messages.

### Matrix ← Channel
- `Matrix.provide_channel(path) -> Channel` — Matrix provides channels to cells.
- `ZenohChannelProvider` — registers channels discovered in remote cells as local proxies.
- `ZenohProxyChannel` — local representation of a remote channel. Commands route over zenoh.
- **The seam**: Matrix.container holds Channel providers. Shell's channel tree populates from these providers.

### Host Bootstrap Order
```
Environment.discover()
  → PackageManifests.from_environment()
  → list_modes_from_root_package()
  → select mode
  → MergedManifests
  → HostAppStore
  → MatrixImpl (zenoh session, cells, container with providers)
  → Host.run()
    → MossRuntimeImpl
      → new_ctml_shell(parent_container=matrix.container, primitives=...)
      → CTML shell ready for execution
```

## Key Insights

1. **Shell stack (CTML+Channel+Shell) is the most complete and battle-tested.** This is L0 — the minimum viable unit.

2. **OS stack (Workspace+Matrix+Host) is working but less mature.** The zenoh bridge exists and system tests prove it works, but the app ecosystem is sparse.

3. **Mindflow and Ghost are architecture designs waiting for implementation.** Mindflow in particular is the most original design — the three-loop full-duplex model has no industry precedent.

4. **The bootstrap order confirms the topology.** Host creates Matrix first, then Shell. Matrix.container is the shared IoC backbone. Shell's channel tree populates from container-registered providers.

5. **The real Ghost is recursive.** The project's own AI collaboration system (`.ai_partners/`, memory, features) is a working Ghost — just external to the code layers. This is self-demonstrating architecture.

## Response to `moss ctml read`

The CTML prompt (`v1_0_0.zh.md`) is the **model's first encounter with MOSS**. It teaches CTML syntax, channel semantics, scope rules, and best practices. It's the "interface" side of the model-oriented design — and arguably the most natural entry point to the topology, since the whole system exists to interpret what the model outputs after reading this prompt.

Positioning: `moss ctml read` could eventually live under `moss docs` as the first AI doc, since it IS the primary AI-facing document. Currently it's under its own command group.
