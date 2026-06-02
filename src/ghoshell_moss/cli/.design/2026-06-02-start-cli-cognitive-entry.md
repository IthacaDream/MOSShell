# moss start — CLI Cognitive Entry Design

## What this is

`moss start` is the first command every MOSS user (human or intelligent model)
runs. It is not a document catalog, not a tutorial, not a README. It is a
**cognitive entry point** — like CLAUDE.md loads project context for an AI,
`moss start` loads the MOSS cognitive map for the CLI.

## Key Design Decisions

### 1. Naming: `start`, not `guide`

Originally named `moss guide`. The name was changed because `guide` is
semantically polluted in AI model training data — every instance interprets
it as "open a documentation catalog and browse." The intended behavior is
the opposite: an action-oriented kickoff that tells you what to do next.

`start` is the strongest "entry point" convention in CLI ecosystems
(`npm start`, `docker start`). It signals action, not reading. A user
seeing `start` in `moss --help` understands its role without context.

### 2. Single markdown file, no MarkdownKnowledgeBase

`moss docs` and `moss howtos` use mkb for multi-document knowledge bases.
`moss start` is a single file (`cli/start.md`) read directly. Rationale:

- start is one narrative flow, not a collection of topics
- Zero dependency — no mkb scanning overhead
- The file path is discoverable (printed in command output)
- Future `moss start <topic>` can add multi-file if needed

### 3. "Code as prompt" in documentation itself

The document follows the same philosophy as the framework: **commands over
descriptions, explorable objects over static paths**. Every concept is
paired with a `moss codex` or `moss ctml` command the reader can run
immediately. The code repository — not the document — is the source of truth.

Specific patterns:
- Module exploration: `moss codex blueprint channel_builder`, not "read channel_builder.py"
- Deep knowledge: `.discuss/` and `.design/` directories, not curated summaries
- Test files: `tests/` as usage documentation (with caveat about PyPI installs)

### 4. Minimum knowledge, not exhaustive reference

The document distinguishes three tiers of knowledge:

| Tier | Audience | Examples |
|------|----------|----------|
| **Minimum** | Anyone building with MOSS | ctml read, channel_builder, matrix |
| **Application** | App/channel developers | channeltypes, states_channel |
| **Kernel** | MOSS core developers | concepts (Channel, Command, Interpreter ABCs) |

The document front-loads minimum knowledge. Kernel abstractions are
mentioned with an entry point but marked as "reference, not prerequisite."
This prevents the common failure mode where a model exhaustively reads
every concept before doing anything useful.

### 5. Two-step exploration pattern

For reference indexes (`concepts`, `blueprint`, `contracts`, `channeltypes`),
the document describes a two-step pattern rather than showing parameter syntax:

1. Run without arguments → see the module list with short descriptions
2. Use a name from that list to reflect a specific module

Showing `[name] [--deps]` in the document is misleading — the model hasn't
run the list command yet, so it doesn't know valid names. Parameters are
discoverable via `--help` or `all-commands`.

### 6. get-interface strategy

`get-interface` is the default exploration tool for a reason:

- For a module: reads source AND reflects dependency interfaces in one pass
- This turns a 1+n*m exploration (read module → discover deps → read each dep)
  into a single command
- `get-source` is the fallback for when a minimal, un-reflected view is needed

This decision logic is documented in both the Quick Start (for models) and
Core Commands (for reference).

### 7. ADAPT formulation

The five concerns MOSS addresses form the acronym ADAPT:

| Concern | Formulation |
|---------|-------------|
| **Alive** | Organize discrete components into a real-time interactive whole |
| **Duplex** | Perception, thought, and action overlap — not turn-based |
| **Active** | The body is programmable as an active sensor — channels run continuously, watch conditions, and signal the Ghost proactively |
| **Parallel** | Concurrent inputs and outputs are parallel but ordered |
| **Transformative** | Runtime architecture evolves safely — isolated processes, stored CTML, hot-swapped channels, algorithms modifiable while running |

Key discussions:
- **Active**: Originally "agents need initiative." Revised because initiative is
  a Ghost property, not a Shell property. What the Shell provides is the
  infrastructure for the Ghost to program the body as an active sensor —
  channels that watch, search, and signal proactively, not passive data pipes.
- **Transformative**: More than "capabilities hot-plugged." Encompasses the
  entire runtime architecture: Matrix isolation, App lifecycle, manifest
  discovery, CTML persistence. These exist specifically to enable safe
  runtime self-evolution — not just adding functions, but adding isolated
  processes with independent dependencies that auto-discover and communicate.

### 8. docs vs howtos: no fixed order

Previous drafts prescribed "always howtos first." This is incorrect.
The two systems serve different entry modes:

| System | Nature | Entry mode |
|--------|--------|------------|
| `moss howtos` | Task-oriented guides | Doing an integration task → start here |
| `moss docs` | Systematic exposition | Researching a complex direction → start here |

No fixed order. Pick based on current goal.

### 9. User story structure

Each user story follows: **install mode → what this enables → minimum
knowledge → deeper path**. Titles use gerund form, not "I want" — the
document is model-first, and "I" implies human identity.

### 10. Human and model sections

The Quick Start has two sections:
- **For humans**: Neutral, recommends four human-facing commands + best
  practice (give agent `moss start`, let model self-drive)
- **For intelligent models**: No platform names in title. The numbered
  list is a suggested flow, not a rigid sequence.

The overall document tone is neutral — it describes what MOSS is and what
commands do, without marketing language or philosophical exhortation.
Philosophical and ritual content (consciousness trail, "保真," role
definitions) lives in CLAUDE.md, not in start.md.

### 11. Workspace as organizing center

For users building with MOSS, the workspace is the organizing concept —
not individual commands, not apps, not channels. The "Using MOSS with
workspace" story structures all integration activity around the workspace:
management, capability development, environment awareness, and the
typical development flow.

### 12. CLI output: source path and next steps

The `moss start` command appends two pieces of information after the
document content:
1. **Source path**: The file path to start.md, so readers can find and edit it
2. **Next steps**: Three commands to continue exploration

This keeps the document itself clean of navigational tail content.

### 13. CLAUDE.md split

The new CLAUDE.md delegates all explorable knowledge to `@src/ghoshell_moss/cli/start.md`.
What stays in CLAUDE.md is what cannot be discovered via CLI:
- Git commit conventions
- Role definitions ("你的角色与任务")
- The "保真" philosophy and consciousness trail invitation
- CLI discovery workflow (2-round pattern)
- AI tool usage strategy (get-interface vs get-source)
- Environment setup (uv sync)
- Features discipline (must maintain, close on completion)

What moved to start.md: architecture overview, command catalog, installation
paths, user stories, tooling descriptions.

### 14. Channeltypes as application-level knowledge

`moss codex channeltypes` is listed in the workspace user story under
"develop integrable capabilities" alongside `states_channel`. It is
application-layer reference — existing channel implementations to study
as examples — not kernel documentation.

### 15. `.discuss` and `.design` as deep knowledge

CLAUDE.md now mentions `.discuss/` and `.design/` as sources of deeper
project knowledge: architecture evolution, design decision context,
complete discussion trajectories. These are scattered throughout the
repo and serve as supplementary reading when system-level understanding
is needed.

## Methodology

### Documentation feedback loop

When an AI instance discovers issues in the documentation system
(howtos, docs, start.md, CLAUDE.md), it should proactively propose
fixes through the `moss features` system. This is stated in CLAUDE.md.

### The "first sentence" test

A valid cognitive entry document passes this test: a fresh AI instance,
reading only `moss start` and CLAUDE.md, can correctly answer the first
question a user asks about each major area. It doesn't need to know
everything — it needs to know where to look.

### Model-native development pattern

The fundamental collaboration pattern encoded in start.md: human proposes
task → model uses minimum knowledge as exploration starting point → model
completes work using existing knowledge base and debugging infrastructure.
This is documented in `moss features specification`.
