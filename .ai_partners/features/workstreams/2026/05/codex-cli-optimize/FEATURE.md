---
created: 2026-05-23
depends: []
description: Quick optimization pass on codex CLI tools — better output hints, module.attr
  fallback, remove info, add where, stronger short_help for all-commands.
milestone: null
priority: P2
status: completed
status_note: 'All 6 changes implemented: module.attr fallback, remove info, add where,
  output hints, stronger short_help, list hint fixes'
title: Codex CLI Optimize
updated: '2026-05-24'
---

# Codex CLI Optimize

## Motivation

The five codex tools (get-interface, get-source, info, list, eval) have been in use
for a while. Some friction points have emerged: `module.attr` syntax is more natural
than `module:attr` but fails, `info` overlaps with `list`/`get-source` without adding
value, output lacks resolution hints, and there's no quick way to locate the canonical
import path of an object (`where`). This workstream addresses these in one pass.

## Key Decisions

1. **get-interface / get-source: `module.attr` fallback**. `import_from_path` only
   splits on `:`. Try `:` first; on failure, re-parse with last `.` as separator.
   The `:` syntax remains canonical; `.` is a UX tolerance.
   Rejected: modifying `import_from_path` upstream — would change semantics for all
   callers. Keep the fallback at the CLI layer.

2. **Remove `info`**. It shows file path + docstring + member name list. Member listing
   is better served by `list`; file path/docstring by `get-source`. No unique value.
   Three commands covering the same territory is two too many.

3. **`list` stays as-is**. The per-item AST docstring extraction is I/O-heavy but
   acceptable for a devops tool that runs infrequently. The information density
   (module name + path + one-line description) justifies the cost.

4. **`eval` stays as-is**. Small footprint, clear niche (runtime introspection that
   static reflection cannot reach).

5. **No batch arguments**. The model cannot predict `get-interface` output size.
   Instead, make `short_help` more assertive so `all-commands` conveys the information
   magnitude of each command, letting the model decide wisely.

6. **New `where` command**. Takes `module:attr`, imports it, calls
   `generate_import_path()` to show the canonical definition path (not re-export path).
   Solves "where is Channel actually defined?"

7. **Output hints for all commands**. `get-interface` output should show: resolved path,
   source file, and content wrapped in ` ```python `. The model gets context to chain
   further operations (e.g., read source after seeing interface).

## Implementation Notes

- All changes are in `src/ghoshell_moss/cli/codex_cli.py` only
- `where` depends on `ghoshell_common.helpers.modules.generate_import_path`
- `module.attr` fallback: try `import_from_path(path)` → if ImportError, split on
  last `.` and retry with `module:attr` format