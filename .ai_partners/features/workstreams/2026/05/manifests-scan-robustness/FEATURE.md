---
created: 2026-05-29
depends: []
description: Add strict mode, error collection, and timeout to the manifests/modes/ghosts
  scan pipeline. Eliminate silent error swallowing while preserving full backward
  compatibility.
milestone: null
priority: P1
status: completed
status_note: '2026-06-02: core implementation done in 29b72d4. CLI display gap fixed:
  shared display_scan_errors in utils.py, manifests_cli shows errors, Host.scan_errors
  now includes manifest errors.'
title: Manifests Scan Robustness
updated: '2026-06-02'
---

# Manifests Scan Robustness

> Use `moss features set-status manifests-scan-robustness <status> -m "note"` to update state.

## Motivation

The scan pipeline (scan_package → type-specific scanners → modes/ghosts discovery → Host assembly)
had four layers of silent error suppression. A single bad import could hang `Host()` indefinitely
with no visibility. Dependency errors were flattened into generic strings. This made debugging
workspace configuration unnecessarily hard — you'd get an empty mode list or partial manifests
with no clue what went wrong.

## Key Decisions

1. **Two orthogonal params: `strict` and `errors`** — not a single flag. `strict=True` fails fast
   (exceptions propagate). `errors=[]` collects without failing. Can combine both: collect errors
   in non-strict mode for logging, or use strict in CI.

2. **Timeout via ThreadPoolExecutor, not subprocess** — covers 95% of real hangs (I/O, deadlocks)
   without serialization complexity. Subprocess isolation left as Phase 2.

3. **Default behavior unchanged** — all new params default to silent mode. Zero breaking changes.

4. **Errors collected at every layer** — `scan_package()`, each scanner, `find_mode_from_package()`,
   `Host.all_modes()` all append to the same `errors` list. Single source of truth for scan health.

5. **New mode defaults to empty apps** — CLI `create_mode` now defaults `--app` to `[]` instead of
   `["*"]`. Only the default mode keeps full access. Prevents accidental capability exposure.

## Implementation Notes

- `ScanError` dataclass carries `module_path`, `exception`, `stage` (scan/import/iterate)
- `ModuleManifest.timeout` field flows from `scan_package(timeout=...)` through the entire pipeline
- Curly quotes (U+201C/U+201D) were introduced by Edit tool and batch-fixed — project should avoid
  full-width punctuation entirely (future cleanup needed)