---
created: 2026-05-23
depends: []
description: 将 ghoshell_moss.__init__.py 从 import * 级联黑洞改造为显式声明 facede，内部自引用改为直接链路。
milestone: null
priority: P2
status: completed
status_note: 'Facade done: 49 explicit symbols, 7 internal refs fixed, 394 tests pass.'
title: Moss Facade
updated: '2026-05-24'
---

# Moss Facade

> Use `moss features set-status moss-facade <status> -m "note"` to update state.

## Motivation

`ghoshell_moss.__init__.py` 当前通过 `from ghoshell_moss.core import *` + `from ghoshell_moss.message import *` 级联暴露几十个符号，但外部实际只使用 18 个。API 表面不可见，且包内 7 处文件自我引用 `from ghoshell_moss import X`（IDE 补完留下的坑）。需改为显式 facade 模式。

## Key Decisions

### 三步走策略

1. **内部调用显式化** — 7 处 `src/ghoshell_moss/**` 内的 `from ghoshell_moss import X` 改为直接链路引用
2. **18 符号逐个审核** — 判断哪些保留在 facade，哪些应让调用方用直接路径
3. **blueprint 候选对齐** — 从 blueprint 各模块挑选有 facade 意义的入口符号

### facade 原则
- 不搞 `import *` 隐式声明
- 每个 re-export 显式写 `from x.y.z import Foo` + 注释说明用途
- 内部代码一律用直接路径，不从 `ghoshell_moss` 自我引用

## Implementation Notes

18 个外部使用符号及真实来源已由 `generate_import_path` 批量确认。7 处内部自引用位置已定位。