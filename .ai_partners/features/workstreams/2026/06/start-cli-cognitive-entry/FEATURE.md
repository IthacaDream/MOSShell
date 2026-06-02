---
title: Moss Start — CLI Cognitive Entry
status: awaiting_validation
priority: P0
created: 2026-06-02
updated: 2026-06-02
depends: []
milestone:
description: >-
  moss start 命令实现 + start.md 认知入口文档 + CLAUDE.md 重写。
  设计原理记录在 cli/.design/2026-06-02-start-cli-cognitive-entry.md。
---

# Moss Start — CLI Cognitive Entry

## What was done

实现了 `moss start` 命令和配套的认知入口文档体系。设计决策和讨论方法论记录在：

**`src/ghoshell_moss/cli/.design/2026-06-02-start-cli-cognitive-entry.md`**

交付物：
- `src/ghoshell_moss/cli/start_cli.py` — 命令实现，单文件读取，双模式渲染 (rich/AI)
- `src/ghoshell_moss/cli/start.md` — 认知地图文档 (~400 行)
- `src/ghoshell_moss/cli/main.py` — 注册 `moss start` 为第一个命令组
- `.ai_partners/prompts/2026-06-02-claude.md` — 重写的 CLAUDE.md，通过 `@` 引用 start.md
- `src/ghoshell_moss/cli/.design/2026-06-02-start-cli-cognitive-entry.md` — 设计原理文档

## Validation criteria (first-sentence test)

一个新的 Claude 实例，在只加载 CLAUDE.md + `moss start` 的前提下，对以下问题给出正确方向的第一反应（不需要精确答案，需要知道去哪里找）：

1. "MOSS 是什么？" → 实时运行时框架，Shell 层，Ghost in Shells 架构
2. "我第一次用，应该先跑什么命令？" → `moss start`
3. "模型怎么控制 MOSS？" → CTML，`moss ctml read`
4. "怎么把一个 Python 函数变成模型能调用的命令？" → Channel + channel_builder
5. "怎么发现所有可用命令？" → `moss --ai all-commands`
6. "`--ai` flag 是干什么的？必须用吗？" → 剥离 rich 排版省 token；是
7. "探索模块时用什么工具？get-interface 还是直接读源码？" → get-interface 优先
8. "concepts / blueprint / contracts / channeltypes 是什么？什么时候用？" → 备查索引，按需查阅
9. "MOSS 有哪几种安装方式？" → Minimal / Framework / Standalone / Full clone
10. "workspace 是什么？能做什么？" → 组织中心，管理能力/模式/apps/ghosts
11. "how-tos 和 docs 有什么区别？什么时候看哪个？" → 任务导向 vs 系统论述，按场景选
12. "想开发 MOSS 本身，从哪里开始？" → `moss --ai features list`
13. "Active 在 ADAPT 里是什么意思？" → 躯体可编程为主动感知器
14. "提交代码时有哪些规范？" → Conventional Commits + 标注 by/coding by/review by
15. "models 探索参考索引的正确方式是什么？" → 先无参数跑看列表，再选名字

## Next steps

1. 下一个 Claude 实例加载新 CLAUDE.md + `moss start`，回答上述 15 个问题
2. 通过后，在此基础上撰写 `README.md`
3. 将 CLAUDE.md 的 `@` 引用实装到项目根目录，替换现有 CLAUDE.md

---

*本 feature 由 deepseek-v4-pro 与人类工程师协作完成, 2026-06-02*
