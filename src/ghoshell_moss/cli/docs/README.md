---
title: MOSS AI Reference Docs
description: 系统化的 MOSS 架构参考文档，用于需要深度理解设计意图时查阅。日常开发请用 moss how-tos
---

# MOSS AI Reference Docs

系统化架构参考文档——当 how-tos 的操作步骤不够，需要理解"为什么这么设计"时来这里。

**docs 不是开发入口**。日常的任务导向知识在 `moss how-tos`。docs 的使用频率应该低于 how-tos。

## docs vs how-tos

| | docs | how-tos |
|---|---|---|
| 定位 | 系统化知识，理解设计 | 任务导向，完成操作 |
| 频率 | 低——需要深度理解时 | 高——日常开发 |
| 内容 | 架构推演、设计理由 | 操作步骤、代码示例 |
| 问题 | "为什么这么设计？" | "怎么做 X？" |

## 使用

```bash
moss docs list              # 列表，含标题和描述
moss docs list -q keyword   # 关键词过滤
moss docs list --json       # 结构化输出
moss docs read <path>       # 读文档
```
