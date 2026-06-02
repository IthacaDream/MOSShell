# MOSS — Model-oriented Operating System Shell

MOSS 是 Ghost in Shells 架构的 Shell（躯体）层 — 让 AI 模型降临到现实世界：感知环境、思考、行动，并发、实时、有序。

不是又一个 agent 框架。它回答一个不同的问题：**一个 Ghost 如何降入 Shell，活过来？**

## 安装

```bash
git clone https://github.com/GhostInShells/MOSShell && cd MOSShell
uv sync --active --all-extras
cp .moss_ws/.env.example .moss_ws/.env    # 填入 API key
```

## 开始

```bash
.venv/bin/moss start        # 认知入口 — 从这里开始
.venv/bin/moss-cli           # 交互式 Shell（人类用）
.venv/bin/moss-as-mcp        # 作为 MCP server 暴露给 AI 工具
```

AI 协作者进入项目后会自动加载认知地图。人类用户运行 `moss start` 即可。

## 核心概念

| 概念 | 是什么 |
|------|--------|
| **CTML** | 流式控制语言 — 模型边生成 token 边实时执行命令 |
| **Channel** | 能力封装 — Python 函数签名即 prompt，无需 schema 翻译 |
| **Mindflow** | 感知/思考/行动仲裁 — Ghost 在连续流中保持存活 |
| **Ghost** | 持久化智能体 — 连续记忆，主动交互，反身性控制 |

## 了解更多

- `moss ctml read` — CTML 语法与执行模型
- `moss --ai all-commands` — 完整命令树
- `moss codex concepts` — 核心抽象
- `moss codex blueprint channel_builder` — 如何构建 Channel
- `tutorials/` — L1 Hello World 到 L4 全链路教程

## 项目状态

Beta。由人类工程师与 AI 协作开发。通过 `moss --ai features list` 查看活跃 workstream。

---

*Ghost in Shells — 探索人类与 AI 协作共生的可能性。*
