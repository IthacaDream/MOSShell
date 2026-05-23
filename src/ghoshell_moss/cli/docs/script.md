---
title: MOSS Script
description: 一次性 Python 脚本入口，通过 Matrix.discover() 接入运行中的 MOSS 网络。需要写探查脚本或调试集成时阅读
---

# MOSS Script

Script 是复用 MOSS 运行时的一次性 Python 脚本。它通过 `Matrix.discover()` 接入运行中的 MOSS 网络，完成任务后退出。

## 1. 它是什么

一个 script 就是一个目录，两个文件：`SCRIPT.md`（清单）+ `main.py`（入口）。通过 `moss script run <name>` 在前台运行，随用随走。

## 2. 为什么需要

MOSS 的能力扩展有两条路径：

| | App | Script |
|---|---|---|
| 运行方式 | 持久化运行时 | 一次性，用完即退 |
| 适用场景 | 提供持续服务，AI 在 moss 操作期间使用 | 工具功能、调试、集成测试、探查运行时状态 |

两者共享同一套 `Matrix` API。

**关键认知**：当 AI 模型通过 MCP 自己就能操作 moss 开发工具时，app 和 script 形成互补——app 交付持续能力，script 是反复调试和验证假设的轻量入口。App 是"造工具"，script 是"用工具理解系统"。

## 3. 最小上手

### 3.1 创建

```bash
moss script init <name>
```

### 3.2 写逻辑

编辑 `main.py`。唯一需要知道的入口：

```python
from ghoshell_moss.core.blueprint.matrix import Matrix

async def main():
    async with Matrix.discover() as matrix:
        # matrix.session  — 通讯总线
        # matrix.container — IoC 容器
        # matrix.manifests — 环境能力声明
        pass

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 3.3 运行与查看

```bash
moss script run <name>       # 运行
moss script list              # 列出所有 script
```

## 4. 展开

### 4.1 Matrix 是面向开发者的 API 承诺

`Matrix` 是 MOSS 对开发者的稳定抽象。无论底层通讯协议如何演进，`matrix.session`、`matrix.container`、`matrix.manifests` 保持语义一致。Script 进程通过 `Matrix.discover()` 获得完整的环境感知能力，不需要手动管理连接或依赖注入。

### 4.2 与 App 的边界

App 由 host 管理生命周期，在 `matrix.list_cells()` 中可见。Script 不以 cell 身份注册，host 不感知其存在。这意味着：
- 可以注入信号到 mindflow，收发 topic 消息
- 生命周期完全自主，不需要 host 参与
- 天然适合一次性的验证和探查任务

### 4.3 深入

- `moss script list` 看已有 script，读 `main.py` 学用法模式
- `moss codex get-interface ghoshell_moss.core.blueprint.matrix` 了解 Matrix 完整 API
- `moss features list` 了解 script 相关 feature 的决策轨迹
