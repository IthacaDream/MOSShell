---
title: How to work with MOSS resources
description: 理解 Resource 体系：访问已有资源、定义新资源并注册到 workspace matrix 环境
---

# How to Work with MOSS Resources

## What is a Resource?

MOSS 用 `scheme://host/path` 作为全局资源句柄。任何数据——图片、文档、记忆、API——都可以通过这个字符串引用和传递。

四个核心抽象，全部定义在一个文件：

```bash
moss codex get-interface ghoshell_moss.contracts.resource
```

- `ResourceMeta` — AI 可读的元信息 (Pydantic)，`host` + `path` + `description`，`locator` 是计算属性
- `ResourceItem[Meta, Data]` — meta 立即可用，`get()` 懒加载实际数据
- `ResourceStorage` — 单 (scheme, host) 的 CRUD 后端，带 `usage()`/`help()` 自解释
- `ResourceRegistry` — 跨 scheme+host 的 VFS 路由层

## Why Register to Matrix?

Matrix 是环境的总线。`ResourceRegistry` 挂在 Matrix 上后，任何 Cell 都能通过 `matrix.resources()` 访问所有已注册的资源 —— 不需要知道数据在哪、用什么协议。

注册方式：写一个 `ResourceRegisterProvider`，环境启动时自动把 Storage 注册进 Registry。

## How to Use Existing Resources

```bash
# 列出所有已注册的 resource providers
moss manifests providers

# 在 REPL 中交互
moss repl
> matrix.resources().schemes()           # 列出所有 scheme
> matrix.resources().hosts("pil-image")  # 列出该 scheme 的 host
> matrix.resources().usage("pil-image")  # 查看用法
```

通过 locator 读取资源：

```python
item = await matrix.resources().get("pil-image://workspace-assets/beach_photo")
meta = item.meta    # ResourceMeta, 立即可用
image = await item.get()  # PIL.Image, 触发磁盘 I/O
```

## How to Define a New Resource

完整实现参考两个开箱示例：

```bash
# 完整 CRUD + JSONL 存储 + Provider
moss codex get-source ghoshell_moss.core.resources.local_image:LocalImageStorage

# 只读目录扫描 + recall agent
moss codex get-source ghoshell_moss.core.resources.markdown_kb:MarkdownKnowledgeBase
```

做三件事：

**1. 实现 Meta + Item + Storage。** 继承 `ResourceMeta`、`ResourceItem`、`ResourceStorage`。`scheme()` 固定为类级别字符串，`host` 是实例标识。`get(path)` 和 `delete(path)` 接受 path 而非 locator。

**2. 实现 ResourceRegisterProvider。** 参考 `LocalImageResourceProvider`：

```bash
moss codex get-source ghoshell_moss.core.resources.local_image:LocalImageResourceProvider
```

`bootstrap()` 已由父类实现：自动从容器取出 Registry 和 Storage，调用 `registry.register(storage)`。

**3. 注册到 workspace manifests。** 在你的 workspace 的 `src/MOSS/manifests/providers.py` 中：

```python
from ghoshell_moss.core.resources.my_module import MyStorageProvider
my_storage_provider = MyStorageProvider()
```

参考 stubs 版本：
```bash
cat host/stubs/workspace/src/MOSS/manifests/providers.py
```

## How to Verify

```bash
# 检查 provider 已被发现
moss manifests providers

# 查看 Registry 在容器中的状态
moss manifests providers --identity ghoshell_moss.contracts.resource:ResourceRegistry

# 单元测试参考
moss codex get-source tests.ghoshell_moss.core.resources.test_local_image
```

## Currently Registered in Environment

```bash
moss manifests providers | grep -i resource
```

| scheme | host | 参考实现 |
|--------|------|----------|
| pil-image | workspace-assets | `core/resources/local_image.py` |
| markdown-kb | moss-howto | `core/resources/markdown_kb/` |

Registry 本身：`core/resources/memory_registry.py`
