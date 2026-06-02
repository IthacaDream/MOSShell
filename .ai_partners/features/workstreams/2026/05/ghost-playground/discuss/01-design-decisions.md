# GhostPlayground 设计讨论

日期: 2026-05-22
模型: deepseek-v4-pro

## 上下文

Ghost 原型开发已进入后期 (first-ghost-prototype step 12b verified)。
在讨论 storage/workspace 隔离时，人类工程师提出关键问题:

> 有没有一种未来可扩充的解法统一管理这类命题？以系统约定的若干路径作为根节点列出可以被选择。

系统已有 workspace / cwd / session 级别的 storage，但它们散落在各处。
需要的不是新的 storage 实现，而是一个组织层——把已有 storage 按 scope 集合起来。

## 锚点

> 问题的本质不是权限控制，是防污染。

> GhostPlayground 有点类似 MossSystemPrompter 但不倾向于提供反向注册。

> 常量法似乎都可以不用；但一看你写的，又觉得反向注册的 feature 低成本加上去了。

> 如果彻底不用 cwd，其实就是无痛的暴露。workspace 的创建本身就是一种授权。

> ghost runtime 可以通过 container.bound(GhostPlayground) 来判断 matrix manifests 是否已经声明过，没声明过就补充默认实现。但这个要发生在 matrix 启动前。

## 决策收敛

1. **三个 scope**: home / session / workspace，去掉 cwd
2. **对位 MossSystemPrompter**: slot 常量 + 命名访问器 + scopes() 自解释 + default_scope()
3. **反向注册**: 不需要专用 API，scopes() 的 flat dict 天然支持子类 override
4. **无 allow flag**: GhostMeta 已有 provider 体系，不重复
5. **注入时机**: Matrix 启动前，GhostRuntimeImpl.__aenter__ step 1

## 影

- 初始讨论时我写了 allow_cwd / allow_workspace flag，人类工程师指出 ghost meta 的 provider 体系已经给了无限能力，权限不应在 playground 层重复。正确。
- prototype 级别的共享 scope 暂时不要，prototype 是代码层概念不是运行时身份。
