# MOSS App

你正在一个 MOSS App 的工作目录中。这里是最小上下文，后续会由 AI 在开发过程中自行完善。

## 关键文件

- `APP.md` — App 的元信息声明（frontmatter 配置 + docstring）。MOSS 通过它发现和注册 App
- `main.py` — App 入口脚本。由 AppWatcher 定义的 executable 启动
- `CLAUDE.md` — 这个文件，给 AI 开发者的上下文

## APP.md 配置

```yaml
---
executable: uv        # 启动器，默认 uv
script: main.py       # 入口脚本
arguments: ""         # 默认启动参数
description: ""       # App 描述
respawn: false        # 崩溃后是否自动重启
workers: 1            # 工作进程数
max_age: null         # 进程最大存活秒数
---
```

## 开发流程

1. 实现 `main.py` 中的业务逻辑
2. 通过 Matrix 暴露 Channel 与其它 Cell 通讯
3. 在 Mode 配置中声明 app 的 include/bringup 规则
4. 用 `moss apps test <group/name>` 前台调试
