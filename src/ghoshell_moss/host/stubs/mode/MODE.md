---
# =============================================================================
# MODE.md — Mode 入口文件，定义权限边界和行为。
#
# 叠加语义：
#   - mode 的 manifests 叠加在全局 manifests 之上
#   - channels: mode 的 __main__ 完全覆盖全局 main channel
#   - 其余类型 (providers/configs/topics/nuclei/resources): 合并叠加
#
# 编辑顺序：
#   1. 先配置 apps 白名单和 bringup 策略
#   2. 再按需添加 manifest 文件（channels.py, providers.py 等）
#   3. 在本正文区域写 mode 的使用说明（会显示在 moss modes show 中）
# =============================================================================

# --- 必填字段 ---

# mode 名称，与目录名一致。moss modes create 自动填充。
name: ''

# 一行描述，moss modes create --desc 填入。
description: ''

# --- 权限边界 ---

# App 白名单。语法：
#   '*/*'      — 允许所有公开 app
#   'group/*'  — 允许某个 group 下所有 app
#   '_internal/*' — _ 前缀表示禁止访问
# moss modes create -a 可重复指定。
apps:
  - '*/*'

# 启动时自动 bringup 的 app 列表，语法同 apps。默认空。
# moss modes create -u 可重复指定。
bringup_apps: []

# --- 可选字段 ---

# CTML 版本，留空使用默认版本。
ctml_version: ''
---

在此写 mode 的使用说明，支持 markdown。
会显示在 moss modes show 的输出中。
对于 AI 模型，这里的文字会作为 mode instruction 注入上下文。
