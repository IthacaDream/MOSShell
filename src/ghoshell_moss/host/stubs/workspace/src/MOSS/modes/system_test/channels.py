"""
system_test mode 的 main channel — 复用全局 main + 实验性原语。

Mode 复用模式：从全局 manifest import main，然后在其上追加改造。
MergedManifests 合并时 mode 的 __main__ 完全覆盖全局 (K5)。
"""
from MOSS.manifests.channels import main
from ghoshell_moss.core.ctml.shell.ctml_main import inject_system_primitives

inject_system_primitives(main, extended=True)
