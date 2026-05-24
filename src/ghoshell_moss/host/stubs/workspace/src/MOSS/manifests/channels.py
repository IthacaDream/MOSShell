# MOSS Main Channel manifest.
#
# FastAPI-like entry point: 创建 CTML shell 的主 channel (__main__)。
# 在这里可以 import_channels（树挂载）、with_module（能力叠加）、with_state（排他切换）。
# 所有组合显式可见，零隐式逻辑。
#
# 如果这个文件没有定义 name == "__main__" 的 channel，MossRuntime 会使用默认空白 main。
# Mode 可以定义自己的 channels.py 来完全覆盖全局的 main channel。
#
# 原语 (primitives) 不再有独立的 manifest 类型 — 直接通过 main.build.add_command() 注册。

from ghoshell_moss import new_main_channel
from ghoshell_moss.core.blueprint.channel_builder import new_command
from ghoshell_moss.core.ctml.shell.primitives import (
    interrupt_command,
    noop,
    observe,
    sleep,
)
from ghoshell_moss.host.app_store_channel import AppStoreChannel

main = new_main_channel(description="Default MOSS main channel with app store")

# -- Shell 原语 --------------------------------------------------
main.build.add_command(new_command(sleep))
main.build.add_command(new_command(noop))
main.build.add_command(new_command(observe))
main.build.add_command(interrupt_command)

# -- App Store ---------------------------------------------------
main.import_channels(AppStoreChannel(name='apps'))
