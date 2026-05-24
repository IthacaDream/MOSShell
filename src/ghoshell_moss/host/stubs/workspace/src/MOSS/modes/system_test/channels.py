"""
system_test mode 的 main channel — 完全覆盖全局 main。

包含所有标准原语 + 实验性原语 (loop, sample, branch)。
"""
from ghoshell_moss import new_main_channel
from ghoshell_moss.core.blueprint.channel_builder import new_command
from ghoshell_moss.core.ctml.shell.primitives import (
    branch,
    interrupt_command,
    loop,
    noop,
    observe,
    sample,
    sleep,
)
from ghoshell_moss.host.app_store_channel import AppStoreChannel

main = new_main_channel(description="system_test main channel with full primitives")

# -- 标准原语 --------------------------------------------------
main.build.add_command(new_command(sleep))
main.build.add_command(new_command(noop))
main.build.add_command(new_command(observe))
main.build.add_command(interrupt_command)

# -- 实验性原语 ------------------------------------------------
main.build.add_command(new_command(loop))
main.build.add_command(new_command(sample))
main.build.add_command(new_command(branch))

# -- App Store -------------------------------------------------
main.import_channels(AppStoreChannel(name='apps'))
