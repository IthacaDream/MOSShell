# Default mode 的 main channel — 从零构建标准 shell main channel。
# 如需复用全局 main，改为 from MOSS.manifests.channels import main。
from ghoshell_moss import new_default_shell_main_channel
from ghoshell_moss.core.blueprint.host import FractalHub

main = new_default_shell_main_channel()

# register fractal hub as channel
factory = FractalHub.channel_factory(auto_start=True, allow_all=False)
main.import_channels(factory)
