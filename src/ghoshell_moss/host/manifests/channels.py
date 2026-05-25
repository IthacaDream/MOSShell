from typing import Dict
from ghoshell_moss.core.codex.discover import scan_package
from ghoshell_moss.core.concepts.channel import Channel

__all__ = ['search_channels_from_package', 'search_main_channel_from_manifest']

MANIFEST_CONFIG_PATH = 'MOSS.manifests.channels'


def search_channels_from_package(
        package_import_path: str = MANIFEST_CONFIG_PATH,
) -> Dict[str, Channel]:
    """
    扫描逻辑：寻找在 manifest 模块中定义的 Channel 实例。
    有重名直接覆盖, 不关心 module name.

    .. deprecated::
        此函数返回所有 Channel 实例，但语义上只有 __main__ 有效。
        新代码应使用 search_main_channel_from_manifest()。
    """
    found: Dict[str, Channel] = {}

    # 递归扫描
    for manifest in scan_package(package_import_path, max_depth=2):
        if manifest.is_package:
            continue

        # 遍历模块内的所有成员
        for name, obj in manifest.module.__dict__.items():
            # 过滤掉私有成员和不符合 ConfigType 的对象
            if name.startswith('_') or not isinstance(obj, Channel):
                continue

            # 这里的逻辑：我们认为在 manifest 包下定义的变量名即为"发现"
            # 以 attr name 作为唯一键
            found[name] = obj

    return found


def search_main_channel_from_manifest(
        package_import_path: str = MANIFEST_CONFIG_PATH,
) -> tuple[Channel, str] | None:
    """
    扫描 manifest 模块，寻找 name == '__main__' 的 Channel。

    Returns:
        (channel, found_module) — found_module 是发现该 channel 的 Python 模块路径，
        如 ``MOSS.manifests.channels``。若未找到返回 None。
    """
    for manifest in scan_package(package_import_path, max_depth=2):
        if manifest.is_package:
            continue
        for name, obj in manifest.module.__dict__.items():
            if name.startswith('_') or not isinstance(obj, Channel):
                continue
            if obj.name() == "__main__":
                return obj, manifest.module.__name__
    return None
