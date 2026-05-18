from ghoshell_moss.core.blueprint.ghost import GhostMeta
from ghoshell_moss.core.codex.discover import scan_package
from importlib import import_module

__all__ = [
    'ROOT_GHOSTS_PACKAGE',
    'find_ghost_from_package',
    'list_ghosts_from_root_package',
]

ROOT_GHOSTS_PACKAGE = 'MOSS.ghosts'


def list_ghosts_from_root_package(
    package_import_path: str = ROOT_GHOSTS_PACKAGE,
) -> dict[str, GhostMeta]:
    """发现根包下所有 Ghost，仅扫描一级子模块/子包."""
    ghosts: dict[str, GhostMeta] = {}
    try:
        for module_manifest in scan_package(package_import_path, max_depth=1):
            if module_manifest.module_path == package_import_path:
                continue
            ghost_meta = find_ghost_from_package(module_manifest.module_path)
            if ghost_meta is not None:
                ghosts[ghost_meta.name()] = ghost_meta
    except ImportError:
        pass
    return ghosts


def find_ghost_from_package(package_import_path: str) -> GhostMeta | None:
    """从模块/包中查找 GhostMeta 实例.

    约定: 模块级变量 `ghost` 或 `__ghost__` 指向一个 GhostMeta 实例.
    """
    try:
        module = import_module(package_import_path)
    except ImportError:
        return None

    for attr in ('ghost', '__ghost__'):
        instance = getattr(module, attr, None)
        if isinstance(instance, GhostMeta):
            return instance

    return None
