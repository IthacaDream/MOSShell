"""
InMemoryRegistry — 基于内存 dict 的 ResourceRegistry 最小实现.
"""

from typing import Sequence

from ghoshell_moss.contracts.resource import (
    ResourceMeta,
    ResourceItem,
    ResourceStorage,
    ResourceRegistry,
    R,
)


class InMemoryRegistry(ResourceRegistry):
    """
    内存级别的 ResourceRegistry 实现.
    单进程, scheme → storage dict.

    用法:
        registry = InMemoryRegistry()
        await registry.register(LocalImageStorage(data_dir="~/.moss/resources"))
        metas = await registry.list_metas("pil-image")
        item = await registry.get_by_scheme("pil-image", "beach_photo")
    """

    def __init__(self):
        self._storages: dict[str, ResourceStorage] = {}

    async def register(self, storage: ResourceStorage) -> None:
        self._storages[storage.scheme()] = storage

    async def unregister(self, scheme: str) -> bool:
        return self._storages.pop(scheme, None) is not None

    async def schemes(self) -> Sequence[str]:
        return list(self._storages.keys())

    async def get_by_scheme(
        self, scheme: str, locator: str
    ) -> ResourceItem | None:
        storage = self._storages.get(scheme)
        if storage is None:
            return None
        return await storage.get(locator)

    async def get_by_item_type(
        self, item_cls: type[R], locator: str
    ) -> R | None:
        return await self.get_by_scheme(item_cls.scheme(), locator)

    async def list_metas(
        self,
        scheme: str,
        query: str | None = None,
        limit: int = 50,
    ) -> Sequence[ResourceMeta]:
        storage = self._storages.get(scheme)
        if storage is None:
            return []
        return await storage.list_metas(query, limit)

    async def help(self, scheme: str, question: str | None = None) -> str:
        storage = self._storages.get(scheme)
        if storage is None:
            return f"scheme '{scheme}' 未注册. 已注册: {list(self._storages.keys())}"
        return await storage.help(question)

    async def usage(self, scheme: str) -> str:
        storage = self._storages.get(scheme)
        if storage is None:
            return f"scheme '{scheme}' 未注册. 已注册: {list(self._storages.keys())}"
        return storage.usage()
