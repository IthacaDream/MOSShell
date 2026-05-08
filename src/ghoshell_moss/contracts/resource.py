"""
资源管理抽象 (Resource Management Abstraction).

-- 万物 Agent or otherwise --

核心概念:
  scheme://locator    全局资源句柄, 字符串即可引用, 跨工具传递.
  ResourceMeta        给 AI 阅读的结构化元信息 (Pydantic).
  ResourceItem        meta + 懒加载的实际数据.
  ResourceStorage     单个 scheme 的资源后端.
  ResourceRegistry    跨 scheme 的路由层 (VFS).

特性:
  - 自解释: usage() + help() 让 AI 自服务发现.
  - 查询语义由后端定义: SQL/keyword/regex/自然语言 均可, usage() 声明.
  - 可扩展: 新 scheme = 新 Storage 实现, 不改接口.
  - 可传递: scheme://locator 作为全局句柄在工具间传递引用.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence

from pydantic import BaseModel, Field

RESOURCE_TYPE = TypeVar("RESOURCE_TYPE")


class ResourceMeta(BaseModel, ABC):
    """
    可寻址的资源元信息.
    全局地址: scheme://locator
    """

    locator: str = Field(
        description="资源唯一标识, 与 scheme 组合为 scheme://locator",
    )
    description: str = Field(
        description="描述信息",
    )

    @classmethod
    @abstractmethod
    def scheme(cls) -> str:
        """
        scheme 名称. scheme://locator 构成全局资源地址.
        """
        pass

    @classmethod
    @abstractmethod
    def scheme_description(cls) -> str:
        """
        scheme 的介绍, 说明这类资源是什么.
        """
        pass

    def as_content(self) -> str:
        """
        返回给 AI 阅读的 JSON 字符串.
        Pydantic BaseModel 天然可序列化, 此方法作为显式约定.
        """
        return self.model_dump_json()


RESOURCE_META = TypeVar("RESOURCE_META", bound=ResourceMeta)


class ResourceItem(Generic[RESOURCE_META, RESOURCE_TYPE], ABC):
    """
    资源项. meta 总是立即可用, get() 懒加载实际数据.

    meta_type() 声明 Item 携带的 Meta 类型.
    scheme() 从 meta_type() 派生, 不需要额外实现.
    """

    @classmethod
    @abstractmethod
    def meta_type(cls) -> type[RESOURCE_META]:
        """
        返回此 Item 所携带的 ResourceMeta 子类.
        """
        pass

    @classmethod
    def scheme(cls) -> str:
        """
        scheme 名称. 从 meta_type() 派生.
        """
        return cls.meta_type().scheme()

    @property
    @abstractmethod
    def meta(self) -> RESOURCE_META:
        """
        返回资源元信息. 始终立即可用, 不触发 I/O.
        """
        pass

    @abstractmethod
    async def get(self) -> RESOURCE_TYPE:
        """
        返回实际数据对象. 可能触发 I/O (文件读取/API 调用等).
        """
        pass


class ClarifyError(Exception):
    """
    需要调用者澄清时抛出.
    question 向上层传递, 上层 (AI 或人类) 补全信息后重试.
    """

    def __init__(self, question: str, *args) -> None:
        self.question = question
        super().__init__(question, *args)


class ResourceStorage(Generic[RESOURCE_META, RESOURCE_TYPE], ABC):
    """
    单一 scheme 的资源存储后端.
    注册到 ResourceRegistry 后, AI 通过 scheme 访问.
    """

    @classmethod
    @abstractmethod
    def scheme(cls) -> str:
        """
        资源的 scheme. 通常委托给 ResourceMeta.scheme().
        """
        pass

    @classmethod
    @abstractmethod
    def scheme_description(cls) -> str:
        """
        scheme 的介绍.
        """
        pass

    @abstractmethod
    def usage(self) -> str:
        """
        解释自身的用法.
        关键内容: query 支持什么语法 (SQL/keyword/regex/自然语言),
        限制, 使用示例, 返回的 ResourceMeta 字段含义.
        AI 调用者应该先读 usage() 再使用.
        """
        pass

    @abstractmethod
    async def help(self, question: str | None = None) -> str:
        """
        类似 CLI 的 --help 或 man.
        question=None 时返回概览, 有值时回答具体问题.
        """
        pass

    @abstractmethod
    async def list_metas(
        self,
        query: str | None = None,
        limit: int = 50,
    ) -> Sequence[RESOURCE_META]:
        """
        浏览或搜索资源.
        :param query: 查询条件. None 表示浏览全量.
                      查询语法由各 Storage 自行定义, 通过 usage() 声明.
        :param limit: 返回数量上限.
        :return: 匹配的资源元信息列表.
        """
        pass

    @abstractmethod
    async def get(
        self, locator: str
    ) -> ResourceItem[RESOURCE_META, RESOURCE_TYPE] | None:
        """
        按 locator 获取资源项. scheme 由 Storage 自身决定.
        :return: None 表示不存在.
        """
        pass

    @abstractmethod
    async def put(
        self, item: ResourceItem[RESOURCE_META, RESOURCE_TYPE]
    ) -> str:
        """
        保存或更新资源. 若是新资源, Storage 负责分配 locator.
        :return: locator, 调用者用此 locator 构造 scheme://locator 句柄.
        """
        pass

    @abstractmethod
    async def delete(self, locator: str) -> bool:
        """
        删除指定资源.
        :return: True 表示删除成功, False 表示资源不存在.
        """
        pass


R = TypeVar("R", bound=ResourceItem)


class ResourceRegistry(ABC):
    """
    跨 scheme 的资源路由层 (VFS).
    以 scheme (str) 为主键索引.

    最小实现 = 内存 dict.
    完整实现 = web hub.

    get_by_item_type() 通过 item_cls.scheme() 路由,
    保留 Python 侧的静态类型.
    """

    @abstractmethod
    async def register(self, storage: ResourceStorage) -> None:
        """
        注册一个 ResourceStorage 实现. 以 storage.scheme() (str) 为键.
        """
        pass

    @abstractmethod
    async def unregister(self, scheme: str) -> bool:
        """
        移除指定 scheme 的存储.
        :return: True 表示移除成功, False 表示该 scheme 未注册.
        """
        pass

    @abstractmethod
    async def schemes(self) -> Sequence[str]:
        """
        列出所有已注册的 scheme.
        """
        pass

    @abstractmethod
    async def get_by_scheme(
        self, scheme: str, locator: str
    ) -> ResourceItem | None:
        """
        按 scheme 字符串获取资源.
        :return: None 表示 scheme 未注册或 locator 不存在.
        """
        pass

    @abstractmethod
    async def get_by_item_type(
        self, item_cls: type[R], locator: str
    ) -> R | None:
        """
        按 ResourceItem 子类获取资源. 保留具体类型.
        :return: None 表示未注册或 locator 不存在.
        """
        pass

    @abstractmethod
    async def list_metas(
        self,
        scheme: str,
        query: str | None = None,
        limit: int = 50,
    ) -> Sequence[ResourceMeta]:
        """
        浏览或搜索指定 scheme 的资源.
        :param scheme: 目标 scheme.
        :param query: 查询条件. None 表示浏览全量.
        :param limit: 返回数量上限.
        :return: 匹配的资源元信息列表.
        """
        pass

    @abstractmethod
    async def help(self, scheme: str, question: str | None = None) -> str:
        """
        获取指定 scheme 的帮助信息.
        """
        pass

    @abstractmethod
    async def usage(self, scheme: str) -> str:
        """
        获取指定 scheme 的用法说明.
        """
        pass
