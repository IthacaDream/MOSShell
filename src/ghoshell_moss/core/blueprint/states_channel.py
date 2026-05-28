from abc import ABC, abstractmethod

from typing import Protocol, runtime_checkable
from typing_extensions import Self

from ghoshell_container import IoCContainer
from ghoshell_moss.message import Message
from ghoshell_moss.core.concepts.command import Command
from ghoshell_moss.core.concepts.channel import Channel, ChannelName
from ghoshell_moss.core.blueprint.channel_builder import Builder, MutableChannel
from PIL.Image import Image

__all__ = [
    'ChannelState', 'ChannelStateBuilder', 'StatefulChannel',
    'new_state_builder', 'new_channel_from_state', 'new_stateful_channel',
    'PrimeChannel', 'new_prime_channel', 'new_shell_main_channel',
    'ChannelModule',
    'new_default_shell_main_channel',
]

"""
how to build a stateful channel
"""


class ChannelModule(Protocol):
    """
    生命周期感知的模块化能力单元。

    通过 BaseStateChannel.with_module() 注册为 channel 的永久能力模块。
    所有 module 同时激活、累积叠加 — 与 with_state() 的排他切换正交。
    PyChannelBuilder 和任意 ChannelState 实现自动满足此 Protocol。

    Protocol 意味着结构子类型 — 只要实现了 name() + own_commands() 的类型就是 ChannelModule，
    不需要显式继承。on_startup/on_close/get_instruction/get_context_messages 是可选的生命周期钩子。

    运行过程中要使用 IoC 容器, 可以通过 channel_builder.CommandUtil.get_contract 获取
    """

    def name(self) -> str: ...

    def own_commands(self) -> dict[str, Command]: ...

    async def on_startup(self) -> None:
        # 可以通过 CommandUtil.get_contract 获取 ioc 绑定依赖.
        pass

    async def on_close(self) -> None:
        # 可以通过 CommandUtil.get_contract 获取 ioc 绑定依赖.
        pass

    async def get_instruction(self) -> str:
        return ""

    async def get_context_messages(self) -> list[Message]:
        # 可以通过 CommandUtil.get_contract 获取 ioc 绑定依赖.
        return []


class ChannelState(ABC):
    """
    Channel 的运行时状态, 用来快速构建一个 StateChannel.

    运行过程中要使用 IoC 容器, 可以通过 bootstrap 函数被调用时获取并持有依赖
    或者 channel_builder.CommandUtil.get_contract 在每个 command 和生命周期函数被调用时获取.
    """

    @abstractmethod
    def name(self) -> str:
        """
        return name of the state
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        return description of the state
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        if the state is available
        """
        pass

    def is_dynamic(self) -> bool:
        """
        if the state is dynamic, need to refresh each time.
        """
        # 既然是有状态的 Channel, 默认是动态的.
        return True

    async def get_instruction(self) -> str:
        """
        return instruction provided by the state
        """
        return ''

    async def get_context_messages(self) -> list[Message | str | Image]:
        """
        return the context messages from the state.
        """
        return []

    async def on_startup(self) -> None:
        """
        when channel startup.
        """
        return None

    async def on_close(self) -> None:
        """
        when channel close.
        """
        return None

    async def on_running(self) -> None:
        """
        when channel is running.
        """
        return None

    async def on_idle(self) -> None:
        """
        when channel is idle, all the commands are done and the children are idle as well
        """
        return None

    @abstractmethod
    def own_commands(self) -> dict[str, Command]:
        """
        return the commands mapping by name
        """
        pass

    @abstractmethod
    def get_own_command(self, name: str) -> Command | None:
        """
        get a command by name
        """
        pass

    def bootstrap(self, container: IoCContainer) -> None:
        """
        register something to the container. or get some contracts from it.
        函数会被 ChannelRuntime 实例化后调用.
        """
        return

    def get_children(self) -> dict[ChannelName, Channel]:
        """
        return the sustain children channel
        """
        return {}

    def get_virtual_children(self) -> dict[ChannelName, Channel]:
        """
        return the virtual children that may be changed during runtime
        """
        return {}


class ChannelStateBuilder(Builder, ChannelState, ABC):
    """
    Channel State which itself is mutable.
    """

    @abstractmethod
    def add_virtual_channel(self, channel: Channel, alias: ChannelName | None = None) -> None:
        """
        add virtual channel during runtime.
        wrap this method into a command
        """
        pass

    @abstractmethod
    def remove_virtual_channel(self, name: str) -> None:
        """
        remove virtual channel during runtime.
        wrap this method into a command
        """
        pass


def new_state_builder(name: str, description: str = "") -> ChannelStateBuilder:
    """
    new state builder
    """
    from ghoshell_moss.core.py_channel import PyChannelBuilder
    return PyChannelBuilder(name=name, description=description)


class StatefulChannel(Channel, ABC):
    """
    Stateful Channel which can switch to one of multiple states.
    """

    @abstractmethod
    def main_state(self) -> ChannelState:
        """
        return the main state of the channel
        """
        pass

    @abstractmethod
    def new_state(self, name: str, description: str) -> ChannelStateBuilder:
        """
        create new substate of the channel
        """
        pass

    @abstractmethod
    def states(self) -> dict[str, ChannelState]:
        """
        return the switchable states, without main states.
        """
        pass

    def modules(self) -> dict[str, ChannelModule]:
        """
        return the permanent capability modules.
        默认返回空 dict，不是每个 StatefulChannel 都需要 module。
        """
        return {}

    @abstractmethod
    def with_module(self, module: ChannelModule) -> Self:
        """
        register a permanent capability module to the channel.
        unlike with_state(), modules are cumulative — all active simultaneously.
        """
        pass

    @abstractmethod
    def with_state(self, state: ChannelState, alias: str | None = None, is_default: bool = False) -> Self:
        """
        register a named substate to the channel.
        """
        pass

    @abstractmethod
    def default_state_name(self) -> str:
        pass


class PrimeChannel(StatefulChannel, MutableChannel, ABC):
    """
    super channel with all abilities.
    """

    @property
    @abstractmethod
    def build(self) -> ChannelStateBuilder:
        pass

    def add_virtual_channel(self, channel: Channel, alias: ChannelName | None = None) -> None:
        """
        add virtual channel during runtime.
        wrap this method into a command
        """
        # 运行时可以执行.
        self.build.add_virtual_channel(channel, alias)

    def remove_virtual_channel(self, name: str) -> None:
        """
        remove virtual channel during runtime.
        wrap this method into a command
        """
        # 运行时可以执行.
        self.build.remove_virtual_channel(name)


def new_channel_from_state(state: ChannelState, id: str | None = None) -> StatefulChannel:
    """
    create new channel by state object
    """
    from ghoshell_moss.core.py_channel import BaseStateChannel
    return BaseStateChannel(state, uid=id)


def new_stateful_channel(name: str, description: str = "") -> StatefulChannel:
    """
    create new stateful channel with builders.
    """
    from ghoshell_moss.core.py_channel import PyChannel
    return PyChannel(name=name, description=description)


def new_prime_channel(name: str, description: str = "") -> PrimeChannel:
    from ghoshell_moss.core.py_channel import PyChannel
    return PyChannel(name=name, description=description)


def new_shell_main_channel(description: str = "") -> PrimeChannel:
    """
    创建 CTML shell 的主 channel (__main__)。

    FastAPI-like 入口。返回空的 main channel，
    可选调用 ``inject_system_primitives(main)`` 注入系统原语，
    可继续 import_channels / with_state / with_module 组合。
    """
    from ghoshell_moss.core.py_channel import PyChannel
    description = description or "MOSS main channel"
    return PyChannel(name="__main__", description=description, blocking=True)


def new_default_shell_main_channel(
        description: str = "",
) -> PrimeChannel:
    """
    创建一个标准的, 默认的 shell main channel.
    提示如何组建 Shell Main Channel.
    """
    from ghoshell_moss.core.ctml.shell.ctml_main import inject_system_primitives
    from ghoshell_moss.core.speech import SpeechChannelModule
    from ghoshell_moss.host.app_store_channel import AppStoreChannel

    main = new_shell_main_channel(description=description)

    # -- 系统原语 --------------------------------------------------
    inject_system_primitives(main, extended=True)

    # -- App Store ---------------------------------------------------
    main.import_channels(AppStoreChannel(name='apps'))

    # -- Speech --------------------------------------------------
    main.with_module(SpeechChannelModule())

    return main


# ---- 面向对象使用思路示范 ---- #

class ChannelStateFactory(ChannelState, ABC):
    """
    如何从 State 类定义开始, 获取一个运行时可以生成 Channel 的 ChannelFactory 对象.

    这不是一个必要的抽象, 仅仅展示如何面向对象的, 以 ChannelState 的思路来定义一个 Channel.

    """

    @classmethod
    @abstractmethod
    def new(cls, container: IoCContainer) -> Self:
        """从 ioc 容器中可以实例化一个 State. 这个函数可以是 object method / class method"""
        pass

    @classmethod
    def factory(cls, container: IoCContainer) -> Channel:
        """factory 函数本身就是一个 channel factory, 所以可以被其它 channel import. """
        state = cls.new(container)
        return new_channel_from_state(state)
