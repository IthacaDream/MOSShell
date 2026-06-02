from ghoshell_container import Container, IoCContainer
from abc import abstractmethod, ABC
from ghoshell_moss.core.blueprint.channel_builder import new_channel, ChannelInterface, Channel
from typing_extensions import Self
import pytest


@pytest.mark.asyncio
async def test_channel_interface_pattern() -> None:
    main = new_channel(name="__main__")

    class FooInterface(ChannelInterface):

        @abstractmethod
        async def foo(self) -> str:
            """hello"""
            pass

        @classmethod
        @abstractmethod
        def new(cls, container: IoCContainer) -> Self:
            pass

        def as_channel(self) -> Channel:
            channel = new_channel(name="foo")
            channel.build.command(
                interface=FooInterface.foo,
            )(self.foo)
            return channel

    class FooImpl(FooInterface):

        def __init__(self, c: IoCContainer):
            self.c = c

        async def foo(self) -> bool:
            """world"""
            return isinstance(self.c, IoCContainer)

        @classmethod
        def new(cls, container: IoCContainer) -> Self:
            return cls(container)

    # 直接将工厂方法注入到通道中.
    main.import_channels(FooImpl.factory)
    container = Container(name="hello")
    async with main.bootstrap(container) as runtime:
        r = await runtime.execute_command('foo:foo')
        assert r
        command = runtime.get_command('foo:foo')
        assert command is not None
        assert 'hello' in command.meta().interface
