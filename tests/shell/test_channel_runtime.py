import logging

from ghoshell_moss.shell.runtime import ChannelRuntimeImpl
from ghoshell_container import Container
from ghoshell_moss import PyChannel, PyCommand, BaseCommandTask
import pytest
import sys


@pytest.mark.asyncio
async def test_channel_runtime_impl_baseline():
    chan = PyChannel(name="")

    @chan.build.command()
    async def foo() -> int:
        return 123

    runtime = ChannelRuntimeImpl(Container(), chan)
    async with runtime:
        assert runtime.is_running()
        assert runtime.is_available()
        await runtime.wait_until_idle()
        assert not runtime.is_busy()

        foo_cmd = runtime.channel.client.get_command("foo")
        task = BaseCommandTask.from_command(foo_cmd)
        runtime.append(task)
        await task.wait()
    assert task.done()
    assert task._result is 123


@pytest.mark.asyncio
async def test_child_channel_runtime_is_running():
    main = PyChannel(name="")
    a = main.new_child('a')

    @a.build.command()
    async def foo() -> int:
        return 123

    runtime = ChannelRuntimeImpl(Container(), main)
    async with runtime:
        assert a.is_running()
        assert main.children().get('a') is a
        assert runtime.children_runtimes.get('a').is_running()
