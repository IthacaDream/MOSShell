import asyncio

import pytest


@pytest.mark.asyncio
async def test_shell_execution_baseline():
    from ghoshell_moss.shell import new_shell

    shell = new_shell()
    a_chan = shell.main_channel.new_child('a')
    b_chan = shell.main_channel.new_child('b')

    @a_chan.build.command()
    async def foo() -> int:
        return 123

    @b_chan.build.command()
    async def bar() -> int:
        await asyncio.sleep(0.01)
        return 456

    async with shell:
        async with shell.interpret() as interpreter:
            interpreter.feed("<foo /><bar />")
            tasks = await interpreter.wait_execution_done()

            assert len(tasks) == 2
            result = []
            for task in tasks.values():
                assert task.done()
                result.append(task.result())
            assert result == [123, 456]
