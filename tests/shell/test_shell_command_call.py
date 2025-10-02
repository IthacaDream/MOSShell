import asyncio

import pytest

from ghoshell_moss import Interpreter


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
        # 晚执行 0.1 秒.
        await asyncio.sleep(0.1)
        return 456

    async with shell:
        interpreter = shell.interpret()
        assert isinstance(interpreter, Interpreter)
        assert shell.is_running()
        async with interpreter:
            interpreter.feed("<a:foo /><b:bar />")
            assert shell.is_running()
            tasks = await interpreter.wait_execution_done(10)

            assert len(tasks) == 2
            result = []
            for task in tasks.values():
                assert task.success()
                result.append(task.result())
            # 获取到结果.
            assert result == [123, 456]
            assert ['a', 'b'] == [t.exec_chan for t in tasks.values()]
            # 验证并发执行.
            task_list = list(tasks.values())
            # 两个任务几乎同时启动.
            running_gap = abs(task_list[0].trace.get('running') - task_list[1].trace.get('running'))
            assert running_gap < 0.01
            done_gap = abs(task_list[1].trace.get('done') - task_list[0].trace.get('done'))
            assert done_gap > 0.05
