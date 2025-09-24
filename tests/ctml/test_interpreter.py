from ghoshell_moss.mocks.outputs import ArrOutput
from ghoshell_moss.ctml.interpreter import CTMLInterpreter
from ghoshell_moss.concepts.command import PyCommand
from collections import deque
import asyncio
import pytest


@pytest.mark.asyncio
async def test_interpreter_baseline():
    async def foo() -> int:
        return 123

    queue = deque()
    interpreter = CTMLInterpreter(
        commands=[PyCommand(foo)],
        stream_id="test",
        output=ArrOutput(),
        callback=queue.append,
    )

    content = "<foo>h</foo>"

    async with interpreter:
        for c in content:
            await interpreter.feed(c)

        await interpreter.wait_until_done()

    # 所有的 input 被 buffer 了.
    assert content == interpreter.inputted()
    assert len(list(interpreter.parsed_tokens())) == 5
    assert len(queue) == 3
