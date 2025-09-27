from ghoshell_moss.concepts.channel import Channel
from ghoshell_moss.channels.py_channel import PyChannel


class ShellMainChannel(PyChannel):
    pass


def create_main_channel() -> Channel:
    return ShellMainChannel(
        name="",
        description="todo",
        block=True,
    )
