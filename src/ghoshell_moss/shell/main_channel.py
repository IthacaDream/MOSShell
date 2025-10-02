from ghoshell_moss.concepts.channel import Channel
from ghoshell_moss.channels.py_channel import PyChannel

__all__ = ['MainChannel']


class MainChannel(PyChannel):
    pass


def create_main_channel() -> Channel:
    return MainChannel(
        name="",
        description="todo",
        block=True,
    )
