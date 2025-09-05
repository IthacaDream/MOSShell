from abc import ABC, abstractmethod
from typing import List
from .channel import Channel
from .interpreter import Interpreter
from .command import CommandTask, CommandCall
from ghoshell_container import IoCContainer


class TextOutput(ABC):

    @abstractmethod
    async def new_batch(self, batch_id: str | None = None, output: bool = False) -> str:
        pass

    @abstractmethod
    async def write(self, batch_id: str, output: str) -> str:
        pass

    @abstractmethod
    async def output(self, batch_id: str) -> None:
        pass

    @abstractmethod
    async def wait_batch_done(self, batch_id: str) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class Controller(ABC):

    @abstractmethod
    async def loop(self, times: int, __text__: str) -> None:
        pass

    @abstractmethod
    async def group(self, __text__: str) -> None:
        pass

    @abstractmethod
    async def clear(self, __text__: str) -> None:
        pass

    @abstractmethod
    async def wait_for(self, __text__: str, timeout: float | None = None) -> None:
        pass

    @abstractmethod
    async def wait_done(self, timeout: float | None = None) -> None:
        pass


# @abstractmethod
# class Stream(ABC):
#     @abstractmethod
#     def write(self, chars: bytes | None) -> None:
#         pass
#
#     @abstractmethod
#     def read(self, wait_until_done: bool = False) -> bytes | None:
#         pass


class MOSSShell(ABC):
    """
    Model-oriented Operating System Shell
    面向模型提供的 Shell, 让 AI 可以操作自身所处的系统.
    """

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @property
    @abstractmethod
    def main(self) -> Channel:
        """
        Shell 自身的主轨. 主轨同时可以用来注册所有的子轨.
        """
        pass

    # --- properties --- #

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        pass

    @property
    @abstractmethod
    def interpreter(self) -> Interpreter:
        pass

    # --- runtime --- #

    @abstractmethod
    def append(self, command: CommandTask) -> None:
        pass

    @abstractmethod
    def prepend(self, command: CommandTask) -> None:
        pass

    @abstractmethod
    def new_command_task(self, call: CommandCall) -> CommandTask:
        pass

    @abstractmethod
    def wait_until_idle(self, timeout: float | None = None) -> None:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def is_idle(self) -> bool:
        pass

    @abstractmethod
    def clear(self, *chans: str) -> None:
        pass

    @abstractmethod
    def defer_clear(self, *chans: str) -> None:
        pass

    @abstractmethod
    def reset(self, *chans: str) -> None:
        pass

    # --- lifecycle --- #

    @abstractmethod
    def bootstrap(self) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass
