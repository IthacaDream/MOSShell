import threading

from .command import CommandToken, CommandTokenStream, CommandTask
from abc import ABC, abstractmethod
from typing import Iterable, Callable, Optional
from threading import Thread, Event

ParserCallback = Callable[[CommandTokenStream | None], None]


class Parser(ABC):
    """
    Parse a string into a CommandToken.
    """

    @classmethod
    @abstractmethod
    def system_prompt(cls) -> str:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def feed(self, tokens: str) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def with_callback(self, callback: Callable[[CommandTokenStream | None], None]) -> None:
        pass

    @abstractmethod
    def parsed(self) -> Iterable[CommandToken]:
        pass

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class Interpreter(ABC):
    interrupt: Event = Event()
    done: Event = Event()
    parser: Parser

    @abstractmethod
    def append(self, text: str) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.done.set()

    @abstractmethod
    def executed(self) -> Iterable[CommandTask]:
        pass

    # def parse(self, tokens: Iterable[str], callback: Optional[ParserCallback] = None) -> Iterable[CommandToken]:
    #     parser = self.new_parser(callback)
    #     stream = CommandTokenStream()
    #
    #     def consumer():
    #         with parser:
    #             for token in tokens:
    #                 parser.add(token)
    #
    #     t = threading.Thread(target=consumer, daemon=True)
    #     t.start()
    #     return stream
    #
    # def parse_string(self, tokens: str) -> Iterable[CommandToken]:
    #     parser = self.new_parser()
    #     buffer = []
    #
    #     def callback(token: CommandToken | None) -> None:
    #         if token is not None:
    #             buffer.append(token)
    #
    #     with parser:
    #         parser.add(tokens)
    #     return buffer
    #
    # def pipe(self, tokens: Iterable[str], callback: Optional[ParserCallback] = None) -> Iterable[str]:
    #     parser = self.new_parser(callback)
    #     with parser:
    #         for token in tokens:
    #             parser.add(token)
    #             yield token
