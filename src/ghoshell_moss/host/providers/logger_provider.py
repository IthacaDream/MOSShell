import logging
from typing import Type, Iterable

from ghoshell_moss.contracts.workspace import Workspace
from ghoshell_moss.contracts.logger import LoggerItf, default_logger_formatter
from ghoshell_container import Provider, IoCContainer, INSTANCE
from logging.handlers import TimedRotatingFileHandler

__all__ = [
    'HostLoggerProvider',
]


class HostLoggerProvider(Provider[LoggerItf]):

    def __init__(
            self,
            *,
            handler_name: str = 'moss_file_handler',
            log_handler: logging.Handler | None = None,
            log_file_name: str = 'moss.log',
    ):
        self._handler_name = handler_name
        self._log_handler = log_handler
        self._log_file_name = log_file_name

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[LoggerItf]:
        return LoggerItf

    def aliases(self) -> Iterable[Type[INSTANCE]]:
        yield logging.Logger

    def factory(self, con: IoCContainer) -> LoggerItf:
        ws = con.get(Workspace)
        if ws is None:
            return logging.getLogger('moss')

        moss_logger = logging.getLogger('moss')

        # 已有非 NullHandler 的 handler 则不重复添加
        for h in moss_logger.handlers:
            if h.get_name() == self._handler_name:
                return moss_logger

        handler = self._log_handler
        if handler is None:
            filename = ws.runtime().sub_storage('logs').abspath().joinpath(self._log_file_name)
            handler = TimedRotatingFileHandler(
                filename=str(filename),
                when='d',
                interval=1,
                backupCount=5,
            )
            handler.set_name(self._handler_name)
            handler.setLevel(logging.INFO)
            handler.setFormatter(default_logger_formatter())

        moss_logger.addHandler(handler)
        return moss_logger
