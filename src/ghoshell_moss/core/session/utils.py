from ghoshell_moss.core.blueprint.session import (
    OutputBuffer, OutputItem
)
from typing import Iterable

import threading
import time

__all__ = ['SimpleOutputBuffer']


class SimpleOutputBuffer(OutputBuffer):

    def __init__(self, maxsize: int, on_change_interval: float = 0.5) -> None:
        self._max = maxsize
        self._on_change_interval = on_change_interval
        self._last_change_at: float = 0.0
        self._closed = False
        self._messages_lock = threading.Lock()
        self._items: list[OutputItem] = []

    def close(self) -> None:
        self._closed = True

    def is_closed(self) -> bool:
        return self._closed

    def add_output(self, item: OutputItem) -> None:
        with self._messages_lock:
            if len(self._items) > 0:
                role = item.role
                last = self._items[-1]
                if last.role == role:
                    last.messages.extend(item.messages)
                else:
                    self._items.append(item)
            else:
                self._items.append(item)
            if len(self._items) > self._max:
                self._items = self._items[len(self._items) - self._max:]
            self._last_change_at = time.time()

    def values(self) -> Iterable[OutputItem]:
        with self._messages_lock:
            items = self._items.copy()
        return items

    def updated_at(self) -> float:
        return self._last_change_at
