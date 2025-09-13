import threading
from xml import sax

import logging
import xml.sax
from xml.sax import saxutils
from typing import List, Iterable, Optional, Dict, Callable
from ghoshell_moss.concepts.command import CommandToken
from ghoshell_moss.concepts.errors import InterpretError, FatalError
from queue import Queue


class CMTLElement:
    """
    Utility class to generate Command Token in XMTL (Command Token Marked Language stream)
    """

    def __init__(self, *, idx: int, stream_id: str, chan: str, name: str, attrs):
        self.idx = idx
        self.ns = chan
        self.name = name
        self.deltas = ""
        self.part_idx = 0
        self._has_delta = False
        self.fullname = self.make_fullname(chan, name)
        self.attrs = dict(attrs)
        self.stream_id = stream_id

    @classmethod
    def make_fullname(cls, ns: str, name: str) -> str:
        return f"{ns}:{name}" if ns else name

    def start_token(self) -> CommandToken:
        attr_expression = []
        for k, v in self.attrs.items():
            quoted_value = saxutils.quoteattr(v)
            attr_expression.append(f"{k}={quoted_value}")
        exp = " " if len(attr_expression) > 0 else ""
        content = f"<{self.fullname}{exp}" + " ".join(attr_expression) + ">"
        part_idx = self.part_idx
        self.part_idx += 1
        return CommandToken(
            name=self.name,
            chan=self.ns,
            idx=self.idx,
            part_idx=part_idx,
            stream_id=self.stream_id,
            type="start",
            kwargs=self.attrs,
            content=content,
        )

    def on_child_command(self):
        if self._has_delta:
            self._has_delta = False
            self.deltas = ""
            self.part_idx += 1

    def add_delta(self, delta: str, gen_token: bool = True) -> Optional[CommandToken]:
        self.deltas += delta
        if not self._has_delta:
            self._has_delta = len(delta.strip()) > 0
            if self._has_delta:
                # fist none empty delta
                delta = self.deltas

        if gen_token and self._has_delta:
            return CommandToken(
                name=self.name,
                chan=self.ns,
                idx=self.idx,
                part_idx=self.part_idx,
                stream_id=self.stream_id,
                type="delta",
                kwargs=None,
                content=delta,
            )
        return None

    def end_token(self) -> CommandToken:
        if self._has_delta:
            self.part_idx += 1
        return CommandToken(
            name=self.name,
            chan=self.ns,
            idx=self.idx,
            part_idx=self.part_idx,
            stream_id=self.stream_id,
            type="end",
            kwargs=None,
            content=f"</{self.fullname}>",
        )


class ParserStopped(Exception):
    pass


class CTMLSaxHandler(xml.sax.ContentHandler, xml.sax.ErrorHandler):

    def __init__(
            self,
            root_tag: str,
            stream_id: str,
            callback: Callable[[CommandToken | Exception | None], None],
            *,
            default_chan: str = "",
            logger: Optional[logging.Logger] = None,
    ):
        """
        :param root_tag: do not send command token with root_tag
        :param stream_id: stream id to mark all the command token
        :param callback: callback function
        :param default_chan: default channel name

        """
        self._root_tag = root_tag
        self._stream_id = stream_id
        self._default_chan = default_chan
        # idx of the command token
        self._idx = 0
        # command token callback
        self._callback = callback
        # get the logger
        self._logger = logger or logging.getLogger("CTMLSaxHandler")
        # simple stack for unfinished element
        self._parsing_element_stack: List[CMTLElement] = []
        # event to notify the parsing is over.
        self.done_event = threading.Event()

    def _send_to_callback(self, token: CommandToken) -> None:
        if not self.done_event.is_set():
            self._callback(token)
        else:
            # todo: log
            pass

    def startElementNS(self, name: tuple[str, str], qname: str, attrs: Dict) -> None:
        if self.done_event.is_set():
            return None
        cmd_chan, cmd_name = name
        element = CMTLElement(
            idx=self._idx,
            stream_id=self._stream_id,
            chan=cmd_chan or self._default_chan,
            name=cmd_name,
            attrs=attrs,
        )
        if len(self._parsing_element_stack) > 0:
            self._parsing_element_stack[-1].on_child_command()

        # using stack to handle elements
        self._parsing_element_stack.append(element)
        self._idx += 1
        if element.fullname != self._root_tag:
            token = element.start_token()
            self._send_to_callback(token)

    def endElementNS(self, name: tuple[str, str], qname):
        if len(self._parsing_element_stack) == 0:
            raise FatalError("CTMLElement end element without existing one")
        element = self._parsing_element_stack.pop(-1)
        if element.fullname != self._root_tag:
            token = element.end_token()
            self._send_to_callback(token)

    def characters(self, content: str):
        if len(self._parsing_element_stack) == 0:
            # something goes wrong, on char without properly start or end
            raise FatalError("CTMLElement on character without properly start or end")
        element = self._parsing_element_stack[-1]
        token = element.add_delta(content)
        if token is not None:
            self._send_to_callback(token)

    def endDocument(self):
        # todo: log
        self.done_event.set()

    def error(self, exception: Exception):
        self._logger.exception(exception)

    def fatalError(self, exception: Exception):
        self._logger.exception(exception)
        if self.done_event.is_set():
            return None
        self._callback(exception)
        self.done_event.set()

    def warning(self, exception):
        self._logger.warning(exception)


class CTMLParser:
    """
    parsing input stream into Command Tokens
    """

    def __init__(
            self,
            stream_id: str = "",
            callback: Callable[[CommandToken], None] | None = None,
            *,
            root_tag: str = "cmtl",
            default_chan: str = "",
            logger: Optional[logging.Logger] = None,
    ):
        self.root_tag = root_tag
        self.logger = logger or logging.getLogger("CTMLParser")
        self._callback = callback
        self._buffer = ""
        self._parsed: List[CommandToken] = []
        self._handler = CTMLSaxHandler(
            root_tag,
            stream_id,
            self._send_callback,
            default_chan=default_chan,
            logger=logger,
        )
        self._sax_parser = xml.sax.make_parser()
        self._sax_parser.setFeature(xml.sax.handler.feature_namespaces, 1)
        self._sax_parser.setContentHandler(self._handler)
        self._sax_parser.setErrorHandler(self._handler)
        self._stopped = False
        self._started = False

    def _send_callback(self, token: CommandToken | Exception | None) -> None:
        if isinstance(token, dict) or isinstance(token, CommandToken):
            self._parsed.append(token)
            if self._callback is not None:
                self._callback(token)
        elif isinstance(token, Exception):
            raise token
        else:
            raise ParserStopped()

    def is_running(self) -> bool:
        return self._started and not self._stopped

    def is_done(self) -> bool:
        return self._handler.done_event.is_set()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._sax_parser.feed(f'<{self.root_tag}>')

    def feed(self, delta: str) -> None:
        if self._stopped:
            raise ParserStopped()
        elif self.is_running():
            self._buffer += delta
            self._sax_parser.feed(delta)
        else:
            return

    def end(self) -> None:
        if not self.is_running():
            return
        self._sax_parser.feed(f'</{self.root_tag}>')

    def stop(self) -> None:
        """
        stop the parser and clear the resources.
        """
        if self._stopped:
            return
        self._stopped = True
        # cancel
        self._handler.done_event.set()
        self._sax_parser.close()

    def buffer(self) -> str:
        return self._buffer

    def parsed(self) -> List[CommandToken]:
        return self._parsed

    def wait_done(self, timeout: float | None = None) -> None:
        """
        wait until the parsing is done
        do not use this before the end_feed in the same thread
        """
        self._handler.done_event.wait(timeout=timeout)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @classmethod
    def parse(
            cls,
            stream: Iterable[str],
            *,
            root_tag: str = "ctml",
            stream_id: str = "",
            default_chan: str = "",
            logger: Optional[logging.Logger] = None,
    ) -> Iterable[CommandToken]:
        """
        simple example of parsing input stream into command token stream with a thread.
        but not a good practice
        """
        if isinstance(stream, str):
            stream = [stream]

        q: Queue[CommandToken | Exception | None] = Queue()

        parser = cls(stream_id, q.put_nowait, root_tag=root_tag, default_chan=default_chan, logger=logger)
        try:

            def _consumer():
                """
                async feeding
                """
                with parser:
                    for delta in stream:
                        parser.feed(delta)
                    parser.end()

            t = threading.Thread(target=_consumer, daemon=True)
            t.start()
            while True:
                item = q.get(block=True)
                if item is None:
                    # reach the end
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            parser.stop()
