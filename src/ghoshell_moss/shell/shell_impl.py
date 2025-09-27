from typing import Dict, Optional, List, Iterable
from typing_extensions import Literal, Self
from ghoshell_moss.concepts.shell import MOSSShell, ChannelRuntime, Output
from ghoshell_moss.concepts.command import Command, CommandTask
from ghoshell_moss.concepts.channel import Channel, ChannelMeta
from ghoshell_moss.concepts.interpreter import Interpreter
from ghoshell_moss.ctml.interpreter import CTMLInterpreter
from ghoshell_moss.mocks.outputs import ArrOutput
from ghoshell_moss.shell.main_channel import ShellMainChannel
from ghoshell_moss.shell.channel_runtime import ChannelRuntimeImpl
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent, TreeNotify
from ghoshell_common.helpers import uuid
from ghoshell_container import IoCContainer, Container
import logging
import asyncio


class ShellImpl(MOSSShell):

    def __init__(
            self,
            *,
            description: Optional[str] = None,
            container: IoCContainer | None = None,
            main_channel: Channel | None = None,
            output: Optional[Output] = None,
    ):
        self.container = Container(parent=container, name=f"MOSShell")
        self.container.set(MOSSShell, self)
        # output
        self._output: Output = output or self.container.get(Output) or ArrOutput()
        self.container.set(Output, self._output)
        # logger
        self.logger = self.container.get(logging.Logger) or logging.getLogger("moss")
        self.container.set(logging.Logger, self.logger)

        # init main channel
        self._main_channel = main_channel or ShellMainChannel(
            name="",
            block=True,
            # todo
            description=description or "",
        )

        # --- lifecycle --- #
        self._starting = False
        self._started = False
        self._closing = False

        self._stop_event = ThreadSafeEvent()
        self._stopped_event = ThreadSafeEvent()
        self._running_loop: Optional[asyncio.AbstractEventLoop] = None
        self._idle_notifier = TreeNotify(name="")

        # --- runtime --- #
        self._main_channel_runtime = ChannelRuntimeImpl(
            container=self.container,
            channel=self._main_channel,
            logger=self.logger,
            stop_event=self._stop_event,
            is_idle_notify=self._idle_notifier,
        )
        self._channel_runtimes: Dict[str, ChannelRuntime] = {
            "": self._main_channel_runtime,
        }
        self._configured_channel_metas: Optional[Dict[str, ChannelMeta]] = None
        self._interpreter: Optional[Interpreter] = None

    def is_running(self) -> bool:
        return self._started and not self._stop_event.is_set()

    def is_idle(self) -> bool:
        return self.is_running() and not self._idle_notifier.is_set()

    async def interpret(
            self,
            kind: Literal['clear', 'defer_clear', 'try'] = "clear",
            *,
            stream_id: Optional[int] = None,
    ) -> Interpreter:
        self._check_running()
        if self._interpreter is not None:
            await self._interpreter.stop()
            self._interpreter = None
            if kind == "defer_clear":
                await self.defer_clear()
            elif kind == "clear":
                await self.clear()
        callback = self._append_command_task if kind != "try" else None
        interpreter = CTMLInterpreter(
            commands=self.commands().values(),
            output=self._output,
            stream_id=stream_id or uuid(),
            callback=callback,
            logger=self.logger,
        )
        if callback is not None:
            self._interpreter = interpreter
        return interpreter

    def _append_command_task(self, task: CommandTask | None) -> None:
        self._check_running()
        self._running_loop.call_soon_threadsafe(self._main_channel_runtime.append, task)

    async def _get_channel_runtime(self, name: str) -> Optional[ChannelRuntime]:
        self._check_running()
        if name in self._channel_runtimes:
            return self._channel_runtimes[name]
        runtime = await self._new_channel_runtime(name)
        if runtime is not None:
            self._channel_runtimes[name] = runtime
        return runtime

    async def _new_channel_runtime(self, name: str) -> Optional[ChannelRuntimeImpl]:
        self._check_running()
        descendants = self._main_channel.descendants()
        if name in descendants:
            channel = descendants[name]
            runtime = ChannelRuntimeImpl(
                container=self.container,
                channel=channel,
                logger=self.logger,
                stop_event=self._stop_event,
                is_idle_notify=self._idle_notifier.child(name),
            )
            await runtime.start()
            self._channel_runtimes[name] = runtime
            return runtime
        return None

    def with_output(self, output: Output) -> None:
        self._output = output

    @property
    def main(self) -> Channel:
        return self._main_channel

    def register(self, parent: str = "", *channels: Channel) -> None:
        if parent == "":
            self._main_channel.with_children(*channels)
        else:
            parent_channel = self._main_channel.descendants().get(parent, None)
            if parent_channel is None:
                raise KeyError(f"Channel {parent} not found")
            parent_channel.with_children(*channels)

    def configure(self, *metas: ChannelMeta) -> None:
        metas = {}
        for meta in metas:
            metas[meta.name] = meta
        if len(metas) > 0:
            self._configured_channel_metas = metas

    async def channel_metas(self) -> Dict[str, ChannelMeta]:
        if self._configured_channel_metas is not None:
            metas = self._configured_channel_metas.copy()
            result = {}
            for name, meta in metas.items():
                runtime = await self._get_channel_runtime(name)
                meta.available = runtime.is_running() and runtime.is_available()
                if meta.available:
                    commands = runtime.client.commands()
                    for command_meta in meta.commands:
                        if command_meta.name not in commands:
                            command_meta.available = False
                        else:
                            command_meta.available = commands[name].is_available()
                    result[name] = meta
            return result

        else:
            result = {}
            for name, runtime in self._channel_runtimes.items():
                if runtime.is_running() and runtime.is_available():
                    meta = runtime.client.meta()
                    result[name] = meta
            return result

    def commands(self) -> Dict[str, Command]:
        """
        动态获取 commands. 因为可能会有变动.
        """
        self._check_running()
        commands = {}
        for channel in self.channels():
            if channel.is_running() and channel.client.is_available():
                for command in channel.client.commands(available_only=True).values():
                    commands[command.name()] = command
        return commands

    async def append(self, *tasks: CommandTask) -> None:
        for task in tasks:
            await self._main_channel_runtime.append(task)

    async def clear(self, *chans: str) -> None:
        self._check_running()
        names = list(chans)
        if len(names) == 0:
            names = [""]
        for name in names:
            if name == self._main_channel_client.name():
                clearing = [self._main_channel_client.clear()]
                break
            else:
                client = self._main_channel_client.get_child_client(name)
                clearing.append(name)

    async def wait_until_idle(self, timeout: float | None = None) -> None:
        self._check_running()
        await self._main_channel_client.wait_until_idle(timeout),

    def get_channel(self, name: str) -> Optional[Channel]:
        if name == self._main_channel.name():
            return self._main_channel
        return self._main_channel.descendants().get(name, None)

    async def defer_clear(self, *chans: str) -> None:
        self._check_running()
        names = list(chans)
        if len(names) == 0:
            await self._main_channel_client.defer_clear()
            return
        # 可以并行执行.
        clearing = []
        for name in names:
            child = await self._main_channel_client.get_child_client(name)
            clearing.append(child.defer_clear())
        await asyncio.gather(*clearing)

    async def system_prompt(self) -> str:
        raise NotImplementedError()

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f"Shell not running")

    def channels(self) -> Iterable[Channel]:
        yield self._main_channel
        yield from self._main_channel.descendants().values()

    async def start(self) -> None:
        if self._closing:
            raise RuntimeError("shell runtime can not re-enter")
        if self._starting:
            return
        self._starting = True
        self._running_loop = asyncio.get_running_loop()
        # 启动容器. 通常已经启动了.
        await asyncio.to_thread(self.container.bootstrap)

        self._main_channel_client = ChannelRuntimeImpl(
            self.container,
            self._main_channel,
            logger=self.logger,
            stop_event=self._stop_event,
        )
        # 启动所有的 runtime.
        await self._main_channel_client.start()
        # 启动自己的 task
        self._started = True

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._stop_event.set()
        # 先关闭所有的 runtime. 递归关闭.
        await self._main_channel_client.close()
        # 然后关闭所有的 channel.
        self._closed = True
