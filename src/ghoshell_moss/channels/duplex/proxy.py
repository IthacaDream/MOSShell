from typing import TypedDict, Dict, Any, ClassVar, Optional, List, Callable, Coroutine
from typing_extensions import Self
from abc import ABC

from ghoshell_moss import ChannelClient
from ghoshell_moss.concepts.channel import Channel, ChannelMeta, Builder, R
from ghoshell_moss.concepts.errors import CommandError
from ghoshell_moss.concepts.command import Command, CommandTask, BaseCommandTask, CommandMeta, CommandWrapper
from ghoshell_moss.channels.py_channel import PyChannel
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent
from .protocol import *
from .connection import *
from ghoshell_common.helpers import uuid
from ghoshell_container import Container, IoCContainer
import logging
import asyncio

__all__ = ['DuplexChannelClient', 'DuplexChannelStub', 'DuplexChannelProxy', 'DuplexChannelProxyClient']


class DuplexChannelClient(ChannelClient):
    """
    实现一个极简的 Duplex Channel, 它核心是可以通过 ChannelMeta 被动态构建出来.
    """

    def __init__(
            self,
            *,
            container: IoCContainer,
            alias: str,
            channel_id: str,
            session_id: str,
            server_meta: ChannelMeta,
            server_event_queue: asyncio.Queue[ChannelEvent | None],
            client_event_queue: asyncio.Queue[ChannelEvent],
            stop_event: Optional[ThreadSafeEvent],
            local_channel: Optional[Channel],
    ) -> None:
        """
        :param alias: channel 别名.
        :param session_id: 唯一的 session id. 和 server 通讯都必须要携带.
        :param server_meta: 从 server 同步过来的 ChannelMeta.
        :param server_event_queue: 从 server 发送来的事件, 经过队列分发到不同的 channel client.
        :param client_event_queue: 向 server 发送事件的队列.
        :param stop_event: 从上一层传递过来的统一关闭事件.
        :param container: ioc 容器.
        :param local_channel: 是否有本地的 channel, 提供额外的本地方法.
        """
        self.alias = alias
        self.session_id = session_id
        self.id = channel_id
        self.py_channel = local_channel
        self.container = Container(parent=container, name=f"moss/duplex_channel/{self.alias}")
        self.server_event_queue = server_event_queue
        self.client_event_queue = client_event_queue

        # meta 的讯息.
        self._server_chan_meta: ChannelMeta = server_meta
        self._cached_meta: Optional[ChannelMeta] = None

        # 运行时参数
        self._started = False
        self._logger: logging.Logger | None = None
        self._stop_event = stop_event or ThreadSafeEvent()

        self._self_close_event = asyncio.Event()
        self._main_loop_task: Optional[asyncio.Task] = None
        self._main_loop_done_event = ThreadSafeEvent()
        # runtime
        self._pending_server_command_calls: Dict[str, CommandTask] = {}

    def is_running(self) -> bool:
        return self._started and not self._stop_event.is_set()

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = self.container.get(logging.Logger) or logging.getLogger("moss")
        return self._logger

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f'Channel client {self.alias} is not running')

    def meta(self, no_cache: bool = False) -> ChannelMeta:
        self._check_running()
        if self._cached_meta is not None and not no_cache:
            return self._cached_meta.model_copy()
        self._cached_meta = self._build_meta()
        return self._cached_meta.model_copy()

    def _build_meta(self) -> ChannelMeta:
        self._check_running()
        meta = self._server_chan_meta.model_copy()
        # 从 server meta 中准备 commands 的原型.
        commands = {}
        for command_meta in self._server_chan_meta.commands:
            command_meta = command_meta.model_copy(update={"chan": self.alias})
            commands[command_meta.name] = command_meta
        # 如果有本地注册的函数, 用它们取代 server 同步过来的.
        if self.py_channel is not None:
            local_meta = self.py_channel.client.meta()
            for command_meta in local_meta.commands:
                commands[command_meta.name] = command_meta
        meta.commands = list(commands.values())
        # 修改别名.
        meta.name = self.alias
        return meta

    def is_available(self) -> bool:
        return self.is_running() and self._server_chan_meta.available

    def commands(self, available_only: bool = True) -> Dict[str, Command]:
        meta = self.meta(no_cache=False)
        result = {}
        for command_meta in meta.commands:
            if not available_only or command_meta.available:
                func = self._get_command_func(command_meta)
                command = CommandWrapper(meta=command_meta, func=func)
                result[command_meta.name] = command
        return result

    def _get_command_func(self, meta: CommandMeta) -> Callable[[...], Coroutine[None, None, Any]] | None:
        name = meta.name
        # 优先尝试从 local channel 中返回.
        if self.py_channel is not None:
            command = self.py_channel.client.get_command(name)
            if command is not None:
                return command.__call__

        # 回调服务端.
        async def _server_caller_as_command(*args, **kwargs):
            task: CommandTask | None = None
            try:
                task = CommandTask.get_from_context()
            except LookupError:
                pass
            cid = task.cid if task else uuid()
            event = CommandCallEvent(
                session_id=self.session_id,
                name=name,
                # channel 名称使用 server 侧的名称
                chan=self._server_chan_meta.name,
                command_id=cid,
                args=list(args),
                kwargs=dict(kwargs),
                tokens=task.tokens if task else "",
                context=task.context if task else {},
            )
            try:
                await self.client_event_queue.put(event.to_channel_event())
            except Exception as e:
                self.logger.exception(e)
                raise e

            wait_result_task = BaseCommandTask(
                meta=meta,
                func=None,
                cid=event.command_id,
                tokens=event.tokens,
                args=event.args,
                kwargs=event.kwargs,
                context=event.context,
            )
            try:
                self._pending_server_command_calls[cid] = wait_result_task
                # 等待异步返回结果.
                await wait_result_task.wait()
                wait_result_task.raise_exception()
                return wait_result_task.result()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(e)
                raise
            finally:
                if cid in self._pending_server_command_calls:
                    t = self._pending_server_command_calls.pop(cid)
                    if not t.done():
                        t.cancel()
                        cancel_event = CommandCancelEvent(
                            session_id=self.session_id,
                            command_id=event.command_id,
                            chan=event.chan,
                        )
                        await self.client_event_queue.put(cancel_event.to_channel_event())

        return _server_caller_as_command

    def get_command(self, name: str) -> Optional[Command]:
        meta = self.meta()
        for command_meta in meta.commands:
            if command_meta.name == name:
                func = self._get_command_func(command_meta)
                return CommandWrapper(meta=command_meta, func=func)
        return None

    async def execute(self, task: CommandTask[R]) -> R:
        self._check_running()
        func = self._get_command_func(task.meta)
        if func is None:
            raise LookupError(f'Channel {self.alias} can find command {task.meta.name}')
        return await func(*task.args, **task.kwargs)

    async def policy_run(self) -> None:
        self._check_running()
        try:
            if self.py_channel is not None:
                await self.py_channel.client.policy_run()

            event = RunPolicyEvent(
                session_id=self.session_id,
                chan=self._server_chan_meta.name,
            )
            await self.client_event_queue.put(event.to_channel_event())
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)

    async def policy_pause(self) -> None:
        self._check_running()
        try:
            if self.py_channel is not None:
                await self.py_channel.client.policy_pause()

            event = PausePolicyEvent(
                session_id=self.session_id,
                chan=self._server_chan_meta.name,
            )
            await self.client_event_queue.put(event.to_channel_event())
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)

    async def clear(self) -> None:
        self._check_running()
        try:
            if self.py_channel is not None:
                await self.py_channel.client.policy_pause()

            event = ClearCallEvent(
                session_id=self.session_id,
                chan=self._server_chan_meta.name,
            )
            await self.client_event_queue.put(event.to_channel_event())
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)

    async def _consume_server_event_loop(self):
        try:
            while not self._stop_event.is_set() and not self._self_close_event.is_set():
                await self._consume_server_event()
        except asyncio.CancelledError:
            # todo: log
            pass
        except Exception as e:
            self.logger.exception(e)
            self._stop_event.set()

    async def _command_peek_loop(self):
        while not self._stop_event.is_set() and not self._self_close_event.is_set():
            if len(self._pending_server_command_calls) > 0:
                tasks = self._pending_server_command_calls.copy()
                for task in tasks.values():
                    peek_event = CommandPeekEvent(
                        chan=task.meta.chan,
                        command_id=task.cid,
                    )
                    await self.client_event_queue.put(peek_event.to_channel_event())
            await asyncio.sleep(1)

    async def _main_loop(self):
        try:
            consume_loop_task = asyncio.create_task(self._consume_server_event_loop())
            command_peek_task = asyncio.create_task(self._command_peek_loop())
            await asyncio.gather(consume_loop_task, command_peek_task)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception(e)
        finally:
            self._main_loop_done_event.set()

    async def _consume_server_event(self):
        try:
            item = await self.server_event_queue.get()
            if item is None:
                self._stop_event.set()
                return

            if model := CommandDoneEvent.from_channel_event(item):
                await self._handle_command_done(model)
            elif model := ChannelMetaUpdateEvent.from_channel_event(item):
                await self._handle_channel_meta_update_event(model)
            elif model := RunPolicyDoneEvent.from_channel_event(item):
                self.logger.info("channel %s run policy is done from event %s", self.name, model)
            elif model := PausePolicyDoneEvent.from_channel_event(item):
                self.logger.info("channel %s pause policy is done from event %s", self.name, model)
            elif model := ClearDoneEvent.from_channel_event(item):
                self.logger.info("channel %s clear is done from event %s", self.name, model)
            else:
                self.logger.info('unknown server event %s', item)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception(e)

    async def _handle_channel_meta_update_event(self, event: ChannelMetaUpdateEvent) -> None:
        for meta in event.metas:
            if meta.name == self._server_chan_meta.name:
                self._server_chan_meta = meta.model_copy()
                break

    async def _handle_command_done(self, event: CommandDoneEvent) -> None:
        command_id = event.command_id
        if command_id in self._pending_server_command_calls:
            task = self._pending_server_command_calls[command_id]
            if event.errcode == 0:
                task.resolve(event.data)
            else:
                error = CommandError(event.errcode, event.errmsg)
                task.fail(error)

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        await asyncio.to_thread(self.container.bootstrap)
        self._main_loop_task = asyncio.create_task(self._main_loop())

    async def close(self) -> None:
        if self._self_close_event.is_set():
            await self._main_loop_done_event.wait()
        self._self_close_event.set()
        self._main_loop_task.cancel()
        await self._main_loop_task
        await asyncio.to_thread(self.container.shutdown)


class DuplexChannelStub(Channel):

    def __init__(
            self,
            *,
            alias: str,
            session_id: str,
            server_meta: ChannelMeta,
            server_event_queue: asyncio.Queue[ChannelEvent | None],
            client_event_queue: asyncio.Queue[ChannelEvent],
            stop_event: Optional[ThreadSafeEvent],
            local_channel: Optional[Channel],
    ) -> None:
        self._alias = alias
        self._session_id = session_id
        self._server_chan_meta = server_meta
        self._server_event_queue = server_event_queue
        self._client_event_queue = client_event_queue
        self._stop_event = stop_event
        self._running_client: Optional[DuplexChannelClient] = None
        self._children: Dict[str, Channel] = {}
        self.local_channel = local_channel or PyChannel(name=self._alias)

    def name(self) -> str:
        return self._alias

    @property
    def client(self) -> ChannelClient:
        if self._running_client is None:
            raise RuntimeError(f'Channel {self} has not been started yet.')
        return self._running_client

    def with_children(self, *children: "Channel", parent: Optional[str] = None) -> Self:
        return self.local_channel.with_children(*children, parent=parent)

    def new_child(self, name: str) -> Self:
        return self.local_channel.new_child(name)

    def children(self) -> Dict[str, "Channel"]:
        return self.local_channel.children()

    def is_running(self) -> bool:
        return self._running_client is not None and self._running_client.is_running()

    def bootstrap(self, container: Optional[IoCContainer] = None, depth: int = 0) -> "ChannelClient":
        if self._running_client is not None:
            raise RuntimeError(f'Channel {self} has already been started.')
        running_client = DuplexChannelClient(
            alias=self._alias,
            session_id=self._session_id,
            channel_id="%s_%s" % (self._session_id, self.name()),
            server_meta=self._server_chan_meta,
            server_event_queue=self._server_event_queue,
            client_event_queue=self._client_event_queue,
            stop_event=self._stop_event,
            local_channel=self.local_channel,
            container=container,
        )
        self._running_client = running_client
        return running_client

    @property
    def build(self) -> Builder:
        return self.local_channel.build


class DuplexChannelProxyClient(ChannelClient):
    """双工通道的主 Client, 它的任务是基于通讯的结果, 生成出不同的 channel stub 和对应的 client. """

    def __init__(
            self,
            *,
            name: str,
            server_connection: Connection,
            local_channel: Channel,
            connect_timeout: float = 10.0,
    ):
        self._name = name
        self._server_connection = server_connection
        self._connect_timeout = connect_timeout
        self._local_channel = local_channel
        self._client: Optional[DuplexChannelProxyClient] = None

        self._starting = False
        self._started_event = asyncio.Event()
        self._session_id: str = ""

        self._root_channel_stub: Optional[DuplexChannelStub] = None
        self._shared_client_queue = asyncio.Queue()
        self._server_event_dispatch_queues: Dict[str, asyncio.Queue[ChannelEvent | None]] = {}

        self._stop_event = ThreadSafeEvent()

    def is_running(self) -> bool:
        is_ran = self._starting and not self._stop_event.is_set()
        return is_ran and not self._server_connection.is_closed() and self._root_channel_stub is not None

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f'Channel {self._name} is not running.')

    def meta(self, no_cache: bool = False) -> ChannelMeta:
        self._check_running()
        return self._root_channel_stub.client.meta(no_cache=no_cache)

    def is_available(self) -> bool:
        if not self.is_running():
            return False
        return self._root_channel_stub.client.is_available()

    def commands(self, available_only: bool = True) -> Dict[str, Command]:
        self._check_running()
        return self._root_channel_stub.client.commands(available_only=available_only)

    def get_command(self, name: str) -> Optional[Command]:
        self._check_running()
        return self._root_channel_stub.client.get_command(name)

    async def execute(self, task: CommandTask[R]) -> R:
        self._check_running()
        return await self._root_channel_stub.client.execute(task)

    async def policy_run(self) -> None:
        self._check_running()
        return await self._root_channel_stub.client.policy_run()

    async def policy_pause(self) -> None:
        self._check_running()
        return await self._root_channel_stub.client.policy_pause()

    async def clear(self) -> None:
        self._check_running()
        return await self._root_channel_stub.client.clear()

    async def start(self) -> None:
        if self._starting:
            await self._started_event.wait()
            return
        self._starting = True
        self._session_id = uuid()

        # 开始创建连接.
        await self._server_connection.start()
        # 解决授权问题.
        # 要求同步所有的 channel meta.
        await self._server_connection.send(SyncChannelMetasEvent().to_channel_event())

        # 等待第一个 event.
        first_event = await self._server_connection.recv(timeout=self._connect_timeout)

        # 如果第一个 event 不是 meta update client
        meta_update_event = ChannelMetaUpdateEvent.from_channel_event(first_event)
        if meta_update_event is None:
            raise RuntimeError(f'Channel {self._name} server can not be connected.')

        # 开始实例化 channel stub.
        meta_map = {meta.name: meta for meta in meta_update_event.metas}
        root_meta = meta_map.get(meta_update_event.root_chan)
        if root_meta is None:
            raise RuntimeError(f'Channel {self._name} server has no root meta.')

        root_queue = asyncio.Queue()
        self._server_event_dispatch_queues[root_meta.name] = root_queue
        self._root_channel_stub = DuplexChannelStub(
            alias=self._name,
            session_id=self._session_id,
            server_meta=root_meta,
            server_event_queue=root_queue,
            client_event_queue=self._shared_client_queue,
            stop_event=self._stop_event,
            local_channel=self._local_channel,
        )
        await self._root_channel_stub.bootstrap(self.container).start()

        # 递归启动子孙.
        start_all_children = []
        for child_name in root_meta.children:
            cor = self._recursive_start_channel_from_metas(self._root_channel_stub, child_name, meta_map)
            start_all_children.append(cor)
        await asyncio.gather(*start_all_children)

        # 标记启动完成.
        self._started_event.set()

    async def _recursive_start_channel_from_metas(
            self,
            parent: Channel,
            name: str,
            meta_maps: Dict[str, ChannelMeta],
    ) -> None:
        local_channel = self._local_channel.get_channel(name)
        current_meta = meta_maps.get(name)
        if current_meta is None:
            return
        queue = asyncio.Queue()
        self._server_event_dispatch_queues[current_meta.name] = queue
        channel_stub = DuplexChannelStub(
            alias=self._name,
            session_id=self._session_id,
            server_meta=current_meta,
            server_event_queue=queue,
            client_event_queue=self._shared_client_queue,
            stop_event=self._stop_event,
            local_channel=local_channel,
        )

        # 父节点挂载自身.
        parent.with_children(channel_stub)
        await channel_stub.bootstrap(self.container).start()

        recursive_start = []
        for child_name in current_meta.children:
            recursive_start.append(self._recursive_start_channel_from_metas(channel_stub, child_name, meta_maps))
        await asyncio.gather(*recursive_start)

    async def close(self) -> None:
        if self._stop_event.is_set():
            return
        # 同时也会通知所有子节点.
        self._stop_event.set()
        await self._root_channel_stub.client.close()


class DuplexChannelProxy(Channel, ABC):

    def __init__(
            self,
            server_connection: Connection,
            block: bool,
            name: str,
            description: str = "",
    ):
        self._name = name
        self._server_connection = server_connection
        self._local_channel = PyChannel(name=name, description=description, block=block)
        self._client: Optional[DuplexChannelProxyClient] = None
        self._children: Dict[str, Channel] = {}

    def name(self) -> str:
        return self._name

    @property
    def client(self) -> ChannelClient:
        if self._client is None:
            raise RuntimeError(f'Channel {self} has not been started yet.')
        return self._client

    def with_children(self, *children: "Channel", parent: Optional[str] = None) -> Self:
        self._local_channel.with_children(*children, parent=parent)
        return self

    def new_child(self, name: str) -> Self:
        return self._local_channel.new_child(name)

    def children(self) -> Dict[str, "Channel"]:
        return self._children.copy()

    def is_running(self) -> bool:
        return self._client is not None and self._client.is_running()

    def bootstrap(self, container: Optional[IoCContainer] = None, depth: int = 0) -> "ChannelClient":
        if self._client is not None and self._client.is_running():
            raise RuntimeError(f'Channel {self} has already been started.')
        client = DuplexChannelProxyClient(
            name=self._name,
            server_connection=self._server_connection,
            local_channel=self._local_channel,
        )
        self._client = client
        return client

    @property
    def build(self) -> Builder:
        return self._local_channel.build
