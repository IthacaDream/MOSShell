import logging

from typing import Dict, Optional, List, Literal, Iterable
from typing_extensions import Self
from ghoshell_moss.concepts.channel import Channel
from ghoshell_moss.concepts.command import CommandTaskSeq, CommandTask, Command
from ghoshell_moss.concepts.errors import FatalError, CommandError
from ghoshell_moss.concepts.interpreter import Interpreter
from ghoshell_moss.concepts.shell import ChannelRuntime
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent, ensure_tasks_done_or_cancel, TreeNotify
from ghoshell_container import IoCContainer, Container
from collections import deque
import asyncio


class ChannelRuntimeImpl(ChannelRuntime):

    def __init__(
            self,
            container: IoCContainer,
            channel: Channel,
            *,
            logger: Optional[logging.Logger] = None,
            stop_event: Optional[ThreadSafeEvent] = None,
            is_idle_notify: Optional[TreeNotify] = None,
    ):
        # 容器应该要已经运行过了. 关键的抽象也被设置过.
        # channel runtime 不需要有自己的容器. 也不需要关闭它.
        self.container = container
        self.logger = logger or logging.getLogger("moss")
        # runtime 级别的关机事件. 会传递给所有的子节点.
        self._stopping_event = stop_event or ThreadSafeEvent()
        self._chan: Channel = channel
        self._channel_server_task: Optional[asyncio.Task] = None
        self._name = channel.name()

        # status
        self._started = False
        self._stopped = False

        # runtime properties
        self._children_clients: Dict[str, ChannelRuntimeImpl] = {}
        # 获取被启动时的 loop, 用来做跨线程的调度.
        self._running_loop: Optional[asyncio.AbstractEventLoop] = None

        # 内部的 运行状态判断. 暂时还不知道会不会用上.
        self._executing_loop_done_event: asyncio.Event = asyncio.Event()
        # 主任务.
        self._main_loop_task: Optional[asyncio.Task] = None

        # 是否闲置状态的树形通知.
        self._is_idle_notify: TreeNotify = is_idle_notify or TreeNotify(self._name, None)
        # 特殊的锁, 可以暂停 consuming loop 的运行.
        self._allow_consume_pending_loop_event: asyncio.Event = asyncio.Event()

        # 输入队列, 只是为了足够快地输入. 当执行 cancel 的时候, executing_queue 会被清空, 但 pending queue 不会被清空.
        # 这种队列是为了 call_soon 的特殊 feature 做准备, 同时又不会在执行时阻塞解析. 解析的速度要求是全并行的.
        self._pending_queue: asyncio.Queue = asyncio.Queue()
        # 消费队列. 如果队列里的数据是 None, 表示这个队列被丢弃了.
        self._executing_queue: asyncio.Queue[CommandTask | None] = asyncio.Queue()
        self._executing_block_task: bool = False

        # 是否是 defer clear 状态.
        # 用 flag 做标记, 因为一旦触发了 clear, 就会递归 clear.
        self._defer_clear: bool = False

        # 运行中的 task group, 方便整体 cancel. 由于版本控制在 3.10, 暂时无法使用 asyncio 的 TaskGroup.
        self._executing_task_group: set = set()
        self._executing_block_task: bool = False

    async def get_child_client(self, name: str) -> Optional["ChannelRuntime"]:
        try:
            # 检查是否已经启动了.
            self._check_running()
            if name == self._name:
                # 兼容来获取自己.
                return self
            # 检查是否已经运行了.
            if name in self._children_clients:
                return self._children_clients[name]

            # 只能从直系血亲里查找.
            child_chan = self._chan.children().get(name)
            if child_chan is None:
                return None
            child_client = self._new_child_client(child_chan)
            # 考虑到要动态启动.
            await child_client.start()
            self._children_clients[name] = child_client
            return child_client
        except Exception as e:
            self.logger.exception(e)
            raise FatalError("Failed to get child client") from e

    def _new_child_client(self, channel: Channel) -> Self:
        child_runtime = ChannelRuntimeImpl(
            self.container,
            channel,
            logger=self.logger,
            stop_event=self._stopping_event,
            is_idle_notify=self._is_idle_notify.child(channel.name()),
        )
        return child_runtime

    async def start(self):
        if self._started:
            return
        self._started = True
        loop = asyncio.get_running_loop()
        self._running_loop = loop
        # 自身的启动.
        await self._self_bootstrap()
        # 启动所有的子 runtime.
        children_start = []
        for child_chan in self._chan.children().values():
            child_client = self._new_child_client(child_chan)
            # 添加所有的 runtime 启动任务.
            children_start.append(child_client.start())
        # 并行启动所有的子节点. 有异常直接抛出.
        if len(children_start) > 0:
            await asyncio.gather(*children_start)
        # 最后才启动主循环.
        self._main_loop_task = asyncio.create_task(self._run_main_loop())

    async def _self_bootstrap(self):
        # 创建主任务.
        if not self._chan.is_running():
            # 启动自身的 channel. 不过这样是效率比较低, 最好提前都启动完了.
            server = self._chan.bootstrap(self.container)
            self._channel_server_task = asyncio.create_task(server.start())

    async def close(self):
        # 已经结束过了.
        if not self._started or self._stopped:
            return
        self._stopped = True
        self._stopping_event.set()
        # 分层递归结束.
        try:
            stop_children = []
            for child in self._children_clients.values():
                stop_children.append(child.close())
            # 等待所有的子节点结束. 从自己开始关闭.
            if len(stop_children) > 0:
                await asyncio.gather(*stop_children)

            # 等待自身的主循环结束. 同时关闭对 server 的调用.
            await self._main_loop_task
            if self._chan.is_running():
                await self._chan.client.close()
                if self._channel_server_task is not None:
                    await self._channel_server_task
        finally:
            self._is_idle_notify.set()
            self._children_clients.clear()

    def _check_running(self):
        if not self._started:
            raise FatalError(f"Channel {self._name} is not running")
        elif self._stopping_event.is_set():
            raise FatalError(f"Channel {self._name} is shutdown")

    def name(self) -> str:
        return self._name

    def is_running(self) -> bool:
        """
        判断 runtime 是否在运行.
        """
        return self._started and not self._stopping_event.is_set()

    def is_available(self) -> bool:
        return self._chan.client.is_available()

    def commands(self) -> Iterable[Command]:
        yield from self._chan.client.commands()

    def is_busy(self) -> bool:
        """
        判断 runtime 是否是 busy 状态. 任何子节点在运行, 都会是 busy 状态.
        """
        if not self.is_running():
            return False
        return not self._is_idle_notify.event.is_set()

    async def wait_until_idle(self, timeout: float | None = None) -> None:
        await asyncio.wait_for(self._is_idle_notify.event.wait(), timeout)

    async def append(self, *tasks: CommandTask) -> None:
        if not self.is_running():
            # todo: log
            return
        for _task in tasks:
            # 快速入队.
            _task.set_state('pending')
            await self._pending_queue.put(_task)

    async def _consume_pending_loop(self) -> None:
        try:
            while not self._stopping_event.is_set():
                _queue = self._pending_queue
                item = await _queue.get()
                if item is None:
                    # 拿到了毒丸, 清空了当前队列.
                    continue
                await self._add_executing_task(item)

        except Exception as e:
            self.logger.exception(e)
            self._stopping_event.set()
        finally:
            self.logger.info('Finished executing loop')

    async def _add_executing_task(self, task: CommandTask) -> None:
        # 推送到等待队列中.
        # 需要在添加命令时就执行
        if task.meta.call_soon:
            # 清空队列先.
            block = task.meta.block
            if block:
                # 先清空.
                await self.cancel_executing()
        # 丢入执行队列中.
        await self._executing_queue.put(task)

    async def _run_main_loop(self) -> None:
        """主循环"""
        # 消费输入的命令
        consume_pending_task = asyncio.create_task(self._consume_pending_loop())
        # 消费确认可执行的命令.
        executing_task = asyncio.create_task(self._executing_loop())

        async def cancel_main_task():
            await self._stopping_event.wait()
            if not executing_task.done():
                executing_task.cancel()
            if not consume_pending_task.done():
                consume_pending_task.cancel()

        try:
            # 如果遇到问题就直接取消.
            await asyncio.gather(consume_pending_task, executing_task, cancel_main_task(), return_exceptions=False)
        finally:
            self.logger.info(f"channel {self._name} main loop done")
            await self._executing_loop_done_event.wait()

    async def _executing_loop(self) -> None:
        try:
            policy_is_running = False
            while not self._stopping_event.is_set():
                # 每次重新去获取 queue. 由于 queue 可能被丢弃, 所以一定要一次只执行一步.
                _queue = self._executing_queue
                # 队列为空的情况.
                if _queue.empty():
                    if not policy_is_running:
                        await self._chan.client.policy_run()
                        continue
                    else:
                        # 设置已经闲了.
                        self._is_idle_notify.set()

                # 用短时间来尝试获取.
                try:
                    item = await asyncio.wait_for(_queue.get(), 0.1)
                except asyncio.TimeoutError:
                    continue
                # 拿到毒丸了, 表示队列已经废弃.
                if item is None:
                    # 进入下一轮循环.
                    continue

                # 有任务在执行, 怎么都 clear 一下.
                self._is_idle_notify.clear()
                if policy_is_running:
                    # 阻塞等待 policy 停止运行.
                    await self._chan.client.policy_pause()

                # 获取最早的一个任务.
                # 运行一个任务. 理论上是很快的调度.
                # 这个任务不运行结束, 不会释放运行状态.
                await self._execute_task(item)
                # 多余的 continue, 方便看一眼.
                continue
        except asyncio.CancelledError:
            self.logger.info(f"channel {self._name} executing loop cancelled")
        except Exception as e:
            self.logger.exception(e)
            self._stopping_event.set()
        finally:
            self._executing_loop_done_event.set()

    async def _wait_allow_consuming(self) -> bool:
        # 其它的逻辑可能导致消费暂时终止.
        if self._allow_consume_pending_loop_event.is_set():
            return True
        try:
            # 加一个轻锁, 用来等待锁.
            await asyncio.wait_for(self._allow_consume_pending_loop_event.wait(), 0.1)
            return True
        except asyncio.TimeoutError:
            return False

    async def _execute_task(self, cmd_task: CommandTask) -> None:
        """执行一个 task. 核心目标是不抛出任何异常. """
        block = cmd_task.meta.block
        task = None
        try:
            # 运行一个任务. 理论上是很快的调度.
            # 这个任务不运行结束, 不会释放运行状态.
            task = asyncio.create_task(self._execute_task_in_stack(cmd_task))
            self._executing_task_group.add(task)
            if block:
                self._executing_block_task = True
                # 用 task group 启动, 方便统一中断.
                await task

        except asyncio.CancelledError:
            # 无所谓, 继续.
            return
        except FatalError as e:
            # 终止队列.
            self.logger.exception(e)
            self._stopping_event.set()
        except CommandError as e:
            # 所有的 command error 都忽视.
            self.logger.info("execute command `%r`error: %s", cmd_task, e)
        except Exception as e:
            # 没有到 Fatal Error 级别的都忽视.
            self.logger.exception(e)
        finally:
            if task and task in self._executing_task_group:
                self._executing_task_group.remove(task)
            if block:
                self._executing_block_task = False

    async def _execute_task_in_stack(self, task: CommandTask) -> None:
        try:
            task.set_state("running")
            sent_to_child = await self._add_executing_task(task)
            if sent_to_child:
                # 不是自己的, 直接退出.
                return

            # 先执行一次 command, 拿到可能的 command_seq, 主要用来做 resolve.
            command_seq = await self._execute_single_task(task)
            if command_seq is None:
                # 正常返回.
                return

            # 非阻塞函数不能有 stack
            if not task.meta.block:
                # todo: 这个是不是 fatal 的问题呢? 应该不是.
                raise CommandError(
                    CommandError.INVALID_USAGE,
                    f"none-block command {task.meta.name} return command sequence",
                )

            stack = deque(command_seq.tasks)
            while len(stack) > 0:
                first = stack.popleft()

                # 先做血缘调度.
                # 根本不是自己的, 要尽快调度. 但如果 task 的 channel 为空, 表示是主轨, 则也可立刻运行.
                sent_to_child = await self._send_task_to_child_if_not_own(first)
                if sent_to_child:
                    continue

                new_stack = await self._execute_single_task(first)
                # 栈操作, 插入到队首.
                if new_stack:
                    stack.extendleft(new_stack.tasks)

            # 有 stack 的 task 再 resolve 一次原始预期的值.
            task.resolve(command_seq.success())
            return

        except asyncio.CancelledError:
            return
        except FatalError as e:
            self.logger.exception(e)
            self._stopping_event.set()
            task.fail(e)
        except CommandError as e:
            self.logger.info("execute command `%r`error: %s", e, e)
            task.fail(e)
        except Exception as e:
            self.logger.exception(e)
            task.fail(e)
        finally:
            # 不要留尾巴?
            if not task.done():
                task.cancel()

    async def _send_task_to_child_if_not_own(self, cmd_task: CommandTask) -> bool:
        # 怕万一没启动呢?
        execution_channel = cmd_task.meta.chan
        if execution_channel == self._name:
            return False
        child_runtime = await self.get_child_client(execution_channel)
        if child_runtime is not None:
            # 直接发送给子节点.
            await child_runtime.append(cmd_task)
            return True

        for child_channel in self._chan.children().values():
            # 判断在不在哪个子节点的所有子孙节点里. 当 channel 特别多的时候, 有性能损耗.
            # 但子 channel 会不会动态注册孙 channel, 目前还不能确认.
            if execution_channel in child_channel.descendants():
                # 还是发送给这个儿女.
                child_runtime = await self.get_child_client(child_channel.name())
                await child_runtime.append(cmd_task)
                return True
        # 丢弃掉不认识的.
        self.logger.error(f"channel {self._name} executing found orphan task {cmd_task}")
        return True

    async def _execute_single_task(self, cmd_task: CommandTask) -> Optional[CommandTaskSeq]:

        try:
            if cmd_task.done():
                cmd_task.raise_exception()
                return cmd_task.result

            cmd_task.set_state('running')

            # 一个出异常了, 就会干扰其它的.
            # 要确保所有的 task 都被送到了 controller 里去运行.
            # 基本逻辑是不用 command task 直接运行, 避免运行完了通知到 interpreter.
            execution = self._chan.client.execute(cmd_task.meta.name, *cmd_task.args, **cmd_task.kwargs)
            execution_task = asyncio.create_task(execution)
            # 有可能 task 被别的地方给 cancel 了.
            results = await ensure_tasks_done_or_cancel(execution_task, cancel=cmd_task.wait)
            # 到这里可能已经被中断了.
            result = results[0]
            if isinstance(result, CommandTaskSeq):
                # 返回一个栈, command task 的结果需要在栈外判断.
                # 等栈运行完了才会赋值.
                return result

            # 这里才真正赋值.
            cmd_task.resolve(result)
            return None

        except asyncio.CancelledError:
            # 保险的方式再运行一次 cancel.
            self.logger.info("task %s cancelled", cmd_task)
            return
        except FatalError as e:
            # 如果有 Fatal Error, 会终止整个 shell 的运行.
            self.logger.exception(e)
            self._stopping_event.set()
            return None
        except CommandError as e:
            cmd_task.fail(e)
            return None
        except Exception as exc:
            # 忽视其它的异常. 仅仅记录.
            self.logger.exception(exc)
            cmd_task.fail(str(exc))
            return None

    async def clear_pending(self) -> None:
        """无锁的清空实现. """
        self._check_running()
        try:
            # 先清空自身的队列.
            _pending_queue = self._pending_queue
            self._pending_queue = asyncio.Queue()
            await _pending_queue.put(None)
            # 然后清空所有子节点的 pending 队列.
            clear_children = []
            for child in self._children_clients.values():
                clear_children.append(child.clear_pending())
            if len(clear_children) > 0:
                await asyncio.gather(*clear_children)
        except asyncio.CancelledError:
            self.logger.info("channel %s clear pending but canceled", self._name)
            raise
        except Exception as exc:
            self.logger.exception(exc)
            # 所有没有管理的异常, 都是致命异常.
            self._stopping_event.set()
            raise exc

    async def cancel_executing(self) -> None:
        self._check_running()
        try:
            # 准备并发 cancel 所有的运行.
            cancel_running = [self._cancel_self_executing()]
            for child in self._children_clients.values():
                # 子节点则是直接 clear.
                cancel_running.append(child.clear())
            await asyncio.gather(*cancel_running)
        except asyncio.CancelledError:
            self.logger.error("channel %s cancel running but canceled", self._name)
        except Exception as exc:
            self.logger.exception(exc)
            self._stopping_event.set()
            raise FatalError("channel %s cancel executing failed" % self._name) from exc

    async def _cancel_self_executing(self) -> None:
        # 发送毒丸.
        old_queue = self._executing_queue
        self._executing_queue = asyncio.Queue()
        await old_queue.put(None)
        # 清除所有运行中的任务. 同步阻塞, 所以不用考虑锁的问题.
        if len(self._executing_task_group) > 0:
            for t in self._executing_task_group:
                t.cancel()
            self._executing_task_group.clear()

    async def clear(self) -> None:
        self._check_running()
        try:
            # 暂停所有的消费动作. 锁了自己, 也就锁了子节点.
            self._allow_consume_pending_loop_event.clear()
            # 先清空队列. 递归地清空.
            await self.clear_pending()
            # 然后清空运行中的任务.
            await self.cancel_executing()
            # 通知自己所有的 channel 清空.
            await self._run_clear_callback()

        except asyncio.CancelledError:
            self.logger.info("channel %s clearing is cancelled", self._name)
            raise
        except FatalError as e:
            self.logger.exception(e)
            self._stopping_event.set()
            raise
        except Exception as exc:
            self.logger.exception(exc)
            raise
        finally:
            self._allow_consume_pending_loop_event.set()

    async def _run_clear_callback(self) -> None:
        """
        回调所有的 channel 已经执行了 clear.
        """
        clear_callbacks = [self._chan.client.clear()]
        for child in self._children_clients.values():
            clear_callbacks.append(child._run_clear_callback())
        await ensure_tasks_done_or_cancel(*clear_callbacks, cancel=self._stopping_event.wait)

    async def defer_clear(self) -> None:
        """
        准备清空运行状态, 如果有指令输入的话.
        """
        # defer clear 不需要递归. 因为所有子节点的任务来自父节点.
        self._defer_clear = True
