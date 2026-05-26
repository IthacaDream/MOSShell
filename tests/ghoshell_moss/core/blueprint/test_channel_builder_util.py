"""channel_builder CommandUtil 的蓝图级单测，通过 CTML shell + MockSession 集成验证."""

import pytest

from ghoshell_moss import CTMLShell, new_ctml_shell
from ghoshell_moss.core.blueprint.channel_builder import CommandUtil, new_command, new_channel
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.mindflow import Signal
from ghoshell_moss.core.concepts.command import Observe, CommandErrorCode
from ghoshell_moss.core.session.mock_session import MockSession
from ghoshell_moss.core.ctml import ctml_shell_test
import time
import asyncio


# ── new_command 纯函数测试 ──────────────────────

@pytest.mark.asyncio
async def test_new_command_basic():
    """new_command 创建可执行的 PyCommand."""

    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    cmd = new_command(add, name="add", doc="Adds two ints")
    assert cmd.name() == "add"
    assert "async def add" in cmd.meta().interface or "add(a" in cmd.meta().interface


@pytest.mark.asyncio
async def test_new_command_sets_metadata():
    """new_command 传递 name/doc/comments 等元数据."""

    async def greet(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}"

    cmd = new_command(greet, name="greet", comments="call this to greet someone")
    assert cmd.name() == "greet"
    iface = cmd.meta().interface
    assert "async def greet" in iface
    # comments 会以 # 形式出现在 interface 中
    assert "call this to greet someone" in iface


# ── CommandUtil.send_signal ─────────────────────

@pytest.mark.asyncio
async def test_command_util_send_signal():
    """CommandUtil.send_signal 将 Signal 发送到 MockSession."""
    mock = MockSession()

    def builder(shell):
        shell.container.set(Session, mock)

        @shell.main_channel.build.command()
        async def emit_signal() -> None:
            sig = Signal(name="test_signal", description="from util")
            CommandUtil.send_signal(sig)

    await ctml_shell_test(builder=builder, ctml="<emit_signal/>")

    assert len(mock.signals) == 1
    assert mock.signals[0].name == "test_signal"
    assert mock.signals[0].description == "from util"


# ── CommandUtil.send_input_signal ───────────────

@pytest.mark.asyncio
async def test_command_util_send_input_signal():
    """CommandUtil.send_input_signal 发送 input 类型 Signal."""
    mock = MockSession()

    def builder(shell):
        shell.container.set(Session, mock)

        @shell.main_channel.build.command()
        async def notify() -> None:
            CommandUtil.send_input_signal("hello", description="greeting")

    await ctml_shell_test(builder=builder, ctml="<notify/>")

    assert len(mock.signals) == 1
    sig = mock.signals[0]
    assert sig.name == "input"
    assert sig.description == "greeting"


# ── CommandUtil.observe ─────────────────────────

@pytest.mark.asyncio
async def test_command_util_observe_returns_observe_object():
    """CommandUtil.observe 返回 Observe 对象，不中断."""
    mock = MockSession()

    def builder(shell):
        shell.container.set(Session, mock)

        @shell.main_channel.build.command()
        async def watch() -> Observe:
            return CommandUtil.observe("noted")

    tasks = await ctml_shell_test(builder=builder, ctml="<watch/>")
    assert len(tasks) == 1
    assert tasks[0].success()
    result = tasks[0].result()
    assert isinstance(result, Observe)
    assert any("noted" in msg.to_content_string() for msg in result.messages)


# ── CommandUtil.raise_observe ───────────────────

@pytest.mark.asyncio
async def test_command_util_raise_observe_interrupts():
    """CommandUtil.raise_observe 抛出 ObserveError，中断执行."""
    mock = MockSession()
    executed_after = []

    def builder(shell):
        shell.container.set(Session, mock)

        @shell.main_channel.build.command()
        async def emergency() -> None:
            CommandUtil.raise_observe("critical!")

        @shell.main_channel.build.command()
        async def never_runs() -> None:
            executed_after.append("should not run")

    tasks = await ctml_shell_test(
        builder=builder,
        ctml="<_><emergency/><never_runs/></_>",
    )

    emergency_task = next(t for t in tasks if t.meta.name == "emergency")
    assert emergency_task.exception() is not None
    assert CommandErrorCode.is_critical(emergency_task.errcode)
    assert "critical!" in emergency_task.errmsg

    # never_runs 被取消
    never_task = next((t for t in tasks if t.meta.name == "never_runs"), None)
    if never_task:
        assert never_task.cancelled()
    assert len(executed_after) == 0


# ── CommandUtil.get_contract ────────────────────

@pytest.mark.asyncio
async def test_command_util_get_contract_returns_registered():
    """CommandUtil.get_contract 从 IoC 获取注册的依赖."""
    mock = MockSession()

    def builder(shell):
        shell.container.set(Session, mock)

        @shell.main_channel.build.command()
        async def fetch_session() -> str:
            sess = CommandUtil.force_get_contract(Session)
            return sess.session_id

    tasks = await ctml_shell_test(builder=builder, ctml="<fetch_session/>")
    assert tasks[0].result() == mock.session_id


@pytest.mark.asyncio
async def test_command_util_get_contract_missing_raises():
    """CommandUtil.get_contract 缺少依赖时抛出错误."""

    def builder(shell):
        # 不注册 Session

        @shell.main_channel.build.command()
        async def fetch_missing() -> None:
            CommandUtil.force_get_contract(Session)

    tasks = await ctml_shell_test(builder=builder, ctml="<fetch_missing/>")
    assert tasks[0].exception() is not None


# ── CommandUtil.logger ──────────────────────────

@pytest.mark.asyncio
async def test_command_util_logger_returns_logger():
    """CommandUtil.logger 返回可用的 logger."""
    mock = MockSession()

    def builder(shell):
        shell.container.set(Session, mock)

        @shell.main_channel.build.command()
        async def get_logger() -> str:
            log = CommandUtil.logger()
            log.info("test log message")
            return type(log).__name__

    tasks = await ctml_shell_test(builder=builder, ctml="<get_logger/>")
    assert tasks[0].success()
    assert "Logger" in tasks[0].result() or "logging" in str(type(tasks[0].result())).lower()


# ── CommandUtil.send_signal 非 Signal 类型 ──────

@pytest.mark.asyncio
async def test_command_util_send_signal_rejects_non_signal():
    """send_signal 传入非 Signal 类型抛出 TypeError."""
    mock = MockSession()

    def builder(shell):
        shell.container.set(Session, mock)

        @shell.main_channel.build.command()
        async def bad_signal() -> None:
            CommandUtil.send_signal("not a signal")  # type: ignore

    tasks = await ctml_shell_test(builder=builder, ctml="<bad_signal/>")
    assert tasks[0].exception() is not None
    assert "only Signal or str is accepted" in tasks[0].errmsg


@pytest.mark.asyncio
async def test_command_util_is_task_done_in_sync_task():
    test_channel = new_channel(name="test")
    data = []
    done = asyncio.Event()

    @test_channel.build.command()
    def foo():
        # 同步函数要用这种方式去做 cancel 逻辑.
        while not CommandUtil.is_task_done():
            time.sleep(0.01)
        data.append(1)
        done.set()

    shell = new_ctml_shell()
    shell.main_channel.import_channels(test_channel)
    async with shell:
        async with shell.interpreter_in_ctx(
        ) as i:
            i.feed("<test:foo />")
            i.commit()
            await i.wait_compiled()
            assert len(i.compiled_tasks()) == 1
            await asyncio.sleep(0.01)
            assert len(data) == 0
        await shell.clear()
        await done.wait()
        assert data == [1]


@pytest.mark.asyncio
async def test_command_util_get_task_context():
    test_channel = new_channel(name="test")
    data = {}

    @test_channel.build.command()
    async def foo():
        # 同步函数要用这种方式去做 cancel 逻辑.
        data.update(CommandUtil.get_task_context())
        return

    shell = new_ctml_shell()
    shell.main_channel.import_channels(test_channel)
    async with shell:
        async with shell.interpreter_in_ctx(task_context={"hello": "world"}) as i:
            i.feed("<test:foo />")
            await i.wait_tasks()
        assert data['hello'] == 'world'
        assert len(data) > 1
