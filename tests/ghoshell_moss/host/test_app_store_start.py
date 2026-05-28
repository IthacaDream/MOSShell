"""HostAppStore.start_app 单测，重点覆盖 add + start: true 的 error 路径.

circusd 真实的返回格式 (controller.py → _dispatch_callback → send_ok/send_error):

    成功 (add + start: true):
        {"status": "ok", "time": ..., "started": [...pids], "kept": [...]}
    成功 (add 不带 start):
        {"status": "ok", "time": ...}
    失败 (任何 exception):
        {"status": "error", "reason": "...", "tb": ..., "time": ..., "errno": ...}

见 circus/commands/base.py: ok() / error().
"""
import pytest
from unittest.mock import MagicMock

from ghoshell_moss.core.concepts.errors import CommandErrorCode, CommandError
from ghoshell_moss.core.blueprint.app import AppInfo, AppWatcher
from ghoshell_moss.host.app_store import HostAppStore


def _make_app(group: str = "tools", name: str = "calculator") -> AppInfo:
    return AppInfo(
        name=name,
        group=group,
        description="test app",
        work_directory="/tmp/test_app",
        watcher=AppWatcher(),
    )


@pytest.fixture
def store():
    """构建一个最小 HostAppStore，mock 掉所有外部依赖."""
    env = MagicMock()
    env.dump_moss_env.return_value = {}
    workspace = MagicMock()
    workspace.root_path.return_value = MagicMock()
    workspace.lock.return_value = MagicMock()

    s = HostAppStore(
        env=env,
        workspace=workspace,
        namespace="test",
        runnable=False,
        include=["*/*"],
    )
    return s


# ---- error 场景: 验证 r1.get('status') == "error" 能正确捕获 ----

@pytest.mark.asyncio
async def test_add_with_start_true_returns_error(store):
    """add + start: true 执行时抛异常，circusd 包装为 status=error 返回."""
    app = _make_app()
    store.get_app_info = MagicMock(return_value=app)

    # circusd 真实错误响应: ok()/error() 总是有 status 键
    async def mock_call(cmd):
        return {
            "status": "error",
            "reason": "arbiter is already running arbiter_add_watcher command",
            "tb": None,
            "time": 1234567890.0,
            "errno": 5,
        }
    store._call_circus = mock_call

    with pytest.raises(CommandError) as exc:
        await store.start_app(app.fullname)
    assert exc.value.code == CommandErrorCode.VALUE_ERROR
    assert "failed to start tools/calculator" in str(exc.value)


@pytest.mark.asyncio
async def test_start_existing_watcher_returns_error(store):
    """watcher 已存在时走 start 命令，circusd 返回 status=error."""
    app = _make_app()
    store.get_app_info = MagicMock(return_value=app)
    store._managed_apps_with_fullname.add(app.fullname)

    async def mock_call(cmd):
        if cmd.get("command") == "start":
            return {
                "status": "error",
                "reason": "arbiter is already running arbiter_start_watchers command",
                "tb": None,
                "time": 1234567890.0,
                "errno": 5,
            }
        return {"status": "ok", "time": 1234567890.0}
    store._call_circus = mock_call

    with pytest.raises(CommandError) as exc:
        await store.start_app(app.fullname)
    assert exc.value.code == CommandErrorCode.VALUE_ERROR
    assert "cause system error" in str(exc.value)


@pytest.mark.asyncio
async def test_start_app_generic_exception(store):
    """非 CommandError 的异常被捕获并转为 VALUE_ERROR."""
    app = _make_app()
    store.get_app_info = MagicMock(return_value=app)

    async def mock_call(cmd):
        raise RuntimeError("unexpected failure")
    store._call_circus = mock_call

    with pytest.raises(CommandError) as exc:
        await store.start_app(app.fullname)
    assert exc.value.code == CommandErrorCode.VALUE_ERROR
    assert "failed to start tools/calculator" in str(exc.value)


# ---- 成功场景: 验证两种路径都能正常通过 ----

@pytest.mark.asyncio
async def test_add_with_start_true_returns_ok(store):
    """add + start: true 成功时 circusd 返回 status=ok + started/kept."""
    app = _make_app()
    store.get_app_info = MagicMock(return_value=app)

    async def mock_call(cmd):
        return {
            "status": "ok",
            "time": 1234567890.0,
            "started": [12345],
            "kept": [],
        }
    store._call_circus = mock_call

    result = await store.start_app(app.fullname)
    assert "Successfully started" in result
    assert app.fullname in store._managed_apps_with_fullname


@pytest.mark.asyncio
async def test_start_existing_watcher_success(store):
    """watcher 已存在时走 start 命令返回 status=ok."""
    app = _make_app()
    store.get_app_info = MagicMock(return_value=app)
    store._managed_apps_with_fullname.add(app.fullname)

    async def mock_call(cmd):
        if cmd.get("command") == "start":
            return {"status": "ok", "time": 1234567890.0}
        return {"status": "ok", "time": 1234567890.0}
    store._call_circus = mock_call

    result = await store.start_app(app.fullname)
    assert "Successfully started" in result
