"""run_ghost 按名称查找路径 — 从 workspace 的 MOSS/ghosts/ 发现后启动."""
import asyncio
from ghoshell_moss.host import Host
from ghoshell_moss.ghosts.mock._runtime import MockArticulator
from ghoshell_moss.core.blueprint.mindflow import Moment

host = Host()

# 确认 mock ghost 已被发现
ghosts = host.all_ghosts()
assert "mock" in ghosts, f"mock ghost not found in {list(ghosts.keys())}"
print(f"discovered: {list(ghosts.keys())}")

gr = host.run_ghost("mock")


async def main():
    async with gr:
        ghost = gr.ghost

        assert ghost.meta.name() == "mock"
        assert ghost.meta.prototype() == "MockGhost"
        assert gr.moss.is_running()

        # articulate
        art = MockArticulator(moment=Moment())
        deltas = [d async for d in ghost.articulate(art)]
        assert deltas == ["hello world"]

        # runtime 属性
        assert gr.moss is not None
        assert gr.meta is ghost.meta
        assert gr.container is not None

        gr.close()

    print("OK")


asyncio.run(main())
