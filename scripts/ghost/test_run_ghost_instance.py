"""run_ghost 环境无关路径 — 传入 GhostMeta 实例，不依赖 workspace 发现."""
import asyncio
from ghoshell_moss.host import Host
from ghoshell_moss.ghosts.mock import MockGhostMeta, MockGhost
from ghoshell_moss.ghosts.mock._runtime import MockArticulator
from ghoshell_moss.core.blueprint.mindflow import Moment

host = Host()
meta = MockGhostMeta(name="test_instance")
gr = host.run_ghost(meta)


async def main():
    async with gr:
        ghost = gr.ghost

        # basic identity
        assert isinstance(ghost, MockGhost)
        assert ghost.meta.name() == "test_instance"
        assert ghost.meta.prototype() == "MockGhost"

        # moss is running
        assert gr.moss.is_running()

        # articulate — default "hello world"
        art = MockArticulator(moment=Moment())
        deltas = [d async for d in ghost.articulate(art)]
        assert deltas == ["hello world"]

        gr.close()

    print("OK")


asyncio.run(main())
