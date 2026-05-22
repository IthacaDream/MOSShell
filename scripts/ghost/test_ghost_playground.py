"""验证 GhostPlayground 通过 IoC 正常注入并可访问三个 scope."""
import asyncio
from ghoshell_moss.host import Host
from ghoshell_moss.core.blueprint.host import GhostPlayground

host = Host()
gr = host.run_ghost("mock")


async def main():
    async with gr:
        playground = gr.container.force_fetch(GhostPlayground)

        # 三个 scope 全部可访问
        home = playground.home()
        session = playground.session()
        workspace = playground.workspace()

        print(f"home:      {home.abspath()}")
        print(f"session:   {session.abspath()}")
        print(f"workspace: {workspace.abspath()}")

        # scopes() 自解释
        scopes = playground.scopes()
        assert set(scopes.keys()) == {"home", "session", "workspace"}, f"unexpected scopes: {set(scopes.keys())}"
        for name, scope in scopes.items():
            assert scope is not None, f"{name} scope is None"
            print(f"  {name}: {scope.abspath()} (exists={scope.exists('')})")

        # default_scope → home
        assert playground.default_scope().abspath() == home.abspath()

        # home 按约定路径: workspace/ghosts/{ghost_name}
        assert home.abspath().name == "mock"
        assert home.abspath().parent.name == "ghosts"

        gr.close()

    print("OK — GhostPlayground all scopes verified")


asyncio.run(main())
