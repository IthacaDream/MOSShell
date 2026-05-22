from typing import Type

from ghoshell_container import IoCContainer, Provider
from ghoshell_moss.core.blueprint.host import GhostPlayground
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.contracts import Storage, Workspace

__all__ = ["GhostPlaygroundImpl", "GhostPlaygroundProvider"]


class GhostPlaygroundImpl(GhostPlayground):
    """GhostPlayground 默认实现 — home 从 workspace 按约定派生."""

    def __init__(
            self,
            *,
            workspace: Workspace,
            session: Session,
            ghost_name: str,
            home_relative_path: str = 'ghosts'
    ):
        self._home = workspace.root().sub_storage(home_relative_path).sub_storage(ghost_name)
        self._session = session.storage
        self._workspace = workspace.root()

    def home(self) -> Storage:
        return self._home

    def session(self) -> Storage:
        return self._session

    def workspace(self) -> Storage:
        return self._workspace


class GhostPlaygroundProvider(Provider[GhostPlayground]):
    """GhostPlayground 的 IoC Provider.

    ghost_name 确定 home 路径约定 (workspace/ghosts/{name}).
    factory 从容器取 Workspace + Session, 构造 GhostPlaygroundImpl.
    """

    def __init__(
            self,
            ghost_name: str,
            home_relative_path: str = 'ghosts'
    ):
        self._ghost_name = ghost_name
        self._home_relative_path = home_relative_path

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[GhostPlayground]:
        return GhostPlayground

    def factory(self, con: IoCContainer) -> GhostPlayground:
        return GhostPlaygroundImpl(
            workspace=con.force_fetch(Workspace),
            session=con.force_fetch(Session),
            ghost_name=self._ghost_name,
            home_relative_path=self._home_relative_path,
        )
