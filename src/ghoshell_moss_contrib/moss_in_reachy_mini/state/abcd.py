from abc import ABC

from ghoshell_moss.core.blueprint.states_channel import ChannelState
from ghoshell_moss.core.concepts.command import Command


class BaseReachyState(ChannelState, ABC):
    """Reachy Mini 状态基类 — 实现 MOSS ChannelState 接口。"""

    NAME: str = ""
    DESCRIPTION: str = ""

    def name(self) -> str:
        return self.NAME

    def description(self) -> str:
        return self.DESCRIPTION

    def is_available(self) -> bool:
        return True

    def is_dynamic(self) -> bool:
        return True

    def own_commands(self) -> dict[str, Command]:
        return {}

    def get_own_command(self, name: str) -> Command | None:
        return self.own_commands().get(name)

    def get_children(self) -> dict[str, "Channel"]:  # noqa: F821
        return {}

    def get_virtual_children(self) -> dict[str, "Channel"]:  # noqa: F821
        return {}
