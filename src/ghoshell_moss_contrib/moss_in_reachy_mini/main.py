import logging

from ghoshell_container import IoCContainer
from reachy_mini import ReachyMini

from ghoshell_moss import new_prime_channel, Message, Text, Channel
from ghoshell_moss.core.blueprint.channel_builder import ChannelCreator
from ghoshell_moss.contracts import Workspace, LoggerItf
from ghoshell_moss.core.concepts.channel import ChannelCtx
from ghoshell_moss.core.py_channel import StatefulChannelRuntime
from ghoshell_moss_contrib.moss_in_reachy_mini.components.antennas import Antennas
from ghoshell_moss_contrib.moss_in_reachy_mini.components.body import Body
from ghoshell_moss_contrib.moss_in_reachy_mini.components.head import Head
from ghoshell_moss_contrib.moss_in_reachy_mini.components.vision import Vision
from ghoshell_moss_contrib.moss_in_reachy_mini.state.asleep import AsleepState
from ghoshell_moss_contrib.moss_in_reachy_mini.state.boring import BoringState
from ghoshell_moss_contrib.moss_in_reachy_mini.state.waken import WakenState

__all__ = ['MossInReachyMini', 'ReachyMiniChannelCreator']


class MossInReachyMini:
    """Reachy Mini 躯体控制器 — 组装 StatefulChannel，状态切换由 MOSS runtime 管理。"""

    def __init__(
            self,
            mini: ReachyMini,
            ws: Workspace,
            logger: LoggerItf,
            default_state: str = "waken",
            name: str = "reachy_mini",
            description: str = "reachy mini robot",
    ):
        self._mini = mini
        self.logger = logger
        self._default_state = default_state
        self._name = name
        self._description = description

        # layer 1: body components
        self._head = Head(mini)
        body = Body(mini, ws, logger)
        antennas = Antennas(mini, logger=logger)

        # layer 2: media channels
        self._vision = Vision(mini, logger=logger)

        # layer 3: states
        self._waken = WakenState(mini, body, self._head, antennas)
        self._boring = BoringState(mini)
        self._asleep = AsleepState(mini)

    # -- context messages -----------------------------------------------------

    async def context_messages(self):
        messages = []

        state_message = Message.new(name="__reachy_mini_state__").with_content(
            Text(text="Reachy Mini body channel — 各状态下可用命令通过 switch_state 切换后暴露"),
        )
        messages.append(state_message)

        return messages

    # -- channel assembly -----------------------------------------------------

    def as_channel(self) -> Channel:
        self.logger.info("MossInReachyMini.as_channel()")
        states = {
            self._waken.name(): self._waken,
            self._boring.name(): self._boring,
            self._asleep.name(): self._asleep,
        }

        channel = new_prime_channel(
            name=self._name or "reachy_mini",
            description=self._description or "reachy mini root channel",
        )

        channel.build.context_messages(self.context_messages)
        channel.build.startup(self.bootstrap)
        channel.build.close(self.aclose)

        default_state = self._default_state
        if not default_state in states:
            default_state = "waken"
        channel.with_state(states[default_state])
        for name, state in states.items():
            if name != default_state:
                channel.with_state(state)

        channel.import_channels(
            self._vision.as_channel(),
        )

        return channel

    # -- lifecycle ------------------------------------------------------------

    async def bootstrap(self):
        self._mini.__enter__()
        runtime = ChannelCtx.runtime()
        if isinstance(runtime, StatefulChannelRuntime):
            await runtime.switch_state("waken")

    async def aclose(self):
        self._mini.__exit__(None, None, None)


class ReachyMiniChannelCreator(ChannelCreator):

    def __init__(
            self,
            name: str = "",
            description: str = "",
            default_state: str = "waken",
    ) -> None:
        self._name = name
        self._description = description
        self._default_state = default_state

    def factory(self, container: IoCContainer) -> Channel:
        ws = container.force_fetch(Workspace)
        logger = container.force_fetch(LoggerItf)
        if not container.bound(ReachyMini):
            reachy_mini = ReachyMini()
            container.set(ReachyMini, reachy_mini)
        else:
            # 完全没必要绑定, 直接拿单例即可.
            reachy_mini = container.force_fetch(ReachyMini)
        reachy_mini = MossInReachyMini(
            mini=reachy_mini,
            ws=ws,
            logger=logger,
            default_state=self._default_state,
            name=self._name,
            description=self._description,
        )
        return reachy_mini.as_channel()
