import logging

from ghoshell_common.contracts import LoggerItf
from reachy_mini import ReachyMini

from ghoshell_moss import PyChannel, Message, Text
from ghoshell_moss.contracts import Workspace
from ghoshell_moss.core.concepts.channel import ChannelCtx
from ghoshell_moss.core.py_channel import StateChannelRuntime
from ghoshell_moss_contrib.moss_in_reachy_mini.components.antennas import Antennas
from ghoshell_moss_contrib.moss_in_reachy_mini.components.body import Body
from ghoshell_moss_contrib.moss_in_reachy_mini.components.head import Head
from ghoshell_moss_contrib.moss_in_reachy_mini.components.vision import Vision
from ghoshell_moss_contrib.moss_in_reachy_mini.state.asleep import AsleepState
from ghoshell_moss_contrib.moss_in_reachy_mini.state.boring import BoringState
from ghoshell_moss_contrib.moss_in_reachy_mini.state.waken import WakenState


class MossInReachyMini:
    """Reachy Mini 躯体控制器 — 组装 StatefulChannel，状态切换由 MOSS runtime 管理。"""

    def __init__(self, mini: ReachyMini, ws: Workspace, logger: LoggerItf):
        self._mini = mini
        self.logger = logger

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

    def as_channel(self) -> PyChannel:
        self.logger.info("MossInReachyMini.as_channel()")

        channel = PyChannel(
            name="reachy_mini",
            description="reachy mini root channel",
            blocking=True,
        )

        channel.build.context_messages(self.context_messages)
        channel.build.startup(self.bootstrap)
        channel.build.close(self.aclose)

        channel.with_state(self._waken)
        channel.with_state(self._boring)
        channel.with_state(self._asleep)

        channel.import_channels(
            self._vision.as_channel(),
        )

        return channel

    # -- lifecycle ------------------------------------------------------------

    async def bootstrap(self):
        self._mini.__enter__()
        runtime = ChannelCtx.runtime()
        if isinstance(runtime, StateChannelRuntime):
            await runtime.switch_state("waken")

    async def aclose(self):
        self._mini.__exit__(None, None, None)
