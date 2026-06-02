import asyncio

from reachy_mini import ReachyMini

from ghoshell_moss_contrib.moss_in_reachy_mini.state.abcd import BaseReachyState


class AsleepState(BaseReachyState):
    NAME = "asleep"
    DESCRIPTION = "休眠状态：电机断电，机器人低头闭眼，不响应交互命令。"

    def __init__(self, mini: ReachyMini, logger = None):
        self._mini = mini
        self.logger = logger

    async def on_startup(self):
        self._mini.set_target_body_yaw(0.0)
        self._mini.goto_sleep()
        self._mini.disable_motors()

    async def on_close(self):
        pass

    async def on_idle(self):
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            pass
