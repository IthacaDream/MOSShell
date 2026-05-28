import asyncio
import logging
from typing import List

from PIL import Image
from reachy_mini import ReachyMini

from ghoshell_moss import Message, Base64Image, Text, PyChannel
from ghoshell_moss.contracts import LoggerItf
from ghoshell_moss.core.concepts.command import CommandTaskResult


class Vision:

    def __init__(
            self,
            mini: ReachyMini,
            logger: LoggerItf = None,
    ):
        self.mini = mini
        self.logger = logger or logging.getLogger("Vision")

    async def look(self, about: str = '', fps: int = 1, n: int = 1):
        """
        主动获取机器人摄像头的视觉信息。

        支持连续拍摄多帧图片(建议fps*n不要超过3)，并将图片交给Agent处理

        :param about: 本次 look 操作的原因或备注信息，默认为空字符串
        :param fps: 每秒采集的帧数，默认为 1, 最大为 3
        :param n: 采集的总帧数，默认为 1, 最大为 3
        """
        fps = min(fps, 3)
        n = min(n, 3)

        while fps * n > 3:
            if fps > 1:
                fps -= 1
            else:
                n -= 1

        interval = 1 / fps
        total_frames = n

        frames: List[Base64Image] = []
        for i in range(total_frames):
            frame = self.mini.media.get_frame()
            frames.append(Base64Image.from_pil_image(Image.fromarray(frame)))
            if total_frames == 1:
                break
            await asyncio.sleep(interval)

        if frames:
            content = about + ' 本次look成功,你只需要说你看到的视觉就可以了,不需要再次调用look'
        else:
            content = about + ' 本次look失败,没有获取到视觉信息'

        return CommandTaskResult(
            observe=True,
            messages=[Message.new(name="__look__").with_content(
                Text(text=content),
                *frames
            )]
        )

    async def context_messages(self):
        frame = self.mini.media.get_frame()
        return [Image.fromarray(frame)]

    def as_channel(self):
        chan = PyChannel(name="vision", description="use camera to look", blocking=True)
        chan.build.command()(self.look)
        chan.build.context_messages(self.context_messages)
        return chan
