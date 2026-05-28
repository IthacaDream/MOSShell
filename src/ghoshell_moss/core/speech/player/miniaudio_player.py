import queue
from typing import Optional

import miniaudio
import numpy as np
from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.core.speech.player.base_player import BaseAudioStreamPlayer

__all__ = ["MiniAudioStreamPlayer"]


class MiniAudioStreamPlayer(BaseAudioStreamPlayer):
    """
    基于 miniaudio 的异步音频播放器实现。
    miniaudio 零系统依赖，跨平台一致，纯 wheel 安装即用。

    miniaudio 1.x 使用 generator 模式：PlaybackDevice.start() 接受一个
    callback generator，内部线程通过 gen.send(frame_count) 请求音频帧。
    """

    def __init__(
        self,
        *,
        sample_rate: int = 44100,
        channels: int = 1,
        logger: LoggerItf | None = None,
        safety_delay: float = 0.1,
    ):
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            logger=logger,
            safety_delay=safety_delay,
        )
        self._playback: Optional[miniaudio.PlaybackDevice] = None

    def _audio_stream_start(self):
        self._data_queue: queue.Queue[bytes] = queue.Queue()
        self._buf = b""  # 未播放的剩余字节

        def _audio_generator():
            bytes_per_frame = self.channels * 2  # SIGNED16 = 2 bytes/sample
            frames_needed = yield b""  # prime
            while not self._stop_event.is_set():
                bytes_needed = (frames_needed or 0) * bytes_per_frame

                # 从队列填充缓冲区直到凑够所需字节
                while len(self._buf) < bytes_needed:
                    try:
                        self._buf += self._data_queue.get_nowait()
                    except queue.Empty:
                        break

                if len(self._buf) >= bytes_needed and bytes_needed > 0:
                    frames_needed = yield self._buf[:bytes_needed]
                    self._buf = self._buf[bytes_needed:]
                else:
                    # 数据不足，用静音补足
                    missing = max(bytes_needed - len(self._buf), 0)
                    result = self._buf + b"\x00" * missing
                    self._buf = b""
                    frames_needed = yield result

        gen = _audio_generator()
        next(gen)  # prime

        self._playback = miniaudio.PlaybackDevice(
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=self.channels,
            sample_rate=self.sample_rate,
        )
        self._playback.start(gen)

    def _audio_stream_write(self, data: np.ndarray):
        self._data_queue.put(data.tobytes())

    def _audio_stream_stop(self):
        if self._playback is not None:
            self._playback.stop()
            self._playback = None
