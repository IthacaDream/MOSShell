import numpy as np

from ghoshell_moss.core.speech.player.base_player import BaseAudioStreamPlayer

__all__ = ["VirtualStreamPlayer"]


class VirtualStreamPlayer(BaseAudioStreamPlayer):
    """无实际音频输出的播放器，仅供测试和降级兜底使用。

    三个抽象方法均为空操作。阻塞行为完全依赖基类的时间估算：
    add() 计算 duration 并累积 _estimated_end_time，
    wait_play_done() 按估算时间 sleep，不产生任何系统音频。
    """

    def _audio_stream_start(self):
        pass

    def _audio_stream_write(self, data: np.ndarray):
        pass

    def _audio_stream_stop(self):
        pass
