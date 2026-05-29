from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.contracts.speech import TTS, Speech, SpeechStream, StreamAudioPlayer
from ghoshell_moss.core.speech.mock import MockSpeech
from ghoshell_moss.core.speech.null import NullSpeech
from ghoshell_moss.core.speech.speech_module import SpeechChannelModule, build_content_command
from ghoshell_moss.core.speech.stream_tts_speech import BaseTTSSpeech, TTSSpeechStream


def make_baseline_tts_speech(
    player: StreamAudioPlayer | None = None,
    tts: TTS | None = None,
    logger: LoggerItf | None = None,
) -> BaseTTSSpeech:
    """
    基线示例.
    """
    from ghoshell_moss.core.speech.player.miniaudio_player import MiniAudioStreamPlayer
    from ghoshell_moss.core.speech.volcengine_tts import VolcengineTTS

    return BaseTTSSpeech(
        player=player or MiniAudioStreamPlayer(),
        tts=tts or VolcengineTTS(),
        logger=logger,
    )
