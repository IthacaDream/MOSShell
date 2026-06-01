from ghoshell_moss.host.providers import (
    HostSessionProvider,
    ZenohTopicServiceProvider,
    HostLoggerProvider,
    HostEnvZenohProvider,
    HostEnvConfigStoreProvider,
)
from ghoshell_moss.host.providers.tts_service_provider import TTSServiceProvider
from ghoshell_moss.host.providers.speech_service_provider import TTSSpeechServiceProvider
from ghoshell_moss.host.providers.audio_player_provider import AudioPlayerProvider
from ghoshell_moss.core.resources.memory_registry import InMemoryResourceRegistryProvider
from ghoshell_moss.host.fractal.zenoh_fractal import ZenohFractalHubProvider, ZenohFractalCellContractProvider

moss_session_provider = HostSessionProvider()

config_store_provider = HostEnvConfigStoreProvider()

zenoh_session_provider = HostEnvZenohProvider()

logger_provider = HostLoggerProvider()

topic_service_provider = ZenohTopicServiceProvider()

# audio player and speech

player_service_provider = AudioPlayerProvider()

tts_service_provider = TTSServiceProvider()

speech_service_provider = TTSSpeechServiceProvider()

resources_provider = InMemoryResourceRegistryProvider()

fractal_hub_provider = ZenohFractalHubProvider()

fractal_node_provider = ZenohFractalCellContractProvider()
