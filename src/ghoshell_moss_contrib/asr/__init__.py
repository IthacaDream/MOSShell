# 异步模块导出
try:
    from framework.listener.async_concepts import (
        AsyncAudioInput,
        AsyncRecognitionCallback,
        AsyncRecognitionBatch,
        AsyncRecognizer,
        AsyncListenerCallback,
        AsyncListenerState,
        AsyncListenerService,
        AsyncListenerStateName,
        Recognition,
    )

    from framework.listener.async_listener_service import AsyncListenerServiceImpl

    from framework.listener.async_states import (
        AsyncAudioInputLoop,
        AsyncDeafState,
        AsyncListeningState,
        AsyncPdtListeningState,
        AsyncPdtWaitingState,
    )

    from framework.listener.async_pyaudio_input import AsyncPyAudioInput, AsyncPyAudioInputConfig

    from framework.listener.async_volcengine_bm import (
        AsyncVocEngineBigModelASR,
        AsyncVocEngineBigModelStreamASRBatch,
    )

    __all_async__ = [
        'AsyncAudioInput',
        'AsyncRecognitionCallback',
        'AsyncRecognitionBatch',
        'AsyncRecognizer',
        'AsyncListenerCallback',
        'AsyncListenerState',
        'AsyncListenerService',
        'AsyncListenerStateName',
        'Recognition',
        'AsyncListenerServiceImpl',
        'AsyncAudioInputLoop',
        'AsyncDeafState',
        'AsyncListeningState',
        'AsyncPdtListeningState',
        'AsyncPdtWaitingState',
        'AsyncPyAudioInput',
        'AsyncPyAudioInputConfig',
        'AsyncVocEngineBigModelASR',
        'AsyncVocEngineBigModelStreamASRBatch',
    ]

except ImportError as e:
    import logging
    logging.getLogger(__name__).debug(f"Async modules not fully available: {e}")
