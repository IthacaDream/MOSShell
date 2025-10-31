import logging
from typing import Optional, Dict, Tuple, Any, Callable

import numpy as np
from typing_extensions import Literal
from pydantic import BaseModel, Field
from ghoshell_moss.speech.tts.volcengine_protocol import (
    Session, AudioParams,
    start_connection, start_session, unwrap_response, finish_session, finish_connection,
    send_full_client_request,
    Response,
)
from ghoshell_moss.concepts.speech import TTS, TTSBatch, TTSInfo, TTSAudioCallback, AudioFormat
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_common.helpers import uuid
from ghoshell_common.contracts import LoggerItf
from websockets.asyncio.connection import Connection
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK
from websockets import connect
from collections import deque
import os
import json
import asyncio

DefaultEmotions = Literal['surprised', 'fear', 'hate', 'happy', 'sad', 'angry', 'excited', 'coldness', 'neutral']


class VoiceConf(BaseModel):
    tone: str = Field(default="zh_female_cancan_mars_bigtts")
    speech_rate: Optional[int] = Field(
        default=None,
        description="语速，取值范围[-50,100]，100代表2.0倍速，-50代表0.5倍数. 0是正常", ge=-50, le=100,
    )
    loudness_rate: Optional[int] = Field(
        default=None,
        description="音量，取值范围[-50,100]，100代表2.0倍音量，-50代表0.5倍音量. 0是正常", ge=-50, le=100,
    )
    emotion: Optional[DefaultEmotions] = Field(
        default=None,
        description="声音情绪, 拥有多种可选择的声音情绪."
    )


class SpeakerConf(BaseModel):
    """
    角色配置, 可以更改.
    """
    description: str = Field(default="", description="角色的描述")
    resource_id: Optional[str] = Field(default=None, description="使用声音复刻的独立的资源")
    voice: VoiceConf = Field(
        default_factory=VoiceConf,
        description="声音配置"
    )

    def to_voice_conf(self) -> Dict:
        return self.model_dump(exclude={"resource_id"})


_Head = Dict[str, Any]
_Url = str


class VolcengineTTSConf(BaseModel):
    """
    火山引擎 tts 基础配置.
    """
    app_key: str = Field(default="$VOLCENGINE_STREAM_TTS_APP")
    access_token: str = Field(default="$VOLCENGINE_STREAM_TTS_ACCESS_TOKEN")
    resource_id: str = Field(default="seed-tts-2.0", description="官方的默认资源")
    sample_rate: int = Field(default=44100, description="生成音频的采样率要求.")
    audio_format: Literal['pcm'] = Field(default="pcm", description="默认可用的数据格式")

    disconnect_on_idle: int = Field(
        default=100,
        description="闲置多少秒后退出",
    )

    disable_markdown_filter: bool = Field(
        default=True, description="支持朗读 markdown 格式. ")
    url: str = Field(
        default="wss://openspeech.bytedance.com/api/v3/tts/bidirection",
        description="火山的流式语音模型的地址",
    )

    speakers: Dict[str, SpeakerConf] = Field(
        default_factory=lambda: dict(
            cancan=SpeakerConf(
                speaker="zh_female_cancan_mars_bigtts",
                description="标准女声",
            )
        ),
        description="the speakers list",
    )
    default_speaker: str = Field(
        default="default",
        description="the default speaker",
    )

    @classmethod
    def unwrap_env(cls, value: str, default: str = "") -> str:
        if value.startswith('$'):
            return os.environ.get(value[1:], default)
        return value or default

    def default_speaker_conf(self) -> SpeakerConf:
        conf = self.speakers.get(self.default_speaker, None)
        if conf is not None:
            return conf.model_copy(deep=True)
        conf = SpeakerConf()
        return conf

    def gen_header(self, *, connection_id: str = "", resource_id: Optional[str] = None) -> _Head:
        """
        :return: (header, url)
        """
        connection_id = connection_id or uuid()
        ws_header = {
            "X-Api-App-Key": self.unwrap_env(self.app_key),
            "X-Api-Access-Key": self.unwrap_env(self.access_token),
            "X-Api-Resource-Id": resource_id or self.resource_id,
            "X-Api-Connect-Id": connection_id,
        }
        return ws_header

    def to_session(self, speaker: SpeakerConf) -> Session:
        # 生成 additions.
        additions_data = dict(
            disable_markdown_filter=self.disable_markdown_filter,
        )
        additions = json.dumps(additions_data)
        return Session(
            speaker=speaker.voice.tone,
            req_params=dict(
                audio_params=AudioParams(
                    format=self.audio_format,
                    sample_rate=self.sample_rate,
                    loudness_rate=speaker.voice.loudness_rate,
                    speech_rate=speaker.voice.speech_rate,
                    emotion=speaker.voice.emotion,
                ),
                speaker=speaker.voice.tone,
                additions=additions,
            ),
        )

    def to_tts_info(self, current_voice: str = "") -> TTSInfo:
        return TTSInfo(
            rate=self.sample_rate,
            channels=1,
            audio_format=AudioFormat.PCM_S16LE.value,
            voice_schema=VoiceConf.model_json_schema(),
            voices={key: value.to_voice_conf() for key, value in self.speakers.items()},
            current_voice=current_voice or self.default_speaker,
        )


class VolcengineTTSBatch(TTSBatch):

    def __init__(
            self,
            *,
            loop: asyncio.AbstractEventLoop,
            speaker: SpeakerConf,
            batch_id: str = "",
            callback: Optional[TTSAudioCallback] = None,
    ):
        self.speaker = speaker
        self.callback = callback
        self.started = ThreadSafeEvent()
        self.committed = False
        self.closed = ThreadSafeEvent()
        self.tts_done = ThreadSafeEvent()
        self.text_buffer = ""
        self.exception: Optional[Exception] = None
        self._running_loop = loop
        self._has_valid_text = False
        self._batch_id = batch_id or uuid()
        self._text_lock = asyncio.Lock()
        self.texts: asyncio.Queue[str | None] = asyncio.Queue()

    def batch_id(self) -> str:
        return self._batch_id

    def with_callback(self, callback: TTSAudioCallback) -> None:
        self.callback = callback

    def fail(self, reason: str) -> None:
        self.exception = RuntimeError(reason)
        self.closed.set()
        self.commit()

    def feed(self, text: str):
        self.text_buffer += text
        # 已经有过数据了.
        if self._has_valid_text:
            self._running_loop.call_soon_threadsafe(self.texts.put_nowait, text)
        # 这里只能 lstrip
        elif stripped := self.text_buffer.lstrip():
            self._running_loop.call_soon_threadsafe(self.texts.put_nowait, stripped)
            self._has_valid_text = True

    def commit(self):
        self.committed = True
        self.texts.put_nowait(None)
        if not self.text_buffer.strip():
            self.closed.set()

    async def close(self) -> None:
        self.commit()
        self.closed.set()

    async def wait_until_done(self, timeout: float | None = None):
        if timeout is not None and timeout > 0.0:
            await asyncio.wait_for(self.tts_done.wait(), timeout=timeout)
        else:
            await self.tts_done.wait()
        # 抛出异常.
        if self.exception is not None:
            raise self.exception


class VolcengineTTS(TTS):

    def __init__(
            self,
            *,
            conf: VolcengineTTSConf | None = None,
            logger: LoggerItf | None = None,
    ):
        self.logger = logger or logging.getLogger("volcengine.tts")

        # ---- 配置状态 --- #
        # 当前生成的 Batches.
        self._conf = conf or VolcengineTTSConf()
        self._current_speaker: str = self._conf.default_speaker
        self._current_speaker_conf: SpeakerConf = self._conf.default_speaker_conf()

        # --- runtime --- #
        self._starting = False
        self._started = True
        self._running_loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_loop_task: Optional[asyncio.Task] = None
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()

        self._tts_connection_conf = self._current_speaker_conf

        self._pending_batches_queue: asyncio.Queue[VolcengineTTSBatch] = asyncio.Queue()
        self._unfinished_batches: deque[VolcengineTTSBatch] = deque()
        self._running_batch: Optional[VolcengineTTSBatch] = None
        self._has_any_batch_event = asyncio.Event()

        self._consume_pending_batches_task: Optional[asyncio.Task] = None

    def get_info(self) -> TTSInfo:
        return self._conf.to_tts_info(self._current_speaker)

    def use_voice(self, config_key: str) -> None:
        if config_key not in self._conf.speakers:
            raise LookupError(f"The voice {config_key} not found")
        conf = self._conf.speakers[config_key]
        self._current_speaker = config_key
        self._current_speaker_conf = conf.model_copy(deep=True)

    def set_voice(self, config: Dict[str, Any]) -> None:
        voice = VoiceConf(**config)
        self._current_speaker_conf.voice = voice

    def _check_running(self) -> None:
        if not self._started or self._closing_event.is_set():
            raise RuntimeError("TTS is closed")

    def new_batch(self, batch_id: str = "", *, callback: TTSAudioCallback | None = None) -> TTSBatch:
        self._check_running()
        batch = self._create_batch(batch_id, callback)
        self._pending_batches_queue.put_nowait(batch)
        self._has_any_batch_event.set()
        return batch

    def _create_batch(self, batch_id: str = "", callback: TTSAudioCallback | None = None) -> VolcengineTTSBatch:
        speaker_conf = self._current_speaker_conf
        return VolcengineTTSBatch(
            loop=self._running_loop,
            speaker=speaker_conf,
            batch_id=batch_id,
            callback=callback,
        )

    async def _main_loop(self):
        """ tts main connection loop"""
        # 没有关闭前, 一直执行这个循环.
        try:
            while not self._closing_event.is_set():
                if len(self._unfinished_batches) > 0:
                    batch = self._unfinished_batches.popleft()
                else:
                    # 等待到有 batch 存在.
                    await self._has_any_batch_event.wait()
                    if self._pending_batches_queue.empty():
                        # 发现实际上没有 batch.
                        self._has_any_batch_event.clear()
                        continue
                    # 等待一个 connection loop 完成. 要求不会抛出任何异常. 除了 cancel.
                    batch = await self._pending_batches_queue.get()
                # 这个 loop 会持续消费 batch, 直到超过等待时间还没有新 batch 为止.
                task = asyncio.create_task(self._start_consuming_batch_loop(batch))
                # 创建一个可以 cancel 的 task. 它自己应该不要抛出 cancel 异常.
                self._consume_pending_batches_task = task
                # 阻塞等待这个消费循环结束.
                await task
        except asyncio.CancelledError:
            # 不需要记录.
            pass
        finally:
            self._consume_pending_batches_task = None
            self.logger.info("TTS main loop is closed")

    async def _start_consuming_batch_loop(self, batch: VolcengineTTSBatch):
        try:
            if batch.closed.is_set():
                # 已经被关闭了.
                return
            speaker = batch.speaker
            # 当前火山的 resource id
            resource_id = speaker.resource_id or self._conf.resource_id
            connection_id = uuid()
            header = self._conf.gen_header(connection_id=connection_id, resource_id=resource_id)
            url = self._conf.url
            # 创建初始连接.
            self.logger.info("prepare to connect to %s with header %s", url, header)
            async with connect(url, additional_headers=header) as ws:
                # 建连确认.
                await start_connection(ws)
                # 接受确认的事件. 完成握手.
                res = unwrap_response(await ws.recv())
                # 建连没有成功.
                if not res.is_connection_started():
                    self.logger.error("TTS connection receive invalid res after init connection: %s", res)
                    # 承认失败. 关闭连接.
                    batch.fail("TTS connection failed")
                    return

                # 消费完第一个 batch.
                goon = await self._consume_batch_in_connection(batch, connection=ws, current_resource_id=resource_id)
                # 消费后续的 batch.
                if goon:
                    await self._consume_pending_batches(connection=ws, resource_id=resource_id)
                # 全部结束了, 就退出来. 等待外层继续调度.
                self.logger.info("consume batch loop %s is done", connection_id)
                # 发送退出信号. 不等待握手了.
                await finish_connection(ws)

        except ConnectionClosedOK:
            self.logger.info("TTS connection closed ok")
        except ConnectionClosed:
            self.logger.info("TTS connection closed")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.exception(e)

    async def _consume_batch_in_connection(
            self,
            batch: VolcengineTTSBatch,
            connection: Connection,
            current_resource_id: str,
    ) -> bool:
        if batch.closed.is_set():
            return True
        try:
            self._running_batch = batch
            resource_id = batch.speaker.resource_id or current_resource_id
            if resource_id != current_resource_id:
                # 连接不一致, 将未完成的 batch 入队, 关闭整个连接.
                self._unfinished_batches.append(batch)
                return False
            session = self._conf.to_session(batch.speaker)
            # 开始发送文本的流程.
            send_task = asyncio.create_task(self._send_batch_text_to_server(batch, session, connection))
            # 开始接受音频的流程.
            receive_task = asyncio.create_task(self._receive_batch_audio_from_server(batch, session, connection))
            # 等两个都完成, 才能进入下一步.
            send_and_receive = asyncio.gather(send_task, receive_task)
            batch_closed = asyncio.create_task(batch.closed.wait())
            done, pending = await asyncio.wait([send_and_receive, batch_closed], return_when=asyncio.FIRST_COMPLETED)
            if batch_closed in done:
                if not receive_task.done():
                    receive_task.cancel()
                if not send_task.done():
                    send_task.cancel()
            await send_and_receive

            # 正常完成返回 true
            return True

        finally:
            batch.closed.set()
            self._running_batch = None

    async def _send_batch_text_to_server(
            self,
            batch: VolcengineTTSBatch,
            session: Session,
            connection: Connection,
    ) -> None:
        batch_id = batch.batch_id()
        try:
            while not batch.closed.is_set():
                # 发送文本.
                text = await batch.texts.get()
                if text is None:
                    # 拿到了毒丸.
                    break
                # 发送给服务端.
                payload = session.to_request_payload_bytes(text)
                await send_full_client_request(
                    connection,
                    batch_id,
                    payload,
                )

            await finish_session(connection, batch_id)
        except asyncio.CancelledError:
            pass
        except (ConnectionClosedOK, ConnectionClosed):
            raise
        except Exception as e:
            self.logger.exception(e)
            batch.fail(str(e))
            # 特殊的错误, 则关闭 batch.
            await batch.close()

    async def _receive_batch_audio_from_server(
            self,
            batch: VolcengineTTSBatch,
            session: Session,
            connection: Connection,
    ) -> None:
        callback = batch.callback
        try:
            batch_id = batch.batch_id()
            while not batch.closed.is_set():
                received = await connection.recv()
                parsed = unwrap_response(received)
                if parsed.is_audio():
                    if parsed.params.sessionId != batch_id:
                        # 有之前 session 遗留的消息.
                        # 由于事件的消费是单一有序的, 所以问题不大.
                        continue
                    audio_data = parsed.get_audio_data()
                    # callback 不是 async 函数.
                    if callback:
                        callback(audio_data)
                elif parsed.is_session_done():
                    # 结束了当前的接收.
                    break
                else:
                    self._validate_response(parsed)
        finally:
            # batch 永远要设置为关闭.
            batch.closed.set()

    def _validate_response(self, res: Response) -> None:
        self.logger.info(f"TTS response is {res}")
        if res.is_connection_done():
            # 需要中断整个连接.
            raise ConnectionClosed

    async def _consume_pending_batches(self, connection: Connection, resource_id: str) -> None:
        while not self._closing_event.is_set():
            # 尝试获取一个最新的 batch.
            try:
                # 等待下一个 batch, 直到超时为止. 超时后关闭 connection, 避免浪费服务端.
                batch = await asyncio.wait_for(self._pending_batches_queue.get(), timeout=self._conf.disconnect_on_idle)
                # 完成对这个 batch 的执行.
                goon = await self._consume_batch_in_connection(batch, connection, resource_id)
                if not goon:
                    # 这个 loop 不行了, 要从头开始运行.
                    break
            except asyncio.TimeoutError:
                # 超时还没拿到新的 batch, tts 就关闭 connection 了.
                self.logger.info(
                    "close connection after disconnect timeout %s",
                    self._conf.disconnect_on_idle,
                )
                return

    async def clear(self) -> None:
        self._check_running()
        # 清空通知事件.
        self._has_any_batch_event.clear()
        # 清空老队列.
        old_queue = self._pending_batches_queue
        self._pending_batches_queue = asyncio.Queue()
        # 取消运行中的消费对象.
        if self._consume_pending_batches_task is not None:
            self._consume_pending_batches_task.cancel()
        # 设置所有元素为废弃. 速度应该挺快的.
        while not old_queue.empty():
            batch = await old_queue.get()
            await batch.close()

    async def start(self) -> None:
        if self._starting:
            return
        self._starting = True
        self._running_loop = asyncio.get_running_loop()
        self._main_loop_task = asyncio.create_task(self._main_loop())
        self._started = True

    async def close(self) -> None:
        if self._closing_event.is_set():
            return
        self._closing_event.set()
        if self._main_loop_task is not None:
            self._main_loop_task.cancel()
            await self._main_loop_task
            self._main_loop_task = None
        self._closed_event.set()


if __name__ == "__main__":
    # 测试验证 tts 可以运行.
    from ghoshell_common.contracts.logger import get_console_logger


    async def baseline():
        tts = VolcengineTTS(logger=get_console_logger("moss"))

        def print_data_len(data: np.ndarray):
            print(len(data.tobytes()))

        async with tts:
            for text in ["Hello World", "Hello World"]:
                batch = tts.new_batch(callback=print_data_len)
                for char in text:
                    batch.feed(char)
                batch.commit()
                await batch.wait_until_done()


    asyncio.run(baseline())
