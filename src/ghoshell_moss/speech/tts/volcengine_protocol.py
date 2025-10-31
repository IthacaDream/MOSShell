# https://www.volcengine.com/docs/6561/1329505#%E7%A4%BA%E4%BE%8Bsamples
from typing import Union
from websockets.asyncio.connection import Connection
from typing import Optional, Dict, Tuple
from typing_extensions import Self, Literal
from pydantic import BaseModel, Field
from ghoshell_common.helpers import uuid
from enum import Enum
import os
import json


class EventCode(int, Enum):
    # Connect 类事件
    START_CONNECTION = 1  # StartConnection, Websocket 阶段申请创建连接（在 HTTP 建立 Upgrade 后）
    FINISH_CONNECTION = 2  # FinishConnection, 结束连接
    CONNECTION_STARTED = 50  # ConnectionStarted, 成功建立
    CONNECTION_FAILED = 51  # ConnectionFailed, 建连失败
    CONNECTION_FINISHED = 52  # ConnectionFinished 结束连接成功

    # Session 类事件
    START_SESSION = 100  # StartSession, Websocket 阶段申请创建会话
    FINISH_SESSION = 102  # FinishSession, 声明结束会话（上行）
    SESSION_STARTED = 150  # SessionStarted, 成功开始会话
    SESSION_CANCELED = 151  # SessionCanceled, 已取消会话
    SESSION_FINISHED = 152  # SessionFinished, 会话已结束（上行&下行）
    SESSION_FAILED = 153  # SessionFailed, 会话失败

    # 数据类事件
    TASK_REQUEST = 200  # TaskRequest, 传输请求内容
    TTS_SENTENCE_START = 350  # TTSSentenceStart, TTS 返回句内容开始
    TTS_SENTENCE_END = 351  # TTSSentenceEnd, TTS 返回句内容结束
    TTS_RESPONSE = 352  # TTSResponse, TTS 返回的音频内容


class User(BaseModel):
    uid: str = Field(default="", description="")


class AudioParams(BaseModel):
    format: Literal["mp3", "pcm", "ogg_opus"] = Field(default="pcm")
    sample_rate: int = Field(
        default=44100, description="8000,16000,22050,24000,32000,44100,48000")
    loudness_rate: Optional[int] = Field(default=0)
    speech_rate: Optional[int] = Field(default=0)
    emotion: Optional[str] = Field(default="neutral")


class ReqParams(BaseModel):
    audio_params: AudioParams = Field(default_factory=AudioParams)
    speaker: str = Field(default="zh_female_cancan_mars_bigtts")
    additions: Optional[str] = Field(default=None)


class Session(BaseModel):
    """
    session 数据.
    """
    user: User = Field(default_factory=User)
    event: int = EventCode.START_SESSION.value
    req_params: ReqParams = Field(default_factory=ReqParams)

    def to_payload_bytes(self) -> bytes:
        config = self
        data = config.model_dump_json(exclude_none=True)
        return data.encode()

    def to_payload_str(self) -> str:
        config = self
        data = config.model_dump_json(exclude_none=True)
        return data

    def to_request_payload_bytes(self, text: str) -> bytes:
        data = self.model_dump(exclude_none=True)
        data["req_params"]["text"] = text
        data["event"] = EventCode.TASK_REQUEST.value
        j = json.dumps(data, ensure_ascii=False)
        return j.encode()


def is_error_frame(frame: bytes) -> bool:
    # 检查协议版本和头部大小
    if frame[0] != 0b00010001:
        return False

    # 检查消息类型
    if frame[1] != 0b11110000:
        return False

    # 检查序列化方法和压缩方法
    if frame[2] != 0b00010000:
        return False

    # 检查保留字节
    if frame[3] != 0b00000000:
        return False

    # 如果以上条件都满足，则是异常帧
    return True


def is_normal_frame(frame: bytes) -> bool:
    # 检查协议版本和头部大小
    if frame[0] != 0b00010001:
        return False

    # 检查消息类型
    if frame[1] not in [0b00000000, 0b00000001, 0b00000010, 0b00000011]:
        return False

    # 检查序列化方法和压缩方法
    if frame[2] not in [0b00000000, 0b00010000, 0b00100000, 0b00110000]:
        return False

    # 检查压缩方法
    if frame[2] & 0b00001111 not in [0b0000, 0b0001]:
        return False

    # 检查保留字节
    if frame[3] != 0b00000000:
        return False

    # 如果以上条件都满足，则是正常帧
    return True


PROTOCOL_VERSION = 0b0001
DEFAULT_HEADER_SIZE = 0b0001

# Message Type:
FULL_CLIENT_REQUEST = 0b0001
AUDIO_ONLY_RESPONSE = 0b1011
FULL_SERVER_RESPONSE = 0b1001
ERROR_INFORMATION = 0b1111

# Message Type Specific Flags
MsgTypeFlagNoSeq = 0b0000  # Non-terminal packet with no sequence
MsgTypeFlagPositiveSeq = 0b1  # Non-terminal packet with sequence > 0
MsgTypeFlagLastNoSeq = 0b10  # last packet with no sequence
MsgTypeFlagNegativeSeq = 0b11  # Payload contains event number (int32)
MsgTypeFlagWithEvent = 0b100
# Message Serialization
NO_SERIALIZATION = 0b0000
JSON = 0b0001
# Message Compression
COMPRESSION_NO = 0b0000
COMPRESSION_GZIP = 0b0001

EVENT_NONE = 0
EVENT_Start_Connection = 1

EVENT_FinishConnection = 2

EVENT_ConnectionStarted = 50  # 成功建连

EVENT_ConnectionFailed = 51  # 建连失败（可能是无法通过权限认证）

EVENT_ConnectionFinished = 52  # 连接结束

# 上行Session事件
EVENT_StartSession = 100

EVENT_FinishSession = 102
# 下行Session事件
EVENT_SessionStarted = 150
EVENT_SessionFinished = 152

EVENT_SessionFailed = 153

# 上行通用事件
EVENT_TaskRequest = 200

# 下行TTS事件
EVENT_TTSSentenceStart = 350

EVENT_TTSSentenceEnd = 351

EVENT_TTSResponse = 352


class Header:
    def __init__(self,
                 protocol_version=PROTOCOL_VERSION,
                 header_size=DEFAULT_HEADER_SIZE,
                 message_type: int = 0,
                 message_type_specific_flags: int = 0,
                 serial_method: int = NO_SERIALIZATION,
                 compression_type: int = COMPRESSION_NO,
                 reserved_data=0):
        self.header_size = header_size
        self.protocol_version = protocol_version
        self.message_type = message_type
        self.message_type_specific_flags = message_type_specific_flags
        self.serial_method = serial_method
        self.compression_type = compression_type
        self.reserved_data = reserved_data

    def as_bytes(self) -> bytes:
        return bytes([
            (self.protocol_version << 4) | self.header_size,
            (self.message_type << 4) | self.message_type_specific_flags,
            (self.serial_method << 4) | self.compression_type,
            self.reserved_data
        ])


class Params:
    def __init__(self, event: int = EVENT_NONE, session_id: str = None, sequence: int = None):
        self.event = event
        self.sessionId = session_id
        self.errorCode: int = 0
        self.connectionId: str | None = None
        self.response_meta_json: str | None = None
        self.sequence = sequence

    # 转成 byte 序列
    def as_bytes(self) -> bytes:
        option_bytes = bytearray()
        if self.event != EVENT_NONE:
            option_bytes.extend(self.event.to_bytes(4, "big", signed=True))
        if self.sessionId is not None:
            session_id_bytes = str.encode(self.sessionId)
            size = len(session_id_bytes).to_bytes(4, "big", signed=True)
            option_bytes.extend(size)
            option_bytes.extend(session_id_bytes)
        if self.sequence is not None:
            option_bytes.extend(self.sequence.to_bytes(4, "big", signed=True))
        return option_bytes


class Response:
    def __init__(self, header: Header, params: Params):
        self.params = params
        self.header = header
        self.payload: bytes | None = None

    def is_audio(self) -> bool:
        return self.params.event == EVENT_TTSResponse and self.header.message_type == AUDIO_ONLY_RESPONSE

    def is_connection_started(self) -> bool:
        return self.params.event == EVENT_ConnectionStarted

    def get_audio_data(self) -> bytes | None:
        return self.payload

    def is_session_done(self) -> bool:
        return self.params.event in (EVENT_SessionFailed, EVENT_SessionFinished)

    def is_connection_done(self) -> bool:
        return self.params.event in (EVENT_ConnectionFailed, EVENT_ConnectionFinished)

    def __str__(self):
        return str(dict(
            optional=self.params.__dict__,
            header=self.header.__dict__,
            payload=len(self.payload) if self.payload else None,
        ))


# 发送事件
async def send_event(
        ws: Connection,
        header: bytes,
        optional: bytes | None = None,
        payload: bytes = None,
):
    """
    send event to websocket server
    """
    full_client_request = bytearray(header)
    if optional is not None:
        full_client_request.extend(optional)
    if payload is not None:
        payload_size = len(payload).to_bytes(4, 'big', signed=True)
        full_client_request.extend(payload_size)
        full_client_request.extend(payload)
    await ws.send(full_client_request)


# 读取 res 数组某段 字符串内容
def read_res_content(res: bytes, offset: int) -> Tuple[str, int]:
    content_size = int.from_bytes(res[offset: offset + 4], 'big')
    offset += 4
    content = res[offset: offset + content_size].decode()
    offset += content_size
    return content, offset


# 读取 payload
def read_res_payload(res: bytes, offset: int) -> Tuple[bytes, int]:
    payload_size = int.from_bytes(res[offset: offset + 4], 'big')
    offset += 4
    payload = res[offset: offset + payload_size]
    offset += payload_size
    return payload, offset


# 解析响应结果
def unwrap_response(res) -> Response:
    if isinstance(res, str):
        raise RuntimeError(res)
    response = Response(Header(), Params())
    # 解析结果
    # header
    header = response.header
    num = 0b00001111
    header.protocol_version = res[0] >> 4 & num
    header.header_size = res[0] & 0x0f
    header.message_type = (res[1] >> 4) & num
    header.message_type_specific_flags = res[1] & 0x0f
    header.serialization_method = res[2] >> num
    header.message_compression = res[2] & 0x0f
    header.reserved = res[3]
    #
    offset = 4
    optional = response.params
    if header.message_type == FULL_SERVER_RESPONSE or AUDIO_ONLY_RESPONSE:
        # read event
        if header.message_type_specific_flags == MsgTypeFlagWithEvent:
            optional.event = int.from_bytes(res[offset:8], 'big')
            offset += 4
            if optional.event == EVENT_NONE:
                return response
            # read connectionId
            elif optional.event == EVENT_ConnectionStarted:
                optional.connectionId, offset = read_res_content(res, offset)
            elif optional.event == EVENT_ConnectionFailed:
                optional.response_meta_json, offset = read_res_content(res, offset)
            elif (optional.event == EVENT_SessionStarted
                  or optional.event == EVENT_SessionFailed
                  or optional.event == EVENT_SessionFinished):
                optional.sessionId, offset = read_res_content(res, offset)
                optional.response_meta_json, offset = read_res_content(res, offset)
            else:
                optional.sessionId, offset = read_res_content(res, offset)
                response.payload, offset = read_res_payload(res, offset)

    elif header.message_type == ERROR_INFORMATION:
        optional.errorCode = int.from_bytes(res[offset:offset + 4], "big", signed=True)
        offset += 4
        response.payload, offset = read_res_payload(res, offset)
    return response


def get_payload_bytes(
        uid='1234',
        event=EVENT_NONE,
        text='',
        speaker='',
        audio_format='mp3',
        audio_sample_rate=24000,
):
    return str.encode(json.dumps(
        {
            "user": {"uid": uid},
            "event": event,
            "namespace": "BidirectionalTTS",
            "req_params": {
                "text": text,
                "speaker": speaker,
                "audio_params": {
                    "format": audio_format,
                    "sample_rate": audio_sample_rate
                }
            }
        }
    ))


async def start_connection(ws: Connection):
    """
    开启 connection.
    """
    header = Header(message_type=FULL_CLIENT_REQUEST, message_type_specific_flags=MsgTypeFlagWithEvent).as_bytes()
    optional = Params(event=EVENT_Start_Connection).as_bytes()
    payload = str.encode("{}")
    return await send_event(ws, header, optional, payload)


async def start_session(
        ws: Connection,
        session_id: str,
        session_payload: bytes,
):
    """
    创建 session.
    """
    header = Header(message_type=FULL_CLIENT_REQUEST,
                    message_type_specific_flags=MsgTypeFlagWithEvent,
                    serial_method=JSON
                    ).as_bytes()
    optional = Params(event=EVENT_StartSession, session_id=session_id).as_bytes()
    return await send_event(ws, header, optional, session_payload)


async def finish_session(
        ws: Connection,
        session_id: str,
):
    """
    关闭一个 session.
    """
    header = Header(message_type=FULL_CLIENT_REQUEST,
                    message_type_specific_flags=MsgTypeFlagWithEvent,
                    serial_method=JSON
                    ).as_bytes()
    optional = Params(event=EVENT_FinishSession, session_id=session_id).as_bytes()
    payload = str.encode('{}')
    return await send_event(ws, header, optional, payload)


async def send_full_client_request(
        ws: Connection,
        session_id: str,
        task_reqeust_payload: bytes,
):
    """
    发送文本.
    """
    header = Header(message_type=FULL_CLIENT_REQUEST,
                    message_type_specific_flags=MsgTypeFlagWithEvent,
                    serial_method=JSON).as_bytes()
    optional = Params(event=EVENT_TaskRequest, session_id=session_id).as_bytes()
    return await send_event(ws, header, optional, task_reqeust_payload)


async def finish_connection(ws: Connection):
    """
    关闭连接.
    """
    header = Header(message_type=FULL_CLIENT_REQUEST,
                    message_type_specific_flags=MsgTypeFlagWithEvent,
                    serial_method=JSON
                    ).as_bytes()
    optional = Params(event=EVENT_FinishConnection).as_bytes()
    payload = str.encode('{}')
    return await send_event(ws, header, optional, payload)
