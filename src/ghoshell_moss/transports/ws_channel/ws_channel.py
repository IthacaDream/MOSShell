
import json
import asyncio
import logging
import fastapi
from typing import Optional
from dataclasses import dataclass

try:
    import websockets
except ImportError:
    raise ImportError(f'Please install websockets by "pip install ghoshell-moss[wss]"')

from ghoshell_moss.core.duplex.connection import Connection, ConnectionClosedError, ConnectionNotAvailable
from ghoshell_moss.core.duplex.protocol import ChannelEvent, HeartbeatEvent
from ghoshell_moss.core.duplex.provider import DuplexChannelProvider
from ghoshell_moss.core.duplex.proxy import DuplexChannelProxy
from ghoshell_container import Container, IoCContainer

logger = logging.getLogger(__name__)


class FastAPIWebSocketConnection(Connection):
    """基于FastAPI启动的WebSocket服务端连接"""

    def __init__(self, ws: fastapi.WebSocket):
        self._ws: fastapi.WebSocket = ws
        self._closed_event = asyncio.Event()
        self._recv_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()

    async def recv(self, timeout: Optional[float] = None) -> ChannelEvent:
        """接收客户端消息"""
        if self._closed_event.is_set():
            raise ConnectionClosedError("Connection closed")

        async with self._recv_lock:
            while not self._closed_event.is_set():
                try:
                    message = await asyncio.wait_for(
                        self._ws.receive_text(),
                        timeout=timeout,
                    )
                    event = json.loads(message)
                    logger.info(f'FastAPIWebSocketConnection Received event: {event}')
                    return event
                except asyncio.TimeoutError:
                    raise
                except fastapi.WebSocketDisconnect as e:
                    raise ConnectionClosedError(f"Connection closed: {e}")
                except Exception as e:
                    logger.warning("Failed to receive message: %s", e)
                    raise

    async def send(self, event: ChannelEvent) -> None:
        """发送消息给客户端"""
        if self._closed_event.is_set():
            raise ConnectionClosedError("Connection closed")
        if self._ws is None:
            raise RuntimeError("Connection not started")
        async with self._send_lock:
            try:
                logger.info(f'FastAPIWebSocketConnection Sending event: {event}')
                await self._ws.send_text(json.dumps(event))
            except Exception as e:
                logger.warning("Failed to send message: %s", e)
                raise

    def is_closed(self) -> bool:
        return self._closed_event.is_set()

    def is_available(self) -> bool:
        return not self.is_closed()

    async def close(self) -> None:
        """关闭连接"""
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._closed_event.set()

    async def start(self) -> None:
        # 交给FastAPI处理
        pass


class FastAPIWebSocketChannelProxy(DuplexChannelProxy):
    """基于FastAPI的WebSocket Channel代理"""

    def __init__(
            self,
            *,
            ws: fastapi.WebSocket,
            name: str,
    ):
        connection = FastAPIWebSocketConnection(ws)
        super().__init__(
            name=name,
            to_server_connection=connection,
        )


@dataclass
class WebSocketConnectionConfig:
    """WebSocket Channel配置"""
    address: str
    headers: Optional[dict] = None


class WebSocketConnection(Connection):
    """客户端连接"""

    def __init__(self, config: WebSocketConnectionConfig):
        self._ws: Optional[websockets.ClientConnection] = None
        self._config = config
        self._closed_event = asyncio.Event()
        self._recv_lock = asyncio.Lock()

    def is_closed(self) -> bool:
        return self._closed_event.is_set()

    def is_available(self) -> bool:
        return not self.is_closed() and self._is_active()

    def _is_active(self) -> bool:
        """判断连接是否活跃（未超时）"""
        return self._ws.state not in (websockets.State.CLOSED, websockets.State.CLOSING)

    async def recv(self, timeout: Optional[float] = None) -> ChannelEvent:
        """接收服务端消息（处理心跳+超时）"""
        if self._closed_event.is_set():
            raise ConnectionClosedError("Connection closed")
        if not self._ws:
            raise RuntimeError("Connection not started")

        async with self._recv_lock:
            while not self.is_closed():
                try:
                    message = await asyncio.wait_for(
                        self._ws.recv(),
                        timeout=timeout,
                    )
                    event: ChannelEvent = json.loads(message)
                    logger.info(f'WebSocketConnection Received event: {event}')
                    return event
                except websockets.ConnectionClosed as e:
                    self._closed_event.set()
                    raise ConnectionClosedError(f"Connection closed: {str(e)}") from e
                except asyncio.TimeoutError:
                    # self._closed_event.set()
                    raise ConnectionClosedError("Connection timeout")

    async def send(self, event: ChannelEvent) -> None:
        """发送消息给服务端"""
        if self._closed_event.is_set():
            raise ConnectionClosedError("Connection closed")
        if not self._ws:
            raise RuntimeError("Connection not started")
        try:
            logger.info(f'WebSocketConnection Sending event: {event}')
            await self._ws.send(json.dumps(event))
        except Exception as e:
            logger.warning("Failed to send message: %s", e)
            raise

    async def start(self) -> None:
        """启动客户端（连接服务端）"""
        if self._ws:
            return

        try:
            self._ws = await websockets.connect(self._config.address, additional_headers=self._config.headers)
            self._ws.start_keepalive()
        except websockets.exceptions.InvalidStatus as e:
            logger.error(f"Connection failed: {e}")
            logger.error(f"Status code: {e.response.status_code}")
            logger.error(f"Response headers: {e.response.headers}")
            self._closed_event.set()
            return
        except Exception as e:
            logger.error(f"Failed to connect to {self._config.address}: {str(e)}")
            self._closed_event.set()
            raise

        logger.info(f"WebSocket client connected to {self._config.address}")

    async def close(self) -> None:
        """关闭客户端（断开服务端连接）"""
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._closed_event.set()


class WebSocketChannelProvider(DuplexChannelProvider):
    """WebSocket Channel提供者"""

    def __init__(
            self,
            config: WebSocketConnectionConfig,
            *,
            container: Optional[IoCContainer] = None,

    ):
        connection = WebSocketConnection(config)
        super().__init__(
            provider_connection=connection,
            container=Container(parent=container, name="WebSocketChannelProvider")
        )
