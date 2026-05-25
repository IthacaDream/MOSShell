from typing import ClassVar, Protocol, runtime_checkable

__all__ = ["BridgeExpr", "NodeChannelBridgeExpr"]


@runtime_checkable
class BridgeExpr(Protocol):
    """Channel bridge key expression 的接口契约。

    Provider/Proxy/Hub 通过此 Protocol 感知 key 体系，
    不依赖具体实现的类继承。
    """

    address: str
    session_scope: str
    bridge_prefix: str
    provider_liveness_key: str
    proxy_liveness_key: str
    provider_receiver_key: str
    proxy_receiver_key: str


class NodeChannelBridgeExpr:
    """
    基线 BridgeExpr 实现。约定优先于配置。

    prefix_parts=None → 约定前缀 "MOSS/{session_scope}/node/{address}/channel_bridge"
    prefix_parts=["MOSS", "{session_scope}", "matrix"] → hub 模式
    """

    NODE_BRIDGE_PREFIX_TEMPLATE: ClassVar[str] = "MOSS/{session_scope}/node/{address}/channel_bridge"

    PROVIDER_LIVENESS_KEY: ClassVar[str] = "provider_liveness"
    PROXY_LIVENESS_KEY: ClassVar[str] = "proxy_liveness"
    PROVIDER_RECEIVER: ClassVar[str] = "provider"
    PROXY_RECEIVER: ClassVar[str] = "proxy"

    def __init__(self, address: str, session_scope: str, *, prefix_parts: list[str] | None = None):
        self.address = address
        self.session_scope = session_scope
        if prefix_parts is not None:
            parts = [p.format(session_scope=session_scope) for p in prefix_parts]
            self.bridge_prefix = "/".join(parts + [address, "channel_bridge"])
        else:
            self.bridge_prefix = self.NODE_BRIDGE_PREFIX_TEMPLATE.format(
                session_scope=self.session_scope,
                address=self.address,
            )
        self.provider_liveness_key: str = "/".join([self.bridge_prefix, self.PROVIDER_LIVENESS_KEY])
        self.proxy_liveness_key: str = "/".join([self.bridge_prefix, self.PROXY_LIVENESS_KEY])

        self.provider_receiver_key: str = "/".join([self.bridge_prefix, self.PROVIDER_RECEIVER])
        '''proxy send to provider'''

        self.proxy_receiver_key: str = "/".join([self.bridge_prefix, self.PROXY_RECEIVER])
        '''provider send to proxy'''
