from typing import Callable
from ghoshell_container import IoCContainer
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.blueprint.fractal import FractalHub
from ghoshell_moss.core.blueprint.matrix import Matrix

__all__ = ['matrix_fractal_hub_channel_factory']


def matrix_fractal_hub_channel_factory(
        description: str = 'fractal moss connection',
        allow_all: bool = False,
        auto_start: bool = False,
) -> Callable[[IoCContainer], Channel]:
    """
    return a channel factory that can be imported by main channel.
    """

    def factory(container: IoCContainer) -> Channel | None:
        hub = container.get(FractalHub)
        if hub is None:
            return None
        if auto_start:
            matrix = container.force_fetch(Matrix)
            # 注册启动生命周期.
            matrix.register_lifecycle_object(hub)
        return hub.as_channel(description, allow_all, auto_start)

    return factory
