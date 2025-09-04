from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union, Callable, Awaitable, Coroutine, List, Type, TypeVar, Dict, ClassVar, Any
from .command import Command, CommandCall, CommandType, CommandMeta, CommandTask
from ghoshell_container import IoCContainer, INSTANCE
from ghoshell_common.helpers import generate_import_path
from pydantic import BaseModel, Field

FunctionCommand = Callable[..., Coroutine]
PolicyCommand = Callable[..., Coroutine[None]]

R = TypeVar('R')


class State(BaseModel):
    version: str = Field(default="", description="state version, optimis lock")
    name: str = Field(description="The name of the state object.")
    description: str = Field(default="", description="The description of the state object.")
    schema: Dict = Field(description="the json schema of the state")
    default: Dict = Field(description="the default value of the state")


class StateModel(BaseModel, ABC):
    state_desc: ClassVar[str] = ""
    state_name: ClassVar[str] = ""

    @classmethod
    def to_state(cls) -> State:
        name = cls.state_name or generate_import_path(cls)
        description = cls.state_desc or cls.__doc__ or ""
        default = cls().model_dump()
        schema = cls.model_json_schema()
        return State(name=name, description=description, schema=schema, default=default)


class Runtime(ABC):

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        """
        提供依赖注入容器, 可以在被 command 标记的函数中用.
        """
        pass

    def make(self, contract: Type[INSTANCE]) -> INSTANCE:
        return self.container.make(contract)

    def call(self, func: Callable[..., R], *args, **kwargs) -> R:
        return self.container.call(func, *args, **kwargs)

    # --- commands --- #

    @abstractmethod
    def push_tail(self, *commands: Command) -> None:
        pass

    @abstractmethod
    def push_head(self, *commands: Command) -> None:
        pass

    # --- states --- #
    @abstractmethod
    def get_state(self, name: str) -> Dict:
        pass

    @abstractmethod
    def get_state_model(self, model: Type[StateModel]) -> StateModel:
        pass

    @abstractmethod
    def set_state(self, name: str, data: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def set_state_model(self, model: Type[StateModel]) -> StateModel:
        pass

    # --- control --- #

    @abstractmethod
    def clear(self) -> None:
        """
        clear the channel's commands, include executing command and pending command.
        after clear, the channel will rerun the policy.
        """
        pass

    @abstractmethod
    def cancel(self) -> bool:
        """
        cancel the running command task
        """
        pass

    @abstractmethod
    def defer_clear(self) -> None:
        """
        clear when any command is pushed into this channel
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        返回初始状态. 包括 policy 也会重置回原始状态.
        """
        pass

    # --- status --- #
    @abstractmethod
    def is_idle(self) -> bool:
        pass

    @abstractmethod
    def wait_until_idle(self, timeout: float | None = None) -> None:
        pass

    # --- call --- #
    @abstractmethod
    def new_task(self, name: str, *args, **kwargs) -> CommandTask:
        pass

    @abstractmethod
    def get_command_metas(self, types: Optional[CommandType] = None) -> Iterable[CommandMeta]:
        pass


class Channel(ABC):

    # --- self-explanation --- #

    @property
    @abstractmethod
    def runtime(self) -> Runtime:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def get_commands(self, types: Optional[CommandType] = None) -> Iterable[Command]:
        pass

    @abstractmethod
    def states(self) -> Iterable[State]:
        pass

    # --- decorators --- #

    @abstractmethod
    def function(
            self,
            *,
            name: str = "",
            doc: Optional[Callable[..., str] | str] = None,
            interface: Optional[str | Callable[..., str]] = None,
            to_thread: bool = False,
            tags: Optional[List[str]] = None,
    ) -> Callable[[FunctionCommand], FunctionCommand]:
        """
        wrap an async function
        """
        pass

    @abstractmethod
    def policy(
            self,
            *,
            name: str = "",
            doc: Optional[Union[Callable[..., str] | str]] = None,
            interface: Optional[str] = None,
            tags: Optional[List[str]] = None,
    ) -> Callable[[PolicyCommand], PolicyCommand]:
        pass

    @abstractmethod
    def with_state_model(self, model: Type[StateModel]) -> None:
        """
        register state model for channel
        """
        pass

    @abstractmethod
    def with_state(self, state: State) -> None:
        """
        register state model for channel
        """
        pass

    # --- lifecycle --- #

    @abstractmethod
    def bootstrap(self, container: IoCContainer) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def join(self) -> None:
        pass
