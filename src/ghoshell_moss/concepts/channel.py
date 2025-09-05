from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union, Callable, Coroutine, List, Type, TypeVar, Dict, ClassVar, Any
from typing_extensions import Self
from .command import Command, CommandCall, CommandType, CommandMeta, CommandTask
from ghoshell_container import IoCContainer, INSTANCE, Provider, BINDING
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
    """
    通过强类型的方式对 State 进行建模.
    """
    state_desc: ClassVar[str] = ""
    state_name: ClassVar[str] = ""

    @classmethod
    def to_state(cls) -> State:
        name = cls.state_name or generate_import_path(cls)
        description = cls.state_desc or cls.__doc__ or ""
        default = cls().model_dump()
        schema = cls.model_json_schema()
        return State(name=name, description=description, schema=schema, default=default)


class ChannelMeta(BaseModel):
    name: str = Field(description="The name of the channel.")
    available: bool = Field(description="Whether the channel is available.")
    description: str = Field(description="The description of the channel.")
    stats: List[State] = Field(default_factory=list, description="The list of state objects.")
    commands: List[CommandMeta] = Field(default_factory=list, description="The list of commands.")


class Runtime(ABC):

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        """
        提供依赖注入容器, 可以在被 command 标记的函数中用.
        """
        pass

    @abstractmethod
    def meta(self) -> ChannelMeta:
        pass

    # --- commands --- #

    @abstractmethod
    def append(self, *commands: CommandTask) -> None:
        pass

    @abstractmethod
    def prepend(self, *commands: CommandTask) -> None:
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

    # --- output --- #

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

    @abstractmethod
    def get_commands(self, types: Optional[CommandType] = None) -> Iterable[Command]:
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

    def default_meta(self) -> ChannelMeta:
        return ChannelMeta(
            name=self.name(),
            description=self.description(),
            available=False,
        )

    @abstractmethod
    def states(self) -> Iterable[State]:
        pass

    # --- children --- #

    @abstractmethod
    def with_children(self, *children: "Channel") -> Self:
        pass

    @abstractmethod
    def children(self) -> Iterable[Self]:
        """
        register children channel.
        """
        pass

    # --- decorators --- #

    @abstractmethod
    def with_description(self, callback: Callable[..., str]) -> Callable[..., str]:
        pass

    @abstractmethod
    def with_function(
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
    def with_policy(
            self,
            *,
            name: str = "",
            doc: Optional[Union[Callable[..., str] | str]] = None,
            interface: Optional[str] = None,
            tags: Optional[List[str]] = None,
    ) -> Callable[[PolicyCommand], PolicyCommand]:
        """
        register policy functions
        """
        pass

    @abstractmethod
    def with_providers(self, *providers: Provider) -> None:
        """
        register default providers for the contracts
        """
        pass

    @abstractmethod
    def with_binding(self, contract: Type[INSTANCE], binding: Optional[BINDING] = None) -> None:
        """
        register default bindings for the given contract.
        """
        pass

    @abstractmethod
    def with_state_policy(self, model: StateModel) -> StateModel:
        pass

    @abstractmethod
    def with_state_model(self, model: Type[StateModel]) -> Type[StateModel]:
        """
        register state model for channel
        """
        pass

    @abstractmethod
    def with_state(self, state: State) -> State:
        """
        register state model for channel
        """
        pass

    # --- lifecycle --- #

    @abstractmethod
    def bootstrap(self, container: Optional[IoCContainer] = None) -> None:
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
