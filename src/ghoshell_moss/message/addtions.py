
from .abcd import Addition
from openai.types.completion_usage import CompletionUsage

__all__ = [
    'CompletionUsageAddition',
]


class CompletionUsageAddition(Addition, CompletionUsage):
    """
    OpenAI 模型调用的数据.
    """

    @classmethod
    def keyword(cls) -> str:
        return "completion_usage"
