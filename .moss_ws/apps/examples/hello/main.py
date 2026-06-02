"""Hello World — the minimal Channel App for MOSS.

Demonstrates the full lifecycle:
  1. moss apps create examples/hello
  2. Write app logic (this file)
  3. moss apps test examples/hello              (foreground debug)
  4. <apps:start fullname="examples/hello" />   (MCP / CTML)
  5. <apps.examples_hello:greet />              (CTML call)
  6. <apps:stop fullname="examples/hello" />    (shutdown)
"""

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import new_channel


# 创建一个 channel — 名字会成为 CTML 调用路径
channel = new_channel(
    name="examples_hello",
    description="A minimal hello-world example channel. Greet someone or check the time.",
)


@channel.build.command()
async def greet(name: str = "World") -> str:
    """Say hello to someone."""
    return f"Hello, {name}! I'm running inside a MOSS App."


@channel.build.command()
async def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


@channel.build.context_messages
async def context() -> list[str]:
    """Provide dynamic context visible to the AI in moss_dynamic."""
    return ["[Hello App] Running and ready."]


async def main(matrix: Matrix):
    # 把 channel 注册到 Matrix 通讯总线
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
