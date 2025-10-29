from abc import ABC, abstractmethod
from typing import List, Optional
from ghoshell_moss.concepts.command import CommandTask

__all__ = [
    'Speech', 'SpeechStream',
]


class SpeechStream(ABC):
    """
    shell 发送文本的专用模块. 本身是非阻塞的.
    todo: 考虑把 OutputStream 通用成 Command.
    """
    id: str
    """所有文本片段都有独立的全局唯一id, 通常是 command_token.part_id"""

    cmd_task: Optional[CommandTask] = None
    """stream 生成的 command task"""

    committed: bool = False
    """是否完成了这个 stream 的提交. """

    def buffer(self, text: str, *, complete: bool = False) -> None:
        """
        添加文本片段到输出流里.
        由于文本可以通过 tts 生成语音, 而 tts 有独立的耗时, 所以通常一边解析 command token 一边 buffer 到 tts 中.
        而音频允许播放的时间则会靠后, 必须等上一段完成后才能开始播放下一段.

        :param text: 文本片段
        :type complete: 输出流是否已经结束.
        """
        if self.committed:
            # 不 buffer.
            return
        if text:
            # 文本不为空.
            self._buffer(text)
            if self.cmd_task is not None:
                # buffer 到 cmd task
                self.cmd_task.tokens = self.buffered()
        if complete:
            # 提交.
            self.commit()

    @abstractmethod
    def _buffer(self, text: str) -> None:
        """
        真实的 buffer 逻辑,
        """
        pass

    def commit(self) -> None:
        if self.committed:
            return
        self.committed = True
        self._commit()

    @abstractmethod
    def _commit(self) -> None:
        """真实的结束 stream 讯号. 如果 stream 通过 tts 实现, 这个讯号会通知 tts 完成输出. """
        pass

    def as_command_task(self, commit: bool = False) -> Optional[CommandTask]:
        """
        将 speech stream 转化为一个 command task, 使之可以发送到 Shell 中阻塞.
        """
        from ghoshell_moss.concepts.command import BaseCommandTask, CommandMeta, CommandWrapper
        if self.cmd_task is not None:
            return self.cmd_task

        if commit:
            # 是否要标记提交. stream 可能在生成 task 的时候, 还没有完成内容的提交.
            self.commit()

        async def _speech_lifecycle() -> None:
            try:
                # 标记开始播放.
                self.start()
                # 等待输入结束, 播放结束.
                await self.wait()
            finally:
                # 关闭播放.
                self.close()

        meta = CommandMeta(
            name="__speech__",
            # 默认主轨运行.
            chan="",
        )

        command = CommandWrapper(meta, _speech_lifecycle)
        task = BaseCommandTask.from_command(
            command,
        )
        task.cid = self.id
        # 添加默认的 tokens.
        task.tokens = self.buffered()
        self.cmd_task = task
        return task

    @abstractmethod
    def buffered(self) -> str:
        """
        返回已经缓冲的文本内容, 可能经过了加工.
        """
        pass

    @abstractmethod
    async def wait(self) -> None:
        """
        阻塞等待到播放完成. start & commit 是两个必要的开关.
        commit 意味着文本片段生成完毕.
        start 意味着允许开始播放.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """
        start to output
        """
        pass

    @abstractmethod
    def close(self):
        """
        关闭一个 Stream.
        """
        pass


class Speech(ABC):
    """
    文本输出模块. 通常和语音输出模块结合.
    """

    @abstractmethod
    def new_stream(self, *, batch_id: Optional[str] = None) -> SpeechStream:
        """
        创建一个新的输出流, 第一个 stream 应该设置为 play
        """
        pass

    @abstractmethod
    def outputted(self) -> List[str]:
        """
        清空之前生成的文本片段, speech 必须能感知到所有输出.
        """
        pass

    @abstractmethod
    def clear(self) -> List[str]:
        """
        清空所有输出中的 output
        """
        pass
