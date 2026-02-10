import logging
import os
from collections import defaultdict
from typing import List, Optional

from PIL import Image
from ghoshell_common.contracts import YamlConfig, WorkspaceConfigs, Workspace, LoggerItf, Storage, FileStorage
from ghoshell_common.helpers import timestamp_ms
from ghoshell_container import IoCContainer, Container
from pydantic import Field, PrivateAttr

from ghoshell_moss_contrib.gui.image_viewer import SimpleImageViewer
from ghoshell_moss.message import Message, Base64Image, Text
from ghoshell_moss import PyChannel, CommandError, CommandErrorCode
import frontmatter


class SlideConfig(YamlConfig):
    relative_path = ".meta.yaml"

    name: str = Field(default="", description="Slide的标识名")  # 面向人和大模型的文本描述
    description: str = Field(default="", description="Slide的描述")  # 面向人和大模型的文本描述

    # system unmodified fields
    created_at: Optional[float] = Field(default_factory=timestamp_ms, description="创建时间")
    updated_at: Optional[float] = Field(default_factory=timestamp_ms, description="更新时间")
    origin_filetype: str = Field(default="", description="Slide的原始文件类型")  # ppt/pptx/pdf/...

    # extension unmarshalled fields
    _dirname: str = PrivateAttr()

    @property
    def origin_filename(self):
        return f".origin.{self.origin_filetype}"

    @property
    def dirname(self):
        return self._dirname


class SlideFrame:
    def __init__(self):
        self.filename: Optional[str] = None
        self.image: Optional[Image.Image] = None  # 页图片展示
        self.title: Optional[str] = None  # 页标题
        self.outline: Optional[str] = None  # 页提纲
        self.content: Optional[str] = None  # 页文本内容


class SlideStudio:
    def __init__(self, viewer: SimpleImageViewer, container: IoCContainer = None):
        # gui
        self.viewer = viewer

        # context
        self._container = Container(parent=container, name="slide_studio")
        self._logger = self._container.get(LoggerItf) or logging.getLogger(__name__)
        self._ws = container.force_fetch(Workspace)

        # studio all slides
        self._slides: List[SlideConfig] = self._load_studio()

        # current playing slide information
        self.is_playing = False
        self.frames: List[SlideFrame] = []
        self.dirname = ""
        self.current_frame_index = -1

    @property
    def logger(self):
        return self._logger

    def _load_studio(self) -> List[SlideConfig]:
        slide_studio_storage: FileStorage | Storage = self._ws.assets().sub_storage("slide_studio")
        slides = []

        with os.scandir(slide_studio_storage.abspath()) as entries:
            for entry in entries:
                if entry.is_dir() and entry.name not in [".", ".."]:
                    slides.append(entry.name)

        res = []
        for slide_dirname in slides:
            slide_storage: FileStorage | Storage = slide_studio_storage.sub_storage(slide_dirname)
            configs = WorkspaceConfigs(slide_storage)
            slide_config = configs.get_or_create(SlideConfig())
            slide_config._dirname = slide_dirname
            if not slide_config.name:
                slide_config.name = slide_dirname
            res.append(slide_config)
        return res

    def _load_frames(self, slide_name: str):
        self.frames.clear()

        dirname = ""
        for s in self._slides:
            if s.name != slide_name:
                continue
            dirname = s.dirname

        if not dirname:
            raise CommandError(CommandErrorCode.NOT_AVAILABLE, message=f"slide {slide_name} not found")

        slide_storage: FileStorage | Storage = self._ws.assets().sub_storage("slide_studio").sub_storage(dirname)
        filenames = slide_storage.dir("", False)

        frame_map: defaultdict[str, SlideFrame] = defaultdict(SlideFrame)
        for filename in filenames:
            if filename.startswith("."):
                continue

            file_abs_path = slide_storage.abspath() + "/" + filename
            image_filename = filename.rstrip(".md")
            frame = frame_map[image_filename]
            frame.filename = image_filename

            if filename.endswith(".md"):
                post = frontmatter.load(file_abs_path)
                frame.title = post.metadata.get("title") or ""
                frame.outline = post.metadata.get("outline") or ""
                frame.content = post.content.strip()
            else:
                frame.image = Image.open(file_abs_path)

        frames = frame_map.values()
        self.frames = sorted(frames, key=lambda f: f.filename)
        self.dirname = dirname

    def _clear_frames(self):
        self.is_playing = False
        self.frames.clear()
        self.dirname = ""
        self.current_frame_index = -1

    def _check_playing(self):
        if not self.is_playing:
            raise CommandError(CommandErrorCode.FAILED, message="slide studio is not playing")

    async def play(self, name: str):
        """
        获取指定ppt的详细信息，只有拿到ppt的详细信息后才能调用其他的command
        """
        try:
            self._load_frames(name)
        except Exception as ex:
            self.logger.warning(f"Player _load {name} failed with: {ex}")
            await self.stop()
            raise
        self.is_playing = True
        return await self.to_page(1)

    async def to_page(self, index: int = 1):
        """
        跳转至指定页数，从1开始，返回True表示跳转成功，返回False表示当前已是最后一页。
        """
        self._check_playing()
        if index >= len(self.frames):
            return False
        self.current_frame_index = index - 1
        self.viewer.set_pil_image(self.current_frame.image)
        return True

    async def to_next_page(self):
        """
        跳转至下一页，返回True表示跳转成功，返回False表示当前已是最后一页。
        """
        self._check_playing()
        if self.is_last():
            return False
        self.current_frame_index += 1
        self.viewer.set_pil_image(self.current_frame.image)
        return True

    async def stop(self):
        """
        退出ppt演示
        """
        self._clear_frames()

    def desc(self):
        self._slides = self._load_studio()
        slides = [f"name:{s.name} description:{s.description}" for s in self._slides]
        if not slides:
            return "There has no slides in Slide Studio"

        slide_desc = f"All assets in Slide Studio:\n{'\n'.join(slides)}"

        if not self.is_playing:
            return f"{slide_desc}\nNot play any slides yet"

        outlines = [f"Page number:{i + 1} Title={f.title} Outline={f.outline}" for i, f in enumerate(self.frames)]
        return f"""{slide_desc}\nCurrent playing slide: {self.dirname}，Progress: {self.current_frame_index + 1}/{len(self.frames)}，Outlines:\n{"\n".join(outlines)}"""

    async def context_messages(self):
        if not self.is_playing:
            return []

        # 创建视觉消息
        message = Message.new(role="user", name="no_ppt_vision").with_content(
            Text(text=f"当前页的概述和图片：{self.current_frame.content}"),
            Base64Image.from_pil_image(self.current_frame.image),
        )

        return [message]

    @property
    def current_frame(self):
        return self.frames[self.current_frame_index]

    @property
    def next_frame(self):
        if not self.is_last():
            return self.frames[self.current_frame_index + 1]
        return None

    def is_last(self):
        return self.current_frame_index == len(self.frames) - 1

    def as_channel(self):
        py_chan = PyChannel(name="slide_studio", block=True)

        py_chan.build.with_description()(self.desc)
        py_chan.build.with_context_messages(self.context_messages)

        py_chan.build.command()(self.play)
        py_chan.build.command()(self.to_page)
        py_chan.build.command()(self.to_next_page)
        py_chan.build.command()(self.stop)

        return py_chan
