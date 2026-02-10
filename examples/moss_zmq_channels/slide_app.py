import asyncio
import sys
from os.path import join, dirname

from PyQt6.QtWidgets import QApplication

from ghoshell_moss_contrib.channels.slide_studio import SlideStudio
from ghoshell_moss.transports.zmq_channel import ZMQChannelProvider
from ghoshell_moss_contrib.example_ws import workspace_container
from ghoshell_moss_contrib.gui.image_viewer import SimpleImageViewer

current_dir = dirname(__file__)
workspace_dir = join(dirname(current_dir), ".workspace")


if __name__ == "__main__":
    with workspace_container(workspace_dir) as _container:
        _app = QApplication(sys.argv)
        _viewer = SimpleImageViewer(window_title="Slide Studio")
        studio = SlideStudio(_viewer, _container)
        provider = ZMQChannelProvider(
            address=f"ipc://{__file__}.sock",
            container=_container,
        )
        provider.run_in_thread(studio.as_channel())
        _viewer.show()
        sys.exit(_app.exec())
