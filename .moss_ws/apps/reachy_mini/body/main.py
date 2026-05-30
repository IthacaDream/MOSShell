from dotenv import load_dotenv
load_dotenv()

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss_contrib.moss_in_reachy_mini.main import provide_channel

if __name__ == "__main__":
    Matrix.discover().run(provide_channel)
