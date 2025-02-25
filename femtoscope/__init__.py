"""
.. include:: ../README.md
"""

from pathlib import Path
from importlib import resources
import importlib.util


spec = importlib.util.find_spec("femtoscope")
if spec and spec.origin:
    femtoscope_path = Path(spec.origin).parent

# Default to the user's current working directory
_BASE_DIR = Path.cwd()

def set_working_directory(path: str):
    """Allows the user to set a custom working directory."""
    global _BASE_DIR
    _BASE_DIR = Path(path).absolute()

# Define paths relative to the working directory
def get_data_dir():
    return _BASE_DIR / "data"

def get_result_dir():
    return get_data_dir() / "result"

def get_mesh_dir():
    return get_data_dir() / "mesh"

def get_images_dir():
    return femtoscope_path / "images"

def get_tmp_dir():
    return get_data_dir() / "tmp"

# Default paths (if the user does not call set_working_directory)
DATA_DIR = get_data_dir()
RESULT_DIR = get_result_dir()
MESH_DIR = get_mesh_dir()
IMAGES_DIR = get_images_dir()
TMP_DIR = get_tmp_dir()
