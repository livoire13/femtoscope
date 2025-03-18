"""
.. include:: ../README.md
"""

from pathlib import Path
import os
import sys


def get_main_script_dir():
    """ Get the directory of the main script being run using sys.argv[0] """
    return Path(sys.argv[0]).resolve().parent


def get_installation_dir():
    """ Get the installation directory of the femtoscope package """
    return Path(__file__).resolve().parent


# Check if we're running in test mode by looking for an environment variable
if os.getenv("FEMTOSCOPE_TEST_MODE") == "1":
    # Use the installation directory for the base dir during tests
    FEMTOSCOPE_BASE_DIR = get_installation_dir()
else:
    # Default to the directory of the main script
    FEMTOSCOPE_BASE_DIR = Path(
        os.getenv('FEMTOSCOPE_BASE_DIR', str(get_main_script_dir())))

# Derive other directories from the base directory
DATA_DIR = FEMTOSCOPE_BASE_DIR / 'data'
RESULT_DIR = DATA_DIR / 'result'
TMP_DIR = DATA_DIR / 'tmp'
MESH_DIR = DATA_DIR / 'mesh'
GEO_DIR = MESH_DIR / 'geo'
INSTALL_DIR = get_installation_dir()
IMAGES_DIR = INSTALL_DIR / 'images'
TEST_DIR = INSTALL_DIR / 'tests'
