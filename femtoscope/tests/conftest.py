import importlib
import os

import femtoscope  # Import here to reload later


def pytest_configure(config):
    """Set FEMTOSCOPE_TEST_MODE before any test modules are imported."""
    os.environ["FEMTOSCOPE_TEST_MODE"] = "1"
    importlib.reload(femtoscope)
