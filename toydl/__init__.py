import importlib.metadata

from . import core

__version__ = importlib.metadata.version("toydl")

__all__ = ["core"]
