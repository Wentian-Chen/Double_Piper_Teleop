"""Top-level package for vla_infer.

Avoid importing heavy subpackages at module import time so scripts can import
only what they need without triggering optional dependencies.
"""

from importlib import import_module
import typing as t

__all__ = ["models", "process", "robots", "zmq", "inference"]

if t.TYPE_CHECKING:
    from vla_infer.src import inference, models, process, robots, zmq


def __getattr__(name: str) -> t.Any:
    if name in __all__:
        module = import_module(f"vla_infer.src.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'vla_infer' has no attribute {name!r}")

