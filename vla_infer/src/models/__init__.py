"""vla_infer model exports with lazy import.

This prevents optional backends from being imported unless explicitly used.
"""

from importlib import import_module
import typing as t

if t.TYPE_CHECKING:
    from .base import BaseVLAModel
    from .dream_adapter_model import DreamAdapterModel
    from .smolvla_model import SmolVLAModel
    from .vla_adapter_model import VlaAdapterModel

    VLAAdapterModel = VlaAdapterModel

__all__ = [
    "BaseVLAModel",
    "SmolVLAModel",
    "VlaAdapterModel",
    "VLAAdapterModel",
    "DreamAdapterModel",
]

_SYMBOL_TO_MODULE = {
    "BaseVLAModel": ".base",
    "SmolVLAModel": ".smolvla_model",
    "VlaAdapterModel": ".vla_adapter_model",
    "VLAAdapterModel": ".vla_adapter_model",
    "DreamAdapterModel": ".dream_adapter_model",
}


def __getattr__(name: str) -> t.Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module 'vla_infer.src.models' has no attribute {name!r}")

    module = import_module(module_name, __name__)
    if name == "VLAAdapterModel":
        value = getattr(module, "VlaAdapterModel")
    else:
        value = getattr(module, name)
    globals()[name] = value
    return value
