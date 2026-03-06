"""动作预处理器实现。"""

from __future__ import annotations

import typing as t

import numpy as np

from .base_process import BasePreprocessor


class ActionPreprocessor(BasePreprocessor):
    """动作预处理流水线（可组合 action ops）。"""

    def __init__(
        self,
        action_key: str = "action",
        action_ops: t.Optional[t.Sequence[t.Callable[[np.ndarray], t.Dict[str, t.Any]]]] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self.action_key = action_key
        self.action_ops = list(action_ops or [])

    def should_process_key(self, key: str, value: t.Any) -> bool:
        return key == self.action_key

    def process_value(self, key: str, value: t.Any) -> t.Any:
        action_array = np.asarray(value)

        for op in self.action_ops:
            op_name = getattr(op, "__name__", op.__class__.__name__)
            op_result = op(action_array)

            if not isinstance(op_result, dict):
                raise TypeError(f"action op {op_name} must return dict, got {type(op_result)}")

            if "action" in op_result:
                action_array = np.asarray(op_result["action"])

        return np.asarray(action_array, dtype=np.float32)


__all__ = [
    "ActionPreprocessor",
]
