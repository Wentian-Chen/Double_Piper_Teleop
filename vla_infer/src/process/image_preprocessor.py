"""图像预处理器实现。"""

from __future__ import annotations

import typing as t

import numpy as np

from .base_process import BasePreprocessor


class ImagePreprocessor(BasePreprocessor):
    """图像预处理流水线（可组合 image ops）。"""

    def __init__(
        self,
        image_keys: t.Optional[t.Sequence[str]] = None,
        image_ops: t.Optional[t.Sequence[t.Callable[[np.ndarray], t.Dict[str, t.Any]]]] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self.image_keys = list(image_keys or ["cam_head", "cam_wrist"])
        self.image_ops = list(image_ops or [])

    def _process_single_image(self, image: np.ndarray) -> np.ndarray:
        """按顺序执行 image ops，仅返回处理后的图像。"""
        processed = image.copy()

        for op in self.image_ops:
            op_name = getattr(op, "__name__", op.__class__.__name__)
            op_result = op(processed)

            if not isinstance(op_result, dict):
                raise TypeError(f"image op {op_name} must return dict, got {type(op_result)}")

            if "image" in op_result:
                processed = op_result["image"]

        return processed

    def should_process_key(self, key: str, value: t.Any) -> bool:
        return key in self.image_keys and isinstance(value, np.ndarray)

    def process_value(self, key: str, value: t.Any) -> t.Any:
        return self._process_single_image(t.cast(np.ndarray, value))


__all__ = [
    "ImagePreprocessor",
]
