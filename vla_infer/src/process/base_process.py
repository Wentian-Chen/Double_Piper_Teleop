"""vla_infer 预处理模块的抽象基类定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
import typing as t


PayloadTransform = t.Callable[[t.Dict[str, t.Any]], t.Dict[str, t.Any]]


class BaseProcess(ABC):
    """统一处理器抽象类。"""

    @abstractmethod
    def preprocess(self, payload: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """执行预处理，输入输出保持字典同构（只处理 value，不改 key 结构）。"""
        raise NotImplementedError


class BasePreprocessor(BaseProcess):
    """预处理器基类，提供按 key/value 逐项处理的默认实现。"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def preprocess(self, payload: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        if not self.enabled:
            return dict(payload)

        return {
            key: self.process_value(key, value)
            if self.should_process_key(key, value)
            else value
            for key, value in payload.items()
        }

    def should_process_key(self, key: str, value: t.Any) -> bool:
        """Return True if the current key/value pair should be transformed."""
        return True

    @abstractmethod
    def process_value(self, key: str, value: t.Any) -> t.Any:
        """Transform one value while preserving the original key."""
        raise NotImplementedError

    @staticmethod
    def validate_required_keys(
        payload: t.Dict[str, t.Any],
        required_keys: t.Sequence[str],
    ) -> t.Dict[str, t.Any]:
        """校验输入字典是否包含必需键。

        返回字典结构：
        - ok: bool，是否通过校验
        - missing_keys: list[str]，缺失键列表
        """
        missing_keys = [key for key in required_keys if key not in payload]
        return {
            "ok": len(missing_keys) == 0,
            "missing_keys": missing_keys,
        }


class ComposeProcess(BasePreprocessor):
    """通用可组合处理器，行为类似 torchvision.transforms.Compose。"""

    def __init__(
        self,
        transforms: t.Optional[t.Sequence[PayloadTransform]] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self.transforms: t.List[PayloadTransform] = list(transforms or [])

    def preprocess(self, payload: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        if not self.enabled:
            return dict(payload)

        current_payload = dict(payload)
        for transform in self.transforms:
            result = transform(current_payload)
            if not isinstance(result, dict):
                transform_name = getattr(transform, "__name__", transform.__class__.__name__)
                raise TypeError(
                    f"transform {transform_name} must return dict, got {type(result)}"
                )
            current_payload = dict(result)

        return current_payload

    def process_value(self, key: str, value: t.Any) -> t.Any:
        return value
