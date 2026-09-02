"""
Weight replacement module for VLA models.

This module provides functionality to replace specific layers/weights in trained models
with weights from source checkpoints. Supports flexible configuration through JSON specs.
"""

import json
import glob
import logging
import typing as t
from pathlib import Path
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)


@dataclass
class WeightReplacementConfig:
    """Configuration for weight replacement functionality.
    
    Attributes:
        enabled: Whether to enable weight replacement. Default: False
        model_path: Path to source model checkpoint containing weights to replace from.
        json_path: Path to JSON configuration file specifying which weights to replace.
            Supports two formats:
            1) Simple: Maps source file names (with optional wildcards) to layer lists.
               Module name is inferred from filename (e.g., "proprio_projector--20000.pt").
            2) Detailed: Maps module names to dicts with "file" and "layers" keys.
        verbose: Enable detailed logging during weight replacement. Default: True
    
    Example JSON (detailed format):
    {
        "proprio_projector": {
            "file": "proprio_projector--20000_checkpoint.pt",
            "layers": ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
        },
        "action_head": {
            "file": "action_head--20000_checkpoint.pt",
            "layers": ["module.model.layer_norm1.weight", "module.model.layer_norm1.bias"]
        }
    }
    
    Example JSON (simple format):
    {
        "proprio_projector--20000.pt": ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"],
        "action_head--20000.pt": ["module.model.layer_norm1.weight"]
    }
    """
    enabled: bool = False
    model_path: t.Optional[t.Union[str, Path]] = None
    json_path: t.Optional[t.Union[str, Path]] = None
    verbose: bool = True

    def validate(self) -> None:
        """Validate configuration. Raises ValueError if invalid."""
        if not self.enabled:
            return
        
        if not self.model_path:
            raise ValueError("model_path is required when weight replacement is enabled")
        if not self.json_path:
            raise ValueError("json_path is required when weight replacement is enabled")
        
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise ValueError(f"model_path does not exist: {model_path}")
        
        json_path = Path(self.json_path)
        if not json_path.exists():
            raise ValueError(f"json_path does not exist: {json_path}")


class WeightReplacer:
    """Handles weight replacement for VLA models.
    
    Supports replacing specific layers in modules with weights from source checkpoints.
    Automatically handles common naming variants like "module." prefix differences.
    
    Usage:
        replacer = WeightReplacer(cfg)
        replacer.register_module("proprio_projector", proprio_projector_module)
        replacer.register_module("action_head", action_head_module)
        replacer.apply_replacements()
    """

    def __init__(self, cfg: WeightReplacementConfig) -> None:
        """Initialize WeightReplacer.
        
        Args:
            cfg: WeightReplacementConfig instance.
        """
        self.cfg = cfg
        self._module_map: t.Dict[str, t.Any] = {}
        self._logger = logger
        # 配置 logger：设置级别和 handler
        self._logger.setLevel(logging.DEBUG if cfg.verbose else logging.INFO)
        # 确保有 handler - 如果没有则添加一个
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        # 禁止向上传播以避免重复日志
        self._logger.propagate = True

    def register_module(self, module_name: str, module: t.Any) -> None:
        """Register a module for weight replacement.
        
        Args:
            module_name: Unique name for the module (e.g., "proprio_projector").
            module: The PyTorch module instance.
        """
        self._module_map[module_name] = module
        self._logger.info(f"Registered module: {module_name}")

    def apply_replacements(self) -> None:
        """Apply weight replacements according to configuration.
        
        Reads JSON config and replaces weights in registered modules.
        
        Raises:
            ValueError: If configuration is invalid.
            FileNotFoundError: If required files don't exist.
            RuntimeError: If no modules are registered.
        """
        if not self.cfg.enabled:
            self._logger.info("Weight replacement is disabled")
            return

        self.cfg.validate()

        if not self._module_map:
            raise RuntimeError("No modules registered. Call register_module() first.")

        json_path = Path(self.cfg.json_path)
        spec = self._load_json_spec(json_path)
        base_dir = Path(self.cfg.model_path) if self.cfg.model_path else json_path.parent

        print(f"  📂 Base directory: {base_dir}")
        self._logger.info(f"Applying weight replacements from: {base_dir}")

        replaced_count = 0
        for module_name, entry in spec.items():
            file_pattern = entry["file"]
            layer_list = entry["layers"]

            # Skip empty layer lists
            if not layer_list:
                self._logger.debug(f"Skipping module '{module_name}': empty layer list")
                continue

            # Resolve source file path
            src_path = self._resolve_source_path(base_dir, file_pattern, module_name)
            if not src_path:
                continue

            # Execute replacement
            print(f"  📦 Replacing weights for module '{module_name}' from '{src_path.name}'...")
            self._logger.info(
                f"Replacing {len(layer_list)} layers in '{module_name}' "
                f"from '{src_path.name}'..."
            )
            try:
                replaced = self._replace_module_weights(module_name, src_path, layer_list)
                replaced_count += replaced
            except Exception as e:
                print(f"    ❌ Error: {e}")
                self._logger.error(f"Error replacing weights for module '{module_name}': {e}")
                if self.cfg.verbose:
                    raise

        print(f"  📊 Total layers replaced: {replaced_count}")
        self._logger.info(f"Weight replacement completed. Total layers replaced: {replaced_count}")

    def _resolve_source_path(
        self, base_dir: Path, file_pattern: str, module_name: str
    ) -> t.Optional[Path]:
        """Resolve source file path from pattern.
        
        Supports:
        - Absolute paths
        - Relative paths
        - Glob patterns with wildcards
        - Common filename variants (e.g., with "_checkpoint" suffix)
        
        Args:
            base_dir: Base directory for relative paths.
            file_pattern: File pattern (may include wildcards or variants).
            module_name: Module name (for logging).
        
        Returns:
            Resolved Path if found, None otherwise.
        """
        if Path(file_pattern).is_absolute():
            src_candidates = [Path(file_pattern)]
        else:
            full_pattern = base_dir / file_pattern

            # Try direct filename first
            if full_pattern.exists():
                src_candidates = [full_pattern]
            elif "*" in file_pattern or "?" in file_pattern:
                # Contains wildcards, use glob
                src_candidates = [Path(p) for p in glob.glob(str(full_pattern))]
            else:
                # Try with "_checkpoint" suffix variant
                if full_pattern.suffix:
                    base_name = full_pattern.with_suffix("").name
                    checkpoint_variant = base_dir / f"{base_name}_checkpoint{full_pattern.suffix}"
                    src_candidates = [checkpoint_variant] if checkpoint_variant.exists() else []
                else:
                    src_candidates = []

        if not src_candidates:
            self._logger.warning(
                f"No file found matching pattern '{file_pattern}' for module '{module_name}'. "
                f"Searched in: {base_dir}"
            )
            return None

        if len(src_candidates) > 1:
            self._logger.warning(
                f"Multiple files found for pattern '{file_pattern}': {src_candidates}. "
                f"Using first: {src_candidates[0]}"
            )

        src_path = src_candidates[0]
        if not src_path.exists():
            self._logger.warning(f"Source file not found: {src_path}")
            return None

        return src_path

    def _replace_module_weights(
        self, module_name: str, src_path: Path, layer_list: t.List[str]
    ) -> int:
        """Replace weights in a specific module.
        
        Automatically handles "module." prefix differences between source and target.
        
        Args:
            module_name: Name of registered module.
            src_path: Path to source checkpoint file.
            layer_list: List of layer names to replace.
        
        Returns:
            Number of successfully replaced layers.
        
        Raises:
            ValueError: If module not registered or loading fails.
        """
        if module_name not in self._module_map:
            raise ValueError(
                f"Module '{module_name}' not found. "
                f"Available: {list(self._module_map.keys())}"
            )

        target_module = self._module_map[module_name]

        # Load source weights
        src_state = self._load_checkpoint(src_path)
        target_state = target_module.state_dict()
        target_keys = set(target_state.keys())

        # Debug output
        if self.cfg.verbose:
            self._logger.debug(f"Module '{module_name}':")
            self._logger.debug(f"  Source keys (first 10): {list(src_state.keys())[:10]}")
            self._logger.debug(f"  Target keys (first 10): {list(target_keys)[:10]}")

        replaced = 0
        missing = 0

        for layer in layer_list:
            target_layer = layer
            src_layer = layer

            # Try to find layer in source (with prefix variants)
            if layer not in src_state:
                alt_layer = (
                    layer.replace("module.", "", 1)
                    if layer.startswith("module.")
                    else f"module.{layer}"
                )
                if alt_layer in src_state:
                    src_layer = alt_layer
                else:
                    self._logger.warning(
                        f"Layer '{layer}' not found in source (tried both variants)"
                    )
                    missing += 1
                    continue

            # Try to find layer in target (with prefix variants)
            if target_layer not in target_keys:
                alt_layer = (
                    target_layer.replace("module.", "", 1)
                    if target_layer.startswith("module.")
                    else f"module.{target_layer}"
                )
                if alt_layer in target_keys:
                    target_layer = alt_layer
                else:
                    self._logger.warning(
                        f"Layer '{layer}' not found in target '{module_name}' "
                        f"(tried both variants)"
                    )
                    missing += 1
                    continue

            # Copy weights
            try:
                target_param = target_module.get_parameter(target_layer)
                src_param = src_state[src_layer]

                if target_param.shape != src_param.shape:
                    self._logger.warning(
                        f"Shape mismatch for '{layer}': "
                        f"target {target_param.shape} vs source {src_param.shape}. "
                        f"Skipping."
                    )
                    missing += 1
                    continue

                target_param.data.copy_(src_param.to(target_param.device))
                replaced += 1
                self._logger.debug(f"  ✓ Replaced '{layer}'")

            except Exception as e:
                self._logger.error(f"Error replacing layer '{layer}': {e}")
                missing += 1
                continue

        msg = f"Module '{module_name}': Replaced {replaced} layers, {missing} skipped"
        print(f"    {msg}")
        self._logger.info(msg)
        return replaced

    @staticmethod
    def _load_checkpoint(path: Path) -> t.Dict[str, t.Any]:
        """Load checkpoint from .pt or .safetensors file.
        
        Args:
            path: Path to checkpoint file.
        
        Returns:
            State dict dictionary.
        
        Raises:
            ValueError: If file format is unsupported or loading fails.
        """
        try:
            if path.suffix.lower() == ".safetensors":
                from safetensors.torch import load_file

                return load_file(str(path), device="cpu")
            else:
                state = torch.load(path, map_location="cpu")

                # Handle wrapped states
                if hasattr(state, "state_dict"):
                    state = state.state_dict()
                elif isinstance(state, dict) and "model" in state:
                    state = state["model"]

                return state
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint {path}: {e}") from e

    @staticmethod
    def _load_json_spec(json_path: Path) -> t.Dict[str, t.Any]:
        """Load and normalize JSON configuration.
        
        Supports both simple and detailed formats.
        
        Args:
            json_path: Path to JSON file.
        
        Returns:
            Normalized spec dict with structure:
            {
                "module_name": {
                    "file": "filename.pt",
                    "layers": ["layer1", "layer2", ...]
                },
                ...
            }
        
        Raises:
            ValueError: If JSON format is invalid.
        """
        try:
            with open(json_path, "r") as f:
                spec = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load JSON from {json_path}: {e}") from e

        if not isinstance(spec, dict):
            raise ValueError("JSON file must contain a dictionary")

        normalized = {}
        for key, value in spec.items():
            if isinstance(value, list):
                # Simple format: filename -> layer list
                # Infer module name from filename
                module_name = key.split("--")[0] if "--" in key else key
                module_name = Path(module_name).stem  # Remove extension if present
                normalized[module_name] = {"file": key, "layers": value}

            elif isinstance(value, dict) and "file" in value and "layers" in value:
                # Detailed format: module_name -> {file, layers}
                if not isinstance(value["file"], str):
                    raise ValueError(f"Invalid 'file' value for key '{key}': {value['file']}")
                if not isinstance(value["layers"], list):
                    raise ValueError(
                        f"Invalid 'layers' value for key '{key}': {value['layers']}"
                    )
                normalized[key] = value

            else:
                raise ValueError(f"Invalid entry for key '{key}': {value}")

        return normalized
