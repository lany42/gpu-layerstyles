"""ComfyUI entry point for clone and ZIP installations."""

from .src.gpu_layerstyles import GPULayerStylesExtension, comfy_entrypoint

__all__ = ["GPULayerStylesExtension", "comfy_entrypoint"]
