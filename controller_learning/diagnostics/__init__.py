"""Runtime diagnostics that do not affect package import."""

from controller_learning.diagnostics.gpu import inspect_gpu_environment

__all__ = ["inspect_gpu_environment"]
