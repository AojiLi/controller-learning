"""Process-level allocator settings required before GPU test modules import JAX.

M7 runs JAX/MJX-Warp and PyTorch in one process. JAX's default large preallocation can otherwise
leave no usable allocator headroom for PyTorch even when the actual arrays are small. Formal GPU
scripts set the same variable before importing either framework.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
