"""NVIDIA dependency and MJX-Warp integration diagnostics."""

from __future__ import annotations

import os
from importlib.metadata import version
from typing import Any


def inspect_gpu_environment() -> dict[str, Any]:
    """Verify CUDA libraries and execute one minimal MJX-Warp physics step.

    Returns:
        JSON-serializable dependency, device, and smoke-step details.

    Raises:
        RuntimeError: If JAX, PyTorch, or Warp cannot access an NVIDIA GPU.
    """

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import mujoco
    import mujoco.mjx as mjx
    import mujoco.mjx.warp as mjx_warp
    import numpy as np
    import torch
    import warp as wp

    jax_devices = jax.devices()
    if not jax_devices or not any(device.platform == "gpu" for device in jax_devices):
        raise RuntimeError(f"JAX cannot access a GPU; detected devices: {jax_devices!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")

    wp.init()
    warp_devices = wp.get_cuda_devices()
    if not warp_devices:
        raise RuntimeError("Warp cannot access a CUDA device")

    xml = """
    <mujoco model="gpu_smoke">
      <option timestep="0.01"/>
      <worldbody>
        <geom type="plane" size="2 2 0.1"/>
        <body pos="0 0 0.5">
          <freejoint/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    cpu_model = mujoco.MjModel.from_xml_string(xml)
    model = mjx.put_model(cpu_model, impl="warp")
    data = mjx.make_data(cpu_model, impl="warp", naconmax=16, njmax=16)
    stepped = jax.jit(mjx.step)(model, data)
    qpos = np.asarray(jax.device_get(stepped.qpos))
    if not np.isfinite(qpos).all():
        raise RuntimeError("MJX-Warp smoke step produced non-finite qpos")

    return {
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax_devices],
        "mujoco_version": mujoco.__version__,
        "mjx_warp_module": mjx_warp.__name__,
        "torch_version": torch.__version__,
        "torch_device": torch.cuda.get_device_name(0),
        "warp_version": version("warp-lang"),
        "warp_devices": [str(device) for device in warp_devices],
        "smoke_qpos_finite": True,
        "smoke_qpos_shape": list(qpos.shape),
    }
