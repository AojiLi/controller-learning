"""GPU dependency and MJX-Warp integration tests."""

import pytest

from controller_learning.diagnostics import inspect_gpu_environment


@pytest.mark.gpu
def test_gpu_environment_executes_mjx_warp_step() -> None:
    report = inspect_gpu_environment()

    assert report["smoke_qpos_finite"] is True
    assert report["smoke_qpos_shape"] == [7]
    assert report["jax_devices"]
    assert report["warp_devices"]
