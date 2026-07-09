"""Build and probe the distributable package instead of the editable source tree."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).parents[2]
REQUIRED_WHEEL_PATHS = (
    "controller_learning/py.typed",
    "controller_learning/assets/vehicle/car.xml",
    "controller_learning/physics/model.py",
    "controller_learning/physics/actuation.py",
    "controller_learning/physics/cpu_reference.py",
)


def test_built_wheel_contains_and_loads_the_m1_vehicle(tmp_path: Path) -> None:
    """Require the sdist-to-wheel path to preserve runtime code and MJCF assets."""

    dist_dir = tmp_path / "dist"
    subprocess.run(
        (
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(dist_dir),
        ),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheels = list(dist_dir.glob("*.whl"))
    source_distributions = list(dist_dir.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(source_distributions) == 1

    site_dir = tmp_path / "site"
    with ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        assert wheel.testzip() is None
        assert len(names) == len(set(names))
        for required_path in REQUIRED_WHEEL_PATHS:
            assert names.count(required_path) == 1
        for name in names:
            lowered = f"/{name.lower()}"
            assert "/reference/" not in lowered
            assert not lowered.endswith((".env", ".key", ".pem"))
        wheel.extractall(site_dir)

    probe = """
from importlib.resources import files
from pathlib import Path
import sys

import controller_learning
from controller_learning.config import load_vehicle_config
from controller_learning.physics import load_vehicle_model

site_dir = Path(sys.argv[1]).resolve()
package_file = Path(controller_learning.__file__).resolve()
assert package_file.is_relative_to(site_dir), (package_file, site_dir)
package = files("controller_learning")
assert package.joinpath("py.typed").is_file()
assert package.joinpath("assets", "vehicle", "car.xml").is_file()
config = load_vehicle_config(Path(sys.argv[2]))
model, _ = load_vehicle_model(config)
assert (model.nq, model.nv, model.nu) == (13, 12, 6)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(site_dir)
    environment["PYTHONNOUSERSITE"] = "1"
    subprocess.run(
        (
            sys.executable,
            "-c",
            probe,
            str(site_dir),
            str(PROJECT_ROOT / "configs" / "vehicle.toml"),
        ),
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
