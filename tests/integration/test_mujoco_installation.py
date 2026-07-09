"""CPU MuJoCo dependency smoke tests."""

import mujoco
import numpy as np


def test_cpu_mujoco_loads_and_steps_finite_state() -> None:
    xml = """
    <mujoco model="cpu_smoke">
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
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    for _ in range(20):
        mujoco.mj_step(model, data)

    assert model.opt.timestep == 0.01
    assert np.isclose(data.time, 0.2)
    assert np.isfinite(data.qpos).all()
    assert data.qpos.shape == (7,)
