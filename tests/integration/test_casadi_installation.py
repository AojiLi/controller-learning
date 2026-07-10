"""Integration smoke test for the M6 CasADi/IPOPT Controller dependency."""

from __future__ import annotations

import casadi as ca
import pytest


def test_casadi_ipopt_solves_a_bounded_nonlinear_program() -> None:
    assert ca.__version__ == "3.7.2"
    assert ca.has_nlpsol("ipopt")

    value = ca.MX.sym("value")
    solver = ca.nlpsol(
        "m6_ipopt_smoke",
        "ipopt",
        {"x": value, "f": (value - 1.5) ** 2},
        {"print_time": False, "ipopt.print_level": 0},
    )
    solution = solver(x0=0.0, lbx=-2.0, ubx=2.0)

    assert solver.stats()["success"] is True
    assert float(solution["x"]) == pytest.approx(1.5, abs=1.0e-7)
