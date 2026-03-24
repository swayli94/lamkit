"""Lekhnitskii unloaded hole: API, far-field recovery, isotropic Kt."""

import numpy as np
import pytest

from lamkit.lekhnitskii.unloaded_hole import UnloadedHole
from lamkit.utils import evaluate_unloaded_hole_stress_field
from lamkit.lekhnitskii.utils import generate_meshgrid


def _isotropic_compliance(E: float, nu: float) -> np.ndarray:
    G = E / (2.0 * (1.0 + nu))
    S = np.zeros((3, 3))
    S[0, 0] = S[1, 1] = 1.0 / E
    S[0, 1] = S[1, 0] = -nu / E
    S[2, 2] = 1.0 / G
    return S


def test_hole_rejects_non_3x3_compliance():
    with pytest.raises(ValueError, match="3x3"):
        UnloadedHole(0.0, 0.0, 0.0, radius=1.0, compliance_matrix=np.eye(2))


def test_theoretical_far_field_uniaxial_x():
    E, nu = 200e3, 0.3
    S = _isotropic_compliance(E, nu)
    sig_inf = 100.0
    R = 50.0
    x = np.array([R])
    y = np.array([0.0])
    sxx, syy, txy = evaluate_unloaded_hole_stress_field(
        sig_inf, 0.0, 0.0, hole_radius=1.0, compliance_matrix=S, x=x, y=y
    )
    assert sxx.item() == pytest.approx(sig_inf, rel=0.02)
    assert syy.item() == pytest.approx(0.0, abs=2.0)
    assert txy.item() == pytest.approx(0.0, abs=1.0)


def test_isotropic_stress_concentration_uniaxial_x():
    """Remote sigma_xx^inf: hoop (tangential ~ x) stress at hole crown ~ 3*sigma."""
    E, nu = 200e3, 0.3
    S = _isotropic_compliance(E, nu)
    a = 1.0
    sig_inf = 100.0
    eps = 1e-5
    # Just outside hole on +y axis; hoop direction aligns with x.
    y = a * (1.0 + eps)
    x = 0.0
    sxx, syy, txy = evaluate_unloaded_hole_stress_field(
        sig_inf, 0.0, 0.0, hole_radius=a, compliance_matrix=S, x=x, y=y
    )
    Kt = np.asarray(sxx).item() / sig_inf
    assert Kt == pytest.approx(3.0, rel=0.05)


def test_generate_meshgrid_shapes():
    m = generate_meshgrid(
        hole_radius=1.0,
        plate_radius=5.0,
        n_points_radial=11,
        n_points_angular=21,
        radial_cluster_power=2.0,
    )
    assert m["X"].shape == (11, 21)
    assert m["meshgrid_shape"] == m["X"].shape


def test_stress_method_requires_1d_arrays():
    S = _isotropic_compliance(1e6, 0.3)
    sol = UnloadedHole(10.0, 0.0, 0.0, radius=1.0, compliance_matrix=S)
    with pytest.raises(ValueError, match="1D"):
        sol.stress(np.array([[1.0]]), np.array([[2.0]]))
