import numpy as np
import pytest

from lamkit.analysis.laminate import Laminate
from lamkit.analysis.material import IM7_8551_7, Ply
from lamkit.lekhnitskii.homogenisation import (
    compute_effective_strains,
    compute_homogenised_properties,
    compute_permutation_invariants,
)
from lamkit.lekhnitskii.unloaded_hole import UnloadedHole
from lamkit.lekhnitskii.utils import (
    generate_meshgrid,
    rotate_complex_parameters,
    rotate_material_matrix,
    rotate_strain,
    rotate_stress,
)
from lamkit.utils import evaluate_unloaded_hole_stress_field


def _isotropic_compliance(E: float, nu: float) -> np.ndarray:
    G = E / (2.0 * (1.0 + nu))
    S = np.zeros((3, 3), dtype=float)
    S[0, 0] = S[1, 1] = 1.0 / E
    S[0, 1] = S[1, 0] = -nu / E
    S[2, 2] = 1.0 / G
    return S


def test_rotate_stress_roundtrip() -> None:
    stresses = np.random.RandomState(0).randn(10, 3)
    angle = 0.37
    back = rotate_stress(rotate_stress(stresses, angle=angle), angle=-angle)
    assert np.allclose(back, stresses, rtol=1e-12, atol=1e-12)


def test_rotate_strain_roundtrip() -> None:
    strains = np.random.RandomState(1).randn(10, 3)
    angle = -0.41
    back = rotate_strain(rotate_strain(strains, angle=angle), angle=-angle)
    assert np.allclose(back, strains, rtol=1e-12, atol=1e-12)


def test_rotate_material_matrix_roundtrip() -> None:
    # rotate_material_matrix works on the inverse CLPT A-matrix representation.
    S = _isotropic_compliance(E=200e3, nu=0.3)
    a_inv = np.linalg.inv(S)
    angle = 0.22

    a_inv_back = rotate_material_matrix(
        rotate_material_matrix(a_inv, angle=angle), angle=-angle
    )
    assert np.allclose(a_inv_back, a_inv, rtol=1e-8, atol=1e-8)


def test_rotate_complex_parameters_roundtrip() -> None:
    mu1 = 0.5 + 0.2j
    mu2 = 0.3 - 0.1j
    angle = 0.41

    mu1p, mu2p = rotate_complex_parameters(mu1=mu1, mu2=mu2, angle=angle)
    mu1b, mu2b = rotate_complex_parameters(mu1=mu1p, mu2=mu2p, angle=-angle)

    np.testing.assert_allclose(mu1b, mu1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mu2b, mu2, rtol=1e-12, atol=1e-12)


def test_compute_permutation_invariants_relations() -> None:
    E11 = 150e3
    E22 = 90e3
    G12 = 55e3
    nu12 = 0.25

    inv = compute_permutation_invariants(E11=E11, E22=E22, G12=G12, nu12=nu12)
    assert inv["G12"] == G12
    assert np.isclose(inv["log(1+p3)"], np.log(1.0 + inv["p3"]))
    assert np.isclose(inv["p0"], np.sqrt(E11 * E22))
    assert np.isclose(inv["p1"], inv["p0"] / G12)
    assert np.isclose(inv["p2"], nu12 * np.sqrt(E22 / E11))


def test_compute_effective_strains_requires_at_least_3_points() -> None:
    S = _isotropic_compliance(E=200e3, nu=0.3)
    hole = UnloadedHole(
        sigma_xx_inf=1.0,
        sigma_yy_inf=0.0,
        tau_xy_inf=0.0,
        radius=1.0,
        compliance_matrix=S,
    )
    with pytest.raises(ValueError, match=">= 3"):
        compute_effective_strains(solution=hole, L=4.0, H=4.0, n_points_boundary=2)


def test_compute_homogenised_properties_basic_contract() -> None:
    # Keep resolution low for test speed; we mainly validate shape/consistency.
    S = _isotropic_compliance(E=200e3, nu=0.3)
    out = compute_homogenised_properties(
        HoleType=UnloadedHole,
        L=4.0,
        H=4.0,
        plate_thickness=1.0,
        hole_radius=1.0,
        compliance_matrix=S,
        n_points_boundary=9,
    )
    assert set(out.keys()) >= {
        "A_eff",
        "S_eff",
        "E11_eff",
        "E22_eff",
        "G12_eff",
        "nu12_eff",
        "nu21_eff",
    }
    assert out["S_eff"].shape == (3, 3)
    assert out["A_eff"].shape == (3, 3)

    S_eff = out["S_eff"]
    # Symmetry contract: compliance matrix should be symmetric (numerically approximate).
    assert np.allclose(S_eff, S_eff.T, rtol=1e-6, atol=1e-9)
    assert out["E11_eff"] == pytest.approx(1.0 / S_eff[0, 0], rel=1e-6)
    assert out["E22_eff"] == pytest.approx(1.0 / S_eff[1, 1], rel=1e-6)
    assert out["G12_eff"] == pytest.approx(1.0 / S_eff[2, 2], rel=1e-6)
    assert out["nu12_eff"] == pytest.approx(-S_eff[0, 1] / S_eff[0, 0], rel=1e-6)
    assert out["nu21_eff"] == pytest.approx(-S_eff[1, 0] / S_eff[1, 1], rel=1e-6)


def test_laminate_evaluate_laminate_contract() -> None:
    t = 0.125
    ply = Ply(IM7_8551_7, thickness=t)
    lam = Laminate([0.0, 45.0], [ply, ply])

    N = np.array([1.0, 2.0, 0.5, 0.1, -0.2, 0.05], dtype=float)
    df = lam.evaluate_laminate(N)

    # 2*n_ply rows: bottom/top face per ply.
    assert len(df) == 2 * lam.n_ply
    required_cols = {
        "sigma_x",
        "sigma_y",
        "tau_xy",
        "sigma_1",
        "sigma_2",
        "tau_12",
        "FI_matrix_cracking",
        "FI_max",
        "failure_mode",
    }
    assert required_cols.issubset(set(df.columns))

    assert "epsilon0" in df.attrs
    assert np.asarray(df.attrs["epsilon0"]).shape == (6,)
    assert df.attrs["global_FI_max"] == pytest.approx(df["FI_max"].max())


def test_evaluate_unloaded_hole_stress_field_shapes_and_types() -> None:
    t = 0.125
    ply = Ply(IM7_8551_7, thickness=t)
    lam = Laminate([0.0], [ply])

    x = np.array([2.0], dtype=float)  # keep away from hole boundary (hole_radius=1.0)
    y = np.array([0.0], dtype=float)

    results_by_plies, mid_plane_field = evaluate_unloaded_hole_stress_field(
        laminate=lam,
        hole_radius=1.0,
        sigma_xx_inf=100.0,
        sigma_yy_inf=0.0,
        tau_xy_inf=0.0,
        x=x,
        y=y,
    )

    assert len(results_by_plies) == 2 * lam.n_ply
    assert mid_plane_field["epsilon_x"].shape == x.shape
    assert mid_plane_field["epsilon_y"].shape == x.shape
    assert mid_plane_field["gamma_xy"].shape == x.shape

    sample = results_by_plies[0]
    assert sample["sigma_x"].shape == x.shape
    assert sample["failure_mode"].shape == x.shape
    assert isinstance(sample["failure_mode"][0], str)


def test_generate_meshgrid_smoke() -> None:
    out = generate_meshgrid(
        hole_radius=1.0,
        plate_radius=2.0,
        n_points_radial=5,
        n_points_angular=7,
    )
    assert out["X"].shape == (5, 7)
    assert np.all(np.isfinite(out["X"]))
    assert np.all(np.isfinite(out["Y"]))

