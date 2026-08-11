import numpy as np
import pytest

from lamkit.analysis.buckling import BucklingAnalysis
from lamkit.analysis.laminate import Laminate
from lamkit.analysis.material import IM7_8551_7, Ply


def _build_laminate() -> Laminate:
    ply = Ply(IM7_8551_7, thickness=0.125)
    return Laminate([0.0, 90.0, 0.0], [ply, ply, ply])


def test_buckling_init_validations() -> None:
    lam = _build_laminate()

    with pytest.raises(TypeError, match="Laminate"):
        BucklingAnalysis(laminate="not-laminate", a=100.0, b=80.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="positive"):
        BucklingAnalysis(laminate=lam, a=0.0, b=80.0)

    with pytest.raises(ValueError, match=">= 1"):
        BucklingAnalysis(laminate=lam, a=100.0, b=80.0, m=0, n=2)


def test_compute_constraints_shapes_for_pinned_case() -> None:
    lam = _build_laminate()
    analysis = BucklingAnalysis(laminate=lam, a=100.0, b=80.0, constraints="PINNED", m=2, n=2)

    uidx, vidx, widx = analysis._compute_constraints()

    assert len(uidx) == 16
    assert len(vidx) == 16
    assert len(widx) == 16
    assert len(analysis.su_idx) == 4
    assert len(analysis.sv_idx) == 4
    assert len(analysis.sw_idx) == 4


def test_calc_k_kg_d_matrix_shapes_and_finite_values() -> None:
    lam = _build_laminate()
    analysis = BucklingAnalysis(laminate=lam, a=120.0, b=100.0, Nxx=-1.0, m=2, n=2)

    K, KG = analysis.calc_K_KG_D()

    assert K.shape == (4, 4)
    assert KG.shape == (4, 4)
    assert np.all(np.isfinite(K))
    assert np.all(np.isfinite(KG))


def test_buckling_analysis_returns_eigenpairs() -> None:
    lam = _build_laminate()
    analysis = BucklingAnalysis(laminate=lam, a=120.0, b=100.0, Nxx=-1.0, m=3, n=3)

    eigvals, eigvecs = analysis.buckling_analysis(num_eigvalues=3)

    assert eigvals.shape == (3,)
    assert eigvecs.shape == (9, 3)
    assert np.all(np.isfinite(eigvals))
    assert np.all(np.isfinite(eigvecs))
    assert np.allclose(analysis.eigenvalue, eigvals)
    assert np.allclose(analysis.eigenvector, eigvecs)


def _nx_cr_closed_form_orthotropic(D: np.ndarray, a: float, b: float) -> float:
    """Classical simply supported specially orthotropic plate buckling load under Nx.

    Nx_cr(m, n) = pi^2 * [D11*(m/a)^2 + 2*(D12+2*D66)*(n/b)^2
                          + D22*(n/b)^4/(m/a)^2], minimized over
    integer half-wave numbers m (load direction) and n (width).
    Valid only when D16 = D26 = 0.
    """
    D11, D12, D22, D66 = D[0, 0], D[0, 1], D[1, 1], D[2, 2]
    best = np.inf
    for m in range(1, 13):
        ma2 = (m / a) ** 2
        for n in range(1, 5):
            nb2 = (n / b) ** 2
            value = D11 * ma2 + 2.0 * (D12 + 2.0 * D66) * nb2 + D22 * nb2 ** 2 / ma2
            best = min(best, value)
    return np.pi ** 2 * best


def test_buckling_analysis_matches_closed_form() -> None:
    # What: first Ritz buckling load of a cross-ply [0/90]s plate (D16=D26=0) vs the
    #   Navier closed form for a simply supported specially orthotropic plate.
    # Why: regression test -- the previous eigsh(which="SM", sigma=1.0, mode="cayley")
    #   solve locked onto eigenvalues near +1 and could miss the true first mode by an
    #   order of magnitude; the dense eigh solve must track the analytic value.
    ply = Ply(IM7_8551_7, thickness=0.125)
    lam = Laminate([0.0, 90.0, 90.0, 0.0], [ply] * 4)
    analysis = BucklingAnalysis(
        laminate=lam, a=100.0, b=100.0, constraints="PINNED", Nxx=-1.0, m=6, n=6
    )

    eigvals, _ = analysis.buckling_analysis(num_eigvalues=5)

    nx_cr_ref = _nx_cr_closed_form_orthotropic(lam.D, a=100.0, b=100.0)
    assert np.all(eigvals > 0.0)
    assert np.all(np.diff(eigvals) >= 0.0), "multipliers should be ascending"
    assert np.isclose(eigvals[0], nx_cr_ref, rtol=1e-5), (
        f"Ritz Nx_cr = {eigvals[0]:.6f} N/mm, closed form = {nx_cr_ref:.6f} N/mm"
    )


def test_buckling_analysis_with_lamination_parameter_laminate() -> None:
    # What: BucklingAnalysis on a Laminate built from a lamination-parameter dict
    #   (xiA/xiD/T only, no ply stacking) gives the same multipliers as the
    #   equivalent ply-by-ply laminate.
    # Why: BucklingAnalysis reads laminate.D; the LP-dict path used to return a zero
    #   D matrix there, so LP-defined laminates needed a subclass workaround.
    ply = Ply(IM7_8551_7, thickness=0.125)
    stacking = [0.0, 90.0, 90.0, 0.0]
    lam_ref = Laminate(stacking, [ply] * 4)
    lp = Laminate.get_lamination_parameters(stacking)
    lam_lp = Laminate(
        {"xiA": lp["xiA"], "xiD": lp["xiD"], "T": 4 * 0.125}, ply
    )

    kwargs = dict(a=100.0, b=100.0, constraints="PINNED", Nxx=-1.0, m=6, n=6)
    eigvals_ref, _ = BucklingAnalysis(laminate=lam_ref, **kwargs).buckling_analysis(3)
    eigvals_lp, _ = BucklingAnalysis(laminate=lam_lp, **kwargs).buckling_analysis(3)

    assert np.allclose(eigvals_lp, eigvals_ref, rtol=1e-9, atol=1e-9)
