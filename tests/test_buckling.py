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
