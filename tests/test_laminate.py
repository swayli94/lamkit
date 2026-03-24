"""Classic laminate theory (CLT): laminate.ABD, Q_bar, symmetry, and load-strain consistency."""

from __future__ import annotations

import numpy as np
import pytest

from lamkit.analysis.laminate import Laminate
from lamkit.analysis.material import IM7_8551_7, Ply

RTOL = 1e-9
ATOL = 1e-9


@pytest.fixture(autouse=True, scope="module")
def _np_random_seed():
    np.random.seed(0)
    yield


def _assert_allclose(name: str, a: np.ndarray, b: np.ndarray) -> None:
    # What: strict numeric equality with a readable failure report.
    # Why: one shared tolerance and message format across checks.
    if not np.allclose(a, b, rtol=RTOL, atol=ATOL):
        max_err = np.max(np.abs(a - b))
        raise AssertionError(f"{name}: max |a-b| = {max_err:g}\n{a}\n{b}")


def test_Q_bar_special_angles() -> None:
    # What: Q_bar at 0 deg equals [Q]; at 90 deg the in-plane normal stiffness entries swap (Q11 <-> Q22) for this orthotropic ply.
    # Why: catches sign/ordering bugs in the rotation tensors and degree-vs-radian mistakes with minimal algebra.
    m = IM7_8551_7
    Q = m.Q
    _assert_allclose("Q_bar(0 deg)", m.get_Q_bar(0.0), Q)
    Qb90 = m.get_Q_bar(90.0)
    assert np.isclose(Qb90[0, 0], Q[1, 1], rtol=1e-6, atol=1e-6)
    assert np.isclose(Qb90[1, 1], Q[0, 0], rtol=1e-6, atol=1e-6)


def test_symmetric_laminate_B_zero() -> None:
    # What: symmetric angle stacks about the mid-plane should give extension–bending coupling B ≈ 0.
    # Why: wrong ply ordering (bottom vs top) or z-through-thickness integration errors usually show up first in B for symmetric layups.
    t = 0.125
    ply = Ply(IM7_8551_7, thickness=t)
    for stacking, desc in (
        ([0, 90, 90, 0], "[0/90]s"),
        ([45, -45, -45, 45], "[45/-45]s"),
    ):
        lam = Laminate(stacking, [ply] * len(stacking))
        B = lam.B
        assert np.allclose(B, 0.0, rtol=0, atol=1e-6), f"{desc}: B should be 0\n{B}"


def test_balanced_laminate() -> None:
    # What: equal-thickness [+theta/-theta] pairs should yield zero membrane shear-extension coupling A16 and A26.
    # Why: validates that Q_bar shear/normal coupling terms integrate to zero under mirror angles—a common CLT sanity property.
    t = 0.14
    ply = Ply(IM7_8551_7, thickness=t)
    for theta in (15.0, 30.0, 45.0):
        lam = Laminate([theta, -theta], [ply, ply])
        A = lam.A
        assert abs(A[0, 2]) < 1e-5 and abs(A[1, 2]) < 1e-5, (
            f"theta={theta}: A16,A26 should be ~0, got A={A}"
        )


def test_single_ply_ABD_formulas() -> None:
    # What: one homogeneous ply with mid-plane at laminate mid-thickness gives A=Q_bar*h, B=0, D=Q_bar*h^3/12 (plate formulas).
    # Why: directly tests the z-integration weights (1, z, z²) in A,B,D without multi-ply cancellation masking errors.
    h = 0.2
    theta = 25.0
    lam = Laminate([theta], Ply(IM7_8551_7, thickness=h))
    Qb = lam.Q_layup[0]
    A_ref = Qb * h
    D_ref = Qb * (h**3) / 12.0
    _assert_allclose("single-ply A", lam.A, A_ref)
    _assert_allclose("single-ply B", lam.B, np.zeros((3, 3)))
    _assert_allclose("single-ply D", lam.D, D_ref)


def test_ABD_integration_vs_lamination_parameters() -> None:
    # What: ply-by-ply z-integration (lam.A, lam.B, lam.D) vs closed form from xiA/xiB/xiD + material invariants.
    # Why: guards drift between the two CLT evaluation paths (wrong xi normalization or missing T²/8 on B, etc.).
    t = 0.125
    ply = Ply(IM7_8551_7, thickness=t)
    cases = (
        [0, 45, -45, 90],
        [0, 90, 0],
        [30, -30, 10],
        [0],
    )
    for stacking in cases:
        lam = Laminate(stacking, [ply] * len(stacking))
        _assert_allclose(
            f"A LP vs integration {stacking}",
            lam.get_A_from_lamination_parameters(),
            lam.A,
        )
        _assert_allclose(
            f"B LP vs integration {stacking}",
            lam.get_B_from_lamination_parameters(),
            lam.B,
        )
        _assert_allclose(
            f"D LP vs integration {stacking}",
            lam.get_D_from_lamination_parameters(),
            lam.D,
        )


def test_ABD_round_trip() -> None:
    # What: pick random mid-plane strains/curvatures x, form resultants N=[ABD]x, then solve x'=[ABD]^{-1}N.
    # Why: confirms the assembled 6×6 ABD block and its inverse are mutually consistent (assembly or inversion bugs fail here).
    t = 0.125
    ply = Ply(IM7_8551_7, thickness=t)
    lam = Laminate([0, 45, -45, 90], [ply] * 4)
    abd = lam.ABD
    abd_inv = lam.ABD_inverse_matrix
    for _ in range(5):
        x = np.random.randn(6)
        N = abd @ x
        x_back = abd_inv @ N
        _assert_allclose("ABD round-trip", x, x_back)


def test_physical_trends() -> None:
    # What: coarse inequalities -- more 0-deg plies raise A11 vs all-90; all-45 raises A66 vs all-0 here; doubling each ply thickness scales ||D|| ~8.
    # Why: catches gross stiffness scaling (wrong angle convention, mm vs m, or missing h³ on D) even when analytic equalities still “almost” hold.
    t = 0.125
    ply = Ply(IM7_8551_7, thickness=t)

    lam_0 = Laminate([0, 0, 0, 0], [ply] * 4)
    lam_90 = Laminate([90, 90, 90, 90], [ply] * 4)
    assert lam_0.A[0, 0] > lam_90.A[0, 0], "A11 for all-0 should exceed all-90"

    lam_45 = Laminate([45, 45, 45, 45], [ply] * 4)
    assert lam_45.A[2, 2] > lam_0.A[2, 2], "all-45 should give larger A66 than all-0 here"

    # Each ply thickness x2 => total h x2 => D ~ h^3 => use Frobenius norm ~8 (avoid element-wise ratio on tiny couplings)
    ply_thin = Ply(IM7_8551_7, thickness=t)
    ply_thick = Ply(IM7_8551_7, thickness=2 * t)
    seq = [0, 45, -45, 90]
    lam_s = Laminate(seq, [ply_thin] * 4)
    lam_d = Laminate(seq, [ply_thick] * 4)
    ratio = np.linalg.norm(lam_d.D) / np.linalg.norm(lam_s.D)
    assert 7.0 < ratio < 9.0, f"doubling thickness: ||D|| should be ~8x, got ratio={ratio}"
