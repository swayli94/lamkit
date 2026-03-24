"""Tests for Material and Ply."""

import numpy as np
import pytest

from lamkit.analysis.material import MATERIAL_IM7_8551_7, IM7_8551_7, Material, Ply


def _minimal_props(**overrides):
    base = {k: (0.0 if v is None else v) for k, v in MATERIAL_IM7_8551_7.items()}
    base.update(overrides)
    return base


def test_material_requires_all_keys():
    props = _minimal_props()
    del props["E11"]
    with pytest.raises(ValueError, match="required keys"):
        Material("x", props)


def test_material_nu21_reciprocal_relation():
    m = IM7_8551_7
    nu21 = m.get_property("nu21")
    expected = m.get_property("nu12") * m.get_property("E22") / m.get_property("E11")
    assert nu21 == pytest.approx(expected)


def test_Q_0_symmetric_positive_diagonal():
    Q = IM7_8551_7.Q
    assert Q.shape == (3, 3)
    assert np.allclose(Q, Q.T)
    assert Q[0, 0] > 0 and Q[1, 1] > 0 and Q[2, 2] > 0


def test_ply_auto_name():
    p1 = Ply(IM7_8551_7, thickness=0.125)
    p2 = Ply(IM7_8551_7, thickness=0.125)
    assert p1.name != p2.name
