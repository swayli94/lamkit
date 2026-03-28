"""LaRC05 UVARM smoke tests."""

import numpy as np

from lamkit.analysis.larc05 import LaRC05


def test_get_uvarm_shape_2d():
    larc = LaRC05(nSCply=3, material_properties=LaRC05.get_property())
    stress = np.array([100.0, 10.0, 5.0])
    out = larc.get_uvarm(stress)
    assert out.shape == (larc.NUVARM,)


def test_get_uvarm_non_negative_failure_indices():
    larc = LaRC05(nSCply=3, material_properties=LaRC05.get_property())
    stress = np.array([50.0, 20.0, 0.0])
    out = larc.get_uvarm(stress)
    assert np.all(out[: larc.NFI] >= 0.0)
