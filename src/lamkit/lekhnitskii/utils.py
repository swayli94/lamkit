'''
Utility functions
'''
import numpy as np
from typing import Tuple, Dict, Any


def rotate_stress(stresses: np.ndarray, angle: float = 0.) -> np.ndarray:
    '''
    Rotates 2D stress components by given angle

    The rotation angle is positive counter-clockwise from the positive x-axis in the cartesian xy-plane.

    Parameters
    ----------
    stresses : ndarray of shape (n, 3)
        array of in-plane stresses in the cartesian coordinate system
        [[sx0, sy0, sxy0], [sx1, sy1, sxy1], ... , [sxn, syn, sxyn]]
    angle : float, default 0.
        angle measured counter-clockwise from positive x-axis (radians)

    Returns
    -------
    stresses :ndarray of shape (n, 3)
        array of rotated in-plane stresses in the cartesian coordinate system
    '''
    c = np.cos(angle)
    s = np.sin(angle)
    rotation_matrix = np.array([
        [c**2, s**2, 2*s*c],
        [s**2, c**2, -2*s*c],
        [-s*c, s*c, c**2-s**2]
    ])
    stresses = rotation_matrix @ stresses.T
    return stresses.T

def rotate_strain(strains: np.ndarray, angle: float = 0.) -> np.ndarray:
    '''
    Rotates 2D strain components by given angle

    The rotation angle is positive counter-clockwise from the positive x-axis in the cartesian xy-plane.

    Parameters
    ----------
    strains : np.ndarray of shape (n, 3)
        array of in-plane strains in the cartesian coordinate system
        [[ex0, ey0, exy0], [ex1, ey1, exy1], ... , [exn, eyn, exyn]]
    angle : float, default 0.
        angle measured counter-clockwise from positive x-axis (radians)

    Returns
    -------
    strains : np.ndarray of shape (n, 3)
        array of rotated in-plane strains in the cartesian coordinate system
        [[ex0', ey0', exy0'], [ex1', ey1', exy1'], ... , [exn', eyn', exyn']]
    '''
    c = np.cos(angle)
    s = np.sin(angle)
    rotation_matrix = np.array([
        [c**2, s**2, s*c],
        [s**2, c**2, -s*c],
        [-2*s*c, 2*s*c, c**2 - s**2]
    ])
    strains = rotation_matrix @ strains.T
    return strains.T

def rotate_material_matrix(a_inv: np.ndarray, angle: float = 0.) -> np.ndarray:
    '''
    Rotates the material compliance matrix by given angle.
    
    The rotation angle is positive counter-clockwise from the positive x-axis in the cartesian xy-plane.

    Parameters
    ----------
    a_inv : np.ndarray of shape (3, 3)
        inverse CLPT A-matrix
    angle : float, default 0.
        angle measured counter-clockwise from positive x-axis (radians)

    Returns
    -------
    a_inv_p : np.ndarray of shape (3, 3)
        rotated inverse CLPT A-matrix
    '''
    c = np.cos(angle)
    s = np.sin(angle)
    c2 = np.cos(2*angle)
    s2 = np.sin(2*angle)

    a11 = a_inv[0, 0]
    a12 = a_inv[0, 1]
    a16 = a_inv[0, 2]
    a22 = a_inv[1, 1]
    a26 = a_inv[1, 2]
    a66 = a_inv[2, 2]

    a11p = a11*c**4 + (2*a12 + a66)*s**2*c**2 + a22*s**4 + (a16*c**2 + a26*s**2)*s2
    a22p = a11*s**4 + (2*a12 + a66)*s**2*c**2 + a22*c**4 - (a16*s**2 + a26*c**2)*s2
    a12p = a12 + (a11 + a22 - 2*a12 - a66)*s**2*c**2 + 0.5*(a26 - a16)*s2*c2
    a66p = a66 + 4*(a11 + a22 - 2*a12 - a66)*s**2*c**2 + 2*(a26 - a16)*s2*c2
    a16p = ((a22*s**2 - a11*c**2 + 0.5*(2*a12 + a66)*c2)*s2
            + a16*c**2*(c**2 - 3*s**2) + a26*s**2*(3*c**2 - s**2))
    a26p = ((a22*c**2 - a11*s**2 - 0.5*(2*a12 + a66)*c2)*s2
            + a16*s**2*(3*c**2 - s**2) + a26*c**2*(c**2 - 3*s**2))

    # test invariants (Eq. 9.7 [2]_)
    np.testing.assert_almost_equal(a11p + a22p + 2*a12p, a11 + a22 + 2*a12, decimal=4)
    np.testing.assert_almost_equal(a66p - 4*a12p, a66 - 4*a12, decimal=4)

    return np.array([[a11p, a12p, a16p], [a12p, a22p, a26p], [a16p, a26p, a66p]])

def rotate_complex_parameters(mu1: complex, mu2: complex,
                        angle: float = 0.) -> Tuple[complex, complex]:
    '''
    Rotates the complex parameters by given angle.

    The rotation angle is positive counter-clockwise from the positive x-axis in the cartesian xy-plane.

    Notes
    -----
    Implements Eq. 10.8 [2]_

    Parameters
    ----------
    mu1 : complex
        first complex parameter
    mu2 : complex
        second complex parameter
    angle : float, default 0.
        angle measured counter-clockwise from positive x-axis (radians)

    Returns
    -------
    mu1p, mu2p : complex
        first and second transformed complex parameters
    '''
    c = np.cos(angle)
    s = np.sin(angle)

    mu1p = (mu1*c - s)/(c + mu1*s)
    mu2p = (mu2*c - s)/(c + mu2*s)

    return mu1p, mu2p


def generate_meshgrid(hole_radius: float = 1.0, plate_radius: float = 10.0,
            n_points_radial: int = 101, n_points_angular: int = 101,
            radial_cluster_power: float = 2.0) -> Dict[str, Any]:
    '''
    Generate the meshgrid for the plate.
    Radial points are clustered near hole_radius (power-law: t^radial_cluster_power).
    '''
    t = np.linspace(0, 1, n_points_radial, endpoint=True)
    r = hole_radius + (plate_radius - hole_radius) * (t ** radial_cluster_power)
    theta = np.linspace(0, 2*np.pi, n_points_angular, endpoint=True)
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    return {'X': X, 'Y': Y,
            'R': R, 'Theta': Theta,
            'meshgrid_shape': X.shape}
