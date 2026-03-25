'''
Loaded hole in an infinite anisotropic homogeneous plate
'''

import numpy as np
from typing import Callable, Tuple

from lamkit.lekhnitskii.hole import Hole
from lamkit.lekhnitskii.utils import rotate_material_matrix, rotate_stress


def _remove_bad_displacements(displacement_func: 
    Callable[[object, np.ndarray, np.ndarray], np.ndarray]):
    '''
    Removes displacements that are 180 degrees behind bearing load direction
    '''
    def inner(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # call displacement function
        displacements = displacement_func(self, x, y)
        # check if any points are 180 degrees behind bearing load
        r, angles = self._cartesian_to_polar(x, y)
        bad_angle = np.pi if self.theta == 0 else -1*(np.pi - self.theta)
        # if so, replace those results with np.nan
        displacements[np.isclose(angles, bad_angle)] = np.nan
        return displacements

    return inner 


class LoadedHole(Hole):
    '''
    Class for defining a loaded hole in an infinite anisotropic homogeneous plate

    A cosine bearing load distribution is assumed to apply to the inside of the hole.

    Notes
    -----
    Bearing distribution as shown below Ref. [4]_

    .. image:: ../img/cosine_distribution.png
       :height: 400px

    Parameters
    ----------
    load : float
        bearing force
    diameter : float
        hole diameter
    thickness : float
        plate thickness
    a_inv : array_like
        2D array (3, 3) inverse CLPT A-matrix
    theta : float, optional
        bearing angle counter clock-wise from positive x-axis (radians)

    Attributes
    ----------
    p : float
        bearing force
    theta : float
        bearing angle counter clock-wise from positive x-axis (radians)
    A : float
        real part of equilibrium constant for first stress function
    A_bar : float
        imaginary part of equilibrium constant for first stress function
    B : float
        real part of equilibrium constant for second stress function
    B_bar : float
        imaginary part of equilibrium constant for second stress function

    '''
    FOURIER_TERMS = 45  # number of fourier series terms [3]_

    def __init__(self, load: float, diameter: float, thickness: float,
                 a_inv: np.ndarray, theta: float = 0.) -> None:
        a_inv = rotate_material_matrix(a_inv, angle=theta)
        radius = diameter / 2.0
        super().__init__(radius=radius, compliance_matrix=a_inv)
        self.h = thickness
        self.p = load
        self.theta = theta
        self.A, self.A_bar, self.B, self.B_bar = self.equilibrium_constants()

    def alpha(self) -> np.ndarray:
        '''
        Fourier series coefficients modified for use in stress function equations

        Returns
        -------
        np.ndarray of shape (N,)
            Fourier series coefficients (complex array)
        '''
        h = self.h
        p = self.p
        N = self.FOURIER_TERMS
        m = np.arange(3, N + 1)

        # modification from Eq. 37.2 [2]_
        mod = -1/(h*np.pi)

        alpha = np.zeros(N)
        alpha[:2] = [p*4/(6*np.pi)*mod, p/8*mod]
        alpha[2:] = -2*p*np.sin(np.pi*m/2)/(np.pi*m*(m**2 - 4))*mod

        # (in Ref. 2 Eq. 37.2, alpha is associated with the y-direction. Can someone explain?)
        return alpha

    def beta(self) -> np.ndarray:
        '''
        Fourier series coefficients modified for use in stress function equations

        Returns
        -------
        np.ndarray of shape (N,)
            Fourier series coefficients (complex array)
        '''
        h = self.h
        p = self.p
        N = self.FOURIER_TERMS
        m = np.arange(1, N + 1)

        # modification from Eq. 37.2 [2]_
        mod = 4 / (np.pi*m**2*h)

        beta = np.zeros(N, dtype=complex)
        beta[:2] = [-p*1j/(3*np.pi)*mod[0], -1j*p/8*mod[1]]
        beta[2:] = 1j*p*np.sin(np.pi*m[2:]/2)/(np.pi*(m[2:]**2 - 4))*mod[2:]

        # (in Ref. 2 Eq. 37.2, beta is associated with the x-direction. Can someone explain?)
        return beta

    def equilibrium_constants(self) -> Tuple[float, float, float, float]:
        '''
        Solve for constants of equilibrium

        When the plate has loads applied that are not in equilibrium,
        the unbalanced loads are reacted at infinity.
        This function solves for the constant terms in the stress functions
        that account for these reactions.

        Notes
        -----
        This method implements Eq. 37.5 [2]_.
        Complex terms have been expanded and resolved for
        A, A_bar, B and B_bar (setting Py equal to zero).

        Returns
        -------
        [A, A_bar, B, B_bar] : Tuple[float, float, float, float]
            real and imaginary parts of constants A and B
        '''
        R1, R2 = np.real(self.mu1), np.imag(self.mu1)
        R3, R4 = np.real(self.mu2), np.imag(self.mu2)
        p = self.p
        h = self.h
        s11 = self.s[0, 0]
        s12 = self.s[0, 1]
        s22 = self.s[1, 1]
        s13 = self.s[0, 2]
        pi = np.pi

        mu_mat = np.array([[0., 1, 0., 1.],
                           [R2, R1, R4, R3],
                           [2*R1*R2, (R1**2 - R2**2), 2*R3*R4, (R3**2 - R4**2)],
                           [R2/(R1**2 + R2**2), -R1/(R1**2 + R2**2), R4/(R3**2 + R4**2), -R3/(R3**2 + R4**2)]])

        load_vec = p/(4.*pi*h) * np.array([0.,
                                           1.,
                                           s13/s11,
                                           s12/s22])

        A1, A2, B1, B2 = np.dot(np.linalg.inv(mu_mat), load_vec)
        
        return A1, A2, B1, B2

    def phi_1(self, z1: np.ndarray) -> np.ndarray:
        '''
        Calculates the first stress function

        Parameters
        ----------
        z1 : np.ndarray of shape (n,)
            first mapping parameter (complex array)

        Returns
        -------
        np.ndarray of shape (n,)
            first stress function (complex array)
        '''
        mu1 = self.mu1
        mu2 = self.mu2
        A = self.A + 1j * self.A_bar
        N = self.FOURIER_TERMS
        xi_1, sign_1 = self.xi_1(z1)

        m = np.arange(1, N + 1)
        alpha = self.alpha()
        beta = self.beta()

        # return results for each point in xi_1
        return np.array([(A*np.log(xi_1[i]) + np.sum((beta - mu2 * alpha) / (mu1 - mu2) / xi_1[i] ** m))
                         for i in range(len(xi_1))])

    def phi_2(self, z2: np.ndarray) -> np.ndarray:
        '''
        Calculates the second stress function

        Parameters
        ----------
        z2 : np.ndarray of shape (n,)
            second mapping parameter (complex array)

        Returns
        -------
        np.ndarray of shape (n,)
            second stress function (complex array)
        '''
        mu1 = self.mu1
        mu2 = self.mu2
        B = self.B + 1j * self.B_bar
        N = self.FOURIER_TERMS
        xi_2, sign_2 = self.xi_2(z2)

        m = np.arange(1, N + 1)
        alpha = self.alpha()
        beta = self.beta()

        # return results for each point in xi_2
        return np.array([(B*np.log(xi_2[i]) - np.sum((beta - mu1 * alpha) / (mu1 - mu2) / xi_2[i] ** m))
                         for i in range(len(xi_2))])

    def phi_1_prime(self, z1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calculates derivative of the first stress function

        Parameters
        ----------
        z1 : np.ndarray of shape (n,)
            first mapping parameter (complex array)

        Returns
        -------
        np.ndarray of shape (n,)
            derivative of the first stress function (complex array)
        '''
        mu1 = self.mu1
        mu2 = self.mu2
        a = self.radius
        b = self.radius
        A = self.A + 1j * self.A_bar
        N = self.FOURIER_TERMS
        xi_1, sign_1 = self.xi_1(z1)

        eta_1 = sign_1 * np.sqrt(z1 * z1 - a * a - b * b * mu1 * mu1)

        m = np.arange(1, N + 1)
        alpha = self.alpha()
        beta = self.beta()

        # return results for each point in xi_1
        phi_1_p = np.array([1 / eta_1[i] * (A - np.sum(m * (beta - mu2 * alpha) / (mu1 - mu2) / xi_1[i] ** m))
                        for i in range(len(xi_1))])
        return phi_1_p, sign_1

    def phi_2_prime(self, z2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calculates derivative of the second stress function

        Parameters
        ----------
        z2 : np.ndarray of shape (n,)
            second mapping parameter (complex array)

        Returns
        -------
        np.ndarray of shape (n,)
            derivative of the second stress function (complex array)
        '''
        mu1 = self.mu1
        mu2 = self.mu2
        a = self.radius
        b = self.radius
        B = self.B + 1j * self.B_bar
        N = self.FOURIER_TERMS
        xi_2, sign_2 = self.xi_2(z2)

        eta_2 = sign_2 * np.sqrt(z2 * z2 - a * a - b * b * mu2 * mu2)

        m = np.arange(1, N + 1)
        alpha = self.alpha()
        beta = self.beta()

        # return results for each point in xi_2
        phi_2_p = np.array([1 / eta_2[i] * (B + np.sum(m * (beta - mu1 * alpha) / (mu1 - mu2) / xi_2[i] ** m))
                         for i in range(len(xi_2))])
        return phi_2_p, sign_2
    
    def _cartesian_to_polar(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Converts cartesian points to polar coordinates

        Parameters
        ----------
        x : np.ndarray of shape (n,)
            x locations in the cartesian coordinate system
        y : np.ndarray of shape (n,)
            y locations in the cartesian coordinate system

        Returns
        -------
        radii : np.ndarray of shape (n,)
            radius of each point
        angles : np.ndarray of shape (n,)
            angle of each point

        '''
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        r = np.sqrt(x**2 + y**2)

        # calculate angles and fix signs
        angles = np.arccos(np.array([1, 0]).dot(np.array([x, y])) / r)
        where_vals = np.nonzero(y)[0]
        angles[where_vals] = angles[where_vals] * np.sign(y[where_vals])

        return r, angles

    def _rotate_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Rotates points to account for bearing angle

        Parameters
        ----------
        x : np.ndarray of shape (n,)
            x locations in the cartesian coordinate system
        y : np.ndarray of shape (n,)
            y locations in the cartesian coordinate system

        Returns
        -------
        x' : np.ndarray of shape (n,)
            new x points
        y' : np.ndarray of shape (n,)
            new y points
        '''
        # rotation back to original coordinates
        rotation = -self.theta

        # convert points to polar coordinates
        r, angles = self._cartesian_to_polar(x, y)

        # rotate coordinates by negative theta
        angles += rotation

        # convert back to cartesian
        x = r * np.cos(angles)
        y = r * np.sin(angles)

        return x, y

    def stress(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Calculates the stress at (x, y) points in the plate

        Parameters
        ----------
        x : np.ndarray of shape (n,)
            x locations in the cartesian coordinate system
        y : np.ndarray of shape (n,)
            y locations in the cartesian coordinate system

        Returns
        -------
        stresses: np.ndarray
            (n, 3) array of in-plane stress components in the cartesian coordinate system
            [[sx0, sy0, sxy0], [sx1, sy1, sxy1], ... , [sxn, syn, sxyn]]
        '''
        # rotate points to account for bearing angle
        x, y = self._rotate_points(x, y)

        # calculate stresses and rotate back
        stresses = super().stress(x, y)
        return rotate_stress(stresses, angle=-self.theta)

    @_remove_bad_displacements
    def displacement(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Calculates the displacement at (x, y) points in the plate

        Parameters
        ----------
        x : np.ndarray of shape (n,)
            x locations in the cartesian coordinate system
        y : np.ndarray of shape (n,)
            y locations in the cartesian coordinate system

        Returns
        -------
        displacements: np.ndarray
            (n, 2) array of in-plane displacement components in the cartesian coordinate system
            [[u0, v0], [u1, v1], ... , [un, vn]]
        '''
        # rotate points to account for bearing angle
        x, y = self._rotate_points(x, y)
        return super().displacement(x, y)
