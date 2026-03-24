'''

'''

import abc
import numpy as np
from typing import Tuple, Dict


class Hole(abc.ABC):
    '''
    Abstract parent class for defining a hole in an anisotropic infinite plate

    This class defines shared methods and attributes for 
    anisotropic elasticity solutions of plates with circular holes.

    Notes
    -----
    The following assumptions apply for plates in a state of generalized plane stress.

    1. The plates are homogeneous and a plane of elastic symmetry
    that is parallel to their middle plane exists at every point.
    2. Applied forces act within planes that are parallel and symmetric to the
    middle plane of the plates and have negligible variation through the thickness.
    3. Plate deformations are small.

    Parameters
    ----------
    radius : float
        hole radius
    compliance_matrix : np.ndarray of shape (3, 3)
        compliance matrix of the anisotropic plate

    Attributes
    ----------
    r : float
        the hole radius
    mu1 : float
        real part of first root of characteristic equation
    mu2 : float
        real part of second root of characteristic equation
    mu1_bar : float
        imaginary part of first root of characteristic equation
    mu2_bar : float
        imaginary part of second root of characteristic equation
    '''

    MAPPING_PRECISION = 0.0000001

    def __init__(self, radius: float, compliance_matrix: np.ndarray) -> None:
        
        if compliance_matrix.shape != (3, 3):
            raise ValueError("compliance_matrix must be a 3x3 array")
        
        self.radius = radius
        
        # For elliptical hole, semi-axis lengths (a, b) are:
        self.semi_axis_x = radius
        self.semi_axis_y = radius
        
        self.s = np.array(compliance_matrix, dtype=float)
        self.mu1, self.mu2, self.mu1_bar, self.mu2_bar = self.roots()

    def roots(self) -> Tuple[complex, complex, complex, complex]:
        '''
        Finds the roots to the 4th order characteristic equation.
        
        Returns
        -------
        mu1 : complex
            first root of characteristic equation
        mu1_bar : complex
            second root of characteristic equation
        mu2 : complex
            third root of characteristic equation
        mu2_bar : complex
            fourth root of characteristic equation
        '''
        s11 = self.s[0, 0]
        s12 = self.s[0, 1]
        s13 = self.s[0, 2]
        s22 = self.s[1, 1]
        s23 = self.s[1, 2]
        s33 = self.s[2, 2]

        roots = np.roots([s11, -2 * s13, (2 * s12 + s33), -2 * s23, s22])

        if np.imag(roots[0]) >= 0.0:
            mu2 = roots[0]
            mu2_bar = roots[1]
        elif np.imag(roots[1]) >= 0.0:
            mu2 = roots[1]
            mu2_bar = roots[0]
        else:
            raise ValueError("mu1 cannot be solved")

        if np.imag(roots[2]) >= 0.0:
            mu1 = roots[2]
            mu1_bar = roots[3]
        elif np.imag(roots[3]) >= 0.0:
            mu1 = roots[3]
            mu1_bar = roots[2]
        else:
            raise ValueError("mu2 cannot be solved")

        return mu1, mu2, mu1_bar, mu2_bar

    def xi_1(self, z1s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calculates the first mapping parameters.

        Parameters
        ----------
        z1s : ndarray
            Array of first parameters from the complex plane (any shape).

        Returns
        -------
        xi_1s : ndarray
            Array of the first mapping parameters (same shape as z1s)
        sign_1s : ndarray
            Array of signs producing positive mapping parameters (same shape as z1s)
        '''
        shape = z1s.shape
        z1s_flat = z1s.ravel()

        mu1 = self.mu1
        a = self.semi_axis_x
        b = self.semi_axis_y

        xi_1s = np.zeros(z1s_flat.size, dtype=complex)
        sign_1s = np.zeros(z1s_flat.size, dtype=int)

        xi_1_pos = (z1s_flat + np.sqrt(z1s_flat * z1s_flat - a * a - mu1 * mu1 * b * b)) / (a - 1j * mu1 * b)
        xi_1_neg = (z1s_flat - np.sqrt(z1s_flat * z1s_flat - a * a - mu1 * mu1 * b * b)) / (a - 1j * mu1 * b)

        pos_indices = np.where(np.abs(xi_1_pos) >= (1. - self.MAPPING_PRECISION))[0]
        neg_indices = np.where(np.abs(xi_1_neg) >= (1. - self.MAPPING_PRECISION))[0]

        xi_1s[pos_indices] = xi_1_pos[pos_indices]
        xi_1s[neg_indices] = xi_1_neg[neg_indices]

        # high level check that all indices were mapped
        if not (pos_indices.size + neg_indices.size) == xi_1s.size:
            bad_indices = np.where(xi_1s == 0)[0]
            print(f"xi_1 unsolvable\n Failed Indices: {bad_indices}")

        sign_1s[pos_indices] = 1
        sign_1s[neg_indices] = -1

        return xi_1s.reshape(shape), sign_1s.reshape(shape)

    def xi_2(self, z2s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calculates the second mapping parameters.

        Parameters
        ----------
        z2s : ndarray
            Array of second parameters from the complex plane (any shape).

        Returns
        -------
        xi_2s : np.ndarray
            Array of the second mapping parameters (same shape as z2s)
        sign_2s : np.ndarray
            Array of signs producing positive mapping parameters (same shape as z2s)
        '''
        shape = z2s.shape
        z2s_flat = z2s.ravel()

        mu2 = self.mu2
        a = self.semi_axis_x
        b = self.semi_axis_y

        xi_2s = np.zeros(z2s_flat.size, dtype=complex)
        sign_2s = np.zeros(z2s_flat.size, dtype=int)

        xi_2_pos = (z2s_flat + np.sqrt(z2s_flat * z2s_flat - a * a - mu2 * mu2 * b * b)) / (a - 1j * mu2 * b)
        xi_2_neg = (z2s_flat - np.sqrt(z2s_flat * z2s_flat - a * a - mu2 * mu2 * b * b)) / (a - 1j * mu2 * b)

        pos_indices = np.where(np.abs(xi_2_pos) >= (1. - self.MAPPING_PRECISION))[0]
        neg_indices = np.where(np.abs(xi_2_neg) >= (1. - self.MAPPING_PRECISION))[0]

        xi_2s[pos_indices] = xi_2_pos[pos_indices]
        xi_2s[neg_indices] = xi_2_neg[neg_indices]

        # high level check that all indices were mapped
        if not (pos_indices.size + neg_indices.size) == xi_2s.size:
            bad_indices = np.where(xi_2s == 0)[0]
            print(f"xi_2 unsolvable\n Failed Indices: {bad_indices}")

        sign_2s[pos_indices] = 1
        sign_2s[neg_indices] = -1

        return xi_2s.reshape(shape), sign_2s.reshape(shape)

    @abc.abstractmethod
    def phi_1(self, z1: np.ndarray):
        '''
        Calculates the first stress function
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def phi_2(self, z2: np.ndarray):
        '''
        Calculates the second stress function
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def phi_1_prime(self, z1: np.ndarray):
        '''
        Calculates derivative of the first stress function
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def phi_2_prime(self, z2: np.ndarray):
        '''
        Calculates derivative of the second stress function
        '''
        raise NotImplementedError()

    def stress(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Calculates the stress at (x, y) points in the plate

        Parameters
        ----------
        x : np.ndarray
            1D array x locations in the cartesian coordinate system
        y : np.ndarray
            1D array y locations in the cartesian coordinate system

        Returns
        -------
        stresses: np.ndarray
            (n, 3) in-plane stress components in the cartesian coordinate system
            [[sx0, sy0, sxy0], [sx1, sy1, sxy1], ... , [sxn, syn, sxyn]]
        '''
        mu1 = self.mu1
        mu2 = self.mu2
        
        # Check if x and y are 1D arrays
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1D arrays")
        
        # Check if x and y have the same length
        if x.size != y.size:
            raise ValueError("x and y must have the same length")

        z1 = x + mu1 * y
        z2 = x + mu2 * y

        phi_1_prime, _ = self.phi_1_prime(z1)
        phi_2_prime, _ = self.phi_2_prime(z2)

        sx = 2.0 * np.real(mu1 * mu1 * phi_1_prime + mu2 * mu2 * phi_2_prime)
        sy = 2.0 * np.real(phi_1_prime + phi_2_prime)
        sxy = -2.0 * np.real(mu1 * phi_1_prime + mu2 * phi_2_prime)
        
        stresses = np.array([sx, sy, sxy]).T

        return stresses

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
        s11 = self.s[0, 0]
        s12 = self.s[0, 1]
        s13 = self.s[0, 2]
        s22 = self.s[1, 1]
        s23 = self.s[1, 2]
        mu1 = self.mu1
        mu2 = self.mu2

        p1 = s11*mu1**2 + s12 - s13*mu1
        p2 = s11*mu2**2 + s12 - s13*mu2
        q1 = s12*mu1 + s22/mu1 - s23
        q2 = s12*mu2 + s22/mu2 - s23

        z1 = x + mu1 * y
        z2 = x + mu2 * y

        phi_1 = self.phi_1(z1)
        phi_2 = self.phi_2(z2)

        u = 2.0 * np.real(p1 * phi_1 + p2 * phi_2)
        v = 2.0 * np.real(q1 * phi_1 + q2 * phi_2)

        displacements = np.array([u, v]).T

        return displacements

    # ------------------------------
    # For detailed analysis
    # ------------------------------
    def calculate_stress_results(self, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        '''
        Calculates the stress at (x, y) points in the plate
        
        Parameters
        ----------
        x, y : np.ndarray of shape (n,)
            x and y locations in the cartesian coordinate system
            
        Returns
        -------
        field: Dict[str, np.ndarray]
            Dictionary containing the stress and sign fields
            'sigma_x': np.ndarray of shape (n,)
            'sigma_y': np.ndarray of shape (n,)
            'tau_xy': np.ndarray of shape (n,)
            'sign_xi1': np.ndarray of shape (n,)
            'sign_xi2': np.ndarray of shape (n,)
            'Real(phi_1_prime)': np.ndarray of shape (n,)
            'Real(phi_2_prime)': np.ndarray of shape (n,)
            'Imag(phi_1_prime)': np.ndarray of shape (n,)
            'Imag(phi_2_prime)': np.ndarray of shape (n,)
        '''
        mu1 = self.mu1
        mu2 = self.mu2
        
        # Check if x and y are 1D arrays
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1D arrays")
        
        # Check if x and y have the same length
        if x.size != y.size:
            raise ValueError("x and y must have the same length")

        z1 = x + mu1 * y
        z2 = x + mu2 * y

        phi_1_prime, sign_xi1 = self.phi_1_prime(z1)
        phi_2_prime, sign_xi2 = self.phi_2_prime(z2)

        sigma_x = 2.0 * np.real(mu1 * mu1 * phi_1_prime + mu2 * mu2 * phi_2_prime)
        sigma_y = 2.0 * np.real(phi_1_prime + phi_2_prime)
        tau_xy = -2.0 * np.real(mu1 * phi_1_prime + mu2 * phi_2_prime)
                
        return {
            'sigma_x': sigma_x,
            'sigma_y': sigma_y,
            'tau_xy': tau_xy,
            'Real(phi_1_prime)': np.real(phi_1_prime),
            'Real(phi_2_prime)': np.real(phi_2_prime),
            'Imag(phi_1_prime)': np.imag(phi_1_prime),
            'Imag(phi_2_prime)': np.imag(phi_2_prime),
            'sign_xi1': sign_xi1,
            'sign_xi2': sign_xi2,
            }
