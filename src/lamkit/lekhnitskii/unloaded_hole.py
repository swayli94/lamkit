'''
Unloaded hole in an infinite anisotropic homogeneous plate
'''

import numpy as np
from typing import Tuple, Dict

from .hole import Hole


class UnloadedHole(Hole):
    '''
    Class for defining an unloaded hole in an infinite anisotropic homogeneous plate.

    This class represents an infinite anisotropic plate with a unfilled circular hole
    loaded at infinity with stresses in the x, y and xy (shear) directions.

    Parameters
    ----------
    sigma_xx_inf : float
        applied stress in the x-direction at infinity
    sigma_yy_inf : float
        applied stress in the y-direction at infinity
    tau_xy_inf : float
        applied shear stress at infinity
    radius : float
        hole radius
    compliance_matrix : np.ndarray of shape (3, 3)
        compliance matrix of the anisotropic material
    '''
    def __init__(self, sigma_xx_inf: float, sigma_yy_inf: float, tau_xy_inf: float,
                radius: float, compliance_matrix: np.ndarray) -> None:
        
        super().__init__(radius, compliance_matrix)
        
        self.sigma_xx_inf = sigma_xx_inf
        self.sigma_yy_inf = sigma_yy_inf
        self.tau_xy_inf = tau_xy_inf
        
        self.C1, self.C2 = self.calculate_potential_function_coefficients()
        
    def calculate_potential_function_coefficients(self) -> Tuple[complex, complex]:
        '''
        Calculates the coefficients of the potential functions, C1 and C2.
        
        The coefficients only depend on the applied stresses at infinity
        and the material properties.

        Returns
        -------
        C1 : complex
            coefficient of the first potential function
        C2 : complex
            coefficient of the second potential function
        '''
        alpha = 1j * self.tau_xy_inf * self.radius / 2 - self.sigma_yy_inf * self.radius / 2
        beta = self.tau_xy_inf * self.radius / 2 - 1j * self.sigma_xx_inf * self.radius / 2
        dd = self.mu1 - self.mu2
        C1 = (beta - self.mu2 * alpha) / dd
        C2 = -(beta - self.mu1 * alpha) / dd
        
        return C1, C2
        
    def phi_1(self, z1: np.ndarray) -> np.ndarray:
        '''
        Calculates the first stress function

        Parameters
        ----------
        z1 : np.ndarray
            1D complex array first mapping parameter

        Returns
        -------
        np.ndarray
            1D complex array
        '''
        xi_1, sign_1 = self.xi_1(z1)

        return self.C1 / xi_1

    def phi_2(self, z2: np.ndarray) -> np.ndarray:
        '''
        Calculates the second stress function

        Parameters
        ----------
        z2 : np.ndarray
            1D complex array second mapping parameter

        Returns
        -------
        np.ndarray
            1D complex array
        '''
        xi_2, sign_2 = self.xi_2(z2)

        return self.C2 / xi_2

    def phi_1_prime(self, z1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calculates derivative of the first stress function

        Parameters
        ----------
        z1 : np.ndarray
            1D complex array first mapping parameter

        Returns
        -------
        phi_1_p: np.ndarray
            derivative of the first stress function (complex array)
        sign_1 : np.ndarray
            sign of the first mapping parameter (xi_1)
        '''
        a = self.semi_axis_x
        b = self.semi_axis_y
        
        xi_1, sign_1 = self.xi_1(z1)

        eta1 = sign_1 * np.sqrt(z1 * z1 - a * a - self.mu1 * self.mu1 * b * b)
        kappa1 = 1 / (a - 1j * self.mu1 * b)

        return -self.C1 / (xi_1 ** 2) * (1 + z1 / eta1) * kappa1, sign_1

    def phi_2_prime(self, z2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calculates derivative of the second stress function
        
        Parameters
        ----------
        z2 : np.ndarray
            1D complex array second mapping parameter

        Returns
        -------
        phi_2_p: np.ndarray
            derivative of the second stress function (complex array)
        sign_2 : np.ndarray
            sign of the second mapping parameter (xi_2)
        '''
        a = self.semi_axis_x
        b = self.semi_axis_y
        
        xi_2, sign_2 = self.xi_2(z2)

        eta2 = sign_2 * np.sqrt(z2 * z2 - a * a - self.mu2 * self.mu2 * b * b)
        kappa2 = 1 / (a - 1j * self.mu2 * b)

        return -self.C2 / (xi_2 ** 2) * (1 + z2 / eta2) * kappa2, sign_2

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
            (n, 3) array of in-plane stress components in the cartesian coordinate system
            [[sx0, sy0, sxy0], [sx1, sy1, sxy1], ... , [sxn, syn, sxyn]]
            (n, 3) in-plane stress components in the cartesian coordinate system
        '''
        sx, sy, sxy = super().stress(x, y).T
        
        _stress = np.array([
            sx + self.sigma_xx_inf,
            sy + self.sigma_yy_inf,
            sxy + self.tau_xy_inf]).T
        
        return _stress

    def calculate_field_results(self, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        '''
        Calculates the stress at (x, y) points in the plate
        
        Parameters
        ----------
        x, y : np.ndarray of shape (n,)
            x and y locations in the cartesian coordinate system
            
        Returns
        -------
        field: Dict[str, np.ndarray]
            Same keys as `Hole.calculate_field_results`; stresses include the
            remote field, and strains are `S @ sigma` for that total stress.
        '''
        field = super().calculate_field_results(x, y)

        field['sigma_x'] += self.sigma_xx_inf
        field['sigma_y'] += self.sigma_yy_inf
        field['tau_xy'] += self.tau_xy_inf

        stress_stack = np.stack(
            [field['sigma_x'], field['sigma_y'], field['tau_xy']], axis=0)
        strain_stack = self.s @ stress_stack
        field['epsilon_x'] = strain_stack[0]
        field['epsilon_y'] = strain_stack[1]
        field['gamma_xy'] = strain_stack[2]

        return field

