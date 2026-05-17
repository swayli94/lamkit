'''
Combined bearing and bypass loading in an infinite anisotropic homogeneous plate,
i.e., superposition of unloaded hole and loaded hole solutions.
'''

import numpy as np
from typing import Tuple, Dict

from lamkit.lekhnitskii.hole import Hole, ZeroField
from lamkit.lekhnitskii.unloaded_hole import UnloadedHole
from lamkit.lekhnitskii.loaded_hole import LoadedHole


class CombinedLoadHole(Hole):
    '''
    Class for defining an infinite anisotropic homogeneous plate with a hole,
    under combined bearing and bypass loading.

    The solution is a superposition of an UnloadedHole (bypass far-field stresses)
    and a LoadedHole (cosine bearing distribution) solution.

    Parameters
    ----------
    sigma_xx_inf : float
        Bypass stress (MPa) in the x-direction at infinity
    sigma_yy_inf : float
        Bypass stress (MPa) in the y-direction at infinity
    tau_xy_inf : float
        Bypass shear stress (MPa) at infinity
    load : float
        Bearing force (N)
    theta_degree : float
        Bearing angle counter-clockwise from positive x-axis (degrees)
    radius : float
        Hole radius (mm)
    thickness : float
        Plate thickness (mm)
    compliance_matrix : np.ndarray of shape (3, 3)
        In-plane compliance matrix of the anisotropic plate.
        Stiffness matrix unit is MPa, so compliance unit is 1/MPa.
    '''

    def __init__(self,
                sigma_xx_inf: float, sigma_yy_inf: float, tau_xy_inf: float,
                load: float, theta_degree: float,
                radius: float, thickness: float,
                compliance_matrix: np.ndarray) -> None:

        super().__init__(radius, compliance_matrix)

        if sigma_xx_inf == 0 and sigma_yy_inf == 0 and tau_xy_inf == 0:
            self._bypass = ZeroField()
        else:
            self._bypass = UnloadedHole(
                sigma_xx_inf, sigma_yy_inf, tau_xy_inf,
                radius=radius,
                compliance_matrix=compliance_matrix)
        
        if load == 0:
            self._bearing = ZeroField()
        else:
            self._bearing = LoadedHole(
                load=load,
                radius=radius,
                thickness=thickness,
                compliance_matrix=compliance_matrix,
                theta=np.radians(theta_degree))

    def phi_1(self, z1: np.ndarray) -> np.ndarray:
        return np.zeros_like(z1)
    
    def phi_2(self, z2: np.ndarray) -> np.ndarray:
        return np.zeros_like(z2)
    
    def phi_1_prime(self, z1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros_like(z1), np.zeros_like(z1)
    
    def phi_2_prime(self, z2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros_like(z2), np.zeros_like(z2)

    def stress(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Superposed stress from bypass and bearing solutions.

        Parameters
        ----------
        x, y : np.ndarray of shape (n,)
            x and y locations in the cartesian coordinate system

        Returns
        -------
        stresses : np.ndarray of shape (n, 3)
            [[sx, sy, sxy], ...] in the global cartesian frame
        '''
        return self._bypass.stress(x, y) + self._bearing.stress(x, y)

    def displacement(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Superposed displacement from bypass and bearing solutions.

        Parameters
        ----------
        x, y : np.ndarray of shape (n,)
            x and y locations in the cartesian coordinate system

        Returns
        -------
        displacements : np.ndarray of shape (n, 2)
            [[u, v], ...] in the global cartesian frame
        '''
        return self._bypass.displacement(x, y) + self._bearing.displacement(x, y)

    def calculate_field_results(self, x: np.ndarray, y: np.ndarray,
                    out_shape: Tuple[int, int]|None = None) -> Dict[str, np.ndarray]:
        '''
        Stress, strain, and displacement field for combined loading.

        Parameters
        ----------
        x, y : np.ndarray of shape (n,)
            x and y locations in the cartesian coordinate system
        out_shape : Tuple[int, int], optional
            Shape to which output arrays are reshaped

        Returns
        -------
        field : Dict[str, np.ndarray]
            Same keys as ``Hole.calculate_field_results``; stresses are the superposition
            of bypass (including far-field) and bearing contributions, strains are
            recomputed from the total stress via the compliance matrix.
        '''
        # Bypass field already includes far-field stress contributions
        field = self._bypass.calculate_field_results(x, y, out_shape)

        shape = field['sigma_x'].shape

        # Bearing stresses are returned in the global (unrotated) frame
        bearing_stresses = self._bearing.stress(x, y)  # (n, 3)
        for i, key in enumerate(['sigma_x', 'sigma_y', 'tau_xy']):
            field[key] += bearing_stresses[:, i].reshape(shape)

        # Recompute strains from total combined stress
        stress_stack = np.stack([
            field['sigma_x'].ravel(), field['sigma_y'].ravel(), field['tau_xy'].ravel()])
        strain_stack = self.s @ stress_stack
        field['epsilon_x'], field['epsilon_y'], field['gamma_xy'] = [
            row.reshape(shape) for row in strain_stack]

        # Add bearing displacement
        bearing_disp = self._bearing.displacement(x, y)  # (n, 2)
        for i, key in enumerate(['u', 'v']):
            field[key] += bearing_disp[:, i].reshape(shape)

        return field
