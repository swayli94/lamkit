'''
Utility functions.
'''

import numpy as np
import pandas as pd
from lamkit.analysis.larc05 import LaRC05
from lamkit.lekhnitskii.unloaded_hole import UnloadedHole


def evaluate_laminate_failure_field(stress_field: pd.DataFrame) -> pd.DataFrame:
    '''
    Evaluate the failure field of the laminate.
    
    Parameters
    ----------
    stress_field: pd.DataFrame
        Stress field of the laminate, ply by ply in plate direction and material direction.
        Including the following columns:
        [ply, position, angle, sigmax, sigmay, tauxy, sigma1, sigma2, tau12]
        
    Returns
    -------
    failure_field: pd.DataFrame
        Failure indices of the laminate plies, including the following columns:
        [ply, position, angle, FI_max, FI_matrix_cracking, FI_matrix_splitting, FI_fibre_tension, FI_fibre_kinking, FI_matrix_interface]
    '''
    failure_field = pd.DataFrame(
        columns=['ply', 'position', 'angle', 
                    'FI_max', 'FI_matrix_cracking', 'FI_matrix_splitting', 
                    'FI_fibre_tension', 'FI_fibre_kinking', 'FI_matrix_interface'],
        )
    
    larc05 = LaRC05(nSCply=3) # 2D element, 3 stress components
    
    for index, row in stress_field.iterrows():
        
        ply_stresses = np.array([row['sigma1'], row['sigma2'], row['tau12']])
        failure_indices = larc05.get_uvarm(ply_stresses)
        
        failure_field.at[index, 'ply'] = row['ply']
        failure_field.at[index, 'position'] = row['position']
        failure_field.at[index, 'angle'] = row['angle']
        failure_field.at[index, 'FI_max'] = failure_indices[5]
        failure_field.at[index, 'FI_matrix_cracking'] = failure_indices[0]
        failure_field.at[index, 'FI_matrix_splitting'] = failure_indices[1]
        failure_field.at[index, 'FI_fibre_tension'] = failure_indices[2]
        failure_field.at[index, 'FI_fibre_kinking'] = failure_indices[3]
        failure_field.at[index, 'FI_matrix_interface'] = failure_indices[4]
    
    return failure_field


def evaluate_unloaded_hole_stress_field(
        sigma_xx_inf: float, sigma_yy_inf: float, tau_xy_inf: float,
        hole_radius: float, compliance_matrix: np.ndarray,
        x: float|np.ndarray, y: float|np.ndarray,
        ) -> tuple[float|np.ndarray, float|np.ndarray, float|np.ndarray]:
    '''
    Calculate the stress field around a circular hole in an infinite elastic plate
    subjected to general two-dimensional (2-D) loading.
    
    Parameters
    ----------
    sigma_xx_inf : float
        applied stress in the x-direction at infinity
    sigma_yy_inf : float
        applied stress in the y-direction at infinity
    tau_xy_inf : float
        applied shear stress at infinity
    hole_radius : float
        hole radius
    compliance_matrix : np.ndarray of shape (3, 3)
        compliance matrix of the anisotropic material
    x : float|np.ndarray
        x locations in the cartesian coordinate system
    y : float|np.ndarray
        y locations in the cartesian coordinate system
        
    Returns
    -------
    sigma_xx : float|np.ndarray
        stress in the x-direction
    sigma_yy : float|np.ndarray
        stress in the y-direction
    tau_xy : float|np.ndarray
        shear stress
    '''
    solution = UnloadedHole(sigma_xx_inf, sigma_yy_inf, tau_xy_inf,
                            radius=hole_radius,
                            compliance_matrix=compliance_matrix)
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out_shape = x.shape
    x_flat = np.atleast_1d(x).ravel()
    y_flat = np.atleast_1d(y).ravel()

    stress_field = solution.stress(x_flat, y_flat)

    sigma_xx = stress_field[:, 0].reshape(out_shape)
    sigma_yy = stress_field[:, 1].reshape(out_shape)
    tau_xy = stress_field[:, 2].reshape(out_shape)

    return sigma_xx, sigma_yy, tau_xy



