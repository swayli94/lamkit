'''
Utility functions.
'''

from typing import Dict, Any, Tuple, List
import numpy as np
from lamkit.analysis.larc05 import LaRC05
from lamkit.analysis.material import Ply, Material
from lamkit.analysis.laminate import Laminate
from lamkit.lekhnitskii.unloaded_hole import UnloadedHole


def midplane_stresses_unloaded_hole_plate(
        sigma_xx_inf: float,
        sigma_yy_inf: float,
        tau_xy_inf: float,
        hole_radius: float,
        compliance_matrix: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Mid-plane stress field (sigma_x, sigma_y, tau_xy) for a homogeneous plate
    with given plane-stress compliance and Lekhnitskii unloaded-hole solution.

    For laminate CLT + ply-level LaRC05 fields, use ``evaluate_unloaded_hole_plate``.
    '''
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out_shape = x.shape
    x_flat = np.atleast_1d(x).ravel()
    y_flat = np.atleast_1d(y).ravel()
    solution = UnloadedHole(
        sigma_xx_inf, sigma_yy_inf, tau_xy_inf,
        radius=hole_radius,
        compliance_matrix=compliance_matrix,
    )
    field = solution.calculate_field_results(x_flat, y_flat, out_shape)
    sigma_x = field['sigma_x']
    sigma_y = field['sigma_y']
    tau_xy = field['tau_xy']
    return sigma_x, sigma_y, tau_xy


def evaluate_unloaded_hole_plate(
        laminate: Laminate, hole_radius: float,
        sigma_xx_inf: float, sigma_yy_inf: float, tau_xy_inf: float,
        x: np.ndarray, y: np.ndarray,
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    '''
    Calculate the stress field around a circular hole in an infinite elastic plate
    subjected to general two-dimensional (2-D) loading.
    
    Parameters
    ----------
    laminate: Laminate
        Laminate object (units: MPa, mm)
    sigma_xx_inf : float
        applied stress in the x-direction at infinity
    sigma_yy_inf : float
        applied stress in the y-direction at infinity
    tau_xy_inf : float
        applied shear stress at infinity
    hole_radius : float
        hole radius
    x : np.ndarray
        x locations in the cartesian coordinate system
    y : np.ndarray
        y locations in the cartesian coordinate system
        
    Returns
    -------
    results_by_plies: List[Dict[str, Any]]
        List of dictionaries, each containing the results for a ply.
        Length is `2*n_ply`.
    mid_plane_field: Dict[str, Any]
        Dictionary containing the results for the mid-plane.
    '''

    # Meshgrid
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out_shape = x.shape
    x_flat = np.atleast_1d(x).ravel()
    y_flat = np.atleast_1d(y).ravel()
    n_points = x_flat.shape[0]

    # Failure analysis
    larc05 = LaRC05(nSCply=3, material_properties=laminate.ply_material.properties_dictionary)

    # Unloaded hole solution for mid-plane strains
    solution = UnloadedHole(sigma_xx_inf, sigma_yy_inf, tau_xy_inf,
                            radius=hole_radius,
                            compliance_matrix=laminate.in_plane_compliance_matrix)
    
    mid_plane_field = solution.calculate_field_results(x_flat, y_flat, out_shape)
    
    epsilon_x = mid_plane_field['epsilon_x'].ravel() # (n_points,)
    epsilon_y = mid_plane_field['epsilon_y'].ravel() # (n_points,)
    gamma_xy = mid_plane_field['gamma_xy'].ravel() # (n_points,)
    zeros_kappa = np.zeros_like(x_flat) # (n_points,)
    epsilon0 = np.stack(
        [
            epsilon_x, epsilon_y, gamma_xy,
            zeros_kappa, zeros_kappa, zeros_kappa,
        ],
        axis=1,
    )  # (n_points, 6)

    # Ply-level field dictionary (failure_mode is string; stored separately)
    NUMERIC_KEYS = [
        'sigma_x', 'sigma_y', 'tau_xy', 'sigma_1', 'sigma_2', 'tau_12',
        'epsilon_x', 'epsilon_y', 'gamma_xy', 'epsilon_1', 'epsilon_2', 'gamma_12',
        'FI_matrix_cracking', 'FI_matrix_splitting', 'FI_fibre_tension', 'FI_fibre_kinking', 'FI_matrix_interface', 'FI_max',
    ]

    def _create_dictionary_for_one_ply(
        index_ply: int, index_surface: int,
        z_eval: float, theta: float,
        ) -> Dict[str, Any]:
        '''
        Create a dictionary for one ply.
        '''
        out = {
            'index_ply': index_ply,
            'index_surface': index_surface,
            'z': z_eval,
            'angle': theta}

        for key in NUMERIC_KEYS:
            out[key] = np.zeros(n_points)
        out['failure_mode'] = np.empty(n_points, dtype=object)

        return out
    
    # Ply-level stress/strain/failure field
    z_pos = laminate.z_position

    results_by_plies = []
    for index_ply in range(laminate.n_ply):
        
        z_bottom = z_pos[index_ply]
        z_top = z_pos[index_ply + 1]
        theta, ply_obj = laminate.layup[index_ply]
        theta = float(theta)
        
        for index_surface, z_eval in ((0, z_bottom), (1, z_top)):
            results_by_plies.append(
                _create_dictionary_for_one_ply(
                    index_ply, index_surface, z_eval, theta))

    for i in range(n_points):
        
        results_one_point = laminate.get_ply_level_results(epsilon0[i, :], larc05) # [2*n_ply]

        for index_ply in range(laminate.n_ply):
            zb = z_pos[index_ply]
            zt = z_pos[index_ply + 1]
            for index_surface, _z_eval in ((0, zb), (1, zt)):
                ii = 2 * index_ply + index_surface
                _result_ply = results_by_plies[ii]
                _result_point = results_one_point[ii]
                
                for key in NUMERIC_KEYS:
                    _result_ply[key][i] = _result_point[key]
                _result_ply['failure_mode'][i] = _result_point['failure_mode']

    # Reshape the results to the original shape
    for ply in results_by_plies:
        for key in NUMERIC_KEYS:
            ply[key] = ply[key].reshape(out_shape)
        ply['failure_mode'] = ply['failure_mode'].reshape(out_shape)
        
    return results_by_plies, mid_plane_field


def create_effective_laminate_for_buckling_analysis(
    E11: float, E22: float, G12: float, nu12: float,
    total_thickness: float,
    ) -> Laminate:
    '''
    Create an effective unidirectional laminate with given properties.
    '''
    material = Material(name='Homogenised',
                properties={'E11': E11, 'E22': E22, 'G12': G12, 'nu12': nu12},
                check_larc05=False)
    ply = Ply(material=material, thickness=total_thickness)
    
    laminate = Laminate(stacking=[0.0], plies=[ply])
    return laminate


'''
Provide access to the homogenisation functions.
'''
from lamkit.lekhnitskii.homogenisation import (
    compute_permutation_invariants,
    compute_effective_strains as compute_effective_strains_hole_plate,
    compute_homogenised_properties as compute_homogenised_properties_hole_plate,
)

__all__ = [
    'evaluate_unloaded_hole_plate',
    'compute_permutation_invariants',
    'compute_effective_strains_hole_plate',
    'compute_homogenised_properties_hole_plate',
]