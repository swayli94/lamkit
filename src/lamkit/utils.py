'''
Utility functions.
'''

from typing import Dict, Any, Tuple, List
import numpy as np
from lamkit.analysis.larc05 import LaRC05, FAILURE_MODE_NAMES
from lamkit.analysis.material import Ply, Material
from lamkit.analysis.laminate import Laminate
from lamkit.lekhnitskii.combined_load import CombinedLoadHole


NUMERIC_KEYS = [
    'sigma_x', 'sigma_y', 'tau_xy', 'sigma_1', 'sigma_2', 'tau_12',
    'epsilon_x', 'epsilon_y', 'gamma_xy', 'epsilon_1', 'epsilon_2', 'gamma_12',
]

LARC05_KEYS = [
    'FI_matrix_cracking', 'FI_matrix_splitting', 'FI_fibre_tension',
    'FI_fibre_kinking', 'FI_matrix_interface']


def create_effective_laminate_for_buckling_analysis(
    E11: float, E22: float, G12: float, nu12: float,
    total_thickness: float) -> Laminate:
    '''
    Create an effective unidirectional laminate with given properties.
    '''
    material = Material(name='Homogenised',
                properties={'E11': E11, 'E22': E22, 'G12': G12, 'nu12': nu12},
                check_larc05=False)
    ply = Ply(material=material, thickness=total_thickness)
    
    laminate = Laminate(stacking=[0.0], plies=[ply])
    return laminate


def midplane_stresses_combined_load(
        sigma_xx_inf: float, sigma_yy_inf: float, tau_xy_inf: float,
        load: float, angle_load_degree: float,
        hole_radius: float, thickness: float,
        compliance_matrix: np.ndarray,
        x: np.ndarray, y: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Stress field (sigma_x, sigma_y, tau_xy) for a homogeneous anisotropic plate
    with given plane-stress compliance.
    '''
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out_shape = x.shape
    x_flat = np.atleast_1d(x).ravel()
    y_flat = np.atleast_1d(y).ravel()
    solution = CombinedLoadHole(
        sigma_xx_inf, sigma_yy_inf, tau_xy_inf,
        load=load,
        theta_degree=angle_load_degree,
        radius=hole_radius,
        thickness=thickness,
        compliance_matrix=compliance_matrix,
    )
    field = solution.calculate_field_results(x_flat, y_flat, out_shape)
    sigma_x = field['sigma_x']
    sigma_y = field['sigma_y']
    tau_xy = field['tau_xy']
    return sigma_x, sigma_y, tau_xy


def evaluate_combined_load_plate(
        laminate: Laminate,
        sigma_xx_inf: float, sigma_yy_inf: float, tau_xy_inf: float,
        load: float, angle_load_degree: float,
        hole_radius: float, thickness: float,
        x: np.ndarray, y: np.ndarray,
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    '''
    Calculate the stress field around a hole under combined bypass and bearing loading.

    Parameters
    ----------
    laminate : Laminate
        Laminate object (units: MPa, mm).
    sigma_xx_inf, sigma_yy_inf, tau_xy_inf : float
        Bypass stresses applied at infinity (MPa).
    load : float
        Total bearing force (N).
    angle_load_degree : float
        Bearing-load angle counter-clockwise from the positive x-axis (degrees).    
    hole_radius : float
        Hole radius (mm).
    thickness : float
        Plate thickness (mm).
    x, y : np.ndarray
        Point coordinates in the Cartesian system.

    Returns
    -------
    results_by_plies : List[Dict[str, Any]]
        Length (2*n_ply), containing results for bottom and top surfaces of every ply.
    mid_plane_field : Dict[str, Any]
        Mid-plane stress and strain fields.
    '''
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out_shape = x.shape
    x_flat = np.atleast_1d(x).ravel()
    y_flat = np.atleast_1d(y).ravel()
    n_points = x_flat.shape[0]

    compliance = laminate.in_plane_compliance_matrix

    #* Mid-plane stresses and strains from the combined load solution.
    solution = CombinedLoadHole(
        sigma_xx_inf=sigma_xx_inf,
        sigma_yy_inf=sigma_yy_inf,
        tau_xy_inf=tau_xy_inf,
        load=load,
        theta_degree=angle_load_degree,
        radius=hole_radius,
        thickness=thickness,
        compliance_matrix=compliance,
    )

    stresses = solution.stress(x_flat, y_flat)  # (n, 3)
    strains = solution.get_strain_from_stress(stresses)  # (n, 3)
    mid_plane_field = {
        'sigma_x': stresses[:, 0].reshape(out_shape),
        'sigma_y': stresses[:, 1].reshape(out_shape),
        'tau_xy': stresses[:, 2].reshape(out_shape),
        'epsilon_x': strains[:, 0].reshape(out_shape),
        'epsilon_y': strains[:, 1].reshape(out_shape),
        'gamma_xy': strains[:, 2].reshape(out_shape),
    }

    epsilon0 = np.hstack([strains, np.zeros((n_points, 3))])  # (n_points, 6)

    def _create_dictionary_for_one_ply(
        index_ply: int, index_surface: int,
        z_eval: float, theta_ply: float,
    ) -> Dict[str, Any]:
        out = {
            'index_ply': index_ply,
            'index_surface': index_surface,
            'z': z_eval,
            'angle': theta_ply,
        }
        for key in NUMERIC_KEYS:
            out[key] = np.zeros(n_points)
        return out

    z_pos = laminate.z_position
    results_by_plies = []
    for index_ply in range(laminate.n_ply):
        z_bottom = z_pos[index_ply]
        z_top = z_pos[index_ply + 1]
        theta_ply, ply_obj = laminate.layup[index_ply]
        theta_ply = float(theta_ply)
        for index_surface, z_eval in ((0, z_bottom), (1, z_top)):
            results_by_plies.append(
                _create_dictionary_for_one_ply(index_ply, index_surface, z_eval, theta_ply))

    for i in range(n_points):
        results_one_point = laminate.get_ply_level_results(epsilon0[i, :])
        for ii in range(2 * laminate.n_ply):
            for key in NUMERIC_KEYS:
                results_by_plies[ii][key][i] = results_one_point[ii][key]

    for ply in results_by_plies:
        for key in NUMERIC_KEYS:
            ply[key] = ply[key].reshape(out_shape)

    return results_by_plies, mid_plane_field


def evaluate_larc05_from_results(
        results_by_plies: List[Dict[str, Any]],
        properties_dictionary: Dict[str, Any]) -> List[Dict[str, Any]]:
    '''
    Evaluate LaRC05 failure indices from the ply-level results.
    
    Parameters
    ----------
    results_by_plies : List[Dict[str, Any]]
        List of dictionaries, each containing the ply-level results.
    properties_dictionary : Dict[str, Any]
        Material properties dictionary for the laminate's ply material,
        accessed by `laminate.ply_material.properties_dictionary`.
    '''
    larc05 = LaRC05(nSCply=3, material_properties=properties_dictionary)

    for ply in results_by_plies:
        shape = ply['sigma_1'].shape
        n_points = ply['sigma_1'].size
        sigma_1 = ply['sigma_1'].ravel()
        sigma_2 = ply['sigma_2'].ravel()
        tau_12 = ply['tau_12'].ravel()

        fi_arrays = {key: np.zeros(n_points) for key in LARC05_KEYS}
        fi_max = np.zeros(n_points)
        failure_mode = np.empty(n_points, dtype=object)

        for i in range(n_points):
            uvarm = larc05.evaluate(np.array([sigma_1[i], sigma_2[i], tau_12[i]], dtype=float))
            fi_block = uvarm[:5]
            for j, key in enumerate(LARC05_KEYS):
                fi_arrays[key][i] = fi_block[j]
            fi_max[i] = float(np.max(fi_block))
            failure_mode[i] = FAILURE_MODE_NAMES[int(np.argmax(fi_block)) + 1]

        for key in LARC05_KEYS:
            ply[key] = fi_arrays[key].reshape(shape)
        ply['FI_max'] = fi_max.reshape(shape)
        ply['failure_mode'] = failure_mode.reshape(shape)

    return results_by_plies

