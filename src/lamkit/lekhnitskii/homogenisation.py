'''
Homogenisation of a plate with a circular hole.
'''


import numpy as np
from typing import Dict, Type, Any
from lamkit.lekhnitskii.hole import Hole
from lamkit.lekhnitskii.unloaded_hole import UnloadedHole


def compute_effective_strains(solution: Hole,
                L: float, H: float, n_points_boundary: int = 301) -> np.ndarray:
    """
    Compute effective strains from boundary displacements.
    The plate is treated as a square domain:
    x in [-L/2, +L/2], y in [-H/2, +H/2]

    Parameters
    ----------
    solution: Hole
        The solution of the hole problem.
        Material properties are described in MPa and mm.
    L: float
        The side length (mm) of the square plate in the x-direction.
    H: float
        The side length (mm) of the square plate in the y-direction.
    n_points_boundary: int
        The number of samples to take on the boundary.

    Returns
    -------
    effective_strains: np.ndarray, shape (3,)
        [epsilon_x, epsilon_y, gamma_xy] (engineering gamma_xy).
    """
    if n_points_boundary < 3:
        raise ValueError("n_points_boundary must be >= 3")

    half_x = 0.5 * L
    half_y = 0.5 * H

    x_left = -half_x
    x_right = +half_x
    y_bottom = -half_y
    y_top = +half_y

    # --- epsilon_xx_bar = (1/(L*H)) [ ∫ u(x=+L/2,y) dy - ∫ u(x=-L/2,y) dy ] ---
    y = np.linspace(y_bottom, y_top, int(n_points_boundary), dtype=float)
    x_r = np.full_like(y, x_right, dtype=float)
    x_l = np.full_like(y, x_left, dtype=float)

    disp_r = np.asarray(solution.displacement(x_r, y), dtype=float)
    disp_l = np.asarray(solution.displacement(x_l, y), dtype=float)
    u_r = disp_r[:, 0]
    u_l = disp_l[:, 0]

    int_u_r = float(np.trapz(u_r, y))
    int_u_l = float(np.trapz(u_l, y))
    epsilon_xx_bar = (int_u_r - int_u_l) / (L * H)

    # --- epsilon_yy_bar = (1/(L*H)) [ ∫ v(x,y=+H/2) dx - ∫ v(x,y=-H/2) dx ] ---
    x = np.linspace(x_left, x_right, int(n_points_boundary), dtype=float)
    y_t = np.full_like(x, y_top, dtype=float)
    y_b = np.full_like(x, y_bottom, dtype=float)

    disp_t = np.asarray(solution.displacement(x, y_t), dtype=float)
    disp_b = np.asarray(solution.displacement(x, y_b), dtype=float)
    v_t = disp_t[:, 1]
    v_b = disp_b[:, 1]

    int_v_t = float(np.trapz(v_t, x))
    int_v_b = float(np.trapz(v_b, x))
    epsilon_yy_bar = (int_v_t - int_v_b) / (L * H)

    # --- gamma_xy_bar = (1/(L*H)) [ ∫ (u(x,H)-u(x,0)) dx + ∫ (v(L,y)-v(0,y)) dy ] ---
    # Our coordinates: H => y_top, 0 => y_bottom; L => x_right, 0 => x_left.
    disp_u_t = np.asarray(solution.displacement(x, y_t), dtype=float)
    disp_u_b = np.asarray(solution.displacement(x, y_b), dtype=float)
    u_t = disp_u_t[:, 0]
    u_b = disp_u_b[:, 0]
    term1 = float(np.trapz(u_t - u_b, x))

    disp_v_r = np.asarray(solution.displacement(x_r, y), dtype=float)
    disp_v_l = np.asarray(solution.displacement(x_l, y), dtype=float)
    v_r = disp_v_r[:, 1]
    v_l = disp_v_l[:, 1]
    term2 = float(np.trapz(v_r - v_l, y))

    gamma_xy_bar = (term1 + term2) / (L * H)

    return np.array([epsilon_xx_bar, epsilon_yy_bar, gamma_xy_bar], dtype=float)


def compute_permutation_invariants(E11: float, E22: float,
            G12: float, nu12: float) -> Dict[str, float]:
    '''
    Compute the permutation invariants of the homogenised properties.

    Parameters
    ----------
    E11: float
        The Young's modulus in the 1-direction.
    E22: float
        The Young's modulus in the 2-direction.
    G12: float
        The shear modulus.
    nu12: float
        The Poisson's ratio in the 1-2 plane.
        
    Returns
    -------
    invariants: Dict[str, float]
        The permutation invariants of the material properties.
        Keys: 'delta', 'p0', 'p1', 'p2', 'p3', 'log(1+p3)', 'G12'.
    '''
    delta = E11/G12 - 2*(nu12+np.sqrt(E11/E22))
    p0 = np.sqrt(E11*E22)
    p1 = p0/G12
    p2 = nu12*np.sqrt(E22/E11)
    p3 = delta*np.sqrt(E22/E11)
    log_1_p3 = np.log(1+p3)
    
    invariants = {
        'delta': delta,
        'p0': p0,
        'p1': p1,
        'p2': p2,
        'p3': p3,
        'log(1+p3)': log_1_p3,
        'G12': G12,
    }
    
    return invariants


def compute_homogenised_properties(HoleType: Type[Hole],
                    L: float, H: float, plate_thickness: float,
                    hole_radius: float, compliance_matrix: np.ndarray,
                    n_points_boundary: int = 301) -> Dict[str, Any]:
    '''
    Compute the homogenised properties of a plate with a circular hole.

    Parameters
    ----------
    HoleType: Type[Hole]
        The type of hole problem to solve.
    L: float
        The side length (mm) of the square plate in the x-direction.
    H: float
        The side length (mm) of the square plate in the y-direction.
    plate_thickness: float
        The thickness (mm) of the plate.
    hole_radius: float
        The radius (mm) of the circular hole.
    compliance_matrix: np.ndarray, shape (3, 3)
        The compliance matrix of the anisotropic plate.
    n_points_boundary: int
        The number of samples to take on the boundary.

    Returns
    -------
    homogenised_properties: Dict[str, Any]
        The homogenised properties of the plate.
        Keys: 'A_eff', 'S_eff', 'E11_eff', 'E22_eff', 'G12_eff', 'nu12_eff', 'nu21_eff'.
    '''

    unit_stress_vectors = np.eye(3)
    eps_cols: list[np.ndarray] = []
    
    for (sx_inf, sy_inf, txy_inf) in unit_stress_vectors:

        if HoleType == UnloadedHole:
            solution = UnloadedHole(
                sigma_xx_inf=sx_inf,
                sigma_yy_inf=sy_inf,
                tau_xy_inf=txy_inf,
                radius=hole_radius,
                compliance_matrix=compliance_matrix,
            )
        else:
            raise ValueError(f"HoleType {HoleType} not supported")

        eps_eff = compute_effective_strains(
            solution=solution, L=L, H=H, n_points_boundary=n_points_boundary)
        eps_cols.append(eps_eff)

    # epsilon = S_eff @ sigma_inf, with sigma_inf ordering [sx, sy, txy]
    # No further scaling by thickness is needed for `S_eff`,
    # as the stress is already thickness-averaged.
    S_eff = np.column_stack(eps_cols) 
    
    # `S_eff` maps in-plane stress -> (thickness-averaged) strain: epsilon = S @ sigma.
    # CLT extensional stiffness uses N = A * epsilon with stress defined as sigma_bar = N/h.
    # Therefore A = h * inv(S).
    A_eff = np.linalg.inv(S_eff)
    A_eff = A_eff * plate_thickness  # scale by thickness to get CLT [A].
    
    # Homogenised material properties:
    E11_eff = 1/S_eff[0, 0]
    E22_eff = 1/S_eff[1, 1]
    G12_eff = 1/S_eff[2, 2]
    nu12_eff = -S_eff[0, 1] / S_eff[0, 0]
    nu21_eff = -S_eff[1, 0] / S_eff[1, 1]

    results = {
        'A_eff': A_eff,
        'S_eff': S_eff,
        'E11_eff': E11_eff,
        'E22_eff': E22_eff,
        'G12_eff': G12_eff,
        'nu12_eff': nu12_eff,
        'nu21_eff': nu21_eff,
    }

    return results
