"""
Example: Effective stiffness of a laminate with an open hole.

- Define a laminate with an open hole, radius `r`.
- Assume a square plate with side length `w`, and the hole is at the center.
- Use the Lekhnitskii unloaded-hole solution for an infinite plate.
- Calculate the effective stiffness of the laminate without holes `A_lam`, which is the original `A` matrix.
- Calculate the effective stiffness of the laminate with the hole `A_eff`:
    - Subject the plate to 3 unit far-field stress states:
    `[sigma_xx_inf, sigma_yy_inf, tau_xy_inf] = [1,0,0], [0,1,0], [0,0,1]`.
    - Get the displacement at the plate boundaries, i.e., `u, v`.
    - Calculate the effective (homogenized) strains by integrating the boundary displacements,
    using `compute_effective_strains_from_boundary_displacement`.
    - Calculate the effective compliance matrix `S_eff` from the effective strains and the unit stress states.
    - Calculate the effective stiffness of the laminate with a hole, i.e., `A_eff = inv(S_eff)`.
- Compare the two results, `A_eff` and `A_lam`:
    - the difference should decrease as the `r/w` ratio decreases.

"""

from __future__ import annotations

import os
import sys

import numpy as np
from typing import Dict, Any

import matplotlib

# Use a non-interactive backend so this example also works on headless machines.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(path, "..", "..", "src"))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from lamkit.analysis.material import IM7_8551_7, Ply
from lamkit.analysis.laminate import Laminate
from lamkit.lekhnitskii.unloaded_hole import UnloadedHole
from lamkit.lekhnitskii.homogenisation import compute_homogenised_properties


def calculate_effective_properties(laminate: Laminate,
            lx_plate_mm: float, ly_plate_mm: float, hole_radius_mm: float,
            n_points_boundary: int = 101) -> Dict[str, Any]:
    '''
    Calculate the effective properties of a laminate with an open hole.
    
    Parameters
    ----------
    laminate: Laminate
        The laminate to calculate the effective properties of.
    lx_plate_mm: float
        The length of the plate in the x-direction (mm).
    ly_plate_mm: float
        The length of the plate in the y-direction (mm).
    hole_radius_mm: float
        The radius of the hole (mm).
    n_points_boundary: int
        The number of points to sample on the boundary.
        
    Returns
    -------
    properties: Dict[str, Any]
        The effective properties of the laminate '_lam' and plate '_eff'.
        Keys: 'total_thickness', 'A_*', 'S_*', 'E11_*', 'E22_*', 'G12_*', 'nu12_*'.
    '''
    # --- Laminate reference (no hole) ---
    A_lam = np.asarray(laminate.A, dtype=float)  # 3x3
    S_lam = np.asarray(laminate.in_plane_compliance_matrix, dtype=float)  # 3x3, 1/MPa
    h_mm = float(sum(p.thickness for p in laminate.plies))

    properties_lam = laminate.get_effective_properties()

    # --- With hole: effective compliance from boundary strains ---
    properties_eff = compute_homogenised_properties(
        HoleType=UnloadedHole,
        L=lx_plate_mm,
        H=ly_plate_mm,
        plate_thickness=h_mm,
        hole_radius=hole_radius_mm,
        compliance_matrix=S_lam,
        n_points_boundary=n_points_boundary,
    )
    
    results = {
        'total_thickness': h_mm,
        'layup': laminate.stacking_sequence,
        'lx_plate_mm': lx_plate_mm,
        'ly_plate_mm': ly_plate_mm,
        'hole_radius_mm': hole_radius_mm,
        'A_lam': A_lam,
        'A_eff': properties_eff['A_eff'],
        'S_lam': S_lam,
        'S_eff': properties_eff['S_eff'],
    }
    
    for key in ['E11', 'E22', 'G12', 'nu12']:
        results[f'{key}_eff'] = properties_eff[f'{key}_eff']
        results[f'{key}_lam'] = properties_lam[f'{key}_eff']
    
    return results


def print_comparison(results: Dict[str, Any]) -> None:
    '''
    Print the comparison of the effective properties of the laminate with and without the hole.
    '''
    np.set_printoptions(formatter={'float_kind': '{:10.3e}'.format}, suppress=True)
    
    print("\n=== Effective stiffness (with vs without hole) ===\n")
    print(f"Layup (deg): {results['layup']}")
    print(f"Plate size:  {results['lx_plate_mm']:.2f} mm x {results['ly_plate_mm']:.2f} mm")
    print(f"Hole radius: {results['hole_radius_mm']:.2f} mm")
    print(f"Plate thickness: {results['total_thickness']:.2f} mm")

    print("\n--- Laminate reference ---\n")
    print("In-plane stiffness matrix (A):")
    print(results['A_lam'])

    print("In-plane compliance matrix (S):")
    print(results['S_lam'])

    print("\n--- Open hole plate ---\n")
    print("Effective stiffness matrix (A_eff):")
    print(results['A_eff'])
    
    print("Effective compliance matrix (S_eff):")
    print(results['S_eff'])

    # Error indicators (Frobenius norm).
    frob_err = np.linalg.norm(results['A_eff'] - results['A_lam'])
    err_A = frob_err / (np.linalg.norm(results['A_lam']) + 1e-30)
    
    frob_err_S = np.linalg.norm(results['S_eff'] - results['S_lam'])
    err_S = frob_err_S / (np.linalg.norm(results['S_lam']) + 1e-30)

    print("\n--- Comparison metrics ---\n")
    print(f"Stiffness matrix:  relative Frobenius error = {err_A:.3e}")
    print(f"Compliance matrix: relative Frobenius error = {err_S:.3e}")
    
    for key in ['E11', 'E22', 'G12', 'nu12']:
        ori = results[f'{key}_lam']
        eff = results[f'{key}_eff']
        print(f"{key:4s}: Original= {ori:.3e}, Effective= {eff:.3e}, Relative error= {(eff - ori) / ori:5.2f}")
    
    return None


def main() -> None:
    
    # Example laminate (quasi-isotropic).
    layup = [45, -45, 0, -45, 45, 0, 0, 90, 90, 0, 0, 45, -45, 0, -45, 45]
    ply_thickness_mm = 0.125

    ply = Ply(material=IM7_8551_7, thickness=ply_thickness_mm)
    laminate = Laminate(stacking=layup, plies=ply)

    # Plot ratio curves of effective properties vs. laminate reference.
    #   {prop}_eff / {prop}_lam, where prop in [E11, E22, G12, nu12].
    properties = ["E11", "E22", "G12", "nu12"]

    r_list = [1, 10]  # mm
    w_r_ratios = [2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4, 5] + [i for i in range(6, 21, 2)]  # w/r ratio

    # ratios[prop][i_r, i_wr] = prop_eff / prop_lam
    ratios: Dict[str, np.ndarray] = {
        prop: np.zeros((len(r_list), len(w_r_ratios)), dtype=float)
        for prop in properties
    }

    for i_r, r_mm in enumerate(r_list):
        for i_wr, w_r_ratio in enumerate(w_r_ratios):
            w_plate_mm = float(w_r_ratio) * float(r_mm)

            results = calculate_effective_properties(
                laminate=laminate,
                lx_plate_mm=w_plate_mm,
                ly_plate_mm=w_plate_mm,
                hole_radius_mm=float(r_mm),
                n_points_boundary=101,
            )

            print_comparison(results)

            for prop in properties:
                lam_val = float(results[f"{prop}_lam"])
                eff_val = float(results[f"{prop}_eff"])
                ratios[prop][i_r, i_wr] = eff_val / (lam_val + 1e-30)  # avoid division-by-zero

    # --- Plot (2x2 subplots) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes_flat = axes.ravel()

    for ax, prop in zip(axes_flat, properties):
        for i_r, r_mm in enumerate(r_list):
            ax.plot(
                w_r_ratios,
                ratios[prop][i_r, :],
                marker="o",
                linewidth=1.8,
                label=f"r={r_mm} mm",
            )

        ax.set_title(prop)
        ax.set_xlabel("w/r")
        ax.set_ylabel(f"Ratio of {prop}")
        ax.set_xlim(2, 21)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")

    fig.suptitle(
        f"Open Hole Plate: Ratio of homogenised properties to laminate reference\n"+ \
        f"Layup: {layup}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 1])

    out_path = os.path.join(path, "images", "open_hole_homogenisation.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot: {out_path}")




if __name__ == "__main__":

    os.makedirs(os.path.join(path, "images"), exist_ok=True)
    
    main()
