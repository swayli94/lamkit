'''
Example: Laminate optimization function.

This example demonstrates how to evaluate a laminate open hole plate,
calculating the objective and constraint values.

- Structural design task:
    - Design a laminate open hole plate, size `w = 100 mm`, `h = 100 mm`.
    - The plate is subjected to a uniform compressive load in the x-direction, `Nxx = -100 N/mm`.
    - The ply thickness `t_ply` ranges from 0.1 mm to 0.2 mm.
    - The composite material is IM7/8552-7.

- Objective: minimize the weight of a laminate, i.e., number of plies `n_ply`.
- Constraints:
    - The maximum x-direction displacement is limited to `delta_x_max = 0.05 mm`.
    - The maximum failure index is limited to `FI_max = 0.8`.
    - The minimum buckling load multiplier is limited to `lambda_min = 1.5`.
'''

from __future__ import annotations

import os
import sys
from typing import Any

import time
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(path, "..", "..", "src"))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from lamkit.analysis.buckling import BucklingAnalysis
from lamkit.analysis.laminate import Laminate
from lamkit.analysis.material import IM7_8551_7, Material, Ply
from lamkit.lekhnitskii.unloaded_hole import UnloadedHole
from lamkit.lekhnitskii.utils import generate_meshgrid
from lamkit.lekhnitskii.homogenisation import compute_homogenised_properties
from lamkit.utils import evaluate_unloaded_hole_plate, create_effective_laminate_for_buckling_analysis
from lamkit.layup.requirements import EngineeringRequirements


def evaluate_laminate_design(
    layup: list[float],
    material: Material,
    w_mm: float = 100.0,
    h_mm: float = 100.0,
    nxx_n_per_mm: float = -100.0,
    t_ply_mm: float = 0.125,
    hole_radius_mm: float = 5.0,
    n_points: int = 121,
    delta_x_limit_mm: float = 0.05,
    fi_limit: float = 0.8,
    lambda_min_limit: float = 1.5,
    ) -> dict[str, Any]:
    """
    Evaluate objective + constraints for a laminate open-hole plate design.

    Objective:
        minimize total thickness.
    Constraints:
        max|u| <= delta_x_limit_mm
        FI_max <= fi_limit
        lambda_cr >= lambda_min_limit
    """
    t0 = time.time()
    
    # Engineering requirements for layup
    requirements = EngineeringRequirements(strong_requirement=False)
    requirements._print_violations = True
    is_layup_feasible = requirements(layup)
    
    ply = Ply(material=material, thickness=t_ply_mm)
    laminate = Laminate(stacking=layup, plies=ply)

    n_ply = int(len(layup))
    total_thickness = n_ply * t_ply_mm
    if total_thickness <= 0.0:
        raise ValueError("Laminate total thickness must be positive.")

    # Convert in-plane load resultant (N/mm) to far-field stress (MPa = N/mm^2).
    sigma_xx_inf = nxx_n_per_mm / total_thickness
    sigma_yy_inf = 0.0
    tau_xy_inf = 0.0

    # Unloaded hole solution
    mesh = generate_meshgrid(
        hole_radius=hole_radius_mm,
        plate_radius=0.5 * min(w_mm, h_mm),
        n_points_radial=n_points,
        n_points_angular=n_points,
        radial_cluster_power=2.0,
    )
    x = mesh["X"]
    y = mesh["Y"]

    results_by_plies, mid_plane_field = evaluate_unloaded_hole_plate(
        laminate=laminate,
        hole_radius=hole_radius_mm,
        sigma_xx_inf=sigma_xx_inf,
        sigma_yy_inf=sigma_yy_inf,
        tau_xy_inf=tau_xy_inf,
        x=x,
        y=y,
    )

    max_abs_u = float(np.max(np.abs(mid_plane_field["u"])))
    fi_max = float(max(np.max(ply_result["FI_max"]) for ply_result in results_by_plies))

    # Effective laminate for buckling analysis
    properties_eff = compute_homogenised_properties(HoleType=UnloadedHole,
        L=w_mm,
        H=h_mm,
        plate_thickness=total_thickness,
        hole_radius=hole_radius_mm,
        compliance_matrix=laminate.in_plane_compliance_matrix,
        n_points_boundary=n_points,
    )
    
    laminate_eff = create_effective_laminate_for_buckling_analysis(
        E11=properties_eff['E11_eff'],
        E22=properties_eff['E22_eff'],
        G12=properties_eff['G12_eff'],
        nu12=properties_eff['nu12_eff'],
        total_thickness=total_thickness,
    )

    # Buckling analysis of the effective laminate
    buckling = BucklingAnalysis(
        laminate=laminate_eff,
        a=w_mm,
        b=h_mm,
        constraints="PINNED",
        Nxx=nxx_n_per_mm,
        Nyy=0.0,
        Nxy=0.0,
        m=6,
        n=6,
    )
    eigvals, _ = buckling.buckling_analysis(num_eigvalues=5)
    lambda_cr = float(np.min(eigvals))

    objective = float(total_thickness) * (1 - np.pi * hole_radius_mm**2 / (w_mm * h_mm))
    g_disp = max_abs_u - delta_x_limit_mm
    g_fi = fi_max - fi_limit
    g_buckle = lambda_min_limit - lambda_cr

    int_layup = [int(angle) for angle in layup]
    n_0 = int_layup.count(0)
    n_90 = int_layup.count(90)

    t1 = time.time()

    return {
        "layup_deg": layup,
        "objective": objective,
        "total_thickness": total_thickness,
        "n_ply": n_ply,
        "n_0": n_0,
        "n_90": n_90,
        "xiA": np.round(laminate.xiA, 6).tolist(),
        "xiB": np.round(laminate.xiB, 6).tolist(),
        "xiD": np.round(laminate.xiD, 6).tolist(),
        "E11_eff": properties_eff['E11_eff'],
        "E22_eff": properties_eff['E22_eff'],
        "G12_eff": properties_eff['G12_eff'],
        "nu12_eff": properties_eff['nu12_eff'],
        "max_abs_u_mm": max_abs_u,
        "FI_max": fi_max,
        "lambda_cr": lambda_cr,
        "is_layup_feasible": is_layup_feasible,
        "limits": {
            "delta_x_max_mm": delta_x_limit_mm,
            "FI_max": fi_limit,
            "lambda_min": lambda_min_limit,
        },
        # g <= 0 means feasible for each constraint
        "constraints_g": {
            "g_disp": float(g_disp),
            "g_fi": float(g_fi),
            "g_buckle": float(g_buckle),
        },
        "is_feasible": bool((g_disp <= 0.0) and (g_fi <= 0.0) and (g_buckle <= 0.0) and is_layup_feasible),
        "time_taken_s": t1 - t0,
    }


def main() -> None:
    
    # Candidate layup for demonstration.
    layup = [45, -45, 0, -45, 45, 0, 0, 90, 90, 0, 0, 45, -45, 0, -45, 45]
    t_ply_mm = 0.125
    result = evaluate_laminate_design(layup=layup, material=IM7_8551_7, t_ply_mm=t_ply_mm)

    out_path = os.path.join(path, "output.txt")
    lines = [
        "Laminate optimization function example",
        "====================================",
        "",
        "Layup parameters:",
        f"Layup (deg): {result['layup_deg']}",
        f"Number of plies: {len(result['layup_deg'])}",
        f"Number of 0-degree plies:  {result['n_0']}",
        f"Number of 90-degree plies: {result['n_90']}",
        f"xiA: {result['xiA']}",
        f"xiB: {result['xiB']}",
        f"xiD: {result['xiD']}",
        "",
        "Effective properties:",
        f"  E11_eff:  {result['E11_eff']:.3f}",
        f"  E22_eff:  {result['E22_eff']:.3f}",
        f"  G12_eff:  {result['G12_eff']:.3f}",
        f"  nu12_eff: {result['nu12_eff']:.3f}",
        "",
        f"Objective (min weight): {result['objective']:.3f}",
        "",
        "Computed responses:",
        f"  max|u_x|  [mm] = {result['max_abs_u_mm']:.6f}",
        f"  FI_max    [-]  = {result['FI_max']:.6f}",
        f"  lambda_cr [-]  = {result['lambda_cr']:.6f}",
        "",
        "Constraint limits:",
        f"  max|u_x| <= {result['limits']['delta_x_max_mm']:.6f} mm",
        f"  FI_max   <= {result['limits']['FI_max']:.6f}",
        f"  lambda   >= {result['limits']['lambda_min']:.6f}",
        "",
        "Constraint residuals g (feasible if g <= 0):",
        f"  g_disp   = {result['constraints_g']['g_disp']:.6f}",
        f"  g_fi     = {result['constraints_g']['g_fi']:.6f}",
        f"  g_buckle = {result['constraints_g']['g_buckle']:.6f}",
        "",
        f"Layup feasible:   {result['is_layup_feasible']}",
        f"Overall feasible: {result['is_feasible']}",
        f"Time taken: {result['time_taken_s']:.2f} seconds",
        "",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()

