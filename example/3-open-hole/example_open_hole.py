"""
Laminate open-hole stress field prediction (2D infinite plate assumption).
"""

from __future__ import annotations

import os
import sys
from typing import Any

path = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(path, "..", "..", "src"))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from lamkit.analysis.material import IM7_8551_7, Ply
from lamkit.analysis.larc05 import LaRC05
from lamkit.analysis.laminate import Laminate
from lamkit.lekhnitskii.utils import generate_meshgrid
from lamkit.utils import evaluate_unloaded_hole_stress_field

DPI = 300

# LaRC05 UVARM1–5: dominant mode at each point is argmax of ply-enveloped component FIs.
_FAILURE_MODE_CMAP_BASE = ListedColormap(
    ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
)
_FAILURE_MODE_CMAP_BASE.set_bad("#d8d8d8")
FAILURE_MODE_NAMES = [
    "matrix cracking",
    "matrix splitting",
    "fibre tension",
    "fibre kinking",
    "matrix interface",
]



def evaluate_laminate_open_hole_field(
    layup: list[float],
    sigma_xx_inf: float,
    sigma_yy_inf: float,
    tau_xy_inf: float,
    hole_radius: float = 1.0,
    plot_radius: float = 8.0,
    n_points: int = 181,
    ply_thickness: float = 0.125,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """
    Predict stress field around an open hole (no plotting).

    Returns the mesh + equivalent mid-plane stresses and ply-surface results from
    ``evaluate_unloaded_hole_stress_field`` (second value used to envelope LaRC05 FIs).
    """
    ply = Ply(material=IM7_8551_7, thickness=ply_thickness)
    laminate = Laminate(stacking=layup, plies=ply)
    compliance_matrix = laminate.in_plane_compliance_matrix

    mesh = generate_meshgrid(
        hole_radius=hole_radius,
        plate_radius=plot_radius,
        n_points_radial=n_points,
        n_points_angular=n_points,
        radial_cluster_power=2.0,
    )
    X = mesh["X"]
    Y = mesh["Y"]

    results_by_plies, mid_plane_field = evaluate_unloaded_hole_stress_field(
        laminate,
        hole_radius,
        sigma_xx_inf,
        sigma_yy_inf,
        tau_xy_inf,
        X,
        Y,
    )
    shape = X.shape
    sigma_xx = mid_plane_field["sigma_x"].reshape(shape)
    sigma_yy = mid_plane_field["sigma_y"].reshape(shape)
    tau_xy = mid_plane_field["tau_xy"].reshape(shape)

    field = {
        "X": X,
        "Y": Y,
        "sigma_xx": sigma_xx,
        "sigma_yy": sigma_yy,
        "tau_xy": tau_xy,
        "compliance_matrix": compliance_matrix,
    }
    return field, results_by_plies

def evaluate_hole_boundary_larc05_field(
    layup: list[float],
    sigma_xx_inf: float,
    sigma_yy_inf: float,
    tau_xy_inf: float,
    hole_radius: float = 1.0,
    n_theta: int = 361,
    ply_thickness: float = 0.125,
    ) -> dict[str, np.ndarray]:
    """
    Evaluate stress/FI fields on hole boundary (theta-thickness map).

    The in-plane hole-edge stress is predicted by equivalent laminate anisotropic
    solution, then mapped to each ply local frame for LaRC05 (2D, nSCply=3).
    """
    ply = Ply(material=IM7_8551_7, thickness=ply_thickness)
    laminate = Laminate(stacking=layup, plies=ply)

    theta_deg = np.linspace(0.0, 360.0, n_theta, endpoint=False)
    theta_rad = np.deg2rad(theta_deg)
    r_eval = hole_radius * (1.0 + 1e-6)
    x = r_eval * np.cos(theta_rad)
    y = r_eval * np.sin(theta_rad)

    _, mid_plane_field = evaluate_unloaded_hole_stress_field(
        laminate,
        hole_radius,
        sigma_xx_inf,
        sigma_yy_inf,
        tau_xy_inf,
        x,
        y,
    )
    sigma_xx = mid_plane_field["sigma_x"]
    sigma_yy = mid_plane_field["sigma_y"]
    tau_xy = mid_plane_field["tau_xy"]

    n_plies = len(layup)
    sigma1_map = np.zeros((n_plies, n_theta), dtype=float)
    sigma2_map = np.zeros((n_plies, n_theta), dtype=float)
    tau12_map = np.zeros((n_plies, n_theta), dtype=float)
    fi_max_map = np.zeros((n_plies, n_theta), dtype=float)
    fi_mc_map = np.zeros((n_plies, n_theta), dtype=float)
    fi_ms_map = np.zeros((n_plies, n_theta), dtype=float)
    fi_ft_map = np.zeros((n_plies, n_theta), dtype=float)
    fi_fk_map = np.zeros((n_plies, n_theta), dtype=float)
    fi_mi_map = np.zeros((n_plies, n_theta), dtype=float)

    larc = LaRC05(nSCply=3)
    for i, angle in enumerate(layup):
        s1, s2, t12 = _stress_xy_to_12(sigma_xx, sigma_yy, tau_xy, angle_deg=angle)
        sigma1_map[i, :] = s1
        sigma2_map[i, :] = s2
        tau12_map[i, :] = t12

        for j in range(n_theta):
            uvarm = larc.get_uvarm(np.array([s1[j], s2[j], t12[j]], dtype=float))
            fi_mc_map[i, j] = uvarm[0]
            fi_ms_map[i, j] = uvarm[1]
            fi_ft_map[i, j] = uvarm[2]
            fi_fk_map[i, j] = uvarm[3]
            fi_mi_map[i, j] = uvarm[4]
            fi_max_map[i, j] = uvarm[5]

    z_edges = np.array(laminate.z_position, dtype=float)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    return {
        "theta_deg": theta_deg,
        "z_edges": z_edges,
        "z_centers": z_centers,
        "sigma_xx_hole": sigma_xx,
        "sigma_yy_hole": sigma_yy,
        "tau_xy_hole": tau_xy,
        "sigma1_map": sigma1_map,
        "sigma2_map": sigma2_map,
        "tau12_map": tau12_map,
        "fi_max_map": fi_max_map,
        "fi_matrix_cracking_map": fi_mc_map,
        "fi_matrix_splitting_map": fi_ms_map,
        "fi_fibre_tension_map": fi_ft_map,
        "fi_fibre_kinking_map": fi_fk_map,
        "fi_matrix_interface_map": fi_mi_map,
    }




def plot_laminate_open_hole_stress_and_larc05(
    field: dict[str, np.ndarray],
    fi_field: dict[str, np.ndarray],
    hole_radius: float,
    layup: list[float],
    sigma_xx_inf: float,
    sigma_yy_inf: float,
    tau_xy_inf: float,
    out_path: str | None = None,
    ) -> None:
    """
    Single figure: 3x3 stress components, LaRC05 component FIs (no matrix-interface FI),
    FI_max, and discrete governing failure mode (ply-enveloped argmax of UVARM1–5).
    """
    X = field["X"]
    Y = field["Y"]
    sigma_xx = field["sigma_xx"]
    sigma_yy = field["sigma_yy"]
    tau_xy = field["tau_xy"]

    fig, ax = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(
        (
            f"Open hole — layup={layup}, "
            f"$\\sigma_{{xx}}^\\infty$={sigma_xx_inf}, "
            f"$\\sigma_{{yy}}^\\infty$={sigma_yy_inf}, "
            f"$\\tau_{{xy}}^\\infty$={tau_xy_inf}"
        ),
        fontsize=12,
    )

    stress_titles = [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\tau_{xy}$"]
    stress_data = [sigma_xx, sigma_yy, tau_xy]
    for i in range(3):
        cf = ax[0, i].contourf(X, Y, stress_data[i], levels=60)
        ax[0, i].add_patch(plt.Circle((0, 0), hole_radius, color="black", fill=False))
        ax[0, i].set_title(stress_titles[i])
        ax[0, i].set_aspect("equal")
        ax[0, i].set_xlabel("x")
        ax[0, i].set_ylabel("y")
        fig.colorbar(cf, ax=ax[0, i], shrink=0.85)

    fi_keys_titles = [
        ("FI_matrix_cracking", "FI matrix cracking"),
        ("FI_matrix_splitting", "FI matrix splitting"),
        ("FI_fibre_tension", "FI fibre tension"),
        ("FI_fibre_kinking", "FI fibre kinking"),
        ("FI_max", "FI max"),
    ]
    for k, (key, title) in enumerate(fi_keys_titles):
        r = 1 + k // 3
        c = k % 3
        cf = ax[r, c].contourf(X, Y, fi_field[key], levels=60)
        ax[r, c].add_patch(plt.Circle((0, 0), hole_radius, color="black", fill=False))
        ax[r, c].set_title(title)
        ax[r, c].set_aspect("equal")
        ax[r, c].set_xlabel("x")
        ax[r, c].set_ylabel("y")
        fig.colorbar(cf, ax=ax[r, c], shrink=0.85)

    cf_mode = ax[2, 2].contourf(
        X,
        Y,
        fi_field["failure_mode"],
        levels=np.arange(0.5, 6.5, 1.0),
        cmap=_FAILURE_MODE_CMAP_BASE,
        extend="neither",
    )
    ax[2, 2].add_patch(plt.Circle((0, 0), hole_radius, color="black", fill=False))
    ax[2, 2].set_title("Failure mode (ply-enveloped argmax)")
    ax[2, 2].set_aspect("equal")
    ax[2, 2].set_xlabel("x")
    ax[2, 2].set_ylabel("y")
    cbar_m = fig.colorbar(
        cf_mode, ax=ax[2, 2], shrink=0.85, ticks=[1, 2, 3, 4, 5], extend="neither"
    )
    cbar_m.ax.set_yticklabels(FAILURE_MODE_NAMES)

    fig.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=DPI)
        plt.close(fig)
    else:
        plt.show()


def hole_boundary_kt(sigma_yy_on_hole: np.ndarray, sigma_remote: float) -> float:
    """
    Stress concentration factor based on circumferential traction proxy:
    Kt = max(sigma_yy at r=a) / sigma_remote.
    """
    if abs(sigma_remote) < 1e-12:
        return np.nan
    return float(np.max(sigma_yy_on_hole) / sigma_remote)


def _stress_xy_to_12(
    sigma_xx: np.ndarray, sigma_yy: np.ndarray, tau_xy: np.ndarray, angle_deg: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform in-plane stresses from global x-y to ply local 1-2 coordinates.
    """
    th = np.deg2rad(angle_deg)
    c = np.cos(th)
    s = np.sin(th)
    c2 = c * c
    s2 = s * s
    sc = s * c

    sigma_1 = c2 * sigma_xx + s2 * sigma_yy + 2.0 * sc * tau_xy
    sigma_2 = s2 * sigma_xx + c2 * sigma_yy - 2.0 * sc * tau_xy
    tau_12 = -sc * sigma_xx + sc * sigma_yy + (c2 - s2) * tau_xy
    return sigma_1, sigma_2, tau_12


def plot_theta_thickness_map(
    theta_deg: np.ndarray,
    z_edges: np.ndarray,
    value_map: np.ndarray,
    title: str,
    cbar_label: str,
    out_path: str,
    ) -> None:
    """
    Plot 2D contour map: x=hole-edge angle, y=thickness, color=value.
    """
    dtheta = float(theta_deg[1] - theta_deg[0])
    theta_edges = np.concatenate([theta_deg, [theta_deg[-1] + dtheta]])
    theta_grid, z_grid = np.meshgrid(theta_edges, z_edges)

    fig, ax = plt.subplots(figsize=(10, 5))
    pc = ax.pcolormesh(theta_grid, z_grid, value_map, shading="flat")
    ax.set_xlabel("Hole-edge angle theta (deg)")
    ax.set_ylabel("Thickness coordinate z (mm)")
    ax.set_title(title)
    fig.colorbar(pc, ax=ax, label=cbar_label)
    fig.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close(fig)


def envelope_larc05_fi_from_ply_results(
    results_by_plies: list[dict[str, Any]],
    ) -> dict[str, np.ndarray]:
    """
    Through-thickness envelope of LaRC05 fields from ``evaluate_unloaded_hole_stress_field``.

    Same structure as the former mesh-only LaRC05 helper: max over ply surfaces, then
    governing mode from argmax of enveloped UVARM1–5.
    """
    component_keys = [
        "FI_matrix_cracking",
        "FI_matrix_splitting",
        "FI_fibre_tension",
        "FI_fibre_kinking",
        "FI_matrix_interface",
    ]
    stacked_comp = np.stack(
        [np.stack([p[k] for p in results_by_plies], axis=0) for k in component_keys],
        axis=0,
    )
    # (5, n_surf, nx, ny) -> env (5, nx, ny)
    env = np.max(stacked_comp, axis=1)
    max_env = np.max(env, axis=0)
    mode = np.argmax(env, axis=0).astype(np.float64) + 1.0
    mode[max_env < 1e-15] = np.nan

    stacked_max = np.stack([p["FI_max"] for p in results_by_plies], axis=0)
    fi_max = np.max(stacked_max, axis=0)

    return {
        "FI_matrix_cracking": env[0],
        "FI_matrix_splitting": env[1],
        "FI_fibre_tension": env[2],
        "FI_fibre_kinking": env[3],
        "FI_max": fi_max,
        "failure_mode": mode,
    }


if __name__ == "__main__":
    os.makedirs(os.path.join(path, "images"), exist_ok=True)

    # Example laminate (quasi-isotropic).
    layup = [45, -45, 0, 90, 90, 0, -45, 45]

    field, results_by_plies = evaluate_laminate_open_hole_field(
        layup=layup,
        sigma_xx_inf=10.0,
        sigma_yy_inf=0.0,
        tau_xy_inf=0.0,
        hole_radius=1.0,
        plot_radius=8.0,
        n_points=181,
        ply_thickness=0.125,
    )

    # Boundary concentration estimate at first radial ring (r ~= a).
    sigma_remote = 10.0
    sigma_yy_hole = field["sigma_yy"][0, :]
    kt = hole_boundary_kt(sigma_yy_hole, sigma_remote=sigma_remote)

    print("Equivalent laminate in-plane compliance matrix [S_eq]:")
    print(field["compliance_matrix"])
    print(f"Estimated Kt (using sigma_yy on r=a): {kt:.4f}")

    boundary = evaluate_hole_boundary_larc05_field(
        layup=layup,
        sigma_xx_inf=10.0,
        sigma_yy_inf=0.0,
        tau_xy_inf=0.0,
        hole_radius=1.0,
        n_theta=361,
        ply_thickness=0.125,
    )

    plot_theta_thickness_map(
        theta_deg=boundary["theta_deg"],
        z_edges=boundary["z_edges"],
        value_map=boundary["sigma1_map"],
        title=r"Hole-edge $\sigma_1$ map in ply local coordinates",
        cbar_label=r"$\sigma_1$",
        out_path=os.path.join(path, "images", "hole_theta_thickness_sigma1.png"),
    )

    plot_theta_thickness_map(
        theta_deg=boundary["theta_deg"],
        z_edges=boundary["z_edges"],
        value_map=boundary["fi_max_map"],
        title="Hole-edge LaRC05 max failure index map",
        cbar_label="FI_max",
        out_path=os.path.join(path, "images", "hole_theta_thickness_fi_max.png"),
    )

    print(f"Hole-edge maximum FI_max: {boundary['fi_max_map'].max():.4f}")

    fi_field = envelope_larc05_fi_from_ply_results(results_by_plies)

    plot_laminate_open_hole_stress_and_larc05(
        field=field,
        fi_field=fi_field,
        hole_radius=1.0,
        layup=layup,
        sigma_xx_inf=10.0,
        sigma_yy_inf=0.0,
        tau_xy_inf=0.0,
        out_path=os.path.join(path, "images", "laminate_open_hole_field.png"),
    )

    print(f"Domain maximum FI_max: {fi_field['FI_max'].max():.4f}")
