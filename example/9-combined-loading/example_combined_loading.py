"""
Combined bearing and bypass loading in an infinite anisotropic plate (2D infinite plate assumption).

Superposition of:
  - UnloadedHole: far-field (bypass) stresses at infinity
  - LoadedHole:   cosine bearing load distribution inside the hole

Outputs:
  - Mid-plane stress / strain contour plots
  - Ply-enveloped LaRC05 failure index contour plots
  - Hole-boundary FI distribution across plies and angle
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Dict

path = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(path, "..", "..", "src"))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from lamkit.analysis.material import IM7_8551_7, Ply
from lamkit.analysis.laminate import Laminate
from lamkit.lekhnitskii.utils import generate_meshgrid
from lamkit.utils import evaluate_combined_load_plate, evaluate_larc05_from_results
from lamkit.analysis.larc05 import FAILURE_MODE_NAMES

DPI = 100

_FAILURE_MODE_CMAP = ListedColormap(
    ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
)
_FAILURE_MODE_CMAP.set_bad("#d8d8d8")


# ---------------------------------------------------------------------------
# Field evaluation
# ---------------------------------------------------------------------------

def evaluate_laminate_combined_load_field(
    layup: list[float],
    sigma_xx_inf: float,
    sigma_yy_inf: float,
    tau_xy_inf: float,
    load: float,
    thickness: float,
    theta: float = 0.0,
    hole_radius: float = 1.0,
    plot_radius: float = 8.0,
    n_points: int = 181,
    ply_thickness: float = 0.125,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """
    Compute stress/strain field and ply-by-ply LaRC05 results for
    combined bypass + bearing loading.

    Parameters
    ----------
    layup : list[float]
        Ply angles in degrees.
    sigma_xx_inf, sigma_yy_inf, tau_xy_inf : float
        Bypass stresses applied at infinity (MPa).
    load : float
        Total bearing force (N).
    thickness : float
        Plate thickness (mm).
    theta : float
        Bearing angle counter-clockwise from positive x-axis (radians).
    hole_radius : float
        Hole radius (mm).
    plot_radius : float
        Outer radius of the plot domain (mm).
    n_points : int
        Number of points in each mesh direction.
    ply_thickness : float
        Ply thickness (mm).

    Returns
    -------
    field : dict
        Mesh coordinates and mid-plane stress/strain fields.
    results_by_plies : list[dict]
        Ply-surface results (length ``2 * n_ply``).
    """
    ply = Ply(material=IM7_8551_7, thickness=ply_thickness)
    laminate = Laminate(stacking=layup, plies=ply)

    mesh = generate_meshgrid(
        hole_radius=hole_radius,
        plate_radius=plot_radius,
        n_points_radial=n_points,
        n_points_angular=n_points,
        radial_cluster_power=2.0,
    )
    X = mesh["X"]
    Y = mesh["Y"]

    results_by_plies, mid_plane_field = evaluate_combined_load_plate(
        laminate=laminate,
        sigma_xx_inf=sigma_xx_inf,
        sigma_yy_inf=sigma_yy_inf,
        tau_xy_inf=tau_xy_inf,
        load=load,
        angle_load_degree=np.rad2deg(theta),
        hole_radius=hole_radius,
        thickness=thickness,
        x=X,
        y=Y,
    )
    evaluate_larc05_from_results(results_by_plies, laminate.ply_material.properties_dictionary)

    z_edges = np.array(laminate.z_position, dtype=float)
    field = {
        "X": X,
        "Y": Y,
        "sigma_x": mid_plane_field["sigma_x"],
        "sigma_y": mid_plane_field["sigma_y"],
        "tau_xy": mid_plane_field["tau_xy"],
        "epsilon_x": mid_plane_field["epsilon_x"],
        "epsilon_y": mid_plane_field["epsilon_y"],
        "gamma_xy": mid_plane_field["gamma_xy"],
        "z_edges": z_edges,
    }
    return field, results_by_plies


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def envelope_fi_of_all_plies(
        results_by_plies: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Through-thickness envelope of LaRC05 fields.

    Returns
    -------
    results : dict
        ``FI_matrix_cracking``, ``FI_matrix_splitting``, ``FI_fibre_tension``,
        ``FI_fibre_kinking``, ``FI_max``, ``failure_mode``, ``failure_ply_index``,
        each of shape ``(nx, ny)``.
    """
    component_keys: List[str] = [
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

    env = np.max(stacked_comp, axis=1)        # (5, nx, ny)
    max_env = np.max(env, axis=0)
    mode = np.argmax(env, axis=0).astype(np.float64) + 1.0
    mode[max_env < 1e-15] = np.nan

    imax = np.max(stacked_comp, axis=0)       # (n_surf, nx, ny)
    imax = np.argmax(imax, axis=0)            # (nx, ny)

    stacked_max = np.stack([p["FI_max"] for p in results_by_plies], axis=0)
    fi_max = np.max(stacked_max, axis=0)

    return {
        "FI_matrix_cracking": env[0],
        "FI_matrix_splitting": env[1],
        "FI_fibre_tension": env[2],
        "FI_fibre_kinking": env[3],
        "FI_max": fi_max,
        "failure_mode": mode,
        "failure_ply_index": imax,
    }


def extract_hole_boundary_field(
    field: dict[str, np.ndarray],
    n_plies: int,
    results_by_plies: List[Dict[str, Any]],
) -> dict[str, np.ndarray]:
    """
    Hole-boundary (radial index 0) stress / LaRC05 maps across angle and thickness.

    Returns
    -------
    boundary : dict
        ``theta_deg``, ``z_edges``, ``z_centers``, ``sigma1_map``,
        ``sigma2_map``, ``tau12_map``, ``fi_max_map``,
        ``fi_matrix_cracking_map``, ``fi_matrix_splitting_map``,
        ``fi_fibre_tension_map``, ``fi_fibre_kinking_map``,
        ``fi_matrix_interface_map`` — each of shape ``(2*n_plies, n_theta)``.
    """
    x_mesh = field["X"][0, :]
    y_mesh = field["Y"][0, :]

    theta_rad = np.arctan2(y_mesh, x_mesh)
    theta_rad = np.unwrap(theta_rad)
    theta_deg = np.rad2deg(theta_rad)
    if theta_deg.size > 0 and theta_deg[0] < 0.0:
        theta_deg += 360.0
    n_theta = int(x_mesh.shape[0])

    z_iface = np.asarray(field["z_edges"], dtype=float)
    n_surf = len(results_by_plies)
    if n_surf != 2 * n_plies:
        raise ValueError(
            f"results_by_plies length {n_surf} != 2 * n_plies ({2 * n_plies})"
        )

    z_edges = np.empty(2 * n_plies + 1, dtype=float)
    z_edges[0] = z_iface[0]
    for i in range(n_plies):
        zb, zt = z_iface[i], z_iface[i + 1]
        z_edges[2 * i + 1] = 0.5 * (zb + zt)
        z_edges[2 * i + 2] = zt

    sigma1_map = np.zeros((n_surf, n_theta), dtype=float)
    sigma2_map = np.zeros((n_surf, n_theta), dtype=float)
    tau12_map = np.zeros((n_surf, n_theta), dtype=float)
    fi_max_map = np.zeros((n_surf, n_theta), dtype=float)
    fi_mc_map = np.zeros((n_surf, n_theta), dtype=float)
    fi_ms_map = np.zeros((n_surf, n_theta), dtype=float)
    fi_ft_map = np.zeros((n_surf, n_theta), dtype=float)
    fi_fk_map = np.zeros((n_surf, n_theta), dtype=float)
    fi_mi_map = np.zeros((n_surf, n_theta), dtype=float)

    for k, ply in enumerate(results_by_plies):
        sigma1_map[k, :] = ply["sigma_1"][0, :]
        sigma2_map[k, :] = ply["sigma_2"][0, :]
        tau12_map[k, :] = ply["tau_12"][0, :]
        fi_mc_map[k, :] = ply["FI_matrix_cracking"][0, :]
        fi_ms_map[k, :] = ply["FI_matrix_splitting"][0, :]
        fi_ft_map[k, :] = ply["FI_fibre_tension"][0, :]
        fi_fk_map[k, :] = ply["FI_fibre_kinking"][0, :]
        fi_mi_map[k, :] = ply["FI_matrix_interface"][0, :]
        fi_max_map[k, :] = ply["FI_max"][0, :]

    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    return {
        "theta_deg": theta_deg,
        "z_edges": z_edges,
        "z_centers": z_centers,
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_midplane_field(
    field: dict[str, np.ndarray],
    hole_radius: float,
    layup: list[float],
    sigma_xx_inf: float,
    sigma_yy_inf: float,
    tau_xy_inf: float,
    load: float,
    thickness: float,
    theta: float,
    out_path: str = None,
) -> None:
    """
    2×3 panel: mid-plane σₓₓ, σᵧᵧ, τₓᵧ, εₓ, εᵧ, γₓᵧ contour plots.
    """
    X = field["X"]
    Y = field["Y"]
    theta_deg = np.rad2deg(theta)

    title = (
        f"Combined loading: layup={layup}\n"
        f"bypass: σxx={sigma_xx_inf}, σyy={sigma_yy_inf}, "
        f"τxy={tau_xy_inf} MPa  |  "
        f"bearing: {load} N, t={thickness} mm, θ={theta_deg:.1f}°"
    )

    subplots = [
        (field["sigma_x"], r"$\sigma_{xx}$ (MPa)"),
        (field["sigma_y"], r"$\sigma_{yy}$ (MPa)"),
        (field["tau_xy"], r"$\tau_{xy}$ (MPa)"),
        (field["epsilon_x"], r"$\varepsilon_{x}$"),
        (field["epsilon_y"], r"$\varepsilon_{y}$"),
        (field["gamma_xy"], r"$\gamma_{xy}$"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(title, fontsize=11)

    for ax, (data, label) in zip(axes.ravel(), subplots):
        cf = ax.contourf(X, Y, data, levels=60)
        ax.add_patch(plt.Circle((0, 0), hole_radius, color="black", fill=False))
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(cf, ax=ax, shrink=0.85)

    fig.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=DPI)
        plt.close(fig)
    else:
        plt.show()


def plot_fi_field(
    field: dict[str, np.ndarray],
    fi_field: dict[str, np.ndarray],
    hole_radius: float,
    layup: list[float],
    sigma_xx_inf: float,
    sigma_yy_inf: float,
    tau_xy_inf: float,
    load: float,
    thickness: float,
    theta: float,
    out_path: str = None,
) -> None:
    """
    3×3 panel: mid-plane strains, LaRC05 component FIs, FI_max, and governing
    failure mode (ply-enveloped argmax of UVARM1-5).
    """
    X = field["X"]
    Y = field["Y"]
    theta_deg = np.rad2deg(theta)

    title = (
        f"Combined loading (ply-enveloped FI): layup={layup}\n"
        f"bypass: σxx={sigma_xx_inf}, σyy={sigma_yy_inf}, "
        f"τxy={tau_xy_inf} MPa  |  "
        f"bearing: {load} N, t={thickness} mm, θ={theta_deg:.1f}°  |  "
        f"FI_max={fi_field['FI_max'].max():.2f}"
    )

    fig, ax = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(title, fontsize=11)

    strain_titles = [
        r"Mid-plane $\varepsilon_{0,x}$",
        r"Mid-plane $\varepsilon_{0,y}$",
        r"Mid-plane $\gamma_{0,xy}$",
    ]
    strain_data = [field["epsilon_x"], field["epsilon_y"], field["gamma_xy"]]
    for i in range(3):
        cf = ax[0, i].contourf(X, Y, strain_data[i], levels=60)
        ax[0, i].add_patch(plt.Circle((0, 0), hole_radius, color="black", fill=False))
        ax[0, i].set_title(strain_titles[i])
        ax[0, i].set_aspect("equal")
        ax[0, i].set_xlabel("x (mm)")
        ax[0, i].set_ylabel("y (mm)")
        fig.colorbar(cf, ax=ax[0, i], shrink=0.85)

    fi_keys_titles = [
        ("FI_matrix_cracking", "FI matrix cracking (ply-enveloped max)"),
        ("FI_matrix_splitting", "FI matrix splitting (ply-enveloped max)"),
        ("FI_fibre_tension", "FI fibre tension (ply-enveloped max)"),
        ("FI_fibre_kinking", "FI fibre kinking (ply-enveloped max)"),
        ("FI_max", "FI max (ply-enveloped max)"),
    ]
    for k, (key, fi_title) in enumerate(fi_keys_titles):
        r = 1 + k // 3
        c = k % 3
        cf = ax[r, c].contourf(X, Y, fi_field[key], levels=60)
        ax[r, c].add_patch(plt.Circle((0, 0), hole_radius, color="black", fill=False))
        ax[r, c].set_title(fi_title)
        ax[r, c].set_aspect("equal")
        ax[r, c].set_xlabel("x (mm)")
        ax[r, c].set_ylabel("y (mm)")
        fig.colorbar(cf, ax=ax[r, c], shrink=0.85)

    cf_mode = ax[2, 2].contourf(
        X, Y,
        fi_field["failure_mode"],
        levels=np.arange(0.5, 6.5, 1.0),
        cmap=_FAILURE_MODE_CMAP,
        extend="neither",
    )
    ax[2, 2].add_patch(plt.Circle((0, 0), hole_radius, color="black", fill=False))
    ax[2, 2].set_title("Failure mode (ply-enveloped max)")
    ax[2, 2].set_aspect("equal")
    ax[2, 2].set_xlabel("x (mm)")
    ax[2, 2].set_ylabel("y (mm)")
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


def plot_hole_boundary_face(
    theta_deg: np.ndarray,
    z_edges: np.ndarray,
    value_map: np.ndarray,
    cbar_label: str,
    title: str = None,
    out_path: str = None,
) -> None:
    """
    2D pcolormesh: x=hole-edge angle, y=thickness coordinate, colour=value.
    """
    dtheta = float(theta_deg[1] - theta_deg[0])
    theta_edges = np.concatenate([theta_deg, [theta_deg[-1] + dtheta]])
    theta_grid, z_grid = np.meshgrid(theta_edges, z_edges)

    if title is None:
        title = f"Hole boundary face: FI_max={value_map.max():.2f}"

    fig, ax = plt.subplots(figsize=(10, 5))
    pc = ax.pcolormesh(theta_grid, z_grid, value_map, shading="flat")
    ax.set_xlabel("Hole-edge angle (deg)")
    ax.set_ylabel("Thickness coordinate z (mm)")
    ax.set_title(title)
    fig.colorbar(pc, ax=ax, label=cbar_label)
    fig.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=DPI)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    os.makedirs(os.path.join(path, "images"), exist_ok=True)

    layup = [45, -45, 0, 90, 90, 0, -45, 45]
    ply_thickness = 0.125
    thickness = len(layup) * ply_thickness   # total plate thickness (mm)
    hole_radius = 1.0

    # Bypass: 100 MPa uniaxial tension in x-direction at infinity
    sigma_xx_inf = 100.0
    sigma_yy_inf = 0.0
    tau_xy_inf = 0.0

    # Bearing: 1000 N along +x axis
    load = 1000.0
    theta = 0.0   # radians

    field, results_by_plies = evaluate_laminate_combined_load_field(
        layup=layup,
        sigma_xx_inf=sigma_xx_inf,
        sigma_yy_inf=sigma_yy_inf,
        tau_xy_inf=tau_xy_inf,
        load=load,
        thickness=thickness,
        theta=theta,
        hole_radius=hole_radius,
        plot_radius=8.0,
        n_points=181,
        ply_thickness=ply_thickness,
    )

    fi_field = envelope_fi_of_all_plies(results_by_plies)
    boundary = extract_hole_boundary_field(field, len(layup), results_by_plies)

    plot_midplane_field(
        field=field,
        hole_radius=hole_radius,
        layup=layup,
        sigma_xx_inf=sigma_xx_inf,
        sigma_yy_inf=sigma_yy_inf,
        tau_xy_inf=tau_xy_inf,
        load=load,
        thickness=thickness,
        theta=theta,
        out_path=os.path.join(path, "images", "combined_midplane_field.png"),
    )

    plot_fi_field(
        field=field,
        fi_field=fi_field,
        hole_radius=hole_radius,
        layup=layup,
        sigma_xx_inf=sigma_xx_inf,
        sigma_yy_inf=sigma_yy_inf,
        tau_xy_inf=tau_xy_inf,
        load=load,
        thickness=thickness,
        theta=theta,
        out_path=os.path.join(path, "images", "combined_fi_field.png"),
    )

    plot_hole_boundary_face(
        theta_deg=boundary["theta_deg"],
        z_edges=boundary["z_edges"],
        value_map=boundary["fi_max_map"],
        cbar_label="FI_max",
        title=(
            f"Combined loading - hole boundary FI_max: layup={layup}\n"
            f"bypass σxx={sigma_xx_inf} MPa | bearing {load} N"
        ),
        out_path=os.path.join(path, "images", "combined_boundary_face.png"),
    )
