'''
Example of using the Laminate class (Classical Lamination Theory).

- Plot thickness distribution of stress and strain components.
- LaRC05 failure indices in the same figure as stress/strain (bottom two rows).
- Test different stacking sequences.
- Test different loading conditions.
'''
from __future__ import annotations

from collections.abc import Callable

import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(path, "..", "..", "src"))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lamkit.analysis.laminate import Laminate
from lamkit.analysis.larc05 import LaRC05
from lamkit.analysis.material import IM7_8551_7, Ply

DPI = 300
PLY_T_MM = 0.125
# When max(x) - min(x) is below this, set a symmetric x-window around the data mean.
X_SPAN_SMALL = 1e-3
# |mean| below this is treated as "essentially zero" for the x-axis window.
X_MEAN_NEAR_ZERO = 1e-9
X_LIM_NEAR_ZERO_MEAN = (-0.01, 0.01)
# Non-zero mean: half-width is at least this, and scales with |mean|.
X_HALF_MIN = 0.01
X_HALF_REL = 1e-4  # half >= max(X_HALF_MIN, X_HALF_REL * |mean|)


def _maybe_widen_small_x_range(ax: plt.Axes, x_values: list[float] | np.ndarray) -> None:
    """
    If data span on x is < 1e-3, fix xlim so the (almost) vertical profile is centred.

    - If |mean| is essentially zero: xlim = [-0.01, 0.01].
    - Else: symmetric window [mean - half, mean + half] with
      half = max(X_HALF_MIN, X_HALF_REL * |mean|).
    """
    xv = np.asarray(x_values, dtype=float).ravel()
    if xv.size == 0:
        return
    span = float(np.nanmax(xv) - np.nanmin(xv))
    if span >= X_SPAN_SMALL:
        return
    mean = float(np.nanmean(xv))
    if not np.isfinite(mean):
        return
    if abs(mean) < X_MEAN_NEAR_ZERO:
        ax.set_xlim(X_LIM_NEAR_ZERO_MEAN)
    else:
        half = max(X_HALF_MIN, X_HALF_REL * abs(mean))
        ax.set_xlim(mean - half, mean + half)


def build_connected_profile(
    z_bot: np.ndarray,
    z_top: np.ndarray,
    n_ply: int,
    value_at: Callable[[int, float], float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Piecewise-linear (value, z) path through all plies, monotonic in z.
    Always visits each ply bottom and top so the polyline matches ply_endpoint_markers.

    At an interface where the bottom value of ply i+1 differs from the top value of ply i,
    the path includes both points at the same z (horizontal segment in this axes layout).
    """
    xs: list[float] = []
    zs: list[float] = []
    for i in range(n_ply):
        z_lo = float(z_bot[i])
        z_hi = float(z_top[i])
        v_lo = float(value_at(i, z_lo))
        v_hi = float(value_at(i, z_hi))
        if i == 0:
            xs.extend([v_lo, v_hi])
            zs.extend([z_lo, z_hi])
        else:
            xs.append(v_lo)
            zs.append(z_lo)
            xs.append(v_hi)
            zs.append(z_hi)
    return np.asarray(xs), np.asarray(zs)


def ply_endpoint_markers(
    z_bot: np.ndarray,
    z_top: np.ndarray,
    n_ply: int,
    value_at: Callable[[int, float], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Solid markers at each ply bottom and top (interfaces may appear twice if values jump)."""
    xs: list[float] = []
    zs: list[float] = []
    for i in range(n_ply):
        z_lo = float(z_bot[i])
        z_hi = float(z_top[i])
        xs.append(float(value_at(i, z_lo)))
        zs.append(z_lo)
        xs.append(float(value_at(i, z_hi)))
        zs.append(z_hi)
    return np.asarray(xs), np.asarray(zs)


def plot_laminate_response(
    lam: Laminate,
    N: np.ndarray,
    title: str,
    out_path: str,
) -> None:
    """
    One figure per laminate load case: strains, stresses, and LaRC05 failure indices vs z
    (6 rows × 3 cols; rows 0–3 stress/strain, rows 4–5 LaRC05).
    """
    z_if = np.asarray(lam.z_position, dtype=float)
    z_bot = z_if[:-1]
    z_top = z_if[1:]
    eps6 = lam.get_mid_plane_strains(N)
    larc = LaRC05(nSCply=3, material="IM7/8551-7")

    row_labels = (
        (r"$\varepsilon_x$", r"$\varepsilon_y$", r"$\gamma_{xy}$"),
        (r"$\varepsilon_1$", r"$\varepsilon_2$", r"$\gamma_{12}$"),
        (r"$\sigma_x$", r"$\sigma_y$", r"$\tau_{xy}$"),
        (r"$\sigma_1$", r"$\sigma_2$", r"$\tau_{12}$"),
    )
    row_title = (
        "strain (plate x–y)",
        "strain (material 1–2)",
        "stress (plate x–y), MPa",
        "stress (material 1–2), MPa",
        "LaRC05 FI (cracking, splitting, tension)",
        "LaRC05 FI (kinking, interface, FI_max)",
    )
    fi_labels = (
        "FI matrix cracking",
        "FI matrix splitting",
        "FI fibre tension",
        "FI fibre kinking",
        "FI matrix interface",
        r"FI$_{\mathrm{max}}$ (UVARM6)",
    )

    fig, axes = plt.subplots(6, 3, figsize=(10, 16), sharey=True)
    for ax in axes.flat:
        ax.axhline(0.0, color="k", linewidth=0.5, linestyle=":")
        ax.grid(True, alpha=0.3)

    n_ply = len(lam.layup)

    for j in range(3):
        ax = axes[0, j]
        xs0: list[float] = []
        zs0: list[float] = []
        for k in range(len(z_if)):
            exy_k = Laminate.strain_xy_at_z(eps6, z_if[k])[0]
            xs0.append(float(exy_k[j]))
            zs0.append(float(z_if[k]))
        ax.plot(xs0, zs0, color="C0", linewidth=1.8)
        ax.scatter(xs0, zs0, s=24, c="C0", zorder=5, edgecolors="none")
        _maybe_widen_small_x_range(ax, xs0)
        ax.set_xlabel(row_labels[0][j])

    for row in (1, 2, 3):
        for j in range(3):

            def value_at(i: int, z: float) -> float:
                theta, ply = lam.layup[i]
                exy = Laminate.strain_xy_at_z(eps6, z)[0]
                if row == 1:
                    return float(Laminate.strain_xy_global_to_material(exy, theta)[j])
                if row == 2:
                    return float(
                        Laminate.stress_xy_global_from_strain(
                            exy, ply.get_Q_bar(theta)
                        )[j]
                    )
                return float(
                    Laminate.stress_material_from_strain(
                        exy, ply("Q"), theta
                    )[j]
                )

            xs, zs = build_connected_profile(z_bot, z_top, n_ply, value_at)
            mx, mz = ply_endpoint_markers(z_bot, z_top, n_ply, value_at)
            ax = axes[row, j]
            ax.plot(xs, zs, color="C0", linewidth=1.8)
            ax.scatter(mx, mz, s=24, c="C0", zorder=5, edgecolors="none")
            _maybe_widen_small_x_range(ax, xs)
            ax.set_xlabel(row_labels[row][j])

    for fi_row in (0, 1):
        for j in range(3):
            fi_idx = fi_row * 3 + j
            ax = axes[4 + fi_row, j]

            def value_at(i: int, z: float, _k: int = fi_idx) -> float:
                theta, ply = lam.layup[i]
                exy = Laminate.strain_xy_at_z(eps6, z)[0]
                s123 = Laminate.stress_material_from_strain(exy, ply("Q"), theta)
                uvarm = larc.get_uvarm(np.asarray(s123, dtype=float))
                return float(uvarm[_k])

            xs, zs = build_connected_profile(z_bot, z_top, n_ply, value_at)
            mx, mz = ply_endpoint_markers(z_bot, z_top, n_ply, value_at)
            ax.plot(xs, zs, color="C3", linewidth=1.8)
            ax.scatter(mx, mz, s=24, c="C3", zorder=5, edgecolors="none")
            _maybe_widen_small_x_range(ax, xs)
            ax.set_xlabel(fi_labels[fi_idx], fontsize=9)

    for r in range(6):
        axes[r, 0].set_ylabel("z (mm)")
        axes[r, 1].set_title(row_title[r], fontsize=9, pad=6)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ply = Ply(IM7_8551_7, thickness=PLY_T_MM)

    stacks = {
        "[0/90/90/0]": ([0.0, 90.0, 90.0, 0.0], "0-90-90-0"),
        "[45/-45/-45/45] (symmetric)": ([45.0, -45.0, -45.0, 45.0], "45-pm45-symmetric"),
    }

    # Uniaxial membrane resultant Nxx (N/mm); plies defined in mm, stiffness in MPa => consistent.
    N_pull = np.array([80.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Pure bending about x (N)
    Mxx = 50.0
    N_bend = np.array([0.0, 0.0, 0.0, Mxx, 0.0, 0.0])

    out_dir = os.path.join(path, "images")

    for name, (stacking, slug) in stacks.items():
        lam = Laminate(stacking, [ply] * len(stacking))
        h = sum(p.thickness for _, p in lam.layup)
        print(f"\n=== {name} ===")
        print(f"Total thickness: {h:.3f} mm")
        print(f"A11 = {lam.A[0, 0]:.1f} N/mm,  D11 = {lam.D[0, 0]:.2f} N.mm")
        print("Mid-plane strains under N_pull [ex0, ey0, gxy0, kx, ky, kxy]:")
        print(np.round(lam.get_mid_plane_strains(N_pull), 6))
        df_s = lam.calculate_stress(N_pull)
        print("Ply-wise stress (top & bot surfaces):")
        with pd.option_context("display.max_rows", 12):
            print(df_s.to_string(index=False))

        plot_laminate_response(
            lam,
            N_pull,
            title=f"{name}, membrane Nxx={N_pull[0]:.0f} N/mm",
            out_path=os.path.join(out_dir, f"laminate_membrane_{slug}.png"),
        )

    # Same stack, bending-only loading
    lam_qs = Laminate(stacks["[0/90/90/0]"][0], [ply] * 4)
    plot_laminate_response(
        lam_qs,
        N_bend,
        title=f"[0/90/90/0], bending Mxx={Mxx} N",
        out_path=os.path.join(out_dir, "laminate_bending_0-90-90-0.png"),
    )
    print("\n=== [0/90/90/0] under pure Mxx ===")
    print(np.round(lam_qs.get_mid_plane_strains(N_bend), 6))


if __name__ == "__main__":
    main()
