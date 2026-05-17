'''
Example of LoadedHole (cosine bearing load distribution) using Lekhnitskii theory.
'''
import os
import sys
path = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(path, "..", "..", "src"))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

import numpy as np
import matplotlib.pyplot as plt

from lamkit.lekhnitskii.loaded_hole import LoadedHole
from lamkit.lekhnitskii.utils import generate_meshgrid


DPI = 100


def _draw_mesh_grid(ax: plt.Axes, X: np.ndarray, Y: np.ndarray, step: int = 10,
                   color: str = 'k', linewidth: float = 0.3, alpha: float = 0.6) -> None:
    '''
    Draw data grid lines on the subplot.
    '''
    n_i, n_j = X.shape
    for i in range(0, n_i, max(1, step)):
        ax.plot(X[i, :], Y[i, :], color=color, lw=linewidth, alpha=alpha)
    for j in range(0, n_j, max(1, step)):
        ax.plot(X[:, j], Y[:, j], color=color, lw=linewidth, alpha=alpha)


def plot_stress_field(
    load: float, diameter: float, thickness: float, theta: float,
    compliance_matrix: np.ndarray,
    length_plot: float, n_points: int = 101,
    filename: str = 'loaded_hole_stress.png',
    ) -> None:
    '''
    Plot the stress field (sigma_xx, sigma_yy, tau_xy) for a loaded hole.
    '''
    hole_radius = diameter / 2.0
    meshgrid = generate_meshgrid(hole_radius=hole_radius, plate_radius=length_plot,
                                 n_points_radial=n_points, n_points_angular=n_points,
                                 radial_cluster_power=2.0)
    X = meshgrid['X']
    Y = meshgrid['Y']
    out_shape = X.shape

    solution = LoadedHole(load=load, radius=hole_radius, thickness=thickness,
                          compliance_matrix=compliance_matrix, theta=theta)

    x_flat = np.atleast_1d(X).ravel()
    y_flat = np.atleast_1d(Y).ravel()
    stresses = solution.stress(x_flat, y_flat)
    sigma_xx = stresses[:, 0].reshape(out_shape)
    sigma_yy = stresses[:, 1].reshape(out_shape)
    tau_xy   = stresses[:, 2].reshape(out_shape)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    theta_deg = np.rad2deg(theta)
    title = (fr'$P = {load}\ \mathrm{{N}},\ '
             fr'd = {diameter}\ \mathrm{{mm}},\ '
             fr't = {thickness}\ \mathrm{{mm}},\ '
             fr'\theta = {theta_deg:.0f}^\circ$')
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.90)

    cf0 = ax[0].contourf(X, Y, sigma_xx)
    ax[0].set_title(r'$\sigma_{xx}$')
    ax[0].axis('equal')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].add_patch(plt.Circle((0, 0), hole_radius, color='black', fill=False))
    fig.colorbar(cf0, ax=ax[0])

    cf1 = ax[1].contourf(X, Y, sigma_yy)
    ax[1].set_title(r'$\sigma_{yy}$')
    ax[1].axis('equal')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].add_patch(plt.Circle((0, 0), hole_radius, color='black', fill=False))
    fig.colorbar(cf1, ax=ax[1])

    cf2 = ax[2].contourf(X, Y, tau_xy)
    ax[2].set_title(r'$\tau_{xy}$')
    ax[2].axis('equal')
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')
    ax[2].add_patch(plt.Circle((0, 0), hole_radius, color='black', fill=False))
    fig.colorbar(cf2, ax=ax[2])

    plt.savefig(filename, dpi=DPI)
    plt.close()


def plot_stress_field_details(
    load: float, diameter: float, thickness: float, theta: float,
    compliance_matrix: np.ndarray,
    length_plot: float, n_points: int = 101,
    filename: str = 'loaded_hole_details.png',
    plot_grid_step: int = 1,
    ) -> None:
    '''
    Plot 3x3 detail figure: stresses plus stress-function auxiliary fields.

    ``Hole.calculate_field_results`` is used here, which works correctly in the
    global frame when ``theta = 0``.  For ``theta != 0`` the stress components
    reflect the rotated material frame; use ``plot_stress_field`` for any theta.
    '''
    hole_radius = diameter / 2.0
    meshgrid = generate_meshgrid(hole_radius=hole_radius, plate_radius=length_plot,
                                 n_points_radial=n_points, n_points_angular=n_points,
                                 radial_cluster_power=2.0)
    X = meshgrid['X']
    Y = meshgrid['Y']
    out_shape = meshgrid['meshgrid_shape']

    solution = LoadedHole(load=load, radius=hole_radius, thickness=thickness,
                          compliance_matrix=compliance_matrix, theta=theta)

    x_flat = np.atleast_1d(X).ravel()
    y_flat = np.atleast_1d(Y).ravel()

    field = solution.calculate_field_results(x_flat, y_flat)

    fig, ax = plt.subplots(3, 3, figsize=(18, 16))

    theta_deg = np.rad2deg(theta)
    title = (fr'$P = {load}\ \mathrm{{N}},\ '
             fr'd = {diameter}\ \mathrm{{mm}},\ '
             fr't = {thickness}\ \mathrm{{mm}},\ '
             fr'\theta = {theta_deg:.0f}^\circ$')
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.90)

    cf00 = ax[0, 0].contourf(X, Y, field['sigma_x'].reshape(out_shape))
    ax[0, 0].set_title(r'$\sigma_{xx}$')
    ax[0, 0].axis('equal')
    ax[0, 0].set_xlabel('x')
    ax[0, 0].set_ylabel('y')
    ax[0, 0].add_patch(plt.Circle((0, 0), hole_radius, color='black', fill=False))
    fig.colorbar(cf00, ax=ax[0, 0])

    cf01 = ax[0, 1].contourf(X, Y, field['sigma_y'].reshape(out_shape))
    ax[0, 1].set_title(r'$\sigma_{yy}$')
    ax[0, 1].axis('equal')
    ax[0, 1].set_xlabel('x')
    ax[0, 1].set_ylabel('y')
    ax[0, 1].add_patch(plt.Circle((0, 0), hole_radius, color='black', fill=False))
    fig.colorbar(cf01, ax=ax[0, 1])

    cf02 = ax[0, 2].contourf(X, Y, field['tau_xy'].reshape(out_shape))
    ax[0, 2].set_title(r'$\tau_{xy}$')
    ax[0, 2].axis('equal')
    ax[0, 2].set_xlabel('x')
    ax[0, 2].set_ylabel('y')
    ax[0, 2].add_patch(plt.Circle((0, 0), hole_radius, color='black', fill=False))
    fig.colorbar(cf02, ax=ax[0, 2])

    cf10 = ax[1, 0].contourf(X, Y, field['Real(phi_1_prime)'].reshape(out_shape))
    ax[1, 0].set_title(r"$\text{Real}(\phi_1^\prime)$")
    ax[1, 0].axis('equal')
    ax[1, 0].set_xlabel('x')
    ax[1, 0].set_ylabel('y')
    ax[1, 0].add_patch(plt.Circle((0, 0), hole_radius, color='black', fill=False))
    fig.colorbar(cf10, ax=ax[1, 0])

    cf11 = ax[1, 1].contourf(X, Y, field['Real(phi_2_prime)'].reshape(out_shape))
    ax[1, 1].set_title(r"$\text{Real}(\phi_2^\prime)$")
    ax[1, 1].axis('equal')
    ax[1, 1].set_xlabel('x')
    ax[1, 1].set_ylabel('y')
    ax[1, 1].add_patch(plt.Circle((0, 0), hole_radius, color='black', fill=False))
    fig.colorbar(cf11, ax=ax[1, 1])

    cf12 = ax[1, 2].contourf(X, Y, field['sign_xi1'].reshape(out_shape))
    if plot_grid_step > 0:
        _draw_mesh_grid(ax[1, 2], X, Y, step=plot_grid_step)
    ax[1, 2].set_title(r"$\text{sign}(\xi_1)$")
    ax[1, 2].axis('equal')
    ax[1, 2].set_xlabel('x')
    ax[1, 2].set_ylabel('y')
    ax[1, 2].add_patch(plt.Circle((0, 0), hole_radius, color='black', fill=False))
    fig.colorbar(cf12, ax=ax[1, 2])

    cf20 = ax[2, 0].contourf(X, Y, field['Imag(phi_1_prime)'].reshape(out_shape))
    ax[2, 0].set_title(r"$\text{Imag}(\phi_1^\prime)$")
    ax[2, 0].axis('equal')
    ax[2, 0].set_xlabel('x')
    ax[2, 0].set_ylabel('y')
    ax[2, 0].add_patch(plt.Circle((0, 0), hole_radius, color='black', fill=False))
    fig.colorbar(cf20, ax=ax[2, 0])

    cf21 = ax[2, 1].contourf(X, Y, field['Imag(phi_2_prime)'].reshape(out_shape))
    ax[2, 1].set_title(r"$\text{Imag}(\phi_2^\prime)$")
    ax[2, 1].axis('equal')
    ax[2, 1].set_xlabel('x')
    ax[2, 1].set_ylabel('y')
    ax[2, 1].add_patch(plt.Circle((0, 0), hole_radius, color='black', fill=False))
    fig.colorbar(cf21, ax=ax[2, 1])

    cf22 = ax[2, 2].contourf(X, Y, field['sign_xi2'].reshape(out_shape))
    if plot_grid_step > 0:
        _draw_mesh_grid(ax[2, 2], X, Y, step=plot_grid_step)
    ax[2, 2].set_title(r"$\text{sign}(\xi_2)$")
    ax[2, 2].axis('equal')
    ax[2, 2].set_xlabel('x')
    ax[2, 2].set_ylabel('y')
    ax[2, 2].add_patch(plt.Circle((0, 0), hole_radius, color='black', fill=False))
    fig.colorbar(cf22, ax=ax[2, 2])

    plt.savefig(filename, dpi=DPI)
    plt.close()


if __name__ == '__main__':

    os.makedirs(os.path.join(path, 'images'), exist_ok=True)
    path_images = os.path.join(path, 'images')

    hole_radius = 1.0
    diameter = 2.0 * hole_radius
    thickness = 1.0       # plate thickness (mm)
    load = 1000.0         # bearing force (N)
    length_plot = 10

    # IM7/8551-7 anisotropic
    E1   = 165000.0
    E2   = 8400.0
    G12  = 5600.0
    nu12 = 0.34

    compliance_matrix = np.zeros((3, 3))
    compliance_matrix[0, 0] = 1 / E1
    compliance_matrix[1, 1] = 1 / E2
    compliance_matrix[2, 2] = 1 / G12
    compliance_matrix[0, 1] = -nu12 / E1
    compliance_matrix[1, 0] = -nu12 / E1

    # Anisotropic — theta=0 (bearing along +x)
    plot_stress_field_details(
        load=load, diameter=diameter, thickness=thickness, theta=0.0,
        compliance_matrix=compliance_matrix,
        length_plot=length_plot,
        filename=os.path.join(path_images, 'loaded_hole_1.png'))

    # Anisotropic — theta=90 deg (bearing along +y)
    plot_stress_field(
        load=load, diameter=diameter, thickness=thickness, theta=np.pi / 2,
        compliance_matrix=compliance_matrix,
        length_plot=length_plot,
        filename=os.path.join(path_images, 'loaded_hole_2.png'))

    # Anisotropic — theta=45 deg
    plot_stress_field(
        load=load, diameter=diameter, thickness=thickness, theta=np.pi / 4,
        compliance_matrix=compliance_matrix,
        length_plot=length_plot,
        filename=os.path.join(path_images, 'loaded_hole_3.png'))

    # Isotropic
    E1   = 1.0
    E2   = 1.0
    nu12 = 0.3
    G12  = E1 / (2 * (1 + nu12))

    compliance_matrix = np.zeros((3, 3))
    compliance_matrix[0, 0] = 1 / E1
    compliance_matrix[1, 1] = 1 / E2
    compliance_matrix[2, 2] = 1 / G12
    compliance_matrix[0, 1] = -nu12 / E1
    compliance_matrix[1, 0] = -nu12 / E1

    # Isotropic — theta=0
    plot_stress_field_details(
        load=load, diameter=diameter, thickness=thickness, theta=0.0,
        compliance_matrix=compliance_matrix,
        length_plot=length_plot,
        filename=os.path.join(path_images, 'loaded_hole_4.png'))

    # Isotropic — theta=45 deg
    plot_stress_field(
        load=load, diameter=diameter, thickness=thickness, theta=np.pi / 4,
        compliance_matrix=compliance_matrix,
        length_plot=length_plot,
        filename=os.path.join(path_images, 'loaded_hole_5.png'))
