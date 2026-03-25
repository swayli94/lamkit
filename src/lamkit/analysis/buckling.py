"""
This is a modified version of the composipy package.
It is used to calculate the buckling load of a laminate plate.

Reference:
    https://github.com/rafaelpsilva07/composipy

Author: Runze Li @ Department of Aeronautics, Imperial College London
Date: 2026-03-25
"""

from itertools import product
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import os

from lamkit.analysis.laminate import Laminate
from lamkit.components.functions import sxieta
from lamkit.components.build_k import (
    calc_K11_ijkl,
    calc_k12_ijkl,
    calc_k13_ijkl,
    calc_k21_ijkl,
    calc_k22_ijkl,
    calc_k23_ijkl,
    calc_k31_ijkl,
    calc_k32_ijkl,
    calc_k33_ijkl,
    calc_kG33_ijkl,
)


ConstraintDict = Dict[str, Iterable[str]]


class BucklingAnalysis:
    """
    Buckling analysis of a laminate plate based on Ritz approximation.

    Parameters
    ----------
    laminate : Laminate
        Laminate object that provides `A`, `D`, and `ABD` stiffness matrices.
    a : float
        Plate length along x direction (mm).
    b : float
        Plate length along y direction (mm).
    constraints : str | dict, default "PINNED"
        Boundary-condition definition. Built-in options: "PINNED", "CLAMPED".
        A custom dict can define edge constraints for keys `x0`, `xa`, `y0`, `yb`.
    Nxx, Nyy, Nxy : float, default 0.0
        In-plane pre-buckling loads, force per unit length (N/mm).
        `Fx = Nxx * b`, `Fy = Nyy * a`, `Fxy = Nxy * a * b`.
    m, n : int, default 10
        Number of Ritz terms in x and y directions.
    """

    def __init__(self, laminate: Laminate, a: float, b: float,
                constraints: str | ConstraintDict = "PINNED",
                Nxx: float = 0.0, Nyy: float = 0.0, Nxy: float = 0.0,
                m: int = 10, n: int = 10) -> None:

        if not isinstance(laminate, Laminate):
            raise TypeError("laminate must be a lamkit.analysis.laminate.Laminate instance")
        if float(a) <= 0 or float(b) <= 0:
            raise ValueError("a and b must be positive")
        if int(m) < 1 or int(n) < 1:
            raise ValueError("m and n must be >= 1")

        self._laminate = laminate
        self._a = float(a)
        self._b = float(b)
        self._constraints = constraints
        self._Nxx = float(Nxx)
        self._Nyy = float(Nyy)
        self._Nxy = float(Nxy)
        self._m = int(m)
        self._n = int(n)

        self.su_idx = None
        self.sv_idx = None
        self.sw_idx = None
        self.eigenvalue = None
        self.eigenvector = None

    @property
    def laminate(self) -> Laminate:
        """Return the laminate object used by this analysis."""
        return self._laminate

    @property
    def a(self) -> float:
        """Return plate length in x direction."""
        return self._a

    @property
    def b(self) -> float:
        """Return plate length in y direction."""
        return self._b

    @property
    def constraints(self) -> str | ConstraintDict:
        """Return boundary condition definition."""
        return self._constraints

    @property
    def Nxx(self) -> float:
        """Return in-plane normal load in x direction."""
        return self._Nxx

    @property
    def Nyy(self) -> float:
        """Return in-plane normal load in y direction."""
        return self._Nyy

    @property
    def Nxy(self) -> float:
        """Return in-plane shear load."""
        return self._Nxy

    @property
    def m(self) -> int:
        """Return number of Ritz terms along x."""
        return self._m

    @property
    def n(self) -> int:
        """Return number of Ritz terms along y."""
        return self._n

    def _compute_constraints(self) -> Tuple[list, list, list]:
        """
        Build Ritz index sets that satisfy the selected boundary conditions.

        The method filters basis-function families for in-plane (`u`, `v`) and
        out-of-plane (`w`) fields according to constrained translations/rotations
        on each edge, then builds Cartesian products used for matrix assembly.

        Returns
        -------
        tuple[list, list, list]
            `(uidx, vidx, widx)` index lists for assembling stiffness terms.
        """
        if self.constraints == "PINNED":
            x0 = xa = y0 = yb = ["TX", "TY", "TZ"]
        elif self.constraints == "CLAMPED":
            x0 = xa = y0 = yb = ["TX", "TY", "TZ", "RX", "RY", "RZ"]
        else:
            if not isinstance(self.constraints, dict):
                raise TypeError("constraints must be 'PINNED', 'CLAMPED' or a constraint dict")
            x0 = list(self.constraints.get("x0", []))
            xa = list(self.constraints.get("xa", []))
            y0 = list(self.constraints.get("y0", []))
            yb = list(self.constraints.get("yb", []))

        sm = [i for i in range(self.m + 4)]
        sn = [i for i in range(self.n + 4)]

        um, un = sm.copy(), sn.copy()
        vm, vn = sm.copy(), sn.copy()
        wm, wn = sm.copy(), sn.copy()

        if "TX" in x0:
            um.remove(0)
        if "TY" in x0:
            vm.remove(0)
        if "TZ" in x0:
            wm.remove(0)
        if "RX" in x0:
            um.remove(1)
        if "RY" in x0:
            vm.remove(1)
        if "RZ" in x0:
            wm.remove(1)

        if "TX" in xa:
            um.remove(2)
        if "TY" in xa:
            vm.remove(2)
        if "TZ" in xa:
            wm.remove(2)
        if "RX" in xa:
            um.remove(3)
        if "RY" in xa:
            vm.remove(3)
        if "RZ" in xa:
            wm.remove(3)

        if "TX" in y0:
            un.remove(0)
        if "TY" in y0:
            vn.remove(0)
        if "TZ" in y0:
            wn.remove(0)
        if "RX" in y0:
            un.remove(1)
        if "RY" in y0:
            vn.remove(1)
        if "RZ" in y0:
            wn.remove(1)

        if "TX" in yb:
            un.remove(2)
        if "TY" in yb:
            vn.remove(2)
        if "TZ" in yb:
            wn.remove(2)
        if "RX" in yb:
            un.remove(3)
        if "RY" in yb:
            vn.remove(3)
        if "RZ" in yb:
            wn.remove(3)

        um, un = um[0 : self.m], un[0 : self.n]
        vm, vn = vm[0 : self.m], vn[0 : self.n]
        wm, wn = wm[0 : self.m], wn[0 : self.n]

        uidx = list(product(um, un, um, un))
        vidx = list(product(vm, vn, vm, vn))
        widx = list(product(wm, wn, wm, wn))

        self.su_idx = list(product(um, un))
        self.sv_idx = list(product(vm, vn))
        self.sw_idx = list(product(wm, wn))

        return (uidx, vidx, widx)

    def calc_K_KG_ABD(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble structural and geometric stiffness matrices using full ABD coupling.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            `(K, KG)` where `K` is the structural stiffness matrix and `KG` is
            the geometric stiffness matrix.
        """
        # Read laminate extensional/coupling/bending stiffness entries.
        A11, A12, A16, B11, B12, B16 = self.laminate.ABD[0, :]
        A12, A22, A26, B12, B22, B26 = self.laminate.ABD[1, :]
        A16, A26, A66, B16, B26, B66 = self.laminate.ABD[2, :]
        B11, B12, B16, D11, D12, D16 = self.laminate.ABD[3, :]
        B12, B22, B26, D12, D22, D26 = self.laminate.ABD[4, :]
        B16, B26, B66, D16, D26, D66 = self.laminate.ABD[5, :]

        k11, k12, k13, k21, k22, k23, k31, k32, k33 = [], [], [], [], [], [], [], [], []
        k33g = []

        uidx, vidx, widx = self._compute_constraints()
        size = self.m**2 * self.n**2

        # Loop over Ritz index combinations and evaluate pre-integrated terms.
        for i in range(size):
            ui, uj, uk, ul = uidx[i]
            vi, vj, vk, vl = vidx[i]
            wi, wj, wk, wl = widx[i]

            k11.append(calc_K11_ijkl(self.a, self.b, ui, uj, uk, ul, A11, A16, A66))
            k12.append(calc_k12_ijkl(self.a, self.b, ui, uj, vk, vl, A12, A16, A26, A66))
            k13.append(calc_k13_ijkl(self.a, self.b, ui, uj, wk, wl, B11, B12, B16, B26, B66))
            k21.append(calc_k21_ijkl(self.a, self.b, vi, vj, uk, ul, A12, A16, A26, A66))
            k22.append(calc_k22_ijkl(self.a, self.b, vi, vj, vk, vl, A22, A26, A66))
            k23.append(calc_k23_ijkl(self.a, self.b, vi, vj, wk, wl, B12, B16, B22, B26, B66))
            k31.append(calc_k31_ijkl(self.a, self.b, wi, wj, uk, ul, B11, B12, B16, B26, B66))
            k32.append(calc_k32_ijkl(self.a, self.b, wi, wj, vk, vl, B11, B12, B16, B22, B26, B66))
            k33.append(calc_k33_ijkl(self.a, self.b, wi, wj, wk, wl, D11, D12, D22, D16, D26, D66))
            k33g.append(calc_kG33_ijkl(self.a, self.b, wi, wj, wk, wl, self.Nxx, self.Nyy, self.Nxy))

        dim = self.m * self.n
        k11 = np.array(k11).reshape(dim, dim)
        k12 = np.array(k12).reshape(dim, dim)
        k13 = np.array(k13).reshape(dim, dim)
        k21 = np.array(k21).reshape(dim, dim)
        k22 = np.array(k22).reshape(dim, dim)
        k23 = np.array(k23).reshape(dim, dim)
        k31 = np.array(k31).reshape(dim, dim)
        k32 = np.array(k32).reshape(dim, dim)
        k33 = np.array(k33).reshape(dim, dim)
        k00 = np.zeros((dim, dim))
        k33g = np.array(k33g).reshape(dim, dim)

        # Build global block matrices:
        # K = [[Kuu, Kuv, Kuw], [Kvu, Kvv, Kvw], [Kwu, Kwv, Kww]]
        K = np.vstack([np.hstack([k11, k12, k13]), np.hstack([k21, k22, k23]), np.hstack([k31, k32, k33])])
        KG = np.vstack([np.hstack([k00, k00, k00]), np.hstack([k00, k00, k00]), np.hstack([k00, k00, k33g])])

        return K, KG

    def calc_K_KG_D(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble stiffness matrices using bending-only (`D`) approximation.

        This reduced formulation keeps only the transverse displacement field
        in the buckling eigen problem.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            `(K, KG)` bending stiffness and geometric stiffness matrices.
        """
        # Read bending stiffness matrix entries.
        D11, D12, D16 = self.laminate.D[0, :]
        D12, D22, D26 = self.laminate.D[1, :]
        D16, D26, D66 = self.laminate.D[2, :]

        k33 = []
        k33g = []
        _, _, widx = self._compute_constraints()
        size = self.m**2 * self.n**2

        # Evaluate bending and geometric terms for each Ritz pair.
        for i in range(size):
            wi, wj, wk, wl = widx[i]
            k33.append(calc_k33_ijkl(self.a, self.b, wi, wj, wk, wl, D11, D12, D22, D16, D26, D66))
            k33g.append(calc_kG33_ijkl(self.a, self.b, wi, wj, wk, wl, self.Nxx, self.Nyy, self.Nxy))

        dim = self.m * self.n
        K = np.array(k33).reshape(dim, dim)
        KG = np.array(k33g).reshape(dim, dim)
        return K, KG

    def buckling_analysis(self, num_eigvalues: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the generalized eigenvalue buckling problem.

        The current implementation uses the bending-only matrices from
        `calc_K_KG_D` and solves:
            KG * phi = lambda * K * phi
        then converts to load multipliers as `-1/lambda`.

        Parameters
        ----------
        num_eigvalues : int, default 5
            Requested number of lowest-magnitude eigenvalues.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Eigenvalues (load multipliers) and eigenvectors.
        """
        K, KG = self.calc_K_KG_D()
        K, KG = csr_matrix(K), csr_matrix(KG)

        k = min(int(num_eigvalues), KG.shape[0] - 2)
        eigvals, eigvecs = eigsh(A=KG, k=k, which="SM", M=K, tol=0.0, sigma=1.0, mode="cayley")
        eigvals = -1.0 / eigvals

        self.eigenvalue, self.eigenvector = eigvals, eigvecs
        return eigvals, eigvecs


def plot_buckling_modes(analysis: BucklingAnalysis,
            eigvals: np.ndarray, n_modes: int, 
            ngridx: int, ngridy: int, case_text: str, save_path: str) -> None:
    """
    Plot several buckling modes in one figure with multiple subplots.
    """
    n_modes = min(int(n_modes), int(eigvals.shape[0]))
    n_cols = 2
    n_rows = (n_modes + n_cols - 1) // n_cols

    xi_arr = np.linspace(-1.0, 1.0, ngridx)
    eta_arr = np.linspace(-1.0, 1.0, ngridy)
    xi_mesh, eta_mesh = np.meshgrid(xi_arr, eta_arr)
    x_mesh = (analysis.a / 2.0) * (xi_mesh + 1.0)
    y_mesh = (analysis.b / 2.0) * (eta_mesh + 1.0)

    fig = plt.figure(figsize=(7 * n_cols, 5.5 * n_rows))

    for mode_idx in range(n_modes):
        c_values = analysis.eigenvector[:, mode_idx]
        len_w = len(analysis.sw_idx)
        cw_values = c_values[-len_w:]

        z = np.zeros((ngridx, ngridy))
        for i in range(ngridx):
            for j in range(ngridy):
                sw = sxieta(analysis.sw_idx, xi_mesh[i, j], eta_mesh[i, j])
                z[i, j] = float(sw @ cw_values)

        is_buckled = float(eigvals[mode_idx]) <= 1.0
        buckle_text = "buckled" if is_buckled else "not-buckled"

        ax = fig.add_subplot(n_rows, n_cols, mode_idx + 1, projection="3d")
        ax.plot_surface(x_mesh, y_mesh, z, cmap="coolwarm")
        ax.set_title(f"Mode {mode_idx + 1} | {buckle_text}\n(lambda={eigvals[mode_idx]:.3f})")
        ax.set_xticks(np.linspace(0.0, max(analysis.a, analysis.b), 5))
        ax.set_yticks(np.linspace(0.0, max(analysis.a, analysis.b), 5))
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("w (mm)")

    fig.suptitle(f"Case: {case_text}", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

