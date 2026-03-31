'''
This is a modified version of the composipy package.
It is used to calculate the mechanical properties of a laminate.

Reference:
    https://github.com/rafaelpsilva07/composipy
    
Author: Runze Li @ Department of Aeronautics, Imperial College London
Date: 2025-10-29
'''

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union, Any

from lamkit.analysis.material import Ply
from lamkit.analysis.larc05 import FAILURE_MODE_NAMES, LaRC05


class Laminate():
    '''
    Laminate class for Classical Lamination Theory (CLT).

    Parameters
    ----------
    stacking : list or dict
        To define a angle stacking sequence
            An iterable containing the angles (in degrees) of layup.
        To define a stack based on lamination parameters.
            {xiA: [xiA1, xiA2, xiA3, xiA4],
            xiB: [xiB1, xiB2, xiB3, xiB4],
            xiD: [xiD1, xiD2, xiD3, xiD4],
            T: thickness}
    plies : Ply or list
        A single Ply or a list of Ply object

    Units
    ------
    Angles: degrees
    Thickness: mm
    Load: N
    Moment: N*mm
    Strain: unitless
    Stress: MPa
    Material properties: MPa and mm

    Note
    -----
    The first element of the stacking list corresponds to the BOTTOM OF THE LAYUP, 
    and the last element corresponds to the TOP OF THE LAYUP.
    This is important for non-symmetric laminates.

    Refer to https://github.com/rafaelpsilva07/composipy/issues/28.
    '''

    def __init__(self, stacking: List[float]|Dict[str, List[float]],
                    plies: List[Ply]|Ply) -> None:
        
        self.stacking = stacking

        # Checking layup
        if not isinstance(stacking, dict): # implements angle stacking sequence

            if isinstance(plies, Ply):
                n_plies = len(stacking)
                plies = [plies for _ in range(n_plies)]
            elif len(plies) != len(stacking):
                raise ValueError('Number of plies and number of stacking must match')

            xiA = None
            xiB = None
            xiD = None
            total_thickness = sum([ply.thickness for ply in plies])
            layup = list(zip(stacking, plies)) # [(angle, ply), ...]
        
        else:
            try:
                xiA = stacking['xiA']
            except KeyError:
                xiA = None
            try:             
                xiB = stacking['xiB']
            except KeyError:
                xiB = None
            try: 
                xiD = stacking['xiD']
            except KeyError:
                KeyError('xiD must be a key')
            try:
                total_thickness = stacking['T']
            except KeyError:
                KeyError('T must be a key')
            layup = []

        self.plies = plies
        self.ply_material = plies[0]._material
        self.layup = layup
        self._z_position = None
        self._Q_layup = None
        self._T_layup = None
        self._A = None
        self._B = None
        self._D = None
        self._ABD = None
        self._ABD_inverse_matrix = None
        self._xiA = xiA
        self._xiB = xiB
        self._xiD = xiD
        self._total_thickness = total_thickness
        self._S = None

    def __repr__(self) -> str:
        representation = f'Laminate\n'
        representation += f'stacking = {self.stacking}'
        return representation

    def __eq__(self, other) -> bool:
        if isinstance(other, Laminate):
            return (self.layup == other.layup)
        return NotImplemented

    @property
    def n_ply(self) -> int:
        '''
        Number of plies in the laminate.
        '''
        return len(self.plies)

    @property
    def stacking_sequence(self) -> List[float]:
        '''
        Stacking sequence (ply angle, degrees) of the laminate.
        '''
        return [angle for angle, _ in self.layup]

    @property
    def z_position(self) -> List[float]:
        '''
        Z coordinates of the ply surfaces in the laminate.
        
        Returns
        -------
        z_position: List[float] (n_ply + 1,)
            Z-position of the ply surfaces in the laminate.
        '''
        return np.cumsum([0] + [ply.thickness for ply in self.plies]) - self._total_thickness/2
    
    @property
    def Q_layup(self) -> List[np.ndarray]:
        '''
        Transformed reduced stiffness matrix of each ply in the laminate.
        
        Returns
        -------
        Q_layup: List[np.ndarray [3, 3]]
            Transformed reduced stiffness matrix of each ply in the laminate.
        '''
        if self._Q_layup is None:
            self._Q_layup = [ply.get_Q_bar(theta) for theta, ply in self.layup]
        return self._Q_layup
     
    @property
    def T_layup(self) -> List[np.ndarray]:
        '''
        Transformation matrix of each ply in the laminate.
        
        Returns
        -------
        T_layup: List[Tuple[np.ndarray [3, 3], np.ndarray [3, 3]]]
            Transformation matrix of each ply in the laminate.
            The first ndarray is the transformation matrix for this ply.
            The second ndarray is the engineering transformation matrix for this ply.
        '''
        if self._T_layup is None:
            self._T_layup = []
            for theta in self.layup:
                T_real = self.ply_material.get_rotation_matrix(theta)
                T_engineering = self.ply_material.get_engineering_rotation_matrix(theta)
                self._T_layup.append([T_real,T_engineering])
        return self._T_layup
   
    @property
    def xiA(self) -> np.ndarray:
        '''
        Lamination parameter xiA for extension

        Returns
        -------
        xiA : np.ndarray (4,)
            Lamination parameter xiA
        '''
        xiA = np.zeros(4)
        T = sum([ply.thickness for ply in self.plies])        
        for i, angle in enumerate(self.stacking):
            angle *= np.pi / 180
            zk1 = self.z_position[i+1]
            zk0 = self.z_position[i]

            xiA[0] += (zk1-zk0) * np.cos(2*angle)
            xiA[1] += (zk1-zk0) * np.sin(2*angle)
            xiA[2] += (zk1-zk0) * np.cos(4*angle)
            xiA[3] += (zk1-zk0) * np.sin(4*angle)                        
        
        self._xiA = xiA / T
        return self._xiA

    @property
    def xiB(self) -> np.ndarray:
        '''
        Lamination parameter xiB for extension-bending coupling.

        Returns
        -------
        xiB : np.ndarray (4,)
            (4/T²) Σ_k (z_{k+1}² - z_k²) [cos2θ, sin2θ, cos4θ, sin4θ] per ply k.
        '''
        if isinstance(self.stacking, dict):
            if self._xiB is None:
                raise ValueError(
                    'xiB is required in stacking dict for coupling parameters'
                )
            return np.asarray(self._xiB, dtype=float)

        xiB = np.zeros(4)
        T = sum([ply.thickness for ply in self.plies])
        for i, angle in enumerate(self.stacking):
            angle *= np.pi / 180
            zk1 = self.z_position[i+1]
            zk0 = self.z_position[i]

            dz2 = zk1**2 - zk0**2
            xiB[0] += dz2 * np.cos(2*angle)
            xiB[1] += dz2 * np.sin(2*angle)
            xiB[2] += dz2 * np.cos(4*angle)
            xiB[3] += dz2 * np.sin(4*angle)

        self._xiB = 4 * xiB / T**2
        return self._xiB
    
    @property
    def xiD(self) -> np.ndarray:
        '''
        Lamination parameter xiD for bending
        
        Returns
        -------
        xiD : np.ndarray (4,)
            Lamination parameter xiD
        '''
        xiD = np.zeros(4)
        T = sum([ply.thickness for ply in self.plies])        
        for i, angle in enumerate(self.stacking):
            angle *= np.pi / 180
            zk1 = self.z_position[i+1]
            zk0 = self.z_position[i]

            xiD[0] += (zk1**3-zk0**3) * np.cos(2*angle)
            xiD[1] += (zk1**3-zk0**3) * np.sin(2*angle)
            xiD[2] += (zk1**3-zk0**3) * np.cos(4*angle)
            xiD[3] += (zk1**3-zk0**3) * np.sin(4*angle)
        self._xiD = 4 * xiD / T**3
        return self._xiD

    @property
    def A(self) -> np.ndarray:
        '''
        [A] matrix of the laminate for extension.
        
        Returns
        -------
        A : np.ndarray (3x3)
            [A] Matrix of the laminate
        '''
        if self._A is None:
            self._A = np.zeros(9).reshape(3,3)

            for i in enumerate(self.Q_layup):
                zk1 = self.z_position[i[0]+1]
                zk0 = self.z_position[i[0]]
                self._A += (zk1-zk0) * i[1]
        return self._A
    
    @property
    def B(self) -> np.ndarray:
        '''
        [B] matrix of the laminate for coupling between extension and bending.

        Returns
        -------
        B : np.ndarray (3x3)
            [B] matrix of the laminate
        '''
        if self._B is None:
            self._B = np.zeros((3,3))

            for i in enumerate(self.Q_layup):
                zk1 = self.z_position[i[0]+1]
                zk0 = self.z_position[i[0]]
                self._B += (1/2) * (zk1**2-zk0**2) * i[1]
        return self._B
    
    @property
    def D(self) -> np.ndarray:
        '''
        [D] matrix of the laminate for bending.

        Returns
        -------
        D : np.ndarray (3x3)
            [D] matrix of the laminate
        '''
        if self._D is None:
            self._D = np.zeros((3,3))

            for i in enumerate(self.Q_layup):
                zk1 = self.z_position[i[0]+1]
                zk0 = self.z_position[i[0]]
                self._D += (1/3) * (zk1**3-zk0**3) * i[1]
        return self._D

    @property
    def ABD(self) -> np.ndarray:
        '''
        [ABD] matrix of the laminate

        Returns
        -------
        ABD : np.ndarray (6x6)
            ABD matrix of the laminate
        '''
        if self._ABD is None:
            self._ABD = np.vstack([
                np.hstack([self.A, self.B]),
                np.hstack([self.B, self.D])
                ])
        return self._ABD
    
    @property
    def ABD_inverse_matrix(self) -> np.ndarray:
        '''
        Get the inverse of the ABD matrix of the laminate.
        '''
        if self._ABD_inverse_matrix is None:
            self._ABD_inverse_matrix = np.linalg.inv(self.ABD)
        return self._ABD_inverse_matrix
    
    @property
    def ABD_determinant(self) -> float:
        '''
        Get the determinant of the ABD matrix of the laminate.
        '''
        return np.linalg.det(self.ABD)
    
    @property
    def ABD_eigenvalues(self) -> np.ndarray:
        '''
        Calculate eigenvalues of the ABD matrix.
        
        Returns raw eigenvalues (no normalization) that can be directly compared
        across different layups with the same material.
        
        Returns
        -------
        eigenvalues : np.ndarray (6,)
            Eigenvalues of the ABD matrix, sorted in descending order.
            Returns raw eigenvalues that can be compared directly.
        
        Notes
        -----
        The ABD matrix is symmetric, so all eigenvalues are real.
        The eigenvalues are sorted in descending order (largest first).
        '''
        # Get ABD matrix
        abd = self.ABD
        
        # Calculate eigenvalues (ABD matrix is symmetric, so eigenvalues are real)
        eigenvalues = np.linalg.eigvals(abd)
        
        # Extract real part (should be real for symmetric matrix, but handle numerical precision)
        eigenvalues = np.real(eigenvalues)
        
        # Sort eigenvalues in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        return eigenvalues
   
    @property
    def in_plane_stiffness_matrix(self) -> np.ndarray:
        '''
        Get the in-plane stiffness matrix of the laminate.
        '''
        return self.A
    
    @property
    def in_plane_compliance_matrix(self) -> np.ndarray:
        '''
        Equivalent plane-stress compliance for a homogeneous plate (3x3).

        Returns h * inv(A), i.e. S_eq = (A/h)^(-1), where h is total thickness
        and A is the CLT extensional stiffness. This relates thickness-averaged
        in-plane stress sigma_bar = N/h to mid-plane strain:
        epsilon0 = S_eq @ sigma_bar (with N = A @ epsilon0 under pure membrane
        response, B = 0 and kappa = 0).

        This is NOT inv(A): the latter maps stress resultants N to epsilon0
        (epsilon0 = inv(A) @ N) and has different physical dimensions.

        Notes
        -----
        For 2D hole problems (e.g. Lekhnitskii), the solver expects the material
        compliance S in epsilon = S @ sigma (Pa-level stresses). Using S_eq
        matches that convention for an equivalent homogeneous laminate.
        If B is non-zero, an equivalent S_eq is only an approximation.
        '''
        if self._S is None:
            # S_eq = (A/h)^(-1) for epsilon0 = S_eq @ (N/h); see docstring.
            self._S = self._total_thickness * np.linalg.inv(self.A)
        return self._S
   

    def get_mid_plane_strains(self, N: np.ndarray) -> np.ndarray:
        '''
        Get the mid plane strains of the laminate.

        Parameters
        ------------------
        N: np.ndarray
            Forces and moments, i.e., [Nxx, Nyy, Nxy, Mxx, Myy, Mxy].
            
        Returns
        ------------------
        epsilon0: np.ndarray (6,)
            Mid plane strains, i.e., [epsilon_x0, epsilon_y0, gamma_xy0, kappa_x0, kappa_y0, kappa_xy0].
        '''
        return self.ABD_inverse_matrix @ N

    def _ply_invariants(self) -> np.ndarray:
        '''[U1..U5] from the first ply (uniform-material laminates).'''
        return self.ply_material.get_property('invariants')

    
    def get_A_from_lamination_parameters(self) -> np.ndarray:
        '''
        Calculate the [A] matrix from the lamination parameters.
        
        Returns
        -------
        A: np.ndarray (3x3)
            [A] matrix of the laminate
        '''
        U1, U2, U3, U4, U5 = self._ply_invariants()
        xi1, xi2, xi3, xi4 = self.xiA
        T = self._total_thickness
        A11 = T*(U1 + U2*xi1 + U3*xi3)
        A12 = T*(-U3*xi3 + U4)
        A13 = T*(U2*xi2/2 + U3*xi4)
        A21 = T*(-U3*xi3 + U4)
        A22 = T*(U1 - U2*xi1 + U3*xi3)
        A23 = T*(U2*xi2/2 - U3*xi4)
        A31 = T*(U2*xi2/2 + U3*xi4)
        A32 = T*(U2*xi2/2 - U3*xi4)
        A33 = T*(-U3*xi3 + U5)

        return np.array([[A11, A12, A13],
                        [A21, A22, A23],
                        [A31, A32, A33]])

    def get_B_from_lamination_parameters(self) -> np.ndarray:
        '''
        Calculate the [B] matrix from the lamination parameters.
        
        Returns
        -------
        B: np.ndarray (3x3)
            [B] matrix of the laminate
        '''
        _, U2, U3, _, _ = self._ply_invariants()
        xi1, xi2, xi3, xi4 = self.xiB
        T = self._total_thickness
        fac = T**2 / 8.0
        # Invariant terms proportional to U1, U4, U5 drop out: Σ_k (z_{k+1}² - z_k²) = 0.
        B11 = fac * (U2*xi1 + U3*xi3)
        B12 = fac * (-U3*xi3)
        B13 = fac * (U2*xi2/2 + U3*xi4)
        B21 = B12
        B22 = fac * (-U2*xi1 + U3*xi3)
        B23 = fac * (U2*xi2/2 - U3*xi4)
        B31 = B13
        B32 = B23
        B33 = fac * (-U3*xi3)

        return np.array([[B11, B12, B13],
                        [B21, B22, B23],
                        [B31, B32, B33]])

    def get_D_from_lamination_parameters(self) -> np.ndarray:
        '''
        Calculate the [D] matrix from the lamination parameters.
        
        Returns
        -------
        D: np.ndarray (3x3)
            [D] matrix of the laminate
        '''
        U1, U2, U3, U4, U5 = self._ply_invariants()
        xi1, xi2, xi3, xi4 = self.xiD
        T = self._total_thickness

        D11 = T**3*(U1 + U2*xi1 + U3*xi3)/12
        D12 = T**3*(-U3*xi3 + U4)/12
        D13 = T**3*(U2*xi2/2 + U3*xi4)/12
        D21 = T**3*(-U3*xi3 + U4)/12
        D22 = T**3*(U1 - U2*xi1 + U3*xi3)/12
        D23 = T**3*(U2*xi2/2 - U3*xi4)/12
        D31 = T**3*(U2*xi2/2 + U3*xi4)/12
        D32 = T**3*(U2*xi2/2 - U3*xi4)/12
        D33 = T**3*(-U3*xi3 + U5)/12

        return np.array([[D11, D12, D13],
                        [D21, D22, D23],
                        [D31, D32, D33]])


    def get_effective_properties(self) -> Dict[str, float]:
        '''
        Get the effective properties of the laminate.
        '''
        S_eff = self.in_plane_compliance_matrix
        E11_eff = 1/S_eff[0, 0]
        E22_eff = 1/S_eff[1, 1]
        G12_eff = 1/S_eff[2, 2]
        nu12_eff = -S_eff[0, 1] / S_eff[0, 0]
        nu21_eff = -S_eff[1, 0] / S_eff[1, 1]
        
        return {
            'E11_eff': E11_eff,
            'E22_eff': E22_eff,
            'G12_eff': G12_eff,
            'nu12_eff': nu12_eff,
            'nu21_eff': nu21_eff,
        }
    

    #* Static methods
    @staticmethod
    def get_lamination_parameters(stacking: List[float]) -> Dict[str, float]:
        '''
        Calculate the lamination parameters of the stacking sequence,
        i.e., ply angles (degrees).
        
        Note: the plies are assumed to have the same material and thickness.
        
        Returns
        -------
        lamination_parameters: Dict[str, float]
            Lamination parameters, i.e.,
            {'xiA': [xiA1, xiA2, xiA3, xiA4],
            'xiB': [xiB1, xiB2, xiB3, xiB4],
            'xiD': [xiD1, xiD2, xiD3, xiD4]}.
        '''
        xiA = np.zeros(4)
        xiB = np.zeros(4)
        xiD = np.zeros(4)
        n_ply = len(stacking)
        z_position = np.cumsum([0] + [1.0 for _ in stacking]) - n_ply/2.0
        stacking = np.deg2rad(stacking) # radians

        for i, angle in enumerate(stacking):
            
            zk1 = z_position[i+1]
            zk0 = z_position[i]
            
            c2 = np.cos(2*angle)
            s2 = np.sin(2*angle)
            c4 = np.cos(4*angle)
            s4 = np.sin(4*angle)

            dz = zk1 - zk0
            xiA[0] += dz * c2
            xiA[1] += dz * s2
            xiA[2] += dz * c4
            xiA[3] += dz * s4
        
            dz2 = zk1**2 - zk0**2
            xiB[0] += dz2 * c2
            xiB[1] += dz2 * s2
            xiB[2] += dz2 * c4
            xiB[3] += dz2 * s4

            dz3 = zk1**3 - zk0**3
            xiD[0] += dz3 * c2
            xiD[1] += dz3 * s2
            xiD[2] += dz3 * c4
            xiD[3] += dz3 * s4
        
        xiA = xiA / n_ply
        xiB = 4 * xiB / n_ply**2
        xiD = 4 * xiD / n_ply**3
        
        return {
            'xiA': xiA,
            'xiB': xiB,
            'xiD': xiD,
        }
    
    
    @staticmethod
    def get_epsilon0(ABD: np.ndarray, N: np.ndarray) -> np.ndarray:
        '''
        Get the mid plane strains of the laminate.
        
        Parameters
        ----------
        ABD: np.ndarray (6x6)
            ABD matrix of the laminate
            (Material properties described in MPa and mm)
            
        N: np.ndarray (6,)
            Load vector, [Nxx, Nyy, Nxy, Mxx, Myy, Mxy].
            Nxx, Nyy, Nxy: in-plane forces (N/mm)
            Mxx, Myy, Mxy: bending moments (N)

        Returns
        -------
        epsilon0: np.ndarray (6,)
            Mid plane strains, i.e.,
            [epsilon_x0, epsilon_y0, gamma_xy0, kappa_x0, kappa_y0, kappa_xy0].
        '''
        return np.linalg.inv(ABD) @ N

    @staticmethod
    def strain_xy_at_z(epsilon6: np.ndarray, z: Union[np.ndarray, float]) -> np.ndarray:
        '''
        Global engineering strains [ex, ey, gxy] through the thickness (CLT).

        Parameters
        ----------
        epsilon6 : np.ndarray (6,)
            Mid-plane generalised strains
            [ex0, ey0, gxy0, kx, ky, kxy].
        z : np.ndarray or float
            Through-thickness coordinate(s) (mm), same convention as z_position.

        Returns
        -------
        epsilon_xy : np.ndarray (n_z, 3)
            Rows are [ex, ey, gxy] at each z.
        '''
        zv = np.atleast_1d(np.asarray(z, dtype=float))
        e0 = np.asarray(epsilon6[:3], dtype=float)
        k = np.asarray(epsilon6[3:6], dtype=float)
        return e0 + zv[:, np.newaxis] * k

    @staticmethod
    def strain_xy_global_to_material(epsilon_xy: np.ndarray, theta_deg: float) -> np.ndarray:
        '''
        Transform engineering strains from plate x-y to ply material 1-2.

        Same convention as get_epsilon_plies_123 / NASA handbook.
        '''
        v = np.asarray(epsilon_xy, dtype=float).reshape(3).copy()
        th = np.radians(theta_deg)
        c, s = np.cos(th), np.sin(th)
        v[2] /= 2.0
        T = np.array(
            [
                [c**2, s**2, 2 * c * s],
                [s**2, c**2, -2 * c * s],
                [-c * s, c * s, c**2 - s**2],
            ]
        )
        e123 = T @ v
        e123 = np.asarray(e123, dtype=float).copy()
        e123[2] *= 2.0
        return e123

    @staticmethod
    def stress_xy_global_from_strain(epsilon_xy: np.ndarray, Q_bar: np.ndarray) -> np.ndarray:
        '''Global stresses [sx, sy, txy] from global strains and transformed stiffness [Q_bar].'''
        exy = np.asarray(epsilon_xy, dtype=float).reshape(3)
        return Q_bar @ exy

    @staticmethod
    def stress_material_from_strain(
        epsilon_xy: np.ndarray, Q_material: np.ndarray, theta_deg: float
        ) -> np.ndarray:
        '''Material stresses [s1, s2, t12] from global strains and ply [Q] in material axes.'''
        e123 = Laminate.strain_xy_global_to_material(epsilon_xy, theta_deg)
        return Q_material @ e123


    def get_ply_level_results(self, epsilon0: np.ndarray, larc05: LaRC05 = None) -> List[Dict[str, Any]]:
        '''
        Get the ply-level results of the laminate.
        
        Parameters
        ----------
        epsilon0: np.ndarray (6,)
            Mid-plane strains, i.e.,
            `[epsilon_x0, epsilon_y0, gamma_xy0, kappa_x0, kappa_y0, kappa_xy0]`.
        larc05: LaRC05, optional
            LaRC05 object. If None, the failure indices are not calculated.
            
        Returns
        -------
        results: List[Dict[str, Any]]
            List of dictionaries, each containing the results for a ply.
            Length is `2*n_ply`.
        '''
        z_pos = self.z_position
        results = []
        for index_ply in range(self.n_ply):
            theta, ply_obj = self.layup[index_ply]
            theta = float(theta)
            z_bottom = float(z_pos[index_ply])
            z_top = float(z_pos[index_ply + 1])
            Q_bar = ply_obj.get_Q_bar(theta)
            Q_mat = np.asarray(ply_obj('Q'), dtype=float)

            for index_surface, z_eval in ((0, z_bottom), (1, z_top)):
                exy = Laminate.strain_xy_at_z(epsilon0, z_eval)[0]
                sig_xy = Laminate.stress_xy_global_from_strain(exy, Q_bar)
                s123 = Laminate.stress_material_from_strain(exy, Q_mat, theta)
                e123 = Laminate.strain_xy_global_to_material(exy, theta)

                if larc05 is not None:
                    uvarm = larc05.evaluate(np.asarray(s123, dtype=float))
                    fi_block = uvarm[:5]
                    fi_max = float(np.max(fi_block))
                    mode_idx = int(np.argmax(fi_block)) + 1
                    failure_mode = FAILURE_MODE_NAMES[mode_idx]
                else:
                    fi_block = np.zeros(5)
                    fi_max = 0.0
                    failure_mode = 'not calculated'

                results.append(
                    {
                        'index_ply': index_ply,
                        'index_surface': index_surface,
                        'z': z_eval,
                        'angle': theta,
                        'sigma_x': float(sig_xy[0]),
                        'sigma_y': float(sig_xy[1]),
                        'tau_xy': float(sig_xy[2]),
                        'sigma_1': float(s123[0]),
                        'sigma_2': float(s123[1]),
                        'tau_12': float(s123[2]),
                        'epsilon_x': float(exy[0]),
                        'epsilon_y': float(exy[1]),
                        'gamma_xy': float(exy[2]),
                        'epsilon_1': float(e123[0]),
                        'epsilon_2': float(e123[1]),
                        'gamma_12': float(e123[2]),
                        'FI_matrix_cracking': float(fi_block[0]),
                        'FI_matrix_splitting': float(fi_block[1]),
                        'FI_fibre_tension': float(fi_block[2]),
                        'FI_fibre_kinking': float(fi_block[3]),
                        'FI_matrix_interface': float(fi_block[4]),
                        'FI_max': fi_max,
                        'failure_mode': failure_mode,
                    }
                )
        return results

    def evaluate_laminate(self, N: np.ndarray) -> pd.DataFrame:
        '''
        Evaluate the failure field of the laminate,
        the laminate consists of plies with the same material.
        
        Parameters
        ----------
        N: np.ndarray (6,)
            Load vector, [Nxx, Nyy, Nxy, Mxx, Myy, Mxy].
            Nxx, Nyy, Nxy: in-plane forces (N/mm)
            Mxx, Myy, Mxy: bending moments (N)

        Returns
        -------
        field_results: pd.DataFrame
            One row per ply face, ordered from the bottom of the layup upward (increasing z).
            
            Columns:
            - index_ply: 0-based ply index, same order as `laminate.layup` (0 = bottom ply)
            - index_surface: (0, 1) = bottom/top face of the ply
            - z: z coordinate of the ply (bottom/top) face (mm)
            - angle: ply angle (degree)
            - sigma_x, sigma_y, tau_xy: global stresses (MPa)
            - sigma_1, sigma_2, tau_12: material stresses (MPa)
            - epsilon_x, epsilon_y, gamma_xy: global strains (unitless)
            - epsilon_1, epsilon_2, gamma_12: material strains (unitless)
            - FI_*: LaRC05 failure indices (unitless)
            - FI_max: maximum LaRC05 failure index (unitless)
            - failure_mode: failure mode (string)
            
            Attributes:
            - epsilon0: mid-plane generalized strains (ndarray (6,))
                accessed as `field_results.attrs['epsilon0']`,
                which is `[epsilon_x0, epsilon_y0, gamma_xy0, kappa_x0, kappa_y0, kappa_xy0]`.
            - global_FI_*: maximum failure indices of all plies
        '''
        N = np.asarray(N, dtype=float).reshape(6)
        epsilon0 = self.get_mid_plane_strains(N)
        
        larc05 = LaRC05(nSCply=3, material_properties=self.ply_material.properties_dictionary)
        
        results = self.get_ply_level_results(epsilon0, larc05)

        out = pd.DataFrame.from_records(results)
        
        out.attrs['epsilon0'] = np.asarray(epsilon0, dtype=float)
        out.attrs['global_FI_matrix_cracking'] = np.max(out['FI_matrix_cracking'])
        out.attrs['global_FI_matrix_splitting'] = np.max(out['FI_matrix_splitting'])
        out.attrs['global_FI_fibre_tension'] = np.max(out['FI_fibre_tension'])
        out.attrs['global_FI_fibre_kinking'] = np.max(out['FI_fibre_kinking'])
        out.attrs['global_FI_matrix_interface'] = np.max(out['FI_matrix_interface'])
        out.attrs['global_FI_max'] = np.max(out['FI_max'])
        
        return out


