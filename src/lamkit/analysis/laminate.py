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
from typing import List, Tuple, Dict

from lamkit.analysis.material import Ply
from lamkit.analysis.larc05 import LaRC05


class Laminate():
    '''
    This class creates a Laminate object.
    It requires Ply objects (to define plies) and angle information.

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

    Note
    -----
    The first element of the stacking list corresponds to the BOTTOM OF THE LAYUP, 
    and the last element corresponds to the TOP OF THE LAYUP.
    This is important for non-symmetric laminates.

    Refer to https://github.com/rafaelpsilva07/composipy/issues/28.
    '''

    def __init__(self, 
                stacking: List[float]|Dict[str, List[float]],
                plies: Ply|List[Ply]) -> None:
        
        self.ply_material = plies
        
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
            layup = list(zip(stacking, plies))
        
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

        self.stacking = stacking
        self.plies = plies
        self.layup = layup
        self._z_position = None
        self._Q_layup = None
        self._T_layup = None
        self._A = None
        self._B = None
        self._D = None
        self._ABD = None
        self._ABD_p = None
        self._xiA = xiA
        self._xiB = xiB
        self._xiD = xiD
        self._total_thickness = total_thickness

    def __repr__(self) -> str:
        representation = f'Laminate\n'
        representation += f'stacking = {self.stacking}'
        return representation

    def __eq__(self, other) -> bool:
        if isinstance(other, Laminate):
            return (self.layup == other.layup)
        return NotImplemented

    @property
    def z_position(self) -> List[float]:
    
        total_thickness = 0
        for t in self.layup:
            total_thickness += t[1].thickness
        
        current_z = -total_thickness/2
        ply_position = [current_z]
        for t in self.layup:
            current_z += t[1].thickness
            ply_position.append(current_z)
        
        return ply_position
    
    @property
    def Q_layup(self) -> List[np.ndarray]:    
        if self._Q_layup is None:
            
            self._Q_layup = []
            for theta, ply in self.layup:
                c = np.cos(theta*np.pi/180)
                s = np.sin(theta*np.pi/180)

                T_real = np.array([
                    [c**2, s**2, 2*c*s],
                    [s**2, c**2, -2*c*s],
                    [-c*s, c*s, c**2-s**2]
                    ])
                T_engineering =  np.array([
                    [c**2, s**2, c*s],
                    [s**2, c**2, -c*s],
                    [-2*c*s, 2*c*s, c**2-s**2]
                    ])

                self._Q_layup.append(
                    (np.linalg.inv(T_real))
                    @ ply('Q_0')
                    @ T_engineering
                    )
        return self._Q_layup
     
    @property
    def T_layup(self) -> List[np.ndarray]:
        if self._T_layup is None:
            
            self._T_layup = []
            for theta in self.layup:
                c = np.cos(theta[0]*np.pi/180)
                s = np.sin(theta[0]*np.pi/180)

                T_real = np.array([
                    [c**2, s**2, 2*c*s],
                    [s**2, c**2, -2*c*s],
                    [-c*s, c*s, c**2-s**2]
                    ])

                T_engineering =  np.array([
                    [c**2, s**2, c*s],
                    [s**2, c**2, -c*s],
                    [-2*c*s, 2*c*s, c**2-s**2]
                    ])

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
        '''[A] Matrix as numpy.ndarray '''

        if not self._xiA is None:

            U1, U2, U3, U4, U5 = self.ply_material('invariants')
            xi1, xi2, xi3, xi4 = self._xiA
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

            self._A = np.array([[A11, A12, A13],
                                [A21, A22, A23],
                                [A31, A32, A33]])

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
        Matrix [B] will be zero if defined using lamination parameters.
               
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
        [D] matrix of the laminate

        Returns
        -------
        D : np.ndarray (3x3)
            [D] matrix of the laminate
        '''

        if not self._xiD is None:

            U1, U2, U3, U4, U5 = self.ply_material('invariants')
            xi1, xi2, xi3, xi4 = self._xiD
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

            self._D = np.array([[D11, D12, D13],
                                [D21, D22, D23],
                                [D31, D32, D33]])

        if self._D is None:
            self._D = np.zeros(9).reshape(3,3)

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
        Get the in-plane compliance matrix of the laminate.
        '''
        compliance = self._total_thickness * np.linalg.inv(self.A)
        return compliance
   
    def get_lamination_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Get the lamination parameters of the laminate.
        
        Returns
        ------------------
        xiA: np.ndarray
            Lamination parameters of tension.
            
        xiD: np.ndarray
            Lamination parameters of bending.
        '''
        return self.xiA, self.xiD

    def get_mid_plane_strains(self, N: np.ndarray) -> np.ndarray:
        '''
        Get the mid plane strains of the laminate.

        Parameters
        ------------------
        N: np.ndarray
            Forces and moments, i.e., [Nxx, Nyy, Nxy, Mxx, Myy, Mxy].
            
        Returns
        ------------------
        epsilon0: np.ndarray
            Mid plane strains, i.e., [epsilon_x0, epsilon_y0, epsilon_xy0, kappa_x0, kappa_y0, kappa_xy0].
        '''
        return self.ABD_inverse_matrix @ N
    

    def calculate_strain(self, N: np.ndarray) -> pd.DataFrame:
        '''
        Calculates strain ply by at laminate direction and material direction.

        Parameters
        ----------
        N: np.ndarray (6,)
            Load vector, [Nxx, Nyy, Nxy, Mxx, Myy, Mxy]

        Returns
        -------
        strains : pd.Dataframe
            ply by ply strains in plate direction and material direction       

        Note
        ----
        The sequence of the DataFrame starts from the TOP OF THE LAYUP to the BOTTOM OF THE LAYUP, which is the reverse of the definition order.
        When defining the laminate, the first element of the list corresponds to the bottom-most layer. This is especially important for non-symmetric laminates.
        '''
        return self.get_strain_field(self.ABD, N, 
                                self.stacking, self.z_position)

    def calculate_stress(self, N: np.ndarray) -> pd.DataFrame:
        '''
        Calculates stress ply by at laminate direction and material direction.

        Parameters
        ----------
        N: np.ndarray (6,)
            Load vector, [Nxx, Nyy, Nxy, Mxx, Myy, Mxy]

        Returns
        -------
        stress : pd.Dataframe
            ply by ply stress in plate direction and material direction       

        Note
        ----
        The sequence of the DataFrame starts from the TOP OF THE LAYUP to the BOTTOM OF THE LAYUP, which is the reverse of the definition order.
        When defining the laminate, the first element of the list corresponds to the bottom-most layer. This is especially important for non-symmetric laminates.
        '''
        return self.get_stress_field(self.ABD, N, 
                        self.stacking, self.z_position, self.Q_layup)


    #* Static methods
    
    @staticmethod
    def get_epsilon0(ABD: np.ndarray, N: np.ndarray) -> np.ndarray:
        '''
        Get the mid plane strains of the laminate.
        
        Parameters
        ----------
        ABD: np.ndarray (6x6)
            ABD matrix of the laminate
        '''
        return np.linalg.inv(ABD) @ N

    @staticmethod
    def get_epsilon_plies(epsilon0: np.ndarray, n_ply: int,
                z_positions: List[float]
                ) -> List[Tuple[np.ndarray, np.ndarray]]:
        '''
        Calculates the strains for each ply from the mid-plane strains,
        the strains are in the global coordinate system (x,y).
        
        Parameters
        ----------
        epsilon0 : np.ndarray (6,)
            Strain vector at the mid-plane
            
        n_ply : int
            Number of plies
            
        z_positions : List[float]
            List of z-positions for each ply
            
        Returns
        -------
        epsilon_plies: list of tuples
            For the k-th ply, epsilon_k = (epsilon_top, epsilon_bot), 
            where epsilon_top and epsilon_bot are np.ndarray([epsilon_x, epsilon_y, gamma_xy]).
            The list of tuples is in the order of the plies, from top to bottom.
            For reference refer to page 145 of Daniel equation 5.8
            
        '''
        epsilon_mid = np.array([epsilon0[0], epsilon0[1], epsilon0[2]])
        kappa_mid = np.array([epsilon0[3], epsilon0[4], epsilon0[5]])
        
        # Reverse z_positions so bot is negative and top is positive
        z = z_positions[::-1]
        
        epsilon_plies = []
        
        for i in range(n_ply):
            epsilon_plies.append(
                (epsilon_mid + z[i] * kappa_mid,
                 epsilon_mid + z[i+1] * kappa_mid)
                )
        
        return epsilon_plies

    @staticmethod
    def get_epsilon_plies_123(stacking: List[float],
                epsilon_plies: List[Tuple[np.ndarray, np.ndarray]]
                ) -> List[Tuple[np.ndarray, np.ndarray]]:
        '''
        Transforms strains from global coordinates (x,y) to material coordinates (1,2,3).
        
        Parameters
        ----------
        stacking : List[float]
            List of ply angles in degrees
            
        epsilon_plies : List[Tuple[np.ndarray, np.ndarray]]
            List of strain tuples (epsilon_top, epsilon_bot) for each ply in global coordinates
            
        Returns
        -------
        epsilon_plies_123: List[Tuple[np.ndarray, np.ndarray]]
            For the k-th ply, epsilon_k_123 = (epsilon_top_123, epsilon_bot_123),
            where epsilon_top_123 and epsilon_bot_123 are np.ndarray([epsilon_1, epsilon_2, gamma_12]).
            The list of tuples is in the order of the plies, from top to bottom.
            For reference refer to NASA pg 50
        '''
        epsilon_plies_123 = []
        for theta, epsilon_k in zip(stacking, epsilon_plies):
            epsilon_top, epsilon_bot = epsilon_k
            # Make copies to avoid modifying the originals
            epsilon_top = epsilon_top.copy()
            epsilon_bot = epsilon_bot.copy()
            
            c = np.cos(theta*np.pi/180)
            s = np.sin(theta*np.pi/180)
            epsilon_top[2] /= 2 # engineering shear strain (see nasa pg 50)
            epsilon_bot[2] /= 2

            T = np.array([
                [c**2, s**2, 2*c*s],
                [s**2, c**2, -2*c*s],
                [-c*s, c*s, c**2-s**2]
                ])

            cur_epsilon_top = T @ epsilon_top
            cur_epsilon_bot = T @ epsilon_bot
            cur_epsilon_top[2] *= 2
            cur_epsilon_bot[2] *= 2
            epsilon_plies_123.append((cur_epsilon_top, cur_epsilon_bot)) # engineering shear strain (see nasa pg 50)
        
        return epsilon_plies_123


    @staticmethod
    def get_stress_plies(Q_layup: List[np.ndarray], 
                epsilon_plies: List[Tuple[np.ndarray, np.ndarray]]
                ) -> List[Tuple[np.ndarray, np.ndarray]]:
        '''
        Calculates the stresses for each ply from the strains.
        
        Parameters
        ----------
        Q_layup : List[np.ndarray]
            List of Q matrices (stiffness matrices) for each ply
            
        epsilon_plies : List[Tuple[np.ndarray, np.ndarray]]
            List of strain tuples (epsilon_top, epsilon_bot) for each ply
            
        Returns
        -------
        stress_plies: list of tuples
            For the k-th ply, stress_k = (stress_top, stress_bot),
            where stress_top and stress_bot are np.ndarray([sigma_x, sigma_y, tau_xy]).
            The list of tuples is in the order of the plies, from top to bottom.
            For reference refer to page 145 of Daniel equation 5.8
        '''
        stress_plies = []
        for Q_k, epsilon_k in zip(Q_layup, epsilon_plies):
            epsilon_top, epsilon_bot = epsilon_k
            stress_plies.append(
                (Q_k @ epsilon_top,
                 Q_k @ epsilon_bot)
            )
        return stress_plies
    
    @staticmethod
    def get_stress_plies_123(stacking: List[float], 
                stress_plies: List[Tuple[np.ndarray, np.ndarray]]
                ) -> List[Tuple[np.ndarray, np.ndarray]]:
        '''
        Transforms stresses from global coordinates (x,y) 
        to material coordinates (1,2,3).
        
        Parameters
        ----------
        stacking : List[float]
            List of ply angles in degrees
            
        stress_plies : List[Tuple[np.ndarray, np.ndarray]]
            List of stress tuples (stress_top, stress_bot) for each ply in global coordinates
            
        Returns
        -------
        stress_plies_123: List[Tuple[np.ndarray, np.ndarray]]
            For the k-th ply, stress_k_123 = (stress_top_123, stress_bot_123),
            where stress_top_123 and stress_bot_123 are np.ndarray([sigma_1, sigma_2, tau_12]).
            The list of tuples is in the order of the plies, from top to bottom.
            For reference refer to NASA pg 50
        '''
        stress_plies_123 = []
        for theta, stress in zip(stacking, stress_plies):
            stress_top, stress_bot = stress   
            c = np.cos(theta*np.pi/180)
            s = np.sin(theta*np.pi/180)
            #stress_top[2] /= 2 # engineering shear strain (see nasa pg 50)
            #stress_bot[2] /=

            T = np.array([
                [c**2, s**2, 2*c*s],
                [s**2, c**2, -2*c*s],
                [-c*s, c*s, c**2-s**2]
                ])

            cur_stress_top = T @ stress_top
            cur_stress_bot = T @ stress_bot
            #cur_stresstop[2] *= 2
            #cur_stress_bot[2] *= 2
            stress_plies_123.append((cur_stress_top, cur_stress_bot)) 
        
        return stress_plies_123


    @staticmethod
    def get_strain_field(ABD: np.ndarray, N: np.ndarray, 
                stacking: List[float], 
                z_positions: List[float]) -> pd.DataFrame:
        '''
        Calculates the strain field of the laminate
        
        Parameters
        ----------
        ABD: np.ndarray (6x6)
            ABD matrix of the laminate
            
        N: np.ndarray (6,)
            Load vector, [Nxx, Nyy, Nxy, Mxx, Myy, Mxy]
            
        stacking: List[float]
            List of ply angles in degrees
            
        z_positions: List[float]
            List of z-positions for each ply
            
        Returns
        -------
        strain_field: pd.DataFrame
            Strain field of the laminate, ply by ply in plate direction and material direction
        '''
        n_ply = len(stacking)
        epsilon0 = Laminate.get_epsilon0(ABD, N)
        epsilon_plies = Laminate.get_epsilon_plies(epsilon0, n_ply, z_positions)
        epsilon_plies_123 = Laminate.get_epsilon_plies_123(
                                stacking, epsilon_plies)
        
        cur_ply = 1
        data = {}
        data['ply'] = []
        data['position'] = []
        data['angle'] = []       
        data['epsilonx'] = []
        data['epsilony'] = []
        data['gammaxy'] = []
        data['epsilon1'] = []
        data['epsilon2'] = []
        data['gamma12'] = []
        for epsilon, epsilon123, theta in zip(epsilon_plies, epsilon_plies_123, stacking):
            epsilon_top, epsilon_bot = epsilon
            epsilon_top_123, epsilon_bot_123 = epsilon123

            data['ply'].append(cur_ply)
            data['ply'].append(cur_ply)
            data['position'].append('top')
            data['position'].append('bot')
            data['angle'].append(theta)
            data['angle'].append(theta)            
            data['epsilonx'].append(epsilon_top[0]) #plate direction
            data['epsilonx'].append(epsilon_bot[0])
            data['epsilony'].append(epsilon_top[1])
            data['epsilony'].append(epsilon_bot[1])
            data['gammaxy'].append(epsilon_top[2])
            data['gammaxy'].append(epsilon_bot[2])
            data['epsilon1'].append(epsilon_top_123[0]) #material direction
            data['epsilon1'].append(epsilon_bot_123[0])
            data['epsilon2'].append(epsilon_top_123[1])
            data['epsilon2'].append(epsilon_bot_123[1])
            data['gamma12'].append(epsilon_top_123[2])
            data['gamma12'].append(epsilon_bot_123[2])
            cur_ply += 1
        pd.set_option('display.precision', 2)
        return pd.DataFrame(data)

    @staticmethod
    def get_stress_field(ABD: np.ndarray, N: np.ndarray, 
                stacking: List[float], z_positions: List[float], 
                Q_layup: List[np.ndarray]) -> pd.DataFrame:
        '''
        Calculates the stress field of the laminate
        
        Parameters
        ----------
        ABD: np.ndarray (6x6)
            ABD matrix of the laminate
            
        N: np.ndarray (6,)
            Load vector, [Nxx, Nyy, Nxy, Mxx, Myy, Mxy]
            
        stacking: List[float]
            List of ply angles in degrees
            
        z_positions: List[float]
            List of z-positions for each ply
            
        Q_layup: List[np.ndarray]
            List of Q matrices (stiffness matrices) for each ply
            
        Returns
        -------
        stress_field: pd.DataFrame
            Stress field of the laminate, ply by ply in plate direction and material direction
        '''
        n_ply = len(stacking)
        epsilon0 = Laminate.get_epsilon0(ABD, N)
        epsilon_plies = Laminate.get_epsilon_plies(epsilon0, n_ply, z_positions)
        stress_plies = Laminate.get_stress_plies(Q_layup, epsilon_plies)
        stress_plies_123 = Laminate.get_stress_plies_123(stacking, stress_plies)
        
        cur_ply = 1
        data = {}
        data['ply'] = []
        data['position'] = []
        data['angle'] = []
        data['sigmax'] = []
        data['sigmay'] = []
        data['tauxy'] = []
        data['sigma1'] = []
        data['sigma2'] = []
        data['tau12'] = []
        for sigma, sigma_123, theta in zip(stress_plies, stress_plies_123, stacking):
            stress_top, stress_bot = sigma
            sigma_top_123, sigma_bot_123 = sigma_123

            data['ply'].append(cur_ply)
            data['ply'].append(cur_ply)
            data['position'].append('top')
            data['position'].append('bot')
            data['angle'].append(theta)
            data['angle'].append(theta)
            data['sigmax'].append(stress_top[0]) #plate direction
            data['sigmax'].append(stress_bot[0])
            data['sigmay'].append(stress_top[1])
            data['sigmay'].append(stress_bot[1])
            data['tauxy'].append(stress_top[2])
            data['tauxy'].append(stress_bot[2])
            data['sigma1'].append(sigma_top_123[0]) #material direction
            data['sigma1'].append(sigma_bot_123[0])
            data['sigma2'].append(sigma_top_123[1])
            data['sigma2'].append(sigma_bot_123[1])
            data['tau12'].append(sigma_top_123[2])
            data['tau12'].append(sigma_bot_123[2])
            cur_ply += 1
        pd.set_option('display.precision', 2)
        return pd.DataFrame(data)


def get_failure_field(stress_field: pd.DataFrame) -> pd.DataFrame:
    '''
    Get the failure field of the laminate.
    
    Parameters
    ----------
    stress_field: pd.DataFrame
        Stress field of the laminate, ply by ply in plate direction and material direction.
        Including the following columns:
        [ply, position, angle, sigmax, sigmay, tauxy, sigma1, sigma2, tau12]
        
    Returns
    -------
    failure_field: pd.DataFrame
        Failure indices of the laminate plies, including the following columns:
        [ply, position, angle, FI_max, FI_matrix_cracking, FI_matrix_splitting, FI_fibre_tension, FI_fibre_kinking, FI_matrix_interface]
    '''
    failure_field = pd.DataFrame(
        columns=['ply', 'position', 'angle', 
                    'FI_max', 'FI_matrix_cracking', 'FI_matrix_splitting', 
                    'FI_fibre_tension', 'FI_fibre_kinking', 'FI_matrix_interface'],
        )
    
    larc05 = LaRC05(nSCply=3) # 2D element, 3 stress components
    
    for index, row in stress_field.iterrows():
        
        ply_stresses = np.array([row['sigma1'], row['sigma2'], row['tau12']])
        failure_indices = larc05.get_uvarm(ply_stresses)
        
        failure_field.at[index, 'ply'] = row['ply']
        failure_field.at[index, 'position'] = row['position']
        failure_field.at[index, 'angle'] = row['angle']
        failure_field.at[index, 'FI_max'] = failure_indices[5]
        failure_field.at[index, 'FI_matrix_cracking'] = failure_indices[0]
        failure_field.at[index, 'FI_matrix_splitting'] = failure_indices[1]
        failure_field.at[index, 'FI_fibre_tension'] = failure_indices[2]
        failure_field.at[index, 'FI_fibre_kinking'] = failure_indices[3]
        failure_field.at[index, 'FI_matrix_interface'] = failure_indices[4]
    
    return failure_field

    

