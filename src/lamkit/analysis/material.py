'''
Material class.
'''

import numpy as np
import itertools


class Material(object):
    '''
    Material class.
    
    Parameters
    ----------
    name: str
        Name of the material.
    properties: dict
        Properties of the material.
    '''
    def __init__(self, name: str, properties: dict,
                check_larc05: bool = True) -> None:
        
        self._name = name
        self._properties = properties
        self.check_property(check_larc05=check_larc05)
        
        self._nu21 = self.get_property('nu12') * (self.get_property('E22') / self.get_property('E11'))
        self._Q_0 = None
        self._invariants = None

    def check_property(self, check_larc05: bool = True):
        '''
        Check whether the properties dictionary contains all the required keys.
        '''
        required_keys_elasticity = ['E11', 'E22', 'nu12', 'G12']
        if not all(key in self._properties for key in required_keys_elasticity):
            raise ValueError(f'Properties dictionary for [Elasticity] must contain all the required keys: {required_keys_elasticity}')
        
        # Check LaRC05 failure criteria properties if required.
        if check_larc05:
            required_keys_lrac05 = ['Xt', 'Xc', 'Yt', 'Yc', 'Sl',
                                'a0', 'nL', 'nT', 'ILSS', 'Zt', 
                                'G1cMat', 'G2cMat', 'G1cFibT', 'G1cFibK', 'GAlphaM']
            if not all(key in self._properties for key in required_keys_lrac05):
                raise ValueError(f'Properties dictionary for [LaRC05 Failure Criteria] must contain all the required keys: {required_keys_lrac05}')

    def get_property(self, key: str) -> float | np.ndarray:
        '''
        Get the property value by key.
        '''
        if key == 'nu21':
            return self._nu21
        elif key == 'Q' or key == 'stiffness_matrix':
            return self.Q
        elif key == 'invariants':
            return self.invariants
        elif key == 'compliance_matrix':
            return self.compliance_matrix
        elif key not in self._properties:
            raise ValueError(f'Property {key} not found in the properties dictionary.')
        
        return self._properties[key]
    
    def __call__(self, key: str) -> float:
        '''
        Get the property value by key.
        '''
        return self.get_property(key)

    @property
    def name(self) -> str:
        '''
        Name of the material.
        '''
        return self._name

    @property
    def Q(self) -> np.ndarray:
        '''
        Reduced stiffness matrix, [Q] matrix (3x3) of the Material.
        
        - `[sigma_1, sigma_2, tau_12]^T = [Q] * [epsilon_1, epsilon_2, gamma_12]^T`
        - `epsilon_1 = epsilon0_1 + z*kappa_1`
        - `epsilon_2 = epsilon0_2 + z*kappa_2`
        - `gamma_12 = gamma0_12 + z*kappa_12`
        - `epsilon0_1, epsilon0_2, gamma0_12` are the mid-plane strains.
        - `kappa_1, kappa_2, kappa_12` are the curvatures.
        '''
        if self._Q_0 is None:
            
            m = 1-self.get_property('nu12')*self.get_property('nu21')
            
            Q11 = self.get_property('E11') / m
            Q12 = self.get_property('E22') * self.get_property('nu12') / m
            Q22 = self.get_property('E22') / m
            Q66 = self.get_property('G12')
            self._Q_0 = np.array([[Q11, Q12, 0],
                                   [Q12, Q22, 0],
                                   [0, 0, Q66]])
        return self._Q_0
    
    @property
    def compliance_matrix(self) -> np.ndarray:
        '''
        Compliance matrix (3x3) of the Material.
        '''
        return np.linalg.inv(self.Q)
    
    @property
    def invariants(self) -> np.ndarray:
        '''
        Material invariants, i.e., [U1, U2, U3, U4, U5].
        '''
        if self._invariants is None:
            
            Q11 = self.Q[0, 0]
            Q12 = self.Q[0, 1]
            Q22 = self.Q[1, 1]
            Q66 = self.Q[2, 2]

            U1 = 1/8 * (3*Q11 + 3*Q22 + 2*Q12 + 4*Q66)
            U2 = 1/2 * (Q11 - Q22)
            U3 = 1/8 * (Q11 + Q22 - 2*Q12 - 4*Q66)
            U4 = 1/8 * (Q11 + Q22 + 6*Q12 - 4*Q66)
            U5 = 1/8 * (Q11 + Q22 - 2*Q12 + 4*Q66)
            
            self._invariants = np.array([U1, U2, U3, U4, U5])
            
        return self._invariants


    def get_rotation_matrix(self, angle_degree: float) -> np.ndarray:
        '''
        Get the 2D rotation matrix for the material.
        
        Use this matrix (and its inverse on the appropriate side) when transforming
        **stress-like** Voigt vectors `[sigma_1, sigma_2, tau_12]^T`. It matches the
        tensor rotation with a factor of 2 on the shear coupling terms, consistent with
        `tau_ij` as the engineering shear stress in the stress-strain law.
        
        Parameters
        ----------
        angle_degree: float
            Angle in degrees.
            
        Returns
        -------
        rotation_matrix: np.ndarray [3, 3]
            Rotation matrix.
        '''
        c = np.cos(angle_degree*np.pi/180)
        s = np.sin(angle_degree*np.pi/180)
        return np.array([[c**2, s**2, 2*c*s],
                         [s**2, c**2, -2*c*s],
                         [-c*s, c*s, c**2-s**2]])
    
    def get_inverse_rotation_matrix(self, angle_degree: float) -> np.ndarray:
        '''
        Get the inverse of the 2D rotation matrix for the material.
        
        Pre-multiplies `[Q]` in `[Q_bar]` because stresses transform with the
        inverse of the strain/engineering transform so that
        `[sigma_bar] = [Q_bar][epsilon_bar]` holds in the rotated axes.
        
        Parameters
        ----------
        angle_degree: float
            Angle in degrees.
            
        Returns
        -------
        inverse_rotation_matrix: np.ndarray [3, 3]
            Inverse rotation matrix.
        '''
        c = np.cos(angle_degree*np.pi/180)
        s = np.sin(angle_degree*np.pi/180)
        return np.array([[c**2, s**2, -2*c*s],
                         [s**2, c**2, 2*c*s],
                         [c*s, -c*s, c**2-s**2]])
        
    def get_engineering_rotation_matrix(self, angle_degree: float) -> np.ndarray:
        '''
        Get the 2D engineering rotation matrix for the material.
        
        Use this when transforming **engineering strain**
        `[epsilon_1, epsilon_2, gamma_12]^T`, where `gamma_12 = 2 * epsilon_12`
        (tensor shear). That extra factor changes the third row/column of the
        transform relative to the stress matrix; omitting it would mix tensor and
        engineering shear definitions.
        
        Parameters
        ----------
        angle_degree: float
            Angle in degrees.
            
        Returns
        -------
        engineering_rotation_matrix: np.ndarray [3, 3]
            Engineering rotation matrix.
        '''
        c = np.cos(angle_degree*np.pi/180)
        s = np.sin(angle_degree*np.pi/180)
        return np.array([[c**2, s**2, c*s],
                         [s**2, c**2, -c*s],
                         [-2*c*s, 2*c*s, c**2-s**2]])

    def get_Q_bar(self, angle_degree: float) -> np.ndarray:
        '''
        Transformed reduced stiffness matrix, [Q_bar] matrix (3x3) of the Material.
        
        Implemented as `inv( stress-rotate ) @ Q @ ( engineering-strain rotate )` so
        the same Voigt definitions as in `Q` (stresses vs. `epsilon` and `gamma`)
        apply in the off-axis system.
        
        Parameters
        ----------
        angle_degree: float
            Angle in degrees.
            
        Returns
        -------
        Q_bar: np.ndarray [3, 3]
            Transformed reduced stiffness matrix.
        '''
        return self.get_inverse_rotation_matrix(angle_degree) @ self.Q @ self.get_engineering_rotation_matrix(angle_degree)


class Ply():
    '''
    Ply (or Lamina) class.
    
    Parameters
    ----------
    material: Material
        Material object.
    thickness: float
        Thickness of the ply (mm).
    name: str
        Name of the ply (optional).
    '''

    _ids = itertools.count(1) # counts number of instances in order to generate names

    def __init__(self, material: Material, 
                    thickness: float = 0.125, 
                    name: str = None) -> None:
        
        self._material = material
        
        self._id = next(self._ids)
        self.thickness = thickness
        self._name = name
  
    def get_property(self, key: str) -> float | np.ndarray:
        '''
        Get the property value by key.
        '''
        if key == 'thickness':
            return self.thickness
        elif key == 'name':
            return self.name
        return self._material.get_property(key)
  
    def __call__(self, key: str) -> float:
        '''
        Get the property value by key.
        '''
        return self.get_property(key)

    @property
    def name(self) -> str:
        '''
        Name of the ply.
        '''
        if self._name is None:
            self._name = f'ply_{self._id}'
            return self._name
        else:
            return self._name

    def get_Q_bar(self, angle_degree: float) -> np.ndarray:
        '''
        Get the transformed reduced stiffness matrix of the ply.
        
        Parameters
        ----------
        angle_degree: float
            Angle in degrees.
            
        Returns
        -------
        Q_bar: np.ndarray [3, 3]
            Transformed reduced stiffness matrix.
        '''
        return self._material.get_Q_bar(angle_degree)


MATERIAL_IM7_8551_7 = {
    'E11':      1.6500E+5,  # Young’s modulus along the fibre direction (MPa)
    'E22':      8.4000E+3,  # Young’s modulus along transverse direction (MPa)
    'nu12':     3.4000E-1,  # Poisson’s ratio (unitless)
    'G12':      5.6000E+3,  # In plane shear modulus 1-2 plane (MPa)
    'Xt':       2.5600E+3,  # Longitudinal tensile strength (MPa)
    'Xc':       1.5900E+3,  # Longitudinal compressive strength (MPa)
    'Yt':       7.3000E+1,  # Transverse tensile strength (MPa)
    'Yc':       1.8500E+2,  # Transverse compressive strength (MPa)
    'Sl':       9.0000E+1,  # In plane shear strength (MPa)
    'a0':       5.3000E+1,  # Fracture angle for pure compression (degree)
    'nL':       8.2000E-2,  # Longitudinal shear friction coefficient (unitless)
    'nT':       None,       # Transverse shear friction coefficient (unitless)
    'ILSS':     9.0000E+1,  # Inter-laminar shear strength (MPa)
    'Zt':       6.3000E+1,  # Tensile strength in the 3rd direction (MPa)
    'G1cMat':   2.1000E-1,  # Matrix mode I  critical energy release rate (N/mm, kJ/m^2)
    'G2cMat':   8.0000E-1,  # Matrix mode II critical energy release rate (N/mm, kJ/m^2)
    'G1cFibT':  9.2000E+1,  # Fibre tensile  critical energy release rate (N/mm, kJ/m^2)
    'G1cFibK':  8.0000E+1,  # Fibre kinking  critical energy release rate (N/mm, kJ/m^2)
    'GAlphaM':  1.2100E+0,  # Matrix mixed-mode power law exponent (unitless)
}

'''
MPa usually used along with mm
GPa usually used along with m
'''

IM7_8551_7 = Material('IM7/8551-7', MATERIAL_IM7_8551_7)
