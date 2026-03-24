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
    def __init__(self, name: str, properties: dict) -> None:
        
        self._name = name
        self._properties = properties
        self.check_property()
        
        self._nu21 = self.get_property('nu12') * (self.get_property('E22') / self.get_property('E11'))
        self._Q_0 = None
        self._invariants = None

    def check_property(self):
        '''
        Check whether the properties dictionary contains all the required keys.
        '''
        required_keys = ['E11', 'E22', 'nu12', 'G12', 
                         'Xt', 'Xc', 'Yt', 'Yc', 'Sl',
                         'a0', 'nL', 'nT', 'ILSS', 'Zt', 
                         'G1cMat', 'G2cMat', 'G1cFibT', 'G1cFibK', 'GAlphaM']
        if not all(key in self._properties for key in required_keys):
            raise ValueError(f'Properties dictionary must contain all the required keys: {required_keys}')

    def get_property(self, key: str) -> float | np.ndarray:
        '''
        Get the property value by key.
        '''
        if key == 'nu21':
            return self._nu21
        elif key == 'Q_0' or key == 'stiffness_matrix':
            return self.Q_0
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
    def Q_0(self) -> np.ndarray:
        '''
        Reduced stiffness matrix, [Q] matrix (3x3) of the Material      
        '''
        if self._Q_0 is None:
            Q11 = self.get_property('E11') / (1-self.get_property('nu12')*self.get_property('nu21')) 
            Q12 = (self.get_property('nu12')*self.get_property('E22')) / (1-self.get_property('nu12')*self.get_property('nu21'))
            Q22 = self.get_property('E22') / (1-self.get_property('nu12')*self.get_property('nu21'))
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
        return np.linalg.inv(self.Q_0)

    @property
    def invariants(self) -> np.ndarray:
        '''
        Material invariants, i.e., [U1, U2, U3, U4, U5].
        '''
        if self._invariants is None:
            
            Q11 = self.Q_0[0, 0]
            Q12 = self.Q_0[0, 1]
            Q22 = self.Q_0[1, 1]
            Q66 = self.Q_0[2, 2]

            U1 = 1/8 * (3*Q11 + 3*Q22 + 2*Q12 + 4*Q66)
            U2 = 1/2 * (Q11 - Q22)
            U3 = 1/8 * (Q11 + Q22 - 2*Q12 - 4*Q66)
            U4 = 1/8 * (Q11 + Q22 + 6*Q12 - 4*Q66)
            U5 = 1/8 * (Q11 + Q22 - 2*Q12 + 4*Q66)
            
            self._invariants = np.array([U1, U2, U3, U4, U5])
            
        return self._invariants


class Ply():
    '''
    Ply class.
    
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


MATERIAL_IM7_8551_7 = {
    'E11':      1.6500E+5,  # Young’s modulus along the fibre direction
    'E22':      8.4000E+3,  # Young’s modulus along transverse direction
    'nu12':     3.4000E-1,  # Poisson’s ratio
    'G12':      5.6000E+3,  # In plane shear modulus 1-2 plane
    'Xt':       2.5600E+3,  # Longitudinal tensile strength
    'Xc':       1.5900E+3,  # Longitudinal compressive strength
    'Yt':       7.3000E+1,  # Transverse tensile strength 
    'Yc':       1.8500E+2,  # Transverse compressive strength
    'Sl':       9.0000E+1,  # In plane shear strength
    'a0':       5.3000E+1,  # Fracture angle for pure compression (53 degree)
    'nL':       8.2000E-2,  # Longitudinal shear friction coefficient
    'nT':       None,       # Transverse shear friction coefficient
    'ILSS':     9.0000E+1,  # Inter-laminar shear strength
    'Zt':       6.3000E+1,  # Tensile strength in the 3rd direction
    'G1cMat':   2.1000E-1,  # Matrix mode I  critical energy release rate
    'G2cMat':   8.0000E-1,  # Matrix mode II critical energy release rate
    'G1cFibT':  9.2000E+1,  # Fibre tensile  critical energy release rate
    'G1cFibK':  8.0000E+1,  # Fibre kinking  critical energy release rate
    'GAlphaM':  1.2100E+0,  # Matrix mixed-mode power law exponent (1.21)
}

IM7_8551_7 = Material('IM7/8551-7', MATERIAL_IM7_8551_7)
