'''
Failure criteria class.
'''

import numpy as np


class FailureCriteria(object):
    '''
    Abstract base class for failure criteria.
    '''

    def __init__(self) -> None:
        raise NotImplementedError('Class not implemented')
        
    def evaluate(self, stresses: np.ndarray, **kwargs) -> np.ndarray:
        '''
        Evaluate the failure criteria.
        
        Parameters
        ----------
        stresses: np.ndarray
            Stresses at the point of interest.
            
        **kwargs: dict
            Additional keyword arguments.
            
        Returns
        -------
        failure_indices: np.ndarray
            Failure indices at the point of interest.
        '''
        raise NotImplementedError('Method not implemented')
        
