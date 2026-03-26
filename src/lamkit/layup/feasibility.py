'''
Feasibility rating for layup stacking sequences.
'''

import pandas as pd
from pathlib import Path
import os
import numpy as np
import json
import time
from scipy.spatial import cKDTree
from typing import Final, List, Optional, Dict, Any


DATA_PATH = os.path.join(Path(__file__).resolve().parents[3], "data")
DEFAULT_DATASET_PATH = os.path.join(DATA_PATH, "layup_database-with-attributes-strong28.csv")


class LayupFeasibilityRating(object):
    '''
    Layup feasibility rating class for composite material layup sequences.
    
    Parameters
    ------------------
    path_to_layup_database: str
        Path to the layup database (CSV file).
    weight_xiD: List[float]
        Weight for xiD1, xiD2, xiD3 for distance calculation.
        
    Attributes
    ------------------
    layup_database: pd.DataFrame
        Layup database with attributes.
    data: np.ndarray
        Layup attributes for distance calculation.
    '''
    def __init__(self,
            path_to_layup_database: str = DEFAULT_DATASET_PATH,
            weight: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) -> None:
        
        self.path_to_layup_database: Final[str] = path_to_layup_database
        self._weight: Final[np.ndarray] = np.array(weight, dtype=float)
        
        start_time = time.time()
        
        dataset_path = Path(path_to_layup_database)
        self.layup_database = pd.read_csv(dataset_path)
        
        required = {"layup_id", "n_ply", "sub_id", "stacking", "xiA", "xiB", "xiD", "n_90", "n_0"}
        missing = required.difference(self.layup_database.columns)
        if missing:
            raise ValueError(
                f"Invalid database format in {dataset_path}. Missing columns: {sorted(missing)}"
            )

        # Built once for fast nearest-neighbour distance queries
        self._tree: Optional[cKDTree] = None
        self._features: Optional[np.ndarray] = None

        self._assemble_ndarray()
        
        elapsed_time = time.time() - start_time
        print(f"Layup feasibility rating class initialized in {elapsed_time:.1f} seconds")
    
    @property
    def size(self) -> int:
        '''
        Number of layups in the database.
        '''
        return len(self.layup_database)

    def _assemble_ndarray(self) -> None:
        '''
        Assemble the layup features for distance calculation.
        
        - n_ply: number of plies
        - n_0: number of 0° plies
        - n_90: number of 90° plies
        - xiD1: lamination parameter xiD1
        - xiD2: lamination parameter xiD2
        - xiD3: lamination parameter xiD3
        '''
        self._features = np.zeros((self.size, 6))
        for i in range(self.size):
            xiD = np.array(json.loads(self.layup_database.loc[i, "xiD"]))
            self._features[i, 0] = int(self.layup_database.loc[i, "n_ply"])
            self._features[i, 1] = int(self.layup_database.loc[i, "n_0"])
            self._features[i, 2] = int(self.layup_database.loc[i, "n_90"])
            self._features[i, 3] = xiD[0]
            self._features[i, 4] = xiD[1]
            self._features[i, 5] = xiD[2]

        self._features = self._features * self._weight

        self._tree = cKDTree(self._features, leafsize=32)

    def calculate_distance(self, n_ply: int, n_0: int, n_90: int,
                    xiD1: float, xiD2: float, xiD3: float) -> Dict[str, Any]:
        '''
        Calculate the distance from input values to closest layups in the database.

        Returns
        -------
        distance: float
            Euclidean distance in feature space.
        layup_id: int
            `layup_id` of the closest layup in the database.
        stacking: List[float]
            Stacking sequence of the closest layup in the database.
        '''
        query = np.array([n_ply, n_0, n_90, xiD1, xiD2, xiD3], dtype=float)
        query = query * self._weight

        # Euclidean distance (cKDTree default) to the closest layup in feature space.
        distance, _idx = self._tree.query(query, k=1)
        layup_id = int(self.layup_database.loc[_idx, "layup_id"])
        stacking = json.loads(self.layup_database.loc[_idx, "stacking"])
        stacking = [float(angle) for angle in stacking]
        
        actual_xiD = json.loads(self.layup_database.loc[_idx, "xiD"])
        
        return {
            "distance": float(distance),
            "layup_id": layup_id,
            "stacking": stacking,
            "n_0": int(self.layup_database.loc[_idx, "n_0"]),
            "n_90": int(self.layup_database.loc[_idx, "n_90"]),
            "n_ply": int(self.layup_database.loc[_idx, "n_ply"]),
            "xiD1": float(actual_xiD[0]),
            "xiD2": float(actual_xiD[1]),
            "xiD3": float(actual_xiD[2]),
        }

