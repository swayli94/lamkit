#!/usr/bin/env python3
'''
Engineering requirements filter for composite material layup sequences.
We define the the 0 degree angle as the reference direction, 
which is the primary load carrying direction.

The stacking sequence is determined by the composite guidelines
----------------------------------------------------------------

- Use homogeneous laminate stacking sequence (LSS) for strength-controlled designs.
- LSS should be balanced and symmetric about the mid-plane. Quasi-isotropic patterns (0/45/-45/90)_s or (0/45/90/-45)_s are close to optimum.

- Have at least four distinct ply angles (e.g., 0, +/-45, 90) with a minimum of 10% of the plies oriented at each angle. 
- Use a minimum of 40% of +/-45 plies. 

- Minimize groupings of plies with the same orientation, e.g., 
  stack no more than 4 plies of the same orientation together to reduce or eliminate the fringe delamination. 
- Alternate +/-45 plies through the LSS except for the closest ply either side of the symmetry plane.
- A pair of +/-45 plies should be located as closely as possible while still meeting the other guidelines. 

- Shield primary load carrying plies from exposed surfaces.
    - Avoid 90° surface plies: put ±45° or 0° plies on the outside; keep 90° away from free edges to reduce peel at the edge.
    - For laminates loaded primarily in tension or compression in the 0 degree direction, should start with angle and transverse plies from the surface. 
    - For laminates loaded primarily in shear, should locate +/-45 plies away from the surface.

- Avoid LSS that create high inter-laminar tension stresses at free edges.
    - Minimize the Poisson's ratio mismatch between adjacent laminates that are co-cured or bonded. 
    - Limit angle jumps between adjacent plies: keep adjacent ply angle change ≤ 45° where possible; avoid abrupt 0°↔90° adjacency.
    - Maintain sublaminate symmetry near surfaces: ensure the outer few plies themselves form a symmetric, balanced mini-stack.

- Laminate in-plane properties are dictated by the percentage of layers with different orientation angles
  with respect to some reference direction that make up the total thickness. 
- Bending properties are dictated by the stacking sequence as well as the percentage of layers with different orientations. 
- Percentage of layers with different orientations is more important than the stacking sequence.

Reference:

[1] P. K. Mallick, Composites engineering handbook (1ed), 
CRC Press, 1997, Ch. 12. doi: https://doi.org/10.1201/9781482277739

[2] M. C.-Y. Niu, Composite airframe structures: practical design information and data,
Conmilit Press LTD, Hong Kong, 2005, Ch. 5. doi: https://doi.org/10.1017/S0001924000012471

Author: Runze Li @ Department of Aeronautics, Imperial College London
Date: 2025-11-06
'''

from typing import Final, List, Tuple
from collections import Counter


class EngineeringRequirements(object):
    '''
    Engineering requirements class for composite material layup sequences.
    
    The layup is limited to symmetric layup sequences with [-45, 0, 45, 90] as candidate angles.
    
    Attributes:
    ------------------
    candidate_angles: list
        List of candidate angles in the layup sequence, [-45, 0, 45, 90].
        
    symmetric: bool
        Whether the layup sequence is symmetric.
        
    strong_requirement: bool
        Whether to apply the strong requirement.
        
    max_gap_between_45_plies: int
        Maximum gap between +/-45 plies.
        
    max_n_plies_outer_layer_sub_laminate: int
        The maximum number of plies considered to form a sub-laminate in the outer layer.
        For example, when it is 8, the outer layer sub-laminate is considered to be the first 4-8 plies.
    '''
    
    def __init__(self, strong_requirement: bool = False) -> None:
        
        self.candidate_angles: Final[List[float]] = [-45.0, 0.0, 45.0, 90.0]
        self.symmetric: Final[bool] = True
        
        self.idx_0: Final[int] = self.candidate_angles.index(0.0)
        self.idx_45_p: Final[int] = self.candidate_angles.index(45.0)
        self.idx_45_m: Final[int] = self.candidate_angles.index(-45.0)
        self.idx_90: Final[int] = self.candidate_angles.index(90.0)
        
        self.max_gap_between_45_plies: Final[int] = 2
        self.max_n_plies_outer_layer_sub_laminate: Final[int] = 8
        
        self.strong_requirement = strong_requirement
        
        self._print_violations = False
    
    def __call__(self, layup: List[float]) -> bool:
        '''
        Check if the layup sequence passes the engineering requirements.
        
        Parameters
        ------------------
        layup: List[float]
            Layup sequence with angle values (e.g., [0.0, 45.0, -45.0, 90.0]).
        '''
        return self.filter(layup)
    
    def _angle_to_index(self, angle: float) -> int:
        '''
        Convert angle value to index in candidate_angles list.
        
        Parameters
        ------------------
        angle: float
            Angle value (e.g., -45.0, 0.0, 45.0, 90.0).
        
        Returns
        ------------------
        index: int
            Index of the angle in candidate_angles list.
        
        Raises
        ------------------
        ValueError: If angle is not in candidate_angles list.
        '''
        try:
            return self.candidate_angles.index(angle)
        except ValueError:
            raise ValueError(f"Angle {angle} is not in candidate angles {self.candidate_angles}")
        
    def _angle_difference(self, angle1: float, angle2: float) -> float:
        '''
        Calculate the difference between two angles.
        
        Note that the angle ranges in (-90, 90] degrees, and there is a wrap-around at 90 degrees,
        so that the difference between -45 and 90 is actually 45 degrees.
        
        Parameters
        ------------------
        angle1: float
            First angle.
        
        angle2: float
            Second angle.
        
        Returns
        ------------------
        difference: float
            Difference between two angles.
        '''
        difference = abs(angle1 - angle2)
        
        return min(difference, 180.0 - difference)
    
    def _angle_difference_by_index(self, index1: int, index2: int) -> int:
        '''
        Calculate the difference between two angles by indices.
        
        Note that the indices are in [0, 3], and there is a wrap-around at 3,
        so that the difference between 0 and 3 is actually 1.
        The indices correspond to the candidate angles in the order of [-45, 0, 45, 90].
        
        Parameters
        ------------------
        index1: int
            First angle index.
            
        index2: int
            Second angle index.
        
        Returns
        ------------------
        difference: int
            Difference between two angle indices, range in [0, 2].
        '''
        difference = abs(index1 - index2)
        return min(difference, 4 - difference)
    
    def filter(self, layup: List[float]) -> bool:
        '''
        Check if the layup sequence passes the engineering requirements.
        
        Parameters
        ------------------
        layup: List[float]
            Layup sequence with angle values (e.g., [0.0, 45.0, -45.0, 90.0]).
        
        Returns
        ------------------
        passed: bool
            Whether the layup sequence passes the engineering requirements.
        '''
        # Convert float angles to indices
        try:
            layup_indices = [self._angle_to_index(angle) for angle in layup]
        except ValueError as e:
            # If angle conversion fails, layup is invalid
            if self._print_violations:
                print(f"Angle conversion failed: {e}")
            return False
        
        # Check the symmetry requirements
        passed, half_layup_indices = self._check_symmetry(layup_indices)
        if not passed:
            if self._print_violations:
                print("Symmetry requirements not met.")
            return False
        
        # Apply all engineering requirement checks
        checks = [
            self._check_ply_proportion(half_layup_indices),
            self._check_groupings_of_same_orientation(half_layup_indices),
            self._check_45_degree_alternation(half_layup_indices),
            self._check_outer_layer_requirement(half_layup_indices),
            self._check_poisson_ratio_mismatch(half_layup_indices)
        ]
        if self._print_violations:
            for i, check in enumerate(checks):
                if not check:
                    print(f"Engineering requirement {i+1} not met.")
        return all(checks)
    
    def _check_symmetry(self, layup_indices: List[int]) -> Tuple[bool, List[int]]:
        '''
        Check the symmetry requirements:
        
        - The layup sequence is symmetric.
        
        Parameters
        ------------------
        layup_indices: List[int]
            Layup sequence.
            
        Returns
        ------------------
        passed: bool
            Whether the layup sequence passes the symmetry requirements.
            
        half_layup_indices: List[int]
            Half layup sequence.
        '''
        if len(layup_indices) % 2 != 0:
            return False, []
        
        # Check if the layup sequence is symmetric
        if layup_indices != list(reversed(layup_indices)):
            return False, []
        
        # Get the half layup sequence
        half_layup_indices = layup_indices[:len(layup_indices) // 2]
        
        return True, half_layup_indices
    
    def _check_ply_proportion(self, half_layup_indices: List[int]) -> bool:
        '''
        Check the ply proportion requirements:
        
        - At least 10% of the plies oriented at each angle.
        - At least 40% of +/-45 plies.
        
        Parameters
        ------------------
        half_layup_indices: List[int]
            Layup sequence.
        '''        
        # Count occurrences of each angle
        counter = Counter(half_layup_indices)
        total_plies = len(half_layup_indices)
        
        # Check that each angle has at least 10%
        for angle_idx in range(len(self.candidate_angles)):
            if counter.get(angle_idx, 0) < total_plies * 0.1:
                return False
        
        # Check that +/- 45 plies are paired
        n_45_p = counter.get(self.idx_45_p, 0)
        n_45_m = counter.get(self.idx_45_m, 0)
        if n_45_p != n_45_m:
            return False
        
        # Check that +/-45 plies (indices 0 and 2) make up at least 40%
        # Index 0 is -45°, index 2 is 45°
        if (n_45_p + n_45_m) < total_plies * 0.4:
            return False
        
        return True
        
    def _check_groupings_of_same_orientation(self, half_layup_indices: List[int]) -> bool:
        '''
        Check the ply groupings requirements:
        
        - Minimize groupings of plies with the same orientation, 
          e.g., stack no more than 4 plies of the same orientation together
          to reduce or eliminate the fringe delamination. 

        Parameters
        ------------------
        half_layup_indices: List[int]
            Layup sequence.
        '''
        is_valid = True
        
        # Check for consecutive plies with same orientation
        max_consecutive = 1
        current_consecutive = 1
        
        for i in range(len(half_layup_indices)-1):
            if half_layup_indices[i] == half_layup_indices[i+1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        # No more than 4 plies of the same orientation together
        is_valid = max_consecutive <= 4
        
        # Further check the symmetric plane, i.e.,
        # the last three plies should not be the same orientation.
        # Because this means the actual layup has at least 6 plies of the same orientation in the middle.
        if len(half_layup_indices) >= 3:
            if half_layup_indices[-3] == half_layup_indices[-2] == half_layup_indices[-1]:
                is_valid = False
        
        return is_valid

    def _check_45_degree_alternation(self, half_layup_indices: List[int]) -> bool:
        '''
        Check the 45 degree alternation requirements:
        
        - Alternate +/-45 plies through the LSS except for the closest ply either side of the symmetry plane.
        - A pair of +/-45 plies should be located as closely as possible while still meeting the other guidelines.
        
        Parameters
        ------------------
        half_layup_indices: List[int]
            Layup sequence.
        '''
        # Extract positions of -45 (index 0) and +45 (index 2) plies
        n_45_p = 0
        n_45_m = 0
        angle_45_positions = []
        for i, angle_idx in enumerate(half_layup_indices):
            
            if angle_idx == self.idx_45_m:
                n_45_m += 1
                angle_45_positions.append((i, angle_idx))
            elif angle_idx == self.idx_45_p:
                n_45_p += 1
                angle_45_positions.append((i, angle_idx))
            else:
                continue
        
        # Checked if the +/-45 plies are paired.
        if n_45_p != n_45_m:
            return False
        
        # Check if the two adjacent plies are the same orientation.
        for i in range(len(angle_45_positions) - 1):
            pos1, angle1 = angle_45_positions[i]
            pos2, angle2 = angle_45_positions[i + 1]

            if pos2 == pos1 + 1 and angle1 == angle2:
                return False
        
        # Check if +/-45 plies in each pair are too far apart, i.e., more than 2 plies apart.
        for i in range(n_45_p):
            pos1, angle1 = angle_45_positions[2*i]
            pos2, angle2 = angle_45_positions[2*i + 1]
            if abs(pos2 - pos1) > self.max_gap_between_45_plies:
                return False

        return True    

    def _check_outer_layer_requirement(self, half_layup_indices: List[int]) -> bool:
        '''
        Check the outer layer requirements, 
        which is to shield primary load carrying plies from exposed surfaces:
        
        - Keep 90° plies away from free edges to reduce peel at the edge.
        - For laminates loaded primarily in tension or compression in the 0 degree direction,
          should start with angle and transverse plies from the surface. 
        - For laminates loaded primarily in shear, should locate +/-45 plies away from the surface.

        Parameters
        ------------------
        half_layup_indices: List[int]
            Layup sequence.
        '''        
        # Check that outer surface plies are not 90° (index 3)
        if half_layup_indices[0] == self.idx_90:
            return False
        
        return True
        
    def _check_poisson_ratio_mismatch(self, half_layup_indices: List[int]) -> bool:
        '''
        Check the Poisson's ratio mismatch requirements.
        
        This is a strong requirement if `strong_requirement` is True,
        which is to avoid creating high inter-laminar tension stresses at free edges:
        
        - Minimize the Poisson's ratio mismatch between adjacent laminates that are co-cured or bonded:
            - Means to limit angle jumps between adjacent plies: keep |Δθ| ≤ 45° where possible.
            - Although 45 and -45 have a difference of 90 degrees, they're both shear angles, so they are allowed to be adjacent;
            - Therefore, this only requires to avoid abrupt 0°↔90° adjacency.
        
        - Maintain sub-laminate symmetry near surfaces: 
            - Ensure the outer few plies themselves form a symmetric, balanced mini-stack.
        
        Parameters
        ------------------
        half_layup_indices: List[int]
            Layup sequence.
        '''
        if not self.strong_requirement:
            return True
        
        # Check 0°↔90° adjacency
        for i in range(len(half_layup_indices) - 1):

            if half_layup_indices[i] == self.idx_0 and half_layup_indices[i + 1] == self.idx_90:
                return False
            elif half_layup_indices[i] == self.idx_90 and half_layup_indices[i + 1] == self.idx_0:
                return False
            else:
                continue
        
        # Check the sub-laminate symmetry near surfaces
        if len(half_layup_indices) > self.max_n_plies_outer_layer_sub_laminate:
            
            one_sub_laminate_is_valid = False
            
            for n_plies in range(4, self.max_n_plies_outer_layer_sub_laminate + 1):
                sub_laminate = half_layup_indices[:n_plies]
                if sub_laminate == list(reversed(sub_laminate)):
                    one_sub_laminate_is_valid = True
            
            if not one_sub_laminate_is_valid:
                return False
        
        return True
