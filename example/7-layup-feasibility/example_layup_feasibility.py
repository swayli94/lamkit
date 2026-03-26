'''
Example: feasibility rating for a layup stacking sequence.
'''

import os
import time
from pathlib import Path
from lamkit.layup.feasibility import LayupFeasibilityRating

path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(Path(__file__).resolve().parents[2], "data")
DEFAULT_DATASET_PATH = os.path.join(DATA_PATH, "layup_database-with-attributes-strong28.csv")


if __name__ == "__main__":

    # Load the layup database
    layup_database = LayupFeasibilityRating(
        path_to_layup_database=DEFAULT_DATASET_PATH,
        weight=[1.0, 0.5, 0.5, 10.0, 10.0, 10.0])

    # Calculate the distance to the closest layup in the database
    n_ply = 20
    n_0 = 6
    n_90 = 4
    xiD1 = 0.5
    xiD2 = 0.5
    xiD3 = 0.5
    
    start_time = time.time()
    result = layup_database.calculate_distance(
                n_ply=n_ply, n_0=n_0, n_90=n_90, xiD1=xiD1, xiD2=xiD2, xiD3=xiD3)
    elapsed_time = time.time() - start_time
    
    with open(os.path.join(path, "output.txt"), "w") as f:

        f.write(f"The closest layup is #{result['layup_id']} with a distance of {result['distance']:.3f}\n\n")
        f.write(f"The stacking sequence:   {result['stacking']}\n\n")
        f.write(f"n_0:   actually {result['n_0']:2d}, expected {n_0:2d}\n")
        f.write(f"n_90:  actually {result['n_90']:2d}, expected {n_90:2d}\n")
        f.write(f"n_ply: actually {result['n_ply']:2d}, expected {n_ply:2d}\n")
        f.write(f"xiD1:  actually {result['xiD1']:.3f}, expected {xiD1:.3f}\n")
        f.write(f"xiD2:  actually {result['xiD2']:.3f}, expected {xiD2:.3f}\n")
        f.write(f"xiD3:  actually {result['xiD3']:.3f}, expected {xiD3:.3f}\n\n")
        f.write(f"The distance calculation took {elapsed_time:.2f} seconds\n")

    print(f"Results written to: {os.path.join(path, 'output.txt')}")
    