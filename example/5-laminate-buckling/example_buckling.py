"""
Example: linear buckling analysis of a laminate plate.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(path, "..", "..", "src"))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from lamkit.analysis.buckling import BucklingAnalysis
from lamkit.analysis.laminate import Laminate
from lamkit.analysis.material import IM7_8551_7, Ply
from lamkit.analysis.buckling import plot_buckling_modes


def main() -> None:
    
    # Example laminate (quasi-isotropic).
    layup = [45, -45, 0, -45, 45, 0, 0, 90, 90, 0, 0, 45, -45, 0, -45, 45]
    ply_thickness_mm = 0.125
    ply = Ply(material=IM7_8551_7, thickness=ply_thickness_mm)
    laminate = Laminate(stacking=layup, plies=ply)

    # Plate dimensions (mm) and in-plane loads (N/mm).
    Nxx = -100.0
    Nyy = 0.0
    Nxy = 0.0

    analysis = BucklingAnalysis(
        laminate=laminate, a=100, b=100,
        constraints="PINNED",
        Nxx=Nxx, Nyy=Nyy, Nxy=Nxy,
        m=6, n=6,
    )

    eigvals, _ = analysis.buckling_analysis(num_eigvalues=5)
    
    print("\nBuckling eigenvalues (load multipliers):")
    for i, val in enumerate(eigvals, start=1):
        print(f"  mode {i}: {val:.3f}")

    case_text = f"Nxx={Nxx:.3f}, Nyy={Nyy:.3f}, Nxy={Nxy:.3f} (N/mm)"
    out_dir = os.path.join(path, "images")
    out_file = os.path.join(out_dir, "buckling_modes.png")
    plot_buckling_modes(
        analysis=analysis,
        eigvals=eigvals,
        n_modes=4,
        ngridx=41,
        ngridy=41,
        case_text=case_text,
        save_path=out_file,
    )


if __name__ == "__main__":
    main()
