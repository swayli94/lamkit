# lamkit

`lamkit` is a Python toolkit for composite laminate analysis, focused on:

- Laminate stress/strain response based on Classical Lamination Theory (CLT)
- LaRC05 failure index evaluation
- Lekhnitskii analytical solutions for infinite anisotropic plates with a hole:
  - Unloaded hole (bypass far-field stresses)
  - Loaded hole (cosine bearing-load distribution)
  - Combined loading (`CombinedLoadHole`): superposition of bypass and bearing
- Open-hole plate's effective (homogenisation) properties
- Linear buckling analysis of laminates
- Objective/constraint evaluation for laminate optimization tasks
- Layup engineering-requirement checks and feasibility rating against a layup attribute database

## Installation

### Requirements

- Python `>=3.9`
- Core dependencies: `numpy`, `pandas`, `matplotlib`
- Optional: `scipy` (linear buckling and layup feasibility rating)

### Install from PyPI

```bash
pip install lamkit
```

### Install for local development

```bash
pip install -e .[dev,docs]
```

## Quick Start

The following example creates a laminate and evaluates ply-surface fields under membrane loading:

```python
import numpy as np
from lamkit import Laminate, Ply
from lamkit.analysis.material import IM7_8551_7

ply = Ply(IM7_8551_7, thickness=0.125)  # mm
stacking = [45, -45, 0, 90, 90, 0, -45, 45]
laminate = Laminate(stacking=stacking, plies=ply)

# N = [Nxx, Nyy, Nxy, Mxx, Myy, Mxy]
N = np.array([80.0, 0.0, 0.0, 0.0, 0.0, 0.0])
field = laminate.evaluate_laminate(N)

print(field[["index_ply", "index_surface", "z", "sigma_1", "sigma_2", "tau_12", "FI_max"]].head())
```

A laminate can also be defined directly by lamination parameters, in which case the
A/B/D matrices come from the closed-form lamination-parameter expressions (`xiD` and
`T` are required; `xiA`/`xiB` optional, a missing `xiB` means a symmetric laminate):

```python
from lamkit import Laminate

lp = Laminate.get_lamination_parameters([0.0, 90.0, 90.0, 0.0])
lam_lp = Laminate(stacking={"xiA": lp["xiA"], "xiD": lp["xiD"], "T": 4 * 0.125}, plies=ply)
print(lam_lp.D)  # identical to the ply-by-ply CLT result for that stacking
```

## Examples

The `example/` directory contains runnable scripts:

1. **Laminate response (CLT + LaRC05)**  
   `example/1-laminate/example_laminate.py`  
   Computes through-thickness stress/strain and LaRC05 index distributions with plots.

2. **Lekhnitskii unloaded-hole solution**  
   `example/2-lekhnitskii-solution/example_unloaded_hole.py`  
   Solves and visualizes open-hole stress fields.

3. **Laminate open-hole field analysis**  
   `example/3-open-hole/example_open_hole.py`  
   Couples laminate equivalent compliance with open-hole fields and generates failure envelopes/boundary maps.

4. **Effective stiffness with hole homogenization**  
   `example/4-effective-stiffness/example_effective_stiffness.py`  
   Compares laminate stiffness `A` (without hole) and homogenized stiffness `A_eff` (with hole).

5. **Laminate linear buckling**  
   `example/5-laminate-buckling/example_buckling.py`  
   Computes buckling eigenvalues and saves buckling mode plots.

6. **Optimization objective/constraint evaluation**  
   `example/6-laminate-optimization-task/example_laminate_opt_function.py`  
   Demonstrates combined displacement, failure, and buckling constraint evaluation for design tasks.

7. **Layup feasibility rating**  
   `example/7-layup-feasibility/example_layup_feasibility.py`  
   Scores a candidate stacking (ply counts, bending lamination parameters) by distance to the nearest layups in a CSV database.

8. **Loaded-hole (bearing) field analysis**  
   `example/8-loaded-hole/example_loaded_hole.py`  
   Computes ply-level stress/strain and LaRC05 failure indices around a bearing-loaded hole using `CombinedLoadHole` with zero bypass stresses. Plots mid-plane strains, ply-enveloped failure index fields, and hole-boundary FI maps.

9. **Combined bearing and bypass loading**  
   `example/9-combined-loading/example_combined_loading.py`  
   Superimposes an `UnloadedHole` (far-field bypass) and a `LoadedHole` (cosine bearing distribution) via `CombinedLoadHole`. Generates mid-plane stress contour plots, ply-enveloped LaRC05 failure index maps, and a hole-boundary FI distribution across angle and thickness.

## Common Development Commands

### Run tests

```bash
pytest
```

### Build distribution package

```bash
python -m build
```

### Build documentation

```bash
cd docs
sphinx-build -b html . _build/html
```
