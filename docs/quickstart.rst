Quickstart
==========

The package exports the main classes directly from ``lamkit``:

.. code-block:: python

   from lamkit import (
       Material,
       Ply,
       Laminate,
       LaRC05,
       Hole,
       UnloadedHole,
       EngineeringRequirements,
       LayupFeasibilityRating,
   )

The Lekhnitskii bearing and combined-load classes live in their own submodules:

.. code-block:: python

   from lamkit.lekhnitskii.loaded_hole import LoadedHole
   from lamkit.lekhnitskii.combined_load import CombinedLoadHole

Build a simple laminate and evaluate ply-level response:

.. code-block:: python

   import numpy as np
   from lamkit import Laminate, Ply
   from lamkit.analysis.material import IM7_8551_7

   ply = Ply(IM7_8551_7, thickness=0.125)
   stacking = [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0]
   lam = Laminate(stacking=stacking, plies=ply)

   # [Nxx, Nyy, Nxy, Mxx, Myy, Mxy]
   N = np.array([80.0, 0.0, 0.0, 0.0, 0.0, 0.0])
   field = lam.evaluate_laminate(N)

   print(field[["index_ply", "index_surface", "sigma_1", "FI_max"]].head())
   print("global FI_max =", field.attrs["global_FI_max"])

Evaluate an open-hole plate under combined bypass and bearing loading:

.. code-block:: python

   import numpy as np
   from lamkit.utils import evaluate_combined_load_plate, evaluate_larc05_from_results
   from lamkit.lekhnitskii.utils import generate_meshgrid

   mesh = generate_meshgrid(
       hole_radius=1.0,
       plate_radius=8.0,
       n_points_radial=121,
       n_points_angular=121,
   )

   # bypass: 100 MPa in x; bearing: 1000 N along +x (angle 0 deg)
   results_by_plies, mid_plane = evaluate_combined_load_plate(
       laminate=lam,
       sigma_xx_inf=100.0,
       sigma_yy_inf=0.0,
       tau_xy_inf=0.0,
       load=1000.0,
       angle_load_degree=0.0,
       hole_radius=1.0,
       thickness=lam.total_thickness,
       x=mesh["X"],
       y=mesh["Y"],
   )

   evaluate_larc05_from_results(results_by_plies,
       properties_dictionary=lam.ply_material.properties_dictionary)

   print("mid-plane sigma_x max =", np.max(mid_plane["sigma_x"]))
   print("ply-surface FI_max max =", max(np.max(p["FI_max"]) for p in results_by_plies))

Pass ``load=0.0`` (and keep the bypass stresses) for a pure bypass / open-hole analysis,
or pass ``sigma_xx_inf=sigma_yy_inf=tau_xy_inf=0.0`` for bearing-only.  Both cases use
:class:`~lamkit.lekhnitskii.combined_load.CombinedLoadHole` internally (zero-contribution
components are replaced by a no-op ``ZeroField`` for efficiency).

Layup guidelines and database-based feasibility use ``EngineeringRequirements`` and
``LayupFeasibilityRating``. The rating step builds a KD-tree and requires SciPy_.

.. code-block:: python

   from lamkit import EngineeringRequirements, LayupFeasibilityRating
   from lamkit.layup.feasibility import DEFAULT_DATASET_PATH

   stacking = [45.0, -45.0, 0.0, 90.0, 90.0, 0.0, -45.0, 45.0]
   eng_req = EngineeringRequirements()
   print("passes guidelines:", eng_req(stacking))

   rating = LayupFeasibilityRating(path_to_layup_database=DEFAULT_DATASET_PATH)
   out = rating.calculate_distance(
       n_ply=8, n_0=2, n_90=2, xiD1=0.0, xiD2=0.0, xiD3=0.0,
   )
   print("nearest distance:", out["distance"], "layup_id:", out["layup_id"])

.. _SciPy: https://scipy.org/
