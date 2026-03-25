Examples
========

The repository contains runnable examples in the ``example/`` directory.

1. Laminate response (CLT + LaRC05)
-----------------------------------

- Path: ``example/1-laminate/example_laminate.py``
- Computes laminate response through thickness and plots strain/stress/FI profiles.

2. Lekhnitskii unloaded-hole solution
-------------------------------------

- Path: ``example/2-lekhnitskii-solution/example_unloaded_hole.py``
- Solves open-hole stress and displacement fields for a homogeneous equivalent plate.

3. Open-hole laminate field
---------------------------

- Path: ``example/3-open-hole/example_open_hole.py``
- Couples laminate equivalent compliance with open-hole field and LaRC05 envelopes.

4. Effective stiffness with hole homogenisation
-----------------------------------------------

- Path: ``example/4-effective-stiffness/example_effective_stiffness.py``
- Compares laminate in-plane properties with homogenized open-hole effective properties.

5. Laminate buckling
--------------------

- Path: ``example/5-laminate-buckling/example_buckling.py``
- Runs linear buckling eigenvalue analysis and saves mode-shape plots.

6. Laminate optimization objective/constraints
----------------------------------------------

- Path: ``example/6-laminate-optimization-task/example_laminate_opt_function.py``
- Demonstrates combined displacement/failure/buckling constraint evaluation.
