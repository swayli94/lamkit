Installation
============

Requirements
------------

- Python 3.9+
- pip

Install from source
-------------------

.. code-block:: bash

   pip install -e .[dev,docs]

Install from PyPI
-----------------

.. code-block:: bash

   pip install lamkit

Optional runtime dependencies
-----------------------------

- Linear buckling and layup feasibility rating (``cKDTree``) use SciPy:

  .. code-block:: bash

     pip install scipy

Build documentation locally
---------------------------

Autodoc imports ``lamkit`` and its dependencies (NumPy, pandas, Matplotlib, SciPy, etc.).
Install the package in the **same environment** you use for Sphinx, for example:

.. code-block:: bash

   pip install -e ".[docs]"

Then:

.. code-block:: bash

   cd docs
   sphinx-build -b html . _build/html

Alternatively, from the repo root: ``pip install -r docs/requirements.txt`` (includes the scientific stack plus Sphinx).

If your editor reports ``No module named 'numpy'`` while analyzing ``docs/conf.py``, point the workspace Python interpreter at the venv where you ran the install above.
