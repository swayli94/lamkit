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

- Buckling analysis uses SciPy:

  .. code-block:: bash

     pip install scipy

Build documentation locally
---------------------------

.. code-block:: bash

   cd docs
   sphinx-build -b html . _build/html
