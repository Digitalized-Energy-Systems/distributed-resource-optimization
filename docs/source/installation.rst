Installation
============

Requirements
------------

- Python 3.10 or later
- `numpy <https://numpy.org>`_ ≥ 1.24
- `scipy <https://scipy.org>`_ ≥ 1.10
- `cvxpy <https://www.cvxpy.org>`_ ≥ 1.3 (with the `OSQP <https://osqp.org>`_ solver)

From PyPI
---------

.. code-block:: bash

   pip install distributed-resource-optimization

With mango-agents (for networked deployments)
---------------------------------------------

.. code-block:: bash

   pip install "distributed-resource-optimization[mango]"

From source (development install)
----------------------------------

.. code-block:: bash

   git clone https://github.com/Digitalized-Energy-Systems/mango-optimization
   cd mango-optimization
   pip install -e ".[dev,docs]"

Verify the installation
-----------------------

.. code-block:: python

   import distributed_resource_optimization
   print(distributed_resource_optimization.__version__ if hasattr(distributed_resource_optimization, "__version__") else "installed OK")
