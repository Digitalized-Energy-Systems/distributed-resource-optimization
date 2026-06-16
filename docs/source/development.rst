Development
===========

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/Digitalized-Energy-Systems/mango-optimization
   cd mango-optimization
   pip install -e ".[dev,docs]"

Running Tests
-------------

.. code-block:: bash

   pytest

With coverage:

.. code-block:: bash

   pytest --cov=distributed_resource_optimization --cov-report=term-missing

Linting
-------

.. code-block:: bash

   ruff check .
   ruff format --check .

Building the Docs
-----------------

.. code-block:: bash

   pip install -e ".[docs]"
   make -C docs html
   # Open docs/build/html/index.html

Contributing
------------

1. Fork the repository on GitHub.
2. Create a feature branch: ``git checkout -b feature/my-feature``
3. Make your changes and add tests.
4. Ensure ``pytest`` and ``ruff check .`` both pass.
5. Open a pull request against ``main``.

Please follow the existing code style (line length ≤ 100, type hints on public APIs,
Google-style docstrings).
