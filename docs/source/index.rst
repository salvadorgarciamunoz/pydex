pydex documentation
====================

**pydex** is a Python package for model-based design of experiments
(MBDoE): given a model and a candidate grid of experimental conditions,
it designs the experiments that will most precisely determine your
model's parameters.

This fork formulates and solves the OED problem entirely in Pyomo, adds
V-optimal (prediction-oriented) design via a two-stage workflow, supports
Pyomo.DAE models with parallel Implicit Function Theorem (IFT)
sensitivities, and adds an estimability-ranking method
(:meth:`~pydex.core.designer.Designer.run_estimability`) based on the
Yao/McAuley orthogonalisation.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   examples
   api

Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
