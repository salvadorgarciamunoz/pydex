Examples
========

Every example is a pair of files: a ``*_model.py`` holding the model (a
``simulate()`` function and, where relevant, a ``build_pyomo_model()``), and a
runner script that builds the :class:`~pydex.core.designer.Designer`, sets the
candidate grid, and designs the experiment. Run the runner, not the model:

.. code-block:: bash

   cd examples/ode
   python case_2.py

Most examples solve an NLP, so they need IPOPT on ``PATH`` — see
:doc:`installation`.

Naming scheme
-------------

The ``case_N`` families vary one axis at a time, so the suffix tells you which
sensitivity path a script exercises:

.. list-table::
   :header-rows: 1
   :widths: 24 32 44

   * - suffix
     - sensitivities
     - model solved by
   * - *(none)*
     - exact, via the Implicit Function Theorem (``pyomo_model_fn`` assigned)
     - Pyomo.DAE orthogonal collocation + IPOPT
   * - ``_no_ift``
     - finite differences
     - Pyomo.DAE orthogonal collocation + IPOPT
   * - ``_no_ift_no_collocation``
     - finite differences
     - Pyomo ``Simulator`` (scipy/vode) forward integration

Comparing ``case_1.py`` against ``case_1_no_ift.py`` therefore isolates the
effect of the sensitivity method alone — same model, same grid, same criterion.
The three paths should agree on the design, and the capability suite asserts
this: section 52 compares nine criteria across both paths and requires
agreement within 5%, typically achieving about 0.01%.

ODE examples
------------

Case 1 — first-order reaction, one parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The smallest useful dynamic example: ``dCA/dt = -k*CA``, one parameter, one
control, one response. Start here.

* ``case_1.py`` — IFT sensitivities via PyomoNLP.
* ``case_1_no_ift.py`` — finite differences over the collocation solve.
* ``case_1_no_ift_no_collocation.py`` — finite differences over forward
  integration. Also shows, in a commented block, what happens if
  ``pyomo_model_fn`` is assigned to a model built for the ``Simulator`` path.

Case 2 — A→B with Arrhenius kinetics, four parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``dCA/dt = -k*CA^alpha``, ``dCB/dt = nu*k*CA^alpha``, with
``k = exp(theta_0 + theta_1*(T-273.15)/T)``. Four parameters, two controls
``[CA0, T]``, two responses ``[CA, CB]`` — the first example with multiple
responses.

* ``case_2.py`` — D-optimal, IFT path.
* ``case_2_no_ift.py``, ``case_2_no_ift_no_collocation.py`` — as above.
* ``case_2_ds.py`` — **Ds-optimal**: same model, grid and parameters as
  ``case_2.py``, only the criterion changes. Worth reading even though this
  model's FIM is healthy, because it shows designing for a *subset* of
  parameters while marginalising the rest, which is the usual reason to reach
  for Ds.

.. warning::

   The collocation grid in this family previously admitted a sampling time a
   hair off an existing collocation node, producing a machine-epsilon finite
   element. IPOPT reported "Optimal Solution Found" while returning
   ``CA = 31 mol/L`` from ``CA0 = 5``. If you adapt this example and see a
   physically impossible result reported as optimal, check that every sampling
   time lands exactly on a collocation node — refining ``nfe`` will not help,
   because this is a formulation problem rather than truncation error.

Case 3 — Michaelis-Menten-style network, nine parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``A -> B`` through an inhibited power-law rate,
``r = k1(T)*CA^alpha / (k2(T) + k3(T)*CA^beta)`` with
``ki(T) = exp(theta_i0 + theta_i1*(T-273.15)/T)``. Nine parameters
``[theta_10, theta_11, theta_20, theta_21, theta_30, theta_31, nu, alpha,
beta]``, three controls ``[CA0, T, tau]``, two responses. The largest example,
and the one where the sensitivity path matters most.

* ``case_3.py`` — scipy/finite-difference path. Its own header reports roughly
  350 s in sensitivity analysis, at about 45 model evaluations per candidate.
  The candidate count is not fixed: ``enumerate_candidates`` builds a
  5x5x5 = 125 point grid, then a feasibility filter drops the candidates whose
  conversion falls below ``MIN_CONVERSION`` at the nominal parameters, and the
  script prints how many survived.
* ``case_3_ift.py`` — the same design via exact IFT sensitivities taken from
  the KKT conditions of the collocation NLP, one IPOPT solve per candidate. Its
  header quotes 5-15 s for the same step. This is the fast version, and the
  clearest illustration of why the IFT path exists. Timings are from the
  example's own documentation, so treat them as indicative of the ratio rather
  than as measurements on your hardware.

ASL elimination
---------------

``examples/ASL Elimination/``

* ``asl_elimination_demo.py`` — demonstrates the diagnostic in
  ``pydex.utils.diagnose_asl_elimination``, which checks that every parameter
  ``Var`` survives into the ASL primal vector by name. This is the
  precondition pydex's IFT column-matching relies on;
  :meth:`~pydex.core.designer.Designer.initialize` runs the same check
  automatically when the utility is importable.
* ``pydex_ift_asl_guide.docx`` — background on the IFT/ASL interaction.

Jupyter notebooks
-----------------

The notebooks are not rendered here — building them into these pages would add
a notebook-execution extension to the docs toolchain. They are kept in the
repository with their outputs stored, so they read correctly on GitHub without
being run:

* `pydex_quickstart.ipynb
  <https://github.com/salvadorgarciamunoz/pydex/blob/main/examples/jupyter/pydex_quickstart.ipynb>`_
  — narrated walkthrough of a first design.
* `pydex_ode_model.ipynb
  <https://github.com/salvadorgarciamunoz/pydex/blob/main/examples/jupyter/pydex_ode_model.ipynb>`_
  — the same for a dynamic model.

To run them locally:

.. code-block:: bash

   cd examples/jupyter
   jupyter lab

Test scripts as worked examples
-------------------------------

``testing_scripts/`` doubles as a set of larger examples. They are standalone
scripts, run individually — the capability suite does not execute them, though
it does reuse the three-reaction batch model introduced in
``v_optimal_test_case.py``:

* ``first_order_reaction.py`` / ``first_order_reaction_pyomo.py`` — the same
  problem posed twice, once with a plain ``simulate()`` and once with a Pyomo
  model.
* ``v_optimal_test_case.py`` / ``v_optimal_test_case_pyomo.py`` — the two-stage
  V-optimal workflow.
* ``smoke_test_designer.py`` — the fastest end-to-end check that needs a
  solver: Ds-optimality resolved by name, the A-optimality singular-FIM
  behaviour, Ds succeeding where D-optimal cannot, and the ``regularize_fim``
  path.

Publication code
----------------

``publications/`` holds scripts reproducing figures and results from the papers
behind pydex. These are archival: they are kept as published rather than
updated alongside the API, so treat them as a record rather than as current
usage examples.
