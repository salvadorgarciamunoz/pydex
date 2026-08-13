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

Both scripts run :meth:`~pydex.core.designer.Designer.run_estimability` before
designing, and both act on what it says. The nine-parameter form is structurally
singular: the rate law carries an exact invariance — adding a constant to every
``theta_i0`` at once, or to every ``theta_i1``, leaves every prediction
unchanged — so ``design_experiment()`` refuses it. Each script fixes the two
parameters its own analysis flags as unresolvable, re-runs estimability on the
reduced seven to see what the reduction did and did not fix, and then designs.

``case_3.py`` makes the complementary point, which is that **estimability
analysis does not require a tractable model**.
:meth:`~pydex.core.designer.Designer.run_estimability` reads
:attr:`~pydex.core.designer.Designer.sensitivities` and is indifferent to their
provenance: here they are finite differences over scipy's Radau integrator, and
``simulate()`` could equally be a legacy binary, a commercial process simulator
or a network call.

Two things follow from the lower accuracy of that path. The ``UNRESOLVABLE``
threshold is inferred from the sensitivity method — about ``1e-3`` for finite
differences against ``1e-7`` for exact IFT derivatives — so the flag should be
read as "cannot be resolved by this analysis" rather than as a verdict on the
model. And the two parameters named differ: ``case_3_ift.py`` fixes ``theta_20``
and ``theta_21``, ``case_3.py`` fixes ``theta_21`` and ``theta_31``. Both are
correct. The redundancy is a common shift within an Arrhenius triple, so which
member is held still is a convention, and at finite-difference accuracy the
residuals deciding the order sit close enough together that the tie breaks
differently.

Case 4 / Case 5 — local vs pseudo-Bayesian D-optimal, A -> B -> C network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A different reaction network from cases 1-3, introducing **pseudo-Bayesian
design** — designing under uncertainty in the model parameters rather than
at a single nominal guess. Neither file carries a ``_no_ift`` /
``_no_ift_no_collocation`` suffix even though both use the
finite-difference-over-forward-integration sensitivity path (no
``pyomo_model_fn``); that suffix scheme is specific to the cases 1-3
sensitivity-path comparison, and this family has no IFT/collocation
counterpart.

* ``case_4.py`` — **local** D-optimal design on two rate constants
  ``[k1, k2]`` at a single nominal guess, with one control (feed rate
  ``f_in``). ``f_in`` feeds pure A with no outflow term, so this is a
  continuously-fed semi-batch reactor rather than a closed system: total
  moles grow as ``1 + f_in * t`` once ``f_in > 0``, confirmed numerically.
  ``pseudo_bayesian_type`` only takes effect when ``model_parameters`` is
  2-D, so passing it here (a 1-D nominal vector) has no effect — the
  design stays local. Also demonstrates recovering ``[k1, k2]`` from data
  simulated at the apportioned design, via PyMC — see below.
* ``case_5.py`` — the Arrhenius version of the same network (``f_in`` fixed
  to 0, so this one is closed) under a **genuine pseudo-Bayesian Type-1**
  D-optimal design: ``model_parameters`` is a scenario array drawn from a
  uniform prior over all four kinetic parameters, and the criterion is
  averaged over scenarios rather than over averaged information matrices.
  Ships with ``N_SCR = 20`` (roughly 13 s/scenario measured on this grid,
  so a few minutes total) rather than a much larger ensemble; raise
  ``N_SCR`` for a smoother estimate at the cost of runtime.
  ``save_atomics`` must be passed to ``design_experiment()`` as a keyword,
  not set as ``designer._save_atomics`` beforehand — the keyword's own
  default silently overwrites the attribute either way, the same as
  ``regularize_fim``.

Both models fold their reaction rates directly into the mass balance
rather than defining them as a separate algebraic constraint, which keeps
each system a pure ODE — required by scipy's ``Simulator`` backend, which
only integrates ODEs, not DAEs.

Bayesian inference in ``case_4.py`` -- via PyMC, not pydex
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``case_4.py`` ends by recovering ``[k1, k2]`` from data simulated at the
apportioned design, using **PyMC** (``pip install pymc arviz`` — not a
pydex dependency, only needed for this section): synthetic "observed" data
is simulated at the apportioned design's condition(s) using the same
nominal parameters the design was built on, plus measurement noise drawn
from ``error_cov``, and PyMC samples the posterior over ``[k1, k2]`` given
that data.

This is ordinary downstream analysis written against the public API
(``designer.optimal_candidates``, ``designer.apportionments``,
``error_cov``, ``simulate()``) — pydex has no Bayesian-inference capability
of its own. ``simulate()`` runs through scipy's Simulator, a black box with
no symbolic gradient, so it is wrapped with ``pytensor.wrap_py`` and
sampled with ``pm.Metropolis`` rather than PyMC's default (gradient-based)
NUTS sampler. A 2-parameter local D-optimal design with an unconstrained
budget typically collapses to a single condition run several times rather
than spread across several; 4 chains x (500 tune + 800 draws) in parallel
takes a few minutes and should recover ``k1``/``k2`` close to their true
values, with r_hat typically landing around 1.01-1.04 — inside PyMC's own
"probably fine, more draws would help" range rather than fully converged.
The draw/tune/chain/core counts are constants at the top of that section
if tighter diagnostics are needed, at the cost of runtime.

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
