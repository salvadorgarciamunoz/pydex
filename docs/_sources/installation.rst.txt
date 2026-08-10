Installation
============

.. code-block:: bash

   pip install git+https://github.com/salvadorgarciamunoz/pydex.git

Core dependencies (installed automatically): ``numpy``, ``scipy``,
``pandas``, ``matplotlib``, ``numdifftools``, ``pyomo``, ``joblib``,
``dill``.

Solvers
-------

The OED problem is formulated in Pyomo, so most design criteria need an
NLP solver Pyomo can call. IPOPT is recommended and is looked up on
``PATH`` by default (``solver="ipopt"``); the Implicit Function Theorem
(IFT) sensitivity path additionally uses Pyomo's PyNumero/ASL interface,
which requires the ``pynumero_ASL`` extension to be built
(``pyomo build-extensions``).

Sparsity-enforcing MINLP designs (``min_effort > 0``) require an MINLP
solver such as ``bonmin`` or (via GAMS) ``BARON``.

POUNCE
^^^^^^

`POUNCE <https://github.com/jkitchin/pounce>`_ is a pure-Rust port of IPOPT.
It implements the same interior-point algorithm and follows upstream IPOPT's
option semantics and console output, but its default build needs no Fortran,
no HSL and no system BLAS — the sparse symmetric factorisation is pure Rust
too. That makes it an alternative worth knowing about if the HSL licensing
step above is inconvenient.

Because POUNCE speaks the AMPL NL/SOL protocol, Pyomo drives it exactly as it
drives IPOPT. The ``pyomo-pounce`` package registers it with
``SolverFactory``:

.. code-block:: bash

   pip install pyomo-pounce

.. code-block:: python

   import pyomo_pounce          # registers the 'pounce' solver
   designer.design_experiment(designer.d_opt_criterion, solver="pounce")

The import is what performs the registration, so it must happen before the
design call.

POUNCE covers both of pydex's solver call sites, which are separate mechanisms:

* the **design formulation** — pydex passes ``solver=`` straight through to
  ``pyo.SolverFactory``, so ``solver="pounce"`` works for the criteria built
  symbolically in Pyomo;
* the **IFT sensitivity path** — here the collocation NLP is solved inside your
  own ``pyomo_model_fn``, so the solver is whichever one that function calls.
  Change the ``SolverFactory`` call in your model function, not the
  ``design_experiment`` argument. Exact sensitivities are then extracted from
  that solve through PyomoNLP/ASL as usual.

Checked on ``examples/ode/case_1.py`` with POUNCE 0.9.0 driving both sites: the
D-optimal criterion agreed with IPOPT to 3.6e-15 relative (1.2188792603520995
against 1.218879260352095) on the same support, with the sensitivity method
still reported as ``Pyomo IFT — PyomoNLP / ASL (compiled)``.

POUNCE also accepts IPOPT's option names, so ``solver_options`` written for
IPOPT carry over; a request it cannot honour degrades rather than failing, e.g.
``linear_solver: ma57`` on a default build reports ``running with linear solver
FERAL (ma57 requested but not compiled)`` and proceeds.

The capability suite is run against IPOPT, so that remains the reference
configuration.

.. note::
   ``cyipopt`` is optional. It accelerates the V-optimal
   operating-point step (Stage 1) when present, and is required for the
   PyNumero/cyipopt solve path; without it, Stage 1 falls back to scipy
   SLSQP. Install with ``pip install pydex[ipopt]``.
