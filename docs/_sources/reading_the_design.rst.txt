Reading the design
==================

``design_experiment()`` prints a report and the plotting methods draw
pictures, but the design itself is available as data. Everything below is on
the :class:`~pydex.core.designer.Designer` after a solve, and nothing here
needs the report to have been printed.

The short version
-----------------

``design_experiment()`` **returns** the full result, so the usual idiom is

.. code-block:: python

   result = designer.design_experiment(designer.d_opt_criterion, solver="ipopt")

   result["criterion_value"]     # the objective, as reported
   result["optimal_efforts"]     # the design itself, (n_c, n_spt)

   designer.get_optimal_candidates_table()   # a tidy DataFrame to hand to a lab

The design: ``designer.efforts``
--------------------------------

The design *is* the effort array.

.. code-block:: python

   designer.efforts.shape    # (n_c, n_spt)
   designer.efforts.sum()    # 1.0 -- a unit budget

One row per candidate, in the order you enumerated them; one column per
sampling time. Entries are **fractions of your experimental budget**, not
numbers of runs, and they sum to 1. A zero entry is the design telling you
not to run that candidate at that time -- on a typical problem most entries
are zero.

The column meaning follows the sampling-time case (see ``n_spt`` on
:meth:`~pydex.core.designer.Designer.design_experiment`):

* sampling times optimised (``n_spt`` omitted) -- one column per candidate
  grid time, effort allocated per *measurement*;
* ``n_spt=k`` -- one column per *schedule* of k times, shape
  ``(n_c, n_spt_comb)``;
* fixed grid (``n_spt`` equal to the number of times listed) -- a single
  column, effort allocated per *experiment*;
* a static model -- a single column.

The supported candidates
------------------------

:meth:`~pydex.core.designer.Designer.get_optimal_candidates` collects just
the candidates carrying non-negligible effort:

.. code-block:: python

   opt = designer.get_optimal_candidates(tol=1e-4)

One entry per supported candidate, each holding its index, control values,
sampling times, schedules and efforts. Also cached on
:attr:`~pydex.core.designer.Designer.optimal_candidates`.

For anything you intend to export or hand over, prefer
:meth:`~pydex.core.designer.Designer.get_optimal_candidates_table`, which
returns a tidy :class:`pandas.DataFrame` with one row per experiment to run:

.. code-block:: python

   designer.ti_controls_names = ["x1", "x2"]
   ...
   designer.get_optimal_candidates_table()

.. code-block:: text

    Experiment  Candidate   x1   x2  Effort
             1          1 -1.0 -1.0    0.25
             2          5 -1.0  1.0    0.25
             3         21  1.0 -1.0    0.25
             4         25  1.0  1.0    0.25

``Experiment`` is sequential over the runs being recommended and corresponds
to nothing in the candidate pool; ``Candidate`` is the 1-indexed pool
position, kept so a row can be cross-referenced against plot legends and
solver logs. Setting ``ti_controls_names`` is what turns the control columns
from ``Time-invariant Control 0`` into your own names.

The whole result: ``designer.oed_result``
-----------------------------------------

The dictionary returned by ``design_experiment()`` is also kept on
:attr:`~pydex.core.designer.Designer.oed_result`, and is what
:meth:`~pydex.core.designer.Designer.load_oed_result` reads back. Its keys:

``optimal_efforts``, ``criterion_value``, ``optimality_criterion``,
``solver``, ``ti_controls_candidates``, ``tv_controls_candidates``,
``sampling_times_candidates``, ``model_parameters``, ``prior_fim``,
``prior_fim_mp``, ``prior_n_exp``, ``pseudo_bayesian``,
``pseudo_bayesian_type``, ``regularized``, ``n_spt_spec``,
``optimize_sampling_times``, ``solution_time``, ``optimization_time``,
``sensitivity_analysis_time``.

**This is the supported way to read the criterion value.** There is no public
``designer.criterion_value`` attribute; use ``oed_result["criterion_value"]``,
or :meth:`~pydex.core.designer.Designer.compute_criterion_value` to score a
design under a criterion of your choosing.

Whole runs: ``apportion()``
---------------------------

Efforts are fractions, so a lab cannot run them directly.
:meth:`~pydex.core.designer.Designer.apportion` rounds a solved design to a
budget of whole experiments:

.. code-block:: python

   runs = designer.apportion(8)          # also on designer.apportionments
   designer.rounding_efficiency          # how much the rounding cost

``apportion()`` is a cheap query against an already-solved design, so
sweeping it across several budgets and comparing is a legitimate way to
*choose* a budget. Two things to know:

* the return is **ragged** -- a 1-D object array of per-candidate integer
  arrays -- when supported candidates carry effort at different numbers of
  sampling times. Total it with ``int(np.nansum(a)) for a in app``, never
  ``astype(int)`` on the whole thing;
* :attr:`~pydex.core.designer.Designer.rounding_efficiency` is only computed
  when it would be reported. At ``verbose=0`` it stays ``None``; pass
  ``compute_actual_efficiency=True`` to force it.

Other quantities on the designer
--------------------------------

================================  ==========================================
``designer.fim``                  the information matrix for the current design
``designer.sensitivities``        ``(n_c, n_spt, n_m_r, n_mp)``
``designer.response``             predicted responses, after ``simulate_candidates()``
``designer.apportionments``       the last ``apportion()`` result
``designer.optimal_candidates``   the last ``get_optimal_candidates()`` result
================================  ==========================================

One trap: printing changes the design
-------------------------------------

:meth:`~pydex.core.designer.Designer.print_optimal_candidates` goes through
``get_optimal_candidates()``, which zeroes efforts below ``tol`` and
renormalises what remains. So the printing call has a side effect on
``designer.efforts``, and a criterion value computed before and after the
same print need not agree.

**Capture any value you intend to quote before printing anything.** If you
only ever read the table or the dictionary this does not arise; it matters
when you read ``designer.efforts`` yourself.
