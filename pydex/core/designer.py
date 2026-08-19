# pydex designer.py
# Last patched: 2026-05-11
# Fix: parallel IFT worker fake namespace was missing _dynamic_system=True,
#      causing _spt=None and pyomo_model_fn to be called without sampling_times
#      in every subprocess worker. All 160 candidates were built with the same
#      wrong (or default) spt, making the assembled FIM rank-deficient (rank 4/6
#      instead of 6/6) and producing grossly inflated A-optimal J_V values.
#      Fix: add _dynamic_system=True to the SimpleNamespace in _worker().
from datetime import datetime
from inspect import signature
from os import getcwd, path, makedirs
from pickle import dump, load
from string import Template
from time import time
import itertools
import warnings
import __main__ as main
import dill
import sys

from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import chi2
from pydex.utils.trellis_plotter import TrellisPlotter
from pydex.core.logger import Logger
import matplotlib
import numdifftools as nd
import numpy as np
import pandas as pd
import pyomo.environ as _pyo

try:
    from pyomo.core.expr.calculus.derivatives import differentiate as _pyomo_differentiate
    import scipy.linalg as _scipy_linalg
    _PYOMO_IFT_AVAILABLE = True
except ImportError:
    _PYOMO_IFT_AVAILABLE = False

try:
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP as _PyomoNLP
    _PYNUMERO_ASL_AVAILABLE = True
except Exception:
    _PYNUMERO_ASL_AVAILABLE = False

# NOTE: _PYNUMERO_ASL_AVAILABLE only reflects whether the *Python* PyomoNLP
# class imported successfully. PyNumero's Python interface is pure-Python
# and always importable with Pyomo; the ASL Jacobian machinery it wraps
# depends on a separately-built compiled extension that can be missing or
# broken even when the import above succeeds. So this flag can be True on
# a machine where _PyomoNLP(m) itself raises at call time. Call sites must
# not treat a True value here as a runtime guarantee — see
# _pyomo_ift_fd_jacobian below and its use in _eval_sensitivities_pyomo_ift.


# Default per-parameter finite-difference step controls. See
# _resolve_fd_base_step below for what these mean and why they exist.
_FD_RELATIVE_BASE_STEP = 1e-2
_FD_ABSOLUTE_STEP_FLOOR = 1e-8


def _resolve_fd_base_step(model_parameters,
                          relative_base_step=_FD_RELATIVE_BASE_STEP,
                          absolute_step_floor=_FD_ABSOLUTE_STEP_FLOOR):
    """
    Size a finite-difference step for each model parameter from that
    parameter's OWN nominal magnitude, returning an array numdifftools can
    consume as a per-parameter ``base_step``.

    step_i = max(relative_base_step * abs(theta_i), absolute_step_floor)

    Why this is not simply a constant
    ---------------------------------
    numdifftools builds its step sequence as
    ``step_nom * base_step * step_ratio ** i``, where the default
    ``step_nom`` heuristic is ``max(log(e + |x|), 1)``. That heuristic
    floors at 1, so it scales the step *up* for parameters larger than
    O(1) and does nothing whatsoever for parameters smaller than O(1).
    A flat ``base_step`` therefore perturbs a parameter of nominal value
    0.02 by ~100x its own magnitude — far outside the local linear regime
    Richardson extrapolation assumes — while leaving a parameter of
    nominal value 2.0 correctly scaled. The failure is silent (no warning,
    no exception) and selective, hitting only small-magnitude parameters,
    which disguises it as a structural problem with the model rather than
    a step-size problem. See CHANGELOG.md 0.3.0 for the ground-truth
    verification behind this.

    The floor exists because a pure percentage gives a parameter whose
    nominal value is exactly 0 a step of exactly 0.

    For a pseudo-Bayesian scenario set (shape ``(n_scr, n_mp)``) the
    magnitude is taken as the per-parameter maximum across scenarios, so
    no single scenario's small value silently under-steps relative to the
    others.
    """
    theta = np.asarray(model_parameters, dtype=float)
    theta_mag = (np.max(np.abs(theta), axis=0) if theta.ndim == 2
                 else np.abs(theta))
    return np.maximum(theta_mag * relative_base_step, absolute_step_floor)


def _pyomo_ift_fd_jacobian(all_vars, all_bodies):
    """
    Pure-Python fallback Jacobian: differentiate() every constraint body
    with respect to every variable. Used both when _PYNUMERO_ASL_AVAILABLE
    is False (import-time) and when the ASL backend is available at import
    time but fails at call time (e.g. missing compiled extension) — see
    _eval_sensitivities_pyomo_ift, which catches that failure and routes
    here instead of propagating it.
    """
    n_v = len(all_vars)
    n_c = len(all_bodies)
    J = np.zeros((n_c, n_v))
    for ci, body in enumerate(all_bodies):
        for vi, var in enumerate(all_vars):
            try:
                J[ci, vi] = _pyo.value(_pyomo_differentiate(body, wrt=var))
            except Exception:
                J[ci, vi] = 0.0
    return J

# ── ASL variable ordering diagnostic ─────────────────────────────────────────
# Optional: gracefully absent if pydex.utils is not installed / importable.
# When available, diagnose_asl_elimination() is called during initialize()
# to verify that every parameter Var is reachable (present, by name) in the
# ASL primal vector — the precondition pydex IFT column-matching relies on.
# This is the single source of truth: the same tool users run by hand to vet
# a model is the one initialize() enforces.
try:
    from pydex.utils.diagnose_asl_elimination import (
        diagnose_asl_elimination as _diagnose_asl,
    )
    _DIAGNOSE_ASL_AVAILABLE = True
except ImportError:
    _diagnose_asl = None
    _DIAGNOSE_ASL_AVAILABLE = False


def _safe_tight_layout(fig):
    """
    fig.tight_layout() that stays quiet on 3-D axes.

    Matplotlib cannot compute a tight layout for Axes3D and emits
        UserWarning: Tight layout not applied. The left and right margins
        cannot be made large enough to accommodate all Axes decorations.
    The effort/sensitivity plots use projection='3d', so the call was warning on
    every figure while doing nothing. Skip it there; matplotlib's default 3-D
    margins are already reasonable.
    """
    import warnings as _warnings
    from mpl_toolkits.mplot3d import Axes3D as _Axes3D
    if any(isinstance(ax, _Axes3D) for ax in fig.axes):
        return
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")
        fig.tight_layout()


def _match_nlp_var(vname, nlp_var_names):
    """
    Find the index of an all_vars name within an ASL primals_names() list.

    This is how pydex IFT maps a model Var (e.g. 'k', 'A0', 'A[1.0]') onto its
    column in the ASL Jacobian.  Matching is purely by NAME (and qualified-name
    suffix / final dotted segment), never by position — ASL is free to
    reorder/displace variables in the primal vector; the column reordering
    downstream handles any permutation.  Returns the matched index, or None if
    the name is absent entirely (true ASL elimination — Failure Mode B).

    EXACT MATCH WINS.  The lookup runs in two passes: an exact-equality pass
    first, then a qualified-name fallback pass.  This matters when a model has
    both a top-level Var and a block-nested Var that share a leaf name (e.g.
    'theta' and 'b.theta'): a single-pass scan that accepted any clause in list
    order could alias 'theta' onto 'b.theta' (or vice versa) depending purely
    on how ASL happened to order its primal list — a silent wrong-column bug.
    Running the exact pass first removes that order dependence: if a primal
    equals the name verbatim it is always chosen, and the suffix/leaf clauses
    only ever resolve names that have NO exact counterpart.

    In normal pydex use the suffix clauses are rarely exercised: the model
    builder places the model's own VarData objects into all_vars, and both
    str(var) and PyomoNLP.primals_names() derive from getname(fully_qualified=
    True), so an exact match almost always exists.  The fallbacks exist for
    builders that synthesise names by hand or through a renaming transformation.

    The predicate here is the SINGLE SOURCE OF TRUTH for IFT name matching and
    must stay identical (including the exact-first ordering) to the one used by
    pydex.utils.diagnose_asl_elimination._match_param_name (the gate users run
    before handing a model to the Designer).  If the gate matched differently
    from this matcher, a model could pass the gate and then crash mid-run with
    'Cannot match variable', or worse, bind to a different column silently.

    Defined at module level (not as a method) so it works regardless of whether
    the caller is a real Designer instance or the lightweight SimpleNamespace
    stand-in used inside parallel (loky) sensitivity workers.
    """
    # Pass 1 — exact equality wins, independent of position.
    for i, n in enumerate(nlp_var_names):
        if n == vname:
            return i
    # Pass 2 — qualified-name suffix / final-segment fallbacks.
    leaf = vname.split(".")[-1]
    for i, n in enumerate(nlp_var_names):
        if (n.endswith("." + vname)
                or vname.endswith("." + n)
                or n == leaf):
            return i
    return None


class Designer:
    """
    version = 20260804000000

    An experiment designer with capabilities to do parameter estimation, parameter
    estimability study, and computes both continuous and exact experimental designs.

    Interfaces to optimization solvers via Pyomo, supporting any solver that Pyomo
    knows about (IPOPT, GLPK, Gurobi, CPLEX, Bonmin, SHOT, etc.).  Supports virtually
    any Python function as the model simulator as long as it follows one of the
    supported signatures (see ``simulate`` below).  Special support for ODE models
    solved via Pyomo.DAE: the model and simulator objects can be passed to the designer
    to prevent re-building them on every sensitivity evaluation, significantly reducing
    computation time.

    Designer comes equipped with convenient built-in visualization capabilities
    using matplotlib, and supports the following design criteria:

    .. code-block:: text

        Calibration-oriented (minimise parameter uncertainty):
            D-optimal  — maximises det(FIM),  minimises joint confidence volume
            Ds-optimal — maximises det of the Schur complement of the nuisance-
                         parameter block in the FIM, i.e. D-optimal design for a
                         chosen SUBSET of parameters (see `interest_parameters`)
                         while marginalising out the rest
            A-optimal  — minimises trace(FIM^{-1}), minimises total param variance
            E-optimal  — minimises lambda_max(FIM^{-1}), minimises worst direction

        Prediction-oriented (minimise prediction uncertainty at target conditions):
            V-optimal  — minimises trace(W FIM^{-1} W^T) at user-specified dw

        Prediction-variance family, on PVAR = f·FIM^{-1}·f^T per candidate and
        sampling time ("g" = generalised / worst case, "i" = individual / summed):
            dg, di     — determinant of PVAR          (see the caveat below)
            ag, ai     — trace of PVAR
            eg, ei     — largest eigenvalue of PVAR
            vdi        — as di, over the operating-point grid

        Robustness to parameter uncertainty:
            pseudo-Bayesian — average information (type 0) or average criterion
                              (type 1) over a scenario set
            CVaR-D          — conditional value-at-risk on the D-criterion, for
                              designs judged on their worst-case scenarios

        Other:
            U-optimal  — maximises the squared Frobenius norm of the FIM; needs
                         no inverse, so it survives a singular FIM

    Quick-start
    -----------
    A minimal example for a static model (no time-varying controls):

    >>> import numpy as np
    >>> from pydex.core.designer import Designer
    >>>
    >>> # 1. Define the simulate function (signature type 1)
    >>> def simulate(ti_controls, model_parameters):
    ...     x   = ti_controls[0]        # single input
    ...     a, b = model_parameters     # two parameters to estimate
    ...     return np.array([a * x + b])
    >>>
    >>> # 2. Build the designer
    >>> d = Designer()
    >>> d.simulate            = simulate
    >>> d.model_parameters    = np.array([2.0, 1.0])   # initial guess
    >>> d.ti_controls_candidates = np.linspace(0, 10, 21).reshape(-1, 1)
    >>> d.initialize()
    >>>
    >>> # 3. Run D-optimal design
    >>> d.design_experiment(d.d_opt_criterion, solver="ipopt")
    >>> d.print_optimal_candidates()

    Design types
    ------------
    Every criterion is a bound method taking the effort vector and returning a
    scalar to be MINIMISED; pass the method itself to
    :meth:`design_experiment`::

            d.design_experiment(d.d_opt_criterion, solver="ipopt")

    The sections below cover each family. Which of them run natively in Pyomo
    and which fall back to SLSQP is set out under
    `Which solver actually runs`_.

    D-optimal
    ^^^^^^^^^
    ``d_opt_criterion`` — maximises ``det(FIM)``, equivalently minimises the
    volume of the joint parameter confidence ellipsoid. The default choice, and
    the one to reach for without a specific reason to do otherwise: it is
    invariant to reparameterisation by any invertible linear transform, so the
    design does not depend on whether you estimate a rate constant or its
    logarithm.

    Its weakness is that a determinant multiplies all eigenvalues, so it says
    nothing about WHICH parameter is poorly determined — a design can score well
    while leaving one direction almost undetermined, provided the others
    compensate. When ``det(FIM)`` is zero for every admissible design the
    criterion is infeasible everywhere; that is a structural statement about the
    model and grid, and :meth:`diagnose_fim_structure` names the parameters
    responsible.

    A-optimal
    ^^^^^^^^^
    ``a_opt_criterion`` — minimises ``trace(FIM^-1)``, the sum of the parameter
    variances. More interpretable than D: it is literally the total variance you
    expect across the parameter set. The trade-off is that it is NOT invariant to
    reparameterisation, and because it is a sum it can be dominated by whichever
    parameter happens to carry the largest variance, which in turn depends on the
    units the parameters are expressed in.

    A singular or indefinite FIM returns ``+inf``, never 0 — see
    `Infeasibility conventions`_ for why that distinction matters.

    E-optimal
    ^^^^^^^^^
    ``e_opt_criterion`` — maximises the smallest eigenvalue of the FIM,
    equivalently minimises ``lambda_max(FIM^-1)``. This targets the WORST
    determined direction in parameter space, so it is the criterion to reach for
    when the concern is one specific poorly determined combination rather than
    average precision across the set. Like A, it is not invariant to
    reparameterisation.

    Ds-optimal (subset)
    ^^^^^^^^^^^^^^^^^^^
    Ds-optimality designs for a chosen SUBSET of the model parameters while
    marginalising the remainder ("nuisance" parameters) out through the Schur
    complement of the FIM. Declare the subset BY NAME:

    >>> d.model_parameter_names = ["k", "A0", "c1", "c2"]
    >>> d.interest_parameters   = ["k", "A0"]        # c1, c2 become nuisance
    >>> d.design_experiment(d.ds_opt_criterion, solver="ipopt")

    Names are matched against `model_parameter_names` by exact string equality.
    Numeric indices are rejected: a parameter's POSITION in the FIM follows the
    order of `model_parameters`, which is not guaranteed to track the order in
    which a Pyomo model declares its equations or variables, so position is not
    a stable identifier. Unknown names raise immediately rather than binding
    silently to the wrong parameter.

    Why use it: Ds stays well defined when a nuisance parameter is
    unidentifiable, whereas D-optimality does not. If a nuisance direction
    carries no information then det(FIM) = 0 and D-optimality is infeasible for
    EVERY design, but the Schur complement over the interest parameters remains
    finite and positive definite, so there is still something to optimise.

    Its limit: Ds only helps when the unidentifiable direction lies in the
    NUISANCE subspace. If the interest parameters are themselves collinear after
    marginalisation, the Schur complement is singular too and the criterion
    correctly reports infeasibility. A Ds value of +inf is therefore a
    diagnosis, not a bug: the question to ask is whether the unidentifiability
    sits in the parameters you care about or the ones you do not.

    Practical note: when the nuisance block cannot be made positive definite,
    the native Pyomo formulation is infeasible by construction and the solve
    falls back to SLSQP on the generalised Schur complement. If the Schur
    complement is also non-PD at the STARTING design, SLSQP has an infinite
    objective and no gradient with which to escape it, so the design will not
    move. Passing regularize_fim=True to design_experiment() keeps the solve on
    the native path, where IPOPT can make progress from an infeasible start;
    note the resulting criterion value then depends on `_eps`.


    V-optimal (prediction-oriented, two-stage)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    V-optimal design targets prediction accuracy at a specific operating
    condition ``dw`` rather than minimising global parameter uncertainty.
    It requires two stages.

    *Stage 1 — Process optimisation (find dw):*
    Solve a user-defined constrained optimisation to find the operating
    condition that maximises (or minimises) a process objective subject
    to process constraints.  This is the condition at which the model
    needs to be most accurate — typically the economically optimal point.

    Set the following attributes before calling
    ``find_optimal_operating_point()``::

            designer.process_objective   = my_objective
            designer.process_constraints = my_constraints   # optional
            designer.dw_sense            = "maximize"       # or "minimize"
            designer.dw_bounds_tic       = [(lb, ub), ...]  # one per ti_control
            designer.dw_bounds_tvc       = [(lb, ub), ...]  # one per tv_control

    ``process_objective`` signature: ``callable(tic, tvc, mp) -> float``
        ``tic`` is a 1-D array of ti_controls, ``tvc`` a 1-D array of
        tv_control parameters, ``mp`` the current model_parameters array.
        Returns the scalar value to minimise or maximise.

    ``process_constraints`` signature: ``callable(tic, tvc, mp) -> list of dicts``

        Each dict: ``{"type": "ineq" | "eq", "fun": callable(tic, tvc, mp)}``.
        For ``"ineq"``: ``fun(tic, tvc, mp) >= 0`` means feasible.
        For ``"eq"``: ``fun(tic, tvc, mp) == 0``.
        The constraint structure must be fixed; only values change with x.

    Example Stage 1 setup::

            def my_objective(tic, tvc, mp):
                sol = _solve(tic[0], tic[1], tic[2], mp, np.array([T_FINAL]))
                return float(sol.y[1, 0])   # maximise CB at end of batch

            def my_constraints(tic, tvc, mp):
                def ci_con(tic, tvc, mp):
                    sol = _solve(tic[0], tic[1], tic[2], mp, np.array([T_FINAL]))
                    return CI_MAX - float(sol.y[2, 0])   # CI_final <= CI_MAX
                def jacket_con(tic, tvc, mp):
                    return tic[1] - tic[0]               # Tjacket >= T0
                return [
                    {"type": "ineq", "fun": ci_con},
                    {"type": "ineq", "fun": jacket_con},
                ]

            designer.process_objective   = my_objective
            designer.process_constraints = my_constraints
            designer.dw_sense            = "maximize"
            designer.dw_bounds_tic       = [(45, 75), (50, 85), (0.5, 2.0)]
            designer.dw_bounds_tvc       = []

            dw_tic, dw_tvc = designer.find_optimal_operating_point(
                init_guess = np.array([[60.0, 70.0, 1.0]]),
                optimizer  = "mumps",
            )

    *Stage 2 — V-optimal MBDoE (design experiments):*
    After Stage 1, set ``dw_spt`` — the time point(s) within the optimal
    operating profile at which prediction accuracy is required — then call
    ``design_v_optimal()``.  Note that ``dw_spt`` is a user specification
    (e.g. end of batch), not a degree of freedom; it is distinct from
    ``sampling_times_candidates``, which pydex optimises over::

            designer.dw_spt = np.array([t_final])

            designer.design_v_optimal(
                package               = "ipopt",
                optimizer             = "mumps",
                optimize_sampling_times = True,
            )


    Prediction-variance family (dg, di, ag, ai, eg, ei, vdi)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    These act on the prediction-variance block ``PVAR = f·FIM^-1·f^T`` formed per
    candidate and sampling time, rather than on the FIM directly. The first
    letter is the aggregation over PVAR and the second is the aggregation over
    blocks — ``g`` for generalised (worst case over blocks), ``i`` for individual
    (summed over blocks):

    .. code-block:: text

        dg, di   determinant of PVAR        (see the caveat below)
        ag, ai   trace of PVAR
        eg, ei   largest eigenvalue of PVAR
        vdi      as di, over the operating-point grid

    Use these when the quantity you care about is the precision of the model's
    PREDICTIONS across the design region, rather than of its parameters. Note
    ``vdi_criterion`` collapses to D-optimality when ``n_m_r == n_mp``, because
    ``W`` is then square and ``det(PVAR) = det(W)^2 / det(FIM)``; it is a
    distinct criterion only when there are fewer measured responses than
    parameters.

    These take determinants of PVAR = f·FIM^{-1}·f^T. A determinant MULTIPLIES
    all eigenvalues, so a single near-null direction in the sensitivity block f
    collapses it to numerical noise. Two failure modes follow:

        * the aggregate underflows to a magnitude below the solver's ABSOLUTE
          convergence tolerance (scipy SLSQP defaults to ftol=1e-9), so the
          optimiser declares convergence at iteration 1 and returns the
          starting design untouched;
        * a non-positive-definite block drives a summed log-determinant to +inf,
          destroying all design information.

    When that is detected, a log-PSEUDO-determinant is substituted: the sum of
    log-eigenvalues above a relative cutoff (`_pvar_rcond`). This is well
    defined for a near-singular PSD matrix, is a monotone transform of the
    determinant where both exist, and lives on an O(1) scale the optimiser can
    work with.

    The decision is BEHAVIOURAL — "did the determinant form produce a usable
    number?" — rather than rank-based, because the numerical rank of PVAR is
    tolerance-dependent and can flip between values across effort vectors for
    blocks sitting on the cutoff. It is made once per design run and LATCHED,
    because a branch that flipped mid-solve would make the objective
    discontinuous and break SLSQP. `design_experiment()` clears the latch;
    `reset_pvar_logdet_mode()` clears it manually.

    Consequences worth knowing:

        * where the determinant form IS usable, values are bit-identical to
          previous releases (the same slogdet call is used);
        * where the fallback engages, the reported value is on a LOG scale and
          is NOT comparable with a determinant from a well-conditioned problem;
        * trace-based (ag, ai) and eigenvalue-based (eg, ei) criteria are
          unaffected, being dominated by the healthy directions rather than the
          near-null one. If dg/di are unusable for your model, those are the
          natural alternatives.

    A near-singular f is a MODELLING signal, not merely a numerical one: it
    means the measurable responses are close to linearly dependent in
    sensitivity space, so the response set carries fewer independent directions
    than its size suggests. A warning reports the measured worst
    sv_min/sv_max over all candidate/sampling-time pairs.


    Pseudo-Bayesian designs
    ^^^^^^^^^^^^^^^^^^^^^^^
    All of the above assume the parameters are known well enough to linearise
    around. When they are not, supply a SCENARIO SET — an array of parameter
    draws instead of a single vector — and pydex designs against the whole set.
    Two aggregations, selected by ``pseudo_bayesian_type``:

    .. code-block:: text

        type 0   average information:  f( mean_s FIM_s )
        type 1   average criterion:    mean_s f( FIM_s )

    Type 0 is solved natively in Pyomo because the scenario-averaged information
    matrix is still linear in the efforts; type 1 is not, and falls back to
    SLSQP with finite-differenced gradients — see `Which solver actually runs`_.
    That makes type 0 the cheaper of the two. Type 1 is the more faithful reading
    of "good on average" when the criterion is strongly non-linear in the FIM,
    since averaging the criterion is not the same as evaluating the criterion of
    the average.

    CVaR-D (risk-averse)
    ^^^^^^^^^^^^^^^^^^^^
    ``cvar_d_opt_criterion``, driven by :meth:`solve_cvar_problem` rather than
    :meth:`design_experiment`. Where a pseudo-Bayesian design optimises the
    AVERAGE over scenarios, a CVaR design optimises the average over the WORST
    ``(1 - beta)`` fraction of them, so it buys protection against the scenarios
    where the design performs badly at some cost to the typical case. Sweeping
    ``beta`` traces a bi-objective frontier, which :meth:`plot_pareto_frontier`
    draws.

    Only the D-criterion has a CVaR form. ``solve_cvar_problem`` rejects any
    criterion whose name does not contain ``cvar``.

    U-optimal
    ^^^^^^^^^
    ``u_opt_criterion`` — maximises the squared Frobenius norm of the FIM,
    ``sum(FIM * FIM)``. It uses no inverse and no decomposition, so unlike D, A
    and E it stays finite for a singular FIM. That robustness is also its
    limitation: it rewards total information without regard to how that
    information is distributed across parameters, so a design scoring well here
    can still leave a parameter combination undetermined. Useful as a
    well-behaved starting point or a sanity check, not as a final criterion.

    Tools and helpers
    -----------------
    These answer questions ABOUT a model and a candidate grid rather than
    producing a design. None of them is wired into any design path: they report,
    and what you do with the report is yours to decide.

    Estimability analysis
    ^^^^^^^^^^^^^^^^^^^^^
    :meth:`run_estimability` ranks the parameters from most to least estimable
    on the current grid, and is usually the FIRST thing worth running on a new
    model — before choosing a criterion, and certainly before diagnosing a
    design that will not converge. It answers three separate questions that are
    easy to conflate:

    .. code-block:: text

        abs_info   ABSOLUTE. Pooled Fisher information about each parameter's
                   fractional value. Dimensionless, so it reads on its own:
                   below 1 the whole grid cannot pin the parameter down to
                   within its own magnitude.
        E / E_UD   RELATIVE. Residual norm at selection, scaled so the best
                   parameter is 1. Says which parameters rank low, NOT whether
                   even the best is any good — that is what abs_info is for.
                   E is for weighted least squares / MLE, E_UD for unweighted.
        group      Which parameters are mutually correlated above corr_tol, and
                   therefore interchangeable as far as the data is concerned.

    The mechanism is the orthogonalisation of Yao et al. (2003) as tabulated by
    Wu, McLean, Harris and McAuley (2011), implemented as column-pivoted QR. The
    step that makes it more than a sensitivity ranking is the projection: a
    parameter with large raw sensitivity that merely DUPLICATES the effect of one
    already selected gets a small residual and ranks low. Magnitude and
    correlation are different problems and call for different experiments, and
    this separates them.

    Reading the output, in the order it usually matters:

    * any parameter with ``abs_info < 1`` cannot be determined by this grid at
      all — fix it, reparameterise, or move it to the nuisance set and use
      Ds-optimality;
    * a CORRELATION GROUP means the data determines roughly ONE parameter among
      its members. Estimate or fix your choice and the rest become unestimable
      once you do. Pick on physical grounds — which one is meaningful,
      transferable, or independently known — not on the ranking;
    * a low ``E`` with healthy ``abs_info`` means the parameter is informative
      but redundant; a low ``abs_info`` means it is simply not informed.

    The ranking is a property of THIS grid at THESE parameter values, not of the
    model in the abstract. A parameter inestimable on one grid may be fine on
    another, so re-run it after changing the candidates.

    FIM and sensitivity diagnostics
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    :meth:`diagnose_fim_structure` evaluates the FIM at the fully supported
    design — every candidate at full effort, the most informative matrix the grid
    can produce — and reports its rank, condition number, and the parameter
    composition of any null direction. Because that design is the best case, a
    deficiency there is STRUCTURAL: no choice of efforts will fix it. The report
    names the implicated parameters and lists the options.

    :meth:`diagnose_sensitivity` works per candidate instead, tabulating
    ``A_k[j,j]`` (how many experiments at that candidate would be needed for
    signal-to-noise 1 on parameter j) and the condition number. Use it to see
    WHICH candidates carry information about which parameters, rather than
    whether the grid as a whole is adequate.

    :meth:`design_experiment` runs the structural check itself and REFUSES a
    rank-deficient design by default rather than returning a plausible number
    from a floored Cholesky factor. Override with ``allow_singular_fim=True``.
    Ds-optimality is exempt, since marginalising a singular nuisance block is
    precisely its purpose.

    ASL elimination diagnostics
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Only relevant on the IFT sensitivity path. Exact sensitivities are taken from
    the KKT conditions of the collocation NLP, which requires locating each
    parameter's column in the ASL primal vector BY NAME. If Pyomo's presolve
    eliminates a parameter ``Var`` — because it is fixed, unused, or trivially
    substitutable — that column is not there, and matching by position instead
    would silently return sensitivities for the wrong variable.

    ``pydex.utils.diagnose_asl_elimination`` checks that every parameter
    survives into the primal vector and reports any that do not.
    :meth:`initialize` runs the same check automatically whenever that utility
    is importable, so in normal use this happens without being asked for. Run it
    directly when an IFT design produces sensitivities you do not believe; see
    ``examples/ASL Elimination/``.

    Apportionment
    ^^^^^^^^^^^^^
    A design is a continuous effort vector, but experiments come in whole
    numbers. :meth:`apportion` converts efforts into an integer allocation over
    ``n_exp`` runs and reports the efficiency of the rounded design relative to
    the continuous one. Worth reading rather than assuming: the fewer
    experiments there are to distribute, the less room rounding has to
    approximate the intended efforts, and the reported efficiency is what tells
    you whether ``n_exp`` is large enough.


    Simulate function signatures
    ----------------------------
    Pydex recognises five signatures based on the argument names.  Use
    exactly these names — pydex inspects them with ``inspect.signature``:

    Type 1 — static model, time-invariant controls only:
        simulate(ti_controls, model_parameters)

    Type 2 — dynamic model, time-invariant controls + sampling times:
        simulate(ti_controls, sampling_times, model_parameters)

    Type 3 — dynamic model, time-varying controls + sampling times:
        simulate(tv_controls, sampling_times, model_parameters)

    Type 4 — dynamic model, both control types + sampling times:
        simulate(ti_controls, tv_controls, sampling_times, model_parameters)

    Type 5 — dynamic model, sampling times only (no explicit controls):
        simulate(sampling_times, model_parameters)

    In all cases ``model_parameters`` must be present.  The function must
    return a numpy array:

        - Static (types 1): shape (n_responses,)
        - Dynamic (types 2-5): shape (n_spt, n_responses)

    Control variables
    -----------------
    ``ti_controls`` — time-invariant controls
        Settings fixed for the entire duration of an experiment.
        Examples: initial concentration, reactor pressure, feed ratio.
        Set as ``designer.ti_controls_candidates``, shape (n_c, n_tic).

    ``tv_controls`` — time-varying controls
        Settings that vary during an experiment, represented as a flat
        parameter vector whose interpretation (ramp, step, spline, etc.)
        is defined inside the user's simulate function.
        Examples: temperature ramp rate, feed profile knots.
        Set as ``designer.tv_controls_candidates``, shape (n_c, n_tvc).

    ``sampling_times`` — measurement time points
        The times within a dynamic experiment at which measurements are
        taken.  Set as ``designer.sampling_times_candidates``, shape (n_c, n_spt).
        When ``optimize_sampling_times=True`` is passed to design_experiment,
        pydex optimises the effort allocation over both candidates and
        sampling times simultaneously.

    Infeasibility conventions
    -------------------------
    Every criterion is MINIMISED, so an unusable information matrix must return
    +inf — the worst attainable value — never 0, which for a minimised criterion
    is among the best. Degenerate inputs handled uniformly (returning +inf, with
    a correctly shaped zero gradient when an analytic Jacobian was requested):

        * FIM absent, wrong shape, all-zero, or non-finite
        * FIM not positive definite

    Positive-definiteness is tested by eigenvalue or Cholesky, never by the sign
    of the determinant: det > 0 only requires an EVEN number of negative
    eigenvalues, so an indefinite matrix such as diag(1, 1, -1, -1) passes a
    determinant-sign test while being meaningless as an information matrix. The
    same test is used in the finite-difference and analytic-gradient branches of
    a given criterion, so toggling `_fd_jac` cannot change whether a design is
    judged feasible.

    Which solver actually runs
    --------------------------
    Criteria expressible as static Pyomo expressions are built symbolically and
    handed to the solver named by design_experiment(solver=...) — any
    AMPL-compatible NLP solver (IPOPT, BONMIN, BARON via GAMS, ...). These are
    D, A, E, V, Ds, and pseudo-Bayesian "average information" (type 0).

    Everything else is optimised by scipy's SLSQP acting on the criterion
    callable as a black box, because the criterion receives per-scenario or
    per-block matrices at runtime and cannot be written symbolically. On that
    path:

        * the `solver=` argument is NOT used;
        * `solver_options` is filtered to ftol / maxiter / disp, so IPOPT-style
          options (linear_solver, tol, max_iter) are silently dropped;
        * gradients are finite-differenced by scipy, so the analytic Jacobians
          implemented by several criteria are unused.

    Pseudo-Bayesian type 0 is solved natively because the scenario-averaged
    information matrix is linear in the efforts,

        mean_s FIM_s(e) = mean_s sum_i e_i A_i^(s) = sum_i e_i (mean_s A_i^(s))

    so it has exactly the structure of a local FIM assembled from
    scenario-averaged atomic FIMs, and the existing formulations apply verbatim.
    Type 1 ("average criterion") is mean_s f(FIM_s), which does not reduce this
    way and keeps the SLSQP fallback.

    Numerical controls
    ------------------
    Attributes, with defaults, grouped by what they govern:

    .. code-block:: text

        Ds-optimality (Schur complement)
            _ds_rcond       1e-12  singular-value cutoff for the nuisance solve
            _ds_resid_tol   1e-8   relative residual above which the nuisance
                                   solve is judged inconsistent (non-PSD FIM)
            _ds_cond_warn   1e10   cond(S) above which a warning is emitted

        Prediction-variance determinants (dg / di / vdi)
            _pvar_rcond       1e-10  eigenvalue cutoff for the pseudo-determinant
            _pvar_scale_floor 1e-12  |aggregate| below this is treated as noise
            _pvar_cond_warn   1e-8   sv_min/sv_max below this warns that f is
                                     near-singular

        Regularisation
            _eps                   magnitude of the eps*I added to the FIM when
                                   design_experiment(regularize_fim=True). NOTE
                                   design_experiment overwrites _regularize_fim
                                   from its keyword argument, so setting that
                                   attribute directly is silently discarded.

    """
    def __init__(self):
        """
        Pydex' main class to instantiate an experimental designer. The designer
        serves as the main user-interface to use Pydex to solve experimental design
        problems.

        All details on the experimental design problem is passed to the designer, which
        Pydex then compiles into an optimization problem passed to the optimization
        package it supports to be solved by a numerical optimizer.

        The designer comes with various built-in plotting capabilities through
        matplotlib's plotting features.
        """
        self.__version__ = "0.0.11"

        """ In Silico Experiments """
        self._bayes_pe_time = None
        self.bayesian_pe_samples = None

        """ Prior experimental information (sequential MBDoE) """
        self._prior_fim          = None   # stored prior FIM (n_mp x n_mp, normalized)
        self._prior_fim_mp       = None   # model_parameters at which _prior_fim was computed
        self._prior_n_exp        = 0      # number of prior experiments (for reporting)

        """ Bracketing-optimal (b_opt_criterion) state -- isolated to this
            criterion only; read via getattr(..., default) elsewhere so
            their absence never affects any other criterion. """
        self._b_opt_output_weight       = 0.5    # last-used output_weight, for _b_opt_criterion()
        self._b_opt_min_sep_frac        = 0.0    # anti-clustering threshold; 0 = off
        self._b_opt_selected_idx        = None   # convenience: selected candidate indices
        self._b_opt_apportion_redundant = False  # set True after a b_opt solve
        self._b_opt_termination         = None   # last MINLP termination condition (str)
        self._b_opt_proven_optimal      = None   # True only if that was 'optimal'


        self.error_cov = None
        self._error_cov_defaulted = False
        self.error_fim = None

        """ goal_oriented_ds"""
        self.n_c_go = None
        self.n_spt_go = None
        self.n_tic_go = None
        self.n_r_go = None
        self._candidates_swapped = False

        self.go_simulate = None
        self.go_tic = None
        self.go_tvc = None
        self.go_spt = None
        self.go_sensitivities = None
        self.go_sample_sensitivities_done = False
        self.go_error_cov = None
        self._step_nom = None

        """ CVaR-exclusive """
        self.n_cvar_scr = None
        self.cvar_optimal_candidates = None
        self.cvar_solution_times = None
        self._biobjective_values = None
        self._constrained_cvar = None
        self.beta = None
        self._cvar_problem = None

        """ pseudo-Bayesian exclusive """
        self.pb_atomic_fims = None
        self._scr_sens = None
        self.scr_responses = None
        self._current_scr = None
        self._pseudo_bayesian_type = None
        self.scr_fims = None
        self.scr_criterion_val = None
        self._current_scr_mp = None

        """ Ds-optimality exclusive """
        # NAMES (into model_parameter_names) of the "interest" parameter
        # subset. The complementary parameters are treated as nuisance
        # parameters that are marginalised out via the Schur complement of
        # the FIM. Set via the `interest_parameters` property, e.g.:
        #     designer.interest_parameters = ["Ka", "A0"]
        # Resolved to positional indices lazily (by name, not by position)
        # in _resolve_ds_idx() the first time a Ds-optimal criterion is
        # evaluated, since model_parameter_names may not be populated yet
        # (defaulted in initialize()) at the point interest_parameters is
        # assigned.
        self.ds_interest_names = None
        self.ds_interest_idx = None
        self.ds_nuisance_idx = None
        # Numerical tolerances for the Schur-complement evaluation.
        #   _ds_rcond     : relative singular-value cutoff used when solving
        #                   M_nn W = M_ns. Singular values below
        #                   rcond * max(sv) are treated as zero, which yields
        #                   the minimum-norm (generalised) Schur complement
        #                   when the nuisance block is rank-deficient.
        #   _ds_resid_tol : relative residual above which the nuisance solve
        #                   is judged inconsistent (=> Ds genuinely diverges;
        #                   only possible for a non-PSD FIM).
        #   _ds_cond_warn : cond(S) above which a warning is emitted.
        self._ds_rcond     = 1e-12
        self._ds_resid_tol = 1e-8
        self._ds_cond_warn = 1e10
        self._ds_warned    = set()

        """ Prediction-variance (dg / di / vdi) numerical controls """
        # These criteria take determinants of PVAR = f·FIM^-1·f^T. A determinant
        # MULTIPLIES all eigenvalues, so a single near-null direction in f
        # collapses it to numerical noise -- unlike trace (ag/ai) or lambda_max
        # (eg/ei), which are dominated by the healthy directions and are
        # unaffected. When that happens the determinant form is unusable and a
        # log-pseudo-determinant is substituted; see _pvar_decide_logdet_mode.
        #   _pvar_rcond      : relative eigenvalue cutoff for the pseudo-determinant
        #   _pvar_scale_floor: |aggregate| below this is treated as numerical
        #                      noise. Referenced to scipy SLSQP's default
        #                      ftol=1e-9, which is an ABSOLUTE tolerance on the
        #                      objective: an objective smaller than ftol makes
        #                      the solver declare convergence at iteration 1.
        #   _pvar_cond_warn  : warn when sv_min/sv_max of a sensitivity block
        #                      falls below this (f is near-singular)
        self._pvar_rcond       = 1e-10
        self._pvar_scale_floor = 1e-12
        self._pvar_cond_warn   = 1e-8
        self._pvar_logdet_mode = None     # None | 'det' | 'pdet'  (latched)
        self._pvar_warned      = set()

        """ Logging """
        # options
        self.sens_report_freq = 10
        self._memory_threshold = None  # threshold for large problems in bytes, default: 1 GB
        # store designer status and its verbal level after initialization
        self._status = 'empty'
        self._verbose = 0
        self._sensitivity_analysis_done = False

        """ The current optimal experimental design """
        self.opt_eff = None
        self.opt_tic = None
        self.n_opt_c = None
        self.mp_covar = None

        # exclusive to discrete designs
        self.spt_binary = None

        # exclusive to dynamic systems
        self.opt_tvc = None
        self.opt_spt = None
        self.opt_spt_combs = None
        self.spt_candidates_combs = None

        # experimental
        self.cost = None
        self.cand_cost = None
        self.spt_cost = None
        self._norm_sens_by_params = True

        """" Type of Problem """
        self._invariant_controls = None
        self._specified_n_spt = None
        self._discrete_design = None
        self._pseudo_bayesian = False
        self._large_memory_requirement = False
        self._current_criterion = None
        self._efforts_transformed = False
        self._unconstrained_form = False
        self.normalized_sensitivity = None
        self._dynamic_controls = False
        self._dynamic_system = False

        """ Attributes to determine if re-computation of atomics is necessary """
        self._candidates_changed = None
        self._model_parameters_changed = None
        self._compute_atomics = False
        self._compute_sensitivities = False

        """ Core user-defined Variables """
        self._tvcc = None
        self._ticc = None
        self._sptc = None
        self._model_parameters = None
        self._simulate_signature = 0

        # optional user inputs
        self.measurable_responses = None  # subset of measurable states

        """ Labelling """
        self.candidate_names = None  # plotting names
        self.measurable_responses_names = None
        self.ti_controls_names = None
        self.tv_controls_names = None
        self.model_parameters_names = None
        self.model_parameter_unit_names = None
        self.response_unit_names = None
        self.time_unit_name = None
        self.model_parameter_names = None
        self.response_names = None
        self.use_finite_difference = True
        self.do_sensitivity_analysis = False

        # ── Pyomo IFT exact-sensitivity ───────────────────────────────────────
        # Set use_pyomo_ift = True and assign pyomo_model_fn to enable exact
        # parametric sensitivities via the implicit-function theorem computed
        # from Pyomo's symbolic expression tree (no finite differences needed).
        #
        # pyomo_model_fn(ti_controls, model_parameters) must return:
        #   (model, all_vars, all_bodies, t_sorted)
        # where all_vars has the n_mp parameter Vars FIRST (declared as fixed
        # Var, not Param), followed by state variables. t_sorted should contain
        # only the output time points (not the full collocation grid) so the
        # designer snaps to the correct output variable.
        #
        # pyomo_output_var_name: base name(s) of output Var (str or list[str]).
        # None -> auto-detect as first n_m_r state vars after param vars.
        self.use_pyomo_ift         = None   # None = auto-detect in initialize()
        self.pyomo_model_fn        = None
        self.pyomo_output_var_name = None
        self.n_jobs                = None  # None = auto-detect in initialize()

        """ Core designer outputs """
        self.response = None
        self.sensitivities = None
        self.optimal_candidates = None
        self.atomic_fims = None
        self.apportionments = None
        self.non_trimmed_apportionments = None
        self.n_exp = None
        self.epsilon = None

        # exclusive to prediction-oriented criteria
        self.pvars = None

        """ problem dimension sizes """
        self.n_c = None
        self.n_c_tic = None
        self.n_c_tvc = None
        self.n_c_spt = None
        self.n_tic = None
        self.n_spt = None
        self.n_r = None
        self.n_mp = None
        self.n_e = None
        self.n_m_r = None
        self.n_scr = None
        self.n_spt_comb = None
        self._n_spt_spec = None
        self.max_n_opt_spt = None
        self.n_factor_sups = None

        """ performance-related """
        self.feval_simulation = None
        self.feval_sensitivity = None
        self._fim_eval_time = None
        # temporary for current design
        self._sensitivity_analysis_time = 0
        self._optimization_time = 0

        """ continuous oed-related quantities """
        # sensitivities
        self.efforts = None
        self.F = None  # overall regressor matrix
        self.fim = None  # the information matrix for current experimental design
        self.p_var = None  # the prediction covariance matrix

        """ saving, loading attributes """
        # current oed result
        self.run_no = 1
        self.oed_result = None
        self.result_dir_daily = None
        self.result_dir = None

        """ plotting attributes """
        self.grid = None  # storing grid when create_grid method is used to help
        # generate candidates

        """ [Private]: current candidate within eval_sensitivities() """
        self._current_tic = None
        self._current_tvc = None
        self._current_spt = None
        self._current_res = None

        """ User-specified Behaviour """
        # problem types
        self._sensitivity_is_normalized = None
        self._opt_sampling_times = False
        self._var_n_sampling_time = None
        # numerical options
        self._regularize_fim = None
        self._num_steps = 5
        self._eps = 1e-5
        self._trim_fim = False
        self._fd_jac = True
        self._store_responses_rtol = 1e-5
        self._store_responses_atol = 1e-8

        # solver name (Pyomo SolverFactory string, e.g. "ipopt", "bonmin", "glpk")
        self._solver = "ipopt"
        self._fd_jac = True          # always True; gradient strategy is internal

        # store current criterion value
        self._criterion_value = None

        """ user saving options """
        self._save_sensitivities = False
        self._save_txt = False
        self._save_txt_nc = 0
        self._save_txt_fmt = '% 7.3e'
        self._save_atomics = False

        """ V-optimal design: operating point and W matrix
        =====================================================
        These attributes support the two-stage V-optimal MBDoE workflow.

        Stage 1 — Process optimisation (user sets before calling
        find_optimal_operating_point):

            process_objective   : callable(tic, tvc, mp) -> float
                The scalar process objective to optimise.  Returns a value
                to be minimised or maximised depending on dw_sense.
                Example: return predicted CB at end of batch.

            process_constraints : callable(tic, tvc, mp) -> list of dicts
                Returns process constraints in scipy/IPOPT format.
                Each dict: {"type": "ineq"|"eq", "fun": f(tic, tvc, mp)}
                For "ineq": fun >= 0 means feasible.
                Set to None if no constraints beyond box bounds.

            dw_bounds_tic : list of (lb, ub) tuples, length n_tic
                Box bounds on the ti_controls at the operating point.
                Must be provided before calling find_optimal_operating_point.

            dw_bounds_tvc : list of (lb, ub) tuples, length n_tvc
                Box bounds on the tv_controls at the operating point.
                Set to [] if the model has no tv_controls.

            dw_sense : str, "minimize" or "maximize"
                Direction of the process objective optimisation.

        Operating point (dw) — the condition(s) at which prediction accuracy
        is desired.  Can be set in two ways:

          Option A — via find_optimal_operating_point() (Stage 1):
            The designer solves a process optimisation and stores the result
            in dw_tic / dw_tvc automatically.  Use this when dw is the
            economically optimal operating condition.

          Option B — direct assignment (any point of interest):
            Assign dw_tic (and optionally dw_tvc) directly before calling
            design_v_optimal().  Use this when dw is known from domain
            knowledge, a regulatory specification, or a prior study.

                designer.dw_tic = np.array([[T0, Tjacket, cat_load]])
                designer.dw_tvc = np.array([[]])   # empty if no tvc
                designer.dw_spt = np.array([t_final])
                designer.design_v_optimal(...)

            dw_tic : np.ndarray, shape (r_w, n_tic)
                ti_controls at the operating point(s) of interest.
                r_w > 1 when multiple points are provided simultaneously.
                Setting this attribute automatically sets _dw_fixed = True
                and invalidates any cached W matrix.

            dw_tvc : np.ndarray, shape (r_w, n_tvc)
                tv_controls at the operating point(s) of interest.
                Defaults to an array of empty rows when not set explicitly.

            _dw_fixed : bool
                True once dw_tic has been assigned (either via
                find_optimal_operating_point or direct assignment).
                Guards against calling design_v_optimal with no dw.

        Stage 2 — V-optimal MBDoE (user sets before calling design_v_optimal):

            dw_spt : np.ndarray, shape (n_spt_dw,)
                Time point(s) within the optimal operating profile at which
                prediction accuracy is required.  This is a user specification
                (e.g. end of batch, critical process transition) — it is NOT
                a degree of freedom for the MBDoE optimisation.
                For non-dynamic models, this attribute is ignored.
                Example: designer.dw_spt = np.array([t_final])

            W : np.ndarray, shape (r_w * n_spt_dw * n_m_r, n_mp)
                Scaled sensitivity matrix evaluated at dw, used in the
                V-optimality criterion J_V = trace(W @ FIM^{-1} @ W^T).
                Computed automatically by _eval_W_matrix() on first call
                to design_v_optimal(). Cached thereafter; set to None or
                pass recompute_W=True to force recomputation (e.g. after
                updating model_parameters in a sequential design loop).
        """
        # user-defined process optimization (Stage 1)
        self.process_objective = None       # callable(tic, tvc, mp) -> scalar
        self.process_constraints = None     # callable(tic, tvc, mp) -> list of
                                            #   {"type": "eq"/"ineq", "fun": f(tic,tvc,mp)}
        self.dw_bounds_tic = None           # list of (lb, ub), length n_tic
        self.dw_bounds_tvc = None           # list of (lb, ub), length n_tvc
        self.dw_sense = "minimize"          # "minimize" or "maximize"

        # operating point of interest (set via property setters or find_optimal_operating_point)
        self._dw_tic   = None               # backing store for dw_tic property
        self._dw_tvc   = None               # backing store for dw_tvc property
        self._dw_fixed = False              # True once dw_tic has been assigned

        # W matrix and associated spt for sensitivity evaluation at dw
        self.dw_spt = None                  # shape (n_spt_dw,) — time points for W eval
        self.W = None                       # shape (r_w * n_spt_dw * n_m_r, n_mp)

    @property
    def model_parameters(self):
        """numpy.ndarray: Nominal model parameter values.

        A 1-D array of length ``n_mp`` for a local design, or a 2-D array of
        shape ``(n_scr, n_mp)`` to make the design PSEUDO-BAYESIAN — each row is
        one scenario drawn from the prior, and :meth:`initialize` sets
        ``_pseudo_bayesian`` and ``n_scr`` accordingly.

        Sensitivities are evaluated at these values, so a design is only as good
        as the guess it was built on. Assigning marks the sensitivities stale;
        they are recomputed on the next :meth:`eval_sensitivities`.
        """
        return self._model_parameters

    @model_parameters.setter
    def model_parameters(self, mp):
        self._model_parameters_changed = True
        self._model_parameters = mp

    @property
    def ti_controls_candidates(self):
        """numpy.ndarray: Candidate time-invariant controls, ``(n_c, n_tic)``.

        One row per candidate experiment, one column per control held constant
        for the duration of that experiment — initial concentrations, reactor
        temperature, catalyst loading. The optimiser allocates effort ACROSS
        these rows; it cannot invent conditions between them, so the grid bounds
        what any design can achieve.

        :meth:`enumerate_candidates` builds a full factorial grid. Assigning
        marks the candidate set changed.
        """
        return self._ticc

    @ti_controls_candidates.setter
    def ti_controls_candidates(self, ticc):
        self._candidates_changed = True
        self._ticc = ticc

    @property
    def tv_controls_candidates(self):
        """numpy.ndarray: Candidate time-varying controls, ``(n_c, n_tvc)``.

        One row per candidate, one column per control that varies during the
        experiment — a temperature ramp rate, a feed profile parameter. pydex
        passes the row to :attr:`simulate` as a flat vector and places no
        interpretation on it: how those numbers become a trajectory is entirely
        the model's business.

        Only used by simulate signatures 3 and 4. Assigning marks the candidate
        set changed.
        """
        return self._tvcc

    @tv_controls_candidates.setter
    def tv_controls_candidates(self, tvcc):
        self._candidates_changed = True
        self._tvcc = tvcc

    @property
    def sampling_times_candidates(self):
        """numpy.ndarray: Candidate sampling times, ``(n_c, n_spt)``.

        One row per candidate giving the times at which that experiment may be
        measured. Rows may be padded with ``numpy.nan`` when candidates have
        different numbers of usable times.

        With ``optimize_sampling_times=False`` every listed time is measured.
        With it True the optimiser also chooses WHICH of them to use, and
        ``n_spt=k`` restricts each experiment to k samples — the same candidate
        may then appear under several sampling schedules.

        Assigning marks the candidate set changed.
        """
        return self._sptc

    @sampling_times_candidates.setter
    def sampling_times_candidates(self, sptc):
        self._candidates_changed = True
        self._sptc = sptc

    @property
    def dw_tic(self):
        """numpy.ndarray: Time-invariant controls at the operating point(s) of
        interest, shape ``(r_w, n_tic)``.

        Where you intend to OPERATE, as opposed to where you intend to
        experiment. V-optimal design minimises prediction variance here rather
        than parameter variance everywhere, so these are the conditions the
        design is ultimately serving.

        Either set directly, or obtained from
        :meth:`find_optimal_operating_point`. The setter stores it as 2-D,
        marks ``_dw_fixed``, and invalidates the cached ``W`` matrix so it is
        rebuilt at the new point. Setting ``None`` clears all three.
        """
        return self._dw_tic

    @dw_tic.setter
    def dw_tic(self, value):
        """
        Set the operating point(s) of interest for V-optimal design.

        Accepts a 1-D array (single point) or 2-D array (multiple points).
        Setting this attribute:
          - stores the value as shape (r_w, n_tic)
          - sets _dw_fixed = True so design_v_optimal() can proceed
          - invalidates the cached W matrix (self.W = None) so it will be
            recomputed at the new dw on the next call to design_v_optimal()

        Setting to None resets dw to the unspecified state.

        Examples
        --------
        # Single operating point (direct assignment, no Stage 1 needed):
        designer.dw_tic = np.array([[45.72, 80.07, 2.0]])

        # Multiple points of interest:
        designer.dw_tic = np.array([[45.0, 78.0, 1.5],
                                    [50.0, 80.0, 2.0]])
        """
        if value is None:
            self._dw_tic   = None
            self._dw_fixed = False
            self.W         = None
            return
        value = np.atleast_2d(np.asarray(value, dtype=float))
        self._dw_tic   = value
        self._dw_fixed = True
        self.W         = None   # invalidate cached W — must be recomputed at new dw

    @property
    def dw_tvc(self):
        """numpy.ndarray: Time-varying controls at the operating point(s) of
        interest, shape ``(r_w, n_tvc)``.

        The time-varying counterpart of :attr:`dw_tic`. Leave as an empty array
        for models without time-varying controls, which is the common case —
        :meth:`design_v_optimal` handles an empty ``dw_tvc`` and only requires
        :attr:`dw_tic`.
        """
        return self._dw_tvc

    @dw_tvc.setter
    def dw_tvc(self, value):
        """
        Set the time-varying controls at the operating point(s) of interest.

        For models without tv_controls, set to an empty array:
            designer.dw_tvc = np.array([[]])

        If not set explicitly, _eval_W_matrix() defaults to empty rows.
        """
        if value is None:
            self._dw_tvc = None
            return
        self._dw_tvc = np.atleast_2d(np.asarray(value, dtype=float))

    @property
    def interest_parameters(self):
        """ Names of the model parameters of interest for Ds-optimal design. """
        if self.ds_interest_names is None:
            return None
        return list(self.ds_interest_names)

    @interest_parameters.setter
    def interest_parameters(self, names):
        """
        Set the SUBSET of model parameters that are of interest for
        Ds-optimal design (ds_opt_criterion), given BY NAME.

        Parameters are matched against designer.model_parameter_names by
        exact string equality — never by index/position. This matters
        because the position of a parameter in the FIM, in a Pyomo model's
        internal variable ordering, or in the ASL primal vector can shift
        depending on how the model's equations happen to be declared;
        position is not a stable identifier. Name matching is the same
        principle already used for IFT column-matching elsewhere in this
        file (see _match_nlp_var): the name is the single source of truth,
        position is not assumed.

        The lookup itself is deferred to first use (inside
        _resolve_ds_idx()), because model_parameter_names is only
        guaranteed to be populated once initialize() has run (or the user
        has assigned it explicitly) — interest_parameters may be set before
        that point.

        The complementary parameters (by name) are treated as nuisance
        parameters and marginalised out via the Schur complement of the
        FIM.

        Parameters
        ----------
        names : list[str] or None
            Must match entries of designer.model_parameter_names exactly.
            Set to None to reset (invalidates ds_opt_criterion until re-set).

        Examples
        --------
        designer.model_parameter_names = ["Ka", "A0", "k1", "k2"]
        designer.interest_parameters = ["Ka", "A0"]   # k1, k2 are nuisance
        """
        if names is None:
            self.ds_interest_names = None
            self.ds_interest_idx   = None
            self.ds_nuisance_idx   = None
            return
        names = list(np.atleast_1d(np.asarray(names, dtype=object)))
        if not names or not all(isinstance(nm, str) for nm in names):
            raise TypeError(
                "interest_parameters must be given as a list of parameter "
                "NAMES (str) matching designer.model_parameter_names, e.g. "
                "designer.interest_parameters = ['Ka', 'A0']. Selecting "
                "parameters by numeric index/position is not supported: "
                "position is not guaranteed to be stable across different "
                "orderings of a Pyomo model's equations/variables."
            )
        deduped = list(dict.fromkeys(names))  # de-dup, order-preserving

        # Fail fast: if model_parameter_names is already known at assignment
        # time, validate the subset relationship immediately — BEFORE
        # committing any state — rather than waiting for _resolve_ds_idx()
        # to be reached inside a criterion evaluation (e.g. mid-solve). This
        # also ensures a rejected assignment leaves interest_parameters
        # unchanged rather than partially applied. If model_parameter_names
        # isn't set yet, this check is skipped here and deferred to
        # _resolve_ds_idx().
        if self.model_parameter_names is not None:
            known = set(self.model_parameter_names)
            unknown = [nm for nm in deduped if nm not in known]
            if unknown:
                raise ValueError(
                    f"interest_parameters {unknown} not found in "
                    f"model_parameter_names {list(self.model_parameter_names)}. "
                    f"interest_parameters must be a SUBSET of "
                    f"model_parameter_names, matched by exact name."
                )

        self.ds_interest_names = deduped
        # invalidate any previously resolved positions; re-resolved lazily
        # (by name) the next time a Ds-optimal criterion is evaluated.
        self.ds_interest_idx = None
        self.ds_nuisance_idx = None

    @staticmethod
    def detect_sensitivity_analysis_function():
        """Identify which sensitivity routine is on the call stack.

        Walks the frames to find the caller, so warnings and diagnostics can
        name the routine they came from. Internal utility.

        Returns:
            str: Name of the detected sensitivity function.
        """
        frame = sys._getframe(1)
        while frame:
            if "numdifftools" in frame.f_code.co_filename:
                return False
            elif frame.f_code.co_name == "eval_sensitivities":
                return True
            frame = frame.f_back
        return False

    """ user-defined methods: must be overwritten by user to work """
    def simulate(self, unspecified):
        """Your model. Assign a callable to this attribute before use.

        pydex calls it once per candidate (and once per parameter perturbation
        when finite-differencing), and infers the model type from the ARGUMENT
        NAMES — so use exactly these, in this order:

        1. ``simulate(ti_controls, model_parameters)`` — static
        2. ``simulate(ti_controls, sampling_times, model_parameters)`` — dynamic
        3. ``simulate(tv_controls, sampling_times, model_parameters)`` — dynamic,
           time-varying controls only
        4. ``simulate(ti_controls, tv_controls, sampling_times, model_parameters)``
        5. ``simulate(sampling_times, model_parameters)`` — no controls

        Return a numpy array: shape ``(n_r,)`` for a static model, or
        ``(n_spt, n_r)`` for a dynamic one. ``n_r`` is inferred from the first
        call during :meth:`initialize`.

        For Pyomo models, also set :attr:`pyomo_model_fn` to get exact implicit
        function theorem derivatives instead of finite differences — faster and
        several orders of magnitude more accurate.

        Raises:
            SyntaxError: If called while still unassigned.

        Example:
            >>> def simulate(ti_controls, sampling_times, model_parameters):
            ...     CA0 = ti_controls[0]
            ...     k = model_parameters[0]
            ...     t = np.asarray(sampling_times)
            ...     return (CA0 * np.exp(-k * t)).reshape(-1, 1)
            >>> designer.simulate = simulate
        """
        raise SyntaxError("Don't forget to specify the simulate function.")

    """ core activity interfaces """

    def initialize(self, verbose=0, memory_threshold=int(1e9)):
        """ check for syntax errors, runs one simulation to determine n_r """

        """ check if simulate function has been specified """
        self._check_stats_framework()
        self._handle_simulate_sig()
        self._get_component_sizes()
        self._check_candidate_lengths()
        self._check_missing_components()

        if self._dynamic_system:
            self._check_var_spt()

        self._initialize_names()

        self._check_memory_req(memory_threshold)

        # ── Auto-configure Pyomo IFT + parallelisation ────────────────────────
        # If the user supplied a pyomo_model_fn but left use_pyomo_ift and
        # n_jobs at their __init__ defaults, flip them on automatically.
        # Explicit user overrides (e.g. use_pyomo_ift=False for FD debugging,
        # or n_jobs=1 to force sequential) are always respected.
        if self.pyomo_model_fn is not None:
            if self.use_pyomo_ift is None:       # not explicitly set → auto-enable
                self.use_pyomo_ift = True
                if verbose >= 1:
                    print("[INFO]: pyomo_model_fn detected — use_pyomo_ift set to True.")
            if self.n_jobs is None:              # not explicitly set → auto-parallelise
                self.n_jobs = -1
                if verbose >= 1:
                    print("[INFO]: pyomo_model_fn detected — n_jobs set to -1 (all cores).")
        # If user never set use_pyomo_ift and no pyomo_model_fn, default to False
        if self.use_pyomo_ift is None:
            self.use_pyomo_ift = False
        # If user never set n_jobs and no pyomo_model_fn, default to 1
        if self.n_jobs is None:
            self.n_jobs = 1
        # ─────────────────────────────────────────────────────────────────────

        # ── IFT structural check (delegated to diagnose_asl_elimination) ──────
        # When use_pyomo_ift=True, verify the one precondition the IFT
        # sensitivity path relies on: every parameter Var must be reachable in
        # the ASL primal vector (true ASL elimination — Failure Mode B — would
        # crash eval_sensitivities() with "Cannot match variable" after a long
        # run).  This is delegated to pydex.utils.diagnose_asl_elimination so
        # there is a SINGLE source of truth: the exact tool users run by hand to
        # vet a model is the one initialize() enforces — they can never disagree.
        #
        # The check is structural (constraint-graph topology), so it is
        # independent of which candidate's numeric values are used, and the
        # diagnostic owns the choice of a non-degenerate probe point internally.
        #
        # Result interpretation:
        #   • errored        → could not verify (e.g. only degenerate points);
        #                       warn and proceed, never block.
        #   • eliminated_*   → a parameter is genuinely unreachable → raise.
        #   • otherwise      → pass.
        #
        # Silently skipped when:
        #   • use_pyomo_ift is False             (FD path — no ASL involved)
        #   • _PYNUMERO_ASL_AVAILABLE is False   (pynumero not installed)
        #   • _DIAGNOSE_ASL_AVAILABLE is False   (utils not installed)
        #   • self._skip_asl_check is True       (set by unit tests)
        if (self.use_pyomo_ift
                and _PYNUMERO_ASL_AVAILABLE
                and _DIAGNOSE_ASL_AVAILABLE
                and not getattr(self, '_skip_asl_check', False)):

            if verbose >= 1:
                print("[INFO]: Running ASL parameter-reachability check "
                      "(diagnose_asl_elimination)...")

            # A representative sampling grid for the probe (at most 5 finite
            # times); the structural verdict does not depend on these values.
            try:
                _spt_check = np.asarray(
                    self.sampling_times_candidates[0], dtype=float
                ).flatten()
                _spt_check = _spt_check[np.isfinite(_spt_check)][:5]
            except Exception:
                _spt_check = np.array([])
            if len(_spt_check) < 3:
                _spt_check = np.array([0.001, 0.5, 1.0])

            try:
                _asl = _diagnose_asl(
                    self.pyomo_model_fn,
                    ti_controls      = self.ti_controls_candidates[0],
                    model_parameters = self.model_parameters,
                    sampling_times   = _spt_check,
                    verbose          = False,
                )
            except Exception as _asl_exc:
                # The diagnostic itself failed unexpectedly — warn, don't block.
                if verbose >= 1:
                    print(
                        f"[WARNING]: ASL reachability check could not run "
                        f"({type(_asl_exc).__name__}: {_asl_exc}). "
                        f"IFT structure is unverified — proceeding."
                    )
                _asl = None

            if _asl is not None:
                if _asl.get('errored'):
                    if verbose >= 1:
                        print(
                            f"[WARNING]: ASL reachability check unverified "
                            f"({_asl.get('error')}). Proceeding."
                        )
                elif not _asl.get('ift_ready'):
                    _missing = sorted({
                        name for _, name in (_asl.get('eliminated_full', [])
                                             + _asl.get('eliminated_single', []))
                    })
                    _sep = "=" * 70
                    _body = (
                        "\n" + _sep + "\n"
                        + "  IFT MODEL STRUCTURE ERROR — parameter not reachable in ASL NLP\n"
                        + _sep + "\n"
                        + "  The pyomo_model_fn does not satisfy pydex IFT requirements.\n"
                        + "  One or more parameter Vars are absent from the ASL primal\n"
                        + "  vector (true ASL elimination), so eval_sensitivities()\n"
                        + "  would crash with 'Cannot match variable' after a long run.\n"
                        + "\n"
                        + f"  Unreachable parameter(s): {_missing}\n"
                        + "\n"
                        + "  Root cause: the parameter appears only in singleton equality\n"
                        + "  constraints whose RHS is fully determined by other fixed\n"
                        + "  quantities, so ASL substitutes it away before IPOPT sees it.\n"
                        + "  Ensure each parameter Var participates in at least one\n"
                        + "  constraint involving a free (unfixed) state Var.\n"
                        + "\n"
                        + "  Run diagnose_asl_elimination(verbose=True) for a full report.\n"
                        + _sep
                    )
                    raise RuntimeError(_body)
                elif verbose >= 1:
                    print("[INFO]: ASL reachability check passed — "
                          "all parameters reachable.")
        # ─────────────────────────────────────────────────────────────────────

        if self.error_cov is None:
            print(
                f"[WARNING]: because the error_cov is not given, Pydex defaults to the "
                f"identity matrix of size {self.n_m_r}x{self.n_m_r}.")
            self.error_cov = np.eye(self.n_m_r)
            # Remember that this was a default, not a user statement. Downstream
            # code cannot otherwise tell "the noise really is unit variance"
            # from "nothing was supplied" -- run_estimability() needs the
            # difference to decide whether a noise-weighted ranking is
            # meaningful or fabricated.
            self._error_cov_defaulted = True
        try:
            self.error_fim = np.linalg.inv(self.error_cov)
        except np.linalg.LinAlgError:
            raise SyntaxError(
                "The provided error covariance is singular, please make sure you "
                "have passed in the correct error covariance."
            )

        self._status = 'ready'
        self._verbose = verbose
        if self._verbose >= 2:
            print("".center(100, "="))
        if self._verbose >= 1:
            print('Initialization complete: designer ready.')
        if self._verbose >= 2:
            print("".center(100, "-"))
            print(f"{'Number of model parameters':<40}: {self.n_mp}")
            print(f"{'Number of candidates':<40}: {self.n_c}")
            print(f"{'Number of responses':<40}: {self.n_r}")
            print(f"{'Number of measured responses':<40}: {self.n_m_r}")
            if self._invariant_controls:
                print(f"{'Number of time-invariant controls':<40}: {self.n_tic}")
            if self._dynamic_system:
                print(f"{'Number of sampling time choices':<40}: {self.n_spt}")
            if self._dynamic_controls:
                print(f"{'Number of time-varying controls':<40}: {self.n_tvc}")
            print(f"{'Covariance of measured responses':<40}: \n {self.error_cov}")
            print(f"{'Pyomo IFT sensitivities':<40}: {self.use_pyomo_ift}")
            print(f"{'Parallel workers (n_jobs)':<40}: {self.n_jobs}")
            print("".center(100, "="))

        return self._status

    def simulate_candidates(self, store_predictions=True,
                            plot_simulation_times=False):
        """Run :attr:`simulate` at every candidate and store the responses.

        Populates :attr:`response`, which the prediction plots need. Does not
        compute sensitivities.

        Args:
            store_predictions (bool): Keep the responses on the designer.
            plot_simulation_times (bool): Plot per-candidate wall-clock time —
                useful for spotting candidates where the model struggles.
        """
        self.response = None  # resets response every time simulation is invoked
        self.feval_simulation = 0
        time_list = []
        start = time()
        for i, exp in enumerate(
                zip(self.ti_controls_candidates, self.tv_controls_candidates,
                    self.sampling_times_candidates)):
            self._current_tic = exp[0]
            self._current_tvc = exp[1]
            self._current_spt = exp[2][~np.isnan(exp[2])]
            if not self._current_spt.size > 0:
                raise SyntaxError(
                    'One candidate has an empty list of sampling times, please check '
                    'the specified experimental candidates.'
                )

            """ determine if simulation needs to be re-run: if data on time-invariant 
            control variables is missing, will not run """
            cond_1 = np.any(np.isnan(exp[0]))
            if np.any([cond_1]):
                self._current_res = np.nan
            else:
                start = time()
                response = self._simulate_internal(self._current_tic, self._current_tvc,
                                                   self.model_parameters,
                                                   self._current_spt)
                finish = time()
                self.feval_simulation += 1
                self._current_res = response
                time_list.append(finish - start)

            if store_predictions:
                self._store_current_response()
        if plot_simulation_times:
            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.plot(time_list)
        if self._verbose >= 3:
            print(f"Completed simulation of all candidates in {time() - start} CPU seconds.")
        return self.response

    def simulate_optimal_candidates(self):
        """Run :attr:`simulate` at the SELECTED candidates only.

        Cheaper than :meth:`simulate_candidates` once a design exists. Prompts
        before overwriting stored responses.
        """
        if self.response is not None:
            overwrite = input("Previously stored responses data detected. "
                              "Running this will overwrite stored responses for the "
                              "optimal candidates. "
                              "Proceed? y: yes, n: no ")
            if not any(entry is overwrite for entry in ['y', 'yes']):
                return
        time_list = []
        for i, exp in enumerate(self.optimal_candidates):
            self._current_tic = exp[1]
            self._current_tvc = exp[2]
            self._current_spt = exp[3][~np.isnan(exp[3])]
            if self._current_spt.size <= 0:
                msg = 'One candidate has an empty list of sampling times, please check ' \
                      '' \
                      '' \
                      '' \
                      '' \
                      'the ' \
                      'specified experimental candidates.'
                raise SyntaxError(msg)

            """ 
            determine if simulation needs to be re-run: 
            if data on time-invariant control variables is missing, will not run 
            """
            cond_1 = np.any(np.isnan(exp[0]))
            if np.any([cond_1]):
                self._current_res = np.nan
            else:
                start = time()
                response = self._simulate_internal(self._current_tic, self._current_tvc,
                                                   self.model_parameters,
                                                   self._current_spt)
                finish = time()
                self.feval_simulation += 1
                self._current_res = response
                time_list.append(finish - start)

    # ------------------------------------------------------------------
    # Prior experimental information — sequential MBDoE support
    # ------------------------------------------------------------------

    def set_prior_fim(self, fim, model_parameters):
        """
        Register a Fisher Information Matrix from previously completed experiments
        (Case A: user already has the FIM, e.g. from an external parameter estimation
        routine that returned a parameter covariance matrix Σ_θ → FIM = Σ_θ⁻¹).

        The FIM is stored normalised at ``model_parameters``.  When
        ``design_experiment()`` is called with different (updated) model parameters,
        pydex automatically rescales the prior FIM to the current normalisation
        before adding it to the candidate FIM sum.

        Parameters
        ----------
        fim : array-like, shape (n_mp, n_mp)
            Fisher Information Matrix accumulated from prior experiments.
            Must be expressed in the **same normalisation convention** that
            pydex uses internally, i.e. each element (i,j) is scaled by
            θᵢ · θⱼ (the product of the nominal parameter values).

            If you have a raw (un-normalised) FIM from an external tool and
            your parameter vector is ``theta``, pass::

                fim_normalised = raw_fim * np.outer(theta, theta)

            If you have a parameter covariance matrix Σ_θ, pass::

                fim_normalised = np.linalg.inv(Σ_θ) * np.outer(theta, theta)

        model_parameters : array-like, shape (n_mp,)
            Parameter values at which ``fim`` was computed.  Used to rescale
            the prior FIM when ``designer.model_parameters`` is updated between
            design rounds.

        Examples
        --------
        >>> # From an external covariance matrix
        >>> theta_est  = np.array([0.45, 52000.0, 0.07, 72000.0])
        >>> sigma_theta = np.diag([0.01, 500.0, 0.005, 300.0]) ** 2
        >>> fim_raw = np.linalg.inv(sigma_theta)
        >>> designer.set_prior_fim(
        ...     fim              = fim_raw * np.outer(theta_est, theta_est),
        ...     model_parameters = theta_est,
        ... )

        See Also
        --------
        set_prior_experiments : Case B — compute FIM from raw experimental conditions.
        clear_prior           : Remove all registered prior information.
        """
        fim = np.asarray(fim, dtype=float)
        mp  = np.asarray(model_parameters, dtype=float).flatten()

        if fim.ndim != 2 or fim.shape[0] != fim.shape[1]:
            raise ValueError(
                f"fim must be a square 2-D array; got shape {fim.shape}."
            )
        if mp.size != fim.shape[0]:
            raise ValueError(
                f"model_parameters length ({mp.size}) must match fim dimension "
                f"({fim.shape[0]})."
            )

        self._prior_fim    = fim.copy()
        self._prior_fim_mp = mp.copy()

        if self._verbose >= 1:
            print(
                f"[set_prior_fim] Prior FIM registered "
                f"({fim.shape[0]}×{fim.shape[1]}, "
                f"computed at θ={np.array2string(mp, precision=4, separator=', ')})."
            )

    def set_prior_experiments(
        self,
        ti_controls,
        model_parameters,
        sampling_times  = None,
        tv_controls     = None,
        n_repeats       = None,
    ):
        """
        Compute and register the Fisher Information Matrix from previously
        completed experiments at **arbitrary** conditions (Case B: the conditions
        do not need to be part of any candidate grid).

        pydex evaluates model sensitivities at each supplied experimental
        condition using the same simulate function and finite-difference /
        Pyomo IFT machinery as for candidate-grid evaluations, then assembles:

            FIM_prior = Σₖ  nₖ · Sₖᵀ · Σ_ε⁻¹ · Sₖ

        where nₖ is the number of repeats at condition k, Sₖ is the
        (n_spt × n_r, n_mp) sensitivity matrix, and Σ_ε is ``designer.error_cov``.

        The result is stored exactly as in :meth:`set_prior_fim` and is
        automatically rescaled when ``designer.model_parameters`` is updated.

        Prerequisites
        -------------
        ``designer.initialize()`` must have been called before this method so
        that the simulate function signature is detected and internal dimensions
        are known.

        Parameters
        ----------
        ti_controls : array-like, shape (n_prior, n_tic)
            Time-invariant controls for each prior experiment.
            For a static (non-dynamic) model this encodes the full experimental
            condition.

        model_parameters : array-like, shape (n_mp,)
            Parameter values at which to evaluate sensitivities (your current
            best estimate after fitting the prior experiments).

        sampling_times : array-like or None
            Shape (n_prior, n_spt) or (n_spt,) for all-same timing.
            Required for dynamic models (``_dynamic_system=True``).
            Pass ``None`` for static models.

        tv_controls : array-like or None
            Shape (n_prior, n_tvc) time-varying controls, or ``None``.

        n_repeats : array-like of int or None
            Number of repeats at each condition, shape (n_prior,).
            ``None`` means each condition was run once.

        Examples
        --------
        Static model — three prior experiments, no sampling times:

        >>> designer.set_prior_experiments(
        ...     ti_controls      = np.array([[55.0, 65.0, 1.0],
        ...                                  [60.0, 70.0, 1.5],
        ...                                  [50.0, 60.0, 0.8]]),
        ...     model_parameters = theta_estimated,
        ... )

        Dynamic model — two prior experiments with per-experiment timing:

        >>> designer.set_prior_experiments(
        ...     ti_controls      = np.array([[55.0, 65.0, 1.0],
        ...                                  [60.0, 70.0, 1.5]]),
        ...     sampling_times   = np.array([[0.25, 0.5, 1.0],
        ...                                  [0.25, 0.75, 1.0]]),
        ...     model_parameters = theta_estimated,
        ... )

        With repeats — first condition run twice:

        >>> designer.set_prior_experiments(
        ...     ti_controls      = np.array([[55.0, 65.0, 1.0],
        ...                                  [60.0, 70.0, 1.5]]),
        ...     sampling_times   = np.array([[0.25, 0.5, 1.0],
        ...                                  [0.25, 0.75, 1.0]]),
        ...     model_parameters = theta_estimated,
        ...     n_repeats        = np.array([2, 1]),
        ... )

        See Also
        --------
        set_prior_fim : Case A — register a FIM directly.
        clear_prior   : Remove all registered prior information.
        """
        if self._status == 'empty':
            raise RuntimeError(
                "designer.initialize() must be called before set_prior_experiments()."
            )

        mp  = np.asarray(model_parameters, dtype=float).flatten()
        tic = np.atleast_2d(np.asarray(ti_controls, dtype=float))
        n_prior = tic.shape[0]

        # --- sampling times ---
        if sampling_times is not None:
            spt_arr = np.atleast_2d(np.asarray(sampling_times, dtype=float))
            if spt_arr.shape[0] == 1 and n_prior > 1:
                spt_arr = np.tile(spt_arr, (n_prior, 1))
        else:
            # static model: use a single dummy time point
            spt_arr = np.zeros((n_prior, 1))

        # --- tv_controls ---
        if tv_controls is not None:
            tvc_arr = np.atleast_2d(np.asarray(tv_controls, dtype=float))
            if tvc_arr.shape[0] == 1 and n_prior > 1:
                tvc_arr = np.tile(tvc_arr, (n_prior, 1))
        else:
            tvc_arr = np.zeros((n_prior, 1))

        # --- repeats ---
        if n_repeats is not None:
            repeats = np.asarray(n_repeats, dtype=float).flatten()
            if repeats.size != n_prior:
                raise ValueError(
                    f"n_repeats length ({repeats.size}) must match number of "
                    f"prior experiments ({n_prior})."
                )
        else:
            repeats = np.ones(n_prior)

        # --- error FIM ---
        if self.error_fim is None:
            error_fim = np.eye(self.n_m_r)
        else:
            error_fim = self.error_fim

        # --- save and temporarily override designer state for sensitivity eval ---
        old_tic  = self._current_tic
        old_tvc  = self._current_tvc
        old_spt  = self._current_spt
        old_scr  = self._current_scr_mp

        _use_pyomo_ift = getattr(self, 'use_pyomo_ift', False)
        if not _use_pyomo_ift:
            # Per-parameter step, sized off each parameter's own magnitude —
            # NOT a flat constant. A flat step silently corrupts the
            # sensitivities of any small-magnitude parameter, and here that
            # corruption propagates into the prior FIM that every subsequent
            # sequential design builds on. See _resolve_fd_base_step.
            step_generator = nd.step_generators.MaxStepGenerator(
                base_step    = _resolve_fd_base_step(self.model_parameters),
                step_ratio   = 2,
                num_steps    = self._num_steps,
                step_nom     = self._step_nom,
            )
            jacob_fun = nd.Jacobian(
                fun         = self._sensitivity_sim_wrapper,
                step        = step_generator,
                method      = 'forward',
                full_output = False,
            )

        fim_prior = np.zeros((self.n_mp, self.n_mp))
        if self._verbose >= 1:
            print(f"[set_prior_experiments] Computing sensitivities for "
                  f"{n_prior} prior experiment(s)...")

        for k in range(n_prior):
            self._current_tic    = tic[k]
            self._current_tvc    = tvc_arr[k]
            self._current_spt    = spt_arr[k][~np.isnan(spt_arr[k])]
            self._current_scr_mp = mp

            try:
                if _use_pyomo_ift:
                    _, sens_k = self._eval_sensitivities_pyomo_ift(
                        self._current_tic,
                        mp,
                        store_predictions=False,
                    )
                else:
                    sens_k = jacob_fun(mp, False)
                    # reshape to (n_spt, n_mr, n_mp)
                    n_spt_k = self._current_spt.size
                    if len(sens_k.shape) == 3:
                        sens_k = np.moveaxis(sens_k, 1, 2)
                    elif self.n_spt == 1:
                        if self.n_mp == 1:
                            sens_k = sens_k[:, :, np.newaxis]
                        else:
                            sens_k = sens_k.reshape(n_spt_k, self.n_m_r, self.n_mp)
                    else:
                        sens_k = sens_k.reshape(n_spt_k, self.n_m_r, self.n_mp)

            except Exception as exc:
                raise RuntimeError(
                    f"Sensitivity computation failed for prior experiment {k+1}/{n_prior}.\n"
                    f"  ti_controls : {tic[k]}\n"
                    f"  spt         : {self._current_spt}\n"
                    f"  Error       : {exc}"
                ) from exc

            # apply parameter normalisation (same as eval_sensitivities)
            if self._norm_sens_by_params:
                sens_k = sens_k * mp[None, None, :]

            # accumulate FIM: sum over time points and responses
            # sens_k shape: (n_spt, n_mr, n_mp)
            for t in range(sens_k.shape[0]):
                s = sens_k[t]   # (n_mr, n_mp)
                fim_prior += repeats[k] * (s.T @ error_fim @ s)

            if self._verbose >= 2:
                print(f"  [{k+1}/{n_prior}] tic={tic[k]}  "
                      f"FIM contribution rank={int(np.linalg.matrix_rank(fim_prior))}")

        # restore designer state
        self._current_tic    = old_tic
        self._current_tvc    = old_tvc
        self._current_spt    = old_spt
        self._current_scr_mp = old_scr

        self._prior_fim      = fim_prior
        self._prior_fim_mp   = mp.copy()
        self._prior_n_exp    = int(np.sum(repeats))

        if self._verbose >= 1:
            rank = int(np.linalg.matrix_rank(fim_prior))
            print(
                f"[set_prior_experiments] Prior FIM assembled from "
                f"{n_prior} condition(s) / {self._prior_n_exp} experiment(s).  "
                f"FIM rank: {rank}/{self.n_mp}."
            )

    def clear_prior(self):
        """
        Remove all registered prior experimental information.

        Call this to start a completely fresh design round without any
        prior FIM contribution, e.g. when switching to a different model
        or parameter set.

        See Also
        --------
        set_prior_fim          : Register a prior FIM directly (Case A).
        set_prior_experiments  : Compute prior FIM from experimental conditions (Case B).
        """
        self._prior_fim    = None
        self._prior_fim_mp = None
        self._prior_n_exp  = 0
        if self._verbose >= 1:
            print("[clear_prior] Prior FIM cleared.")

    def _get_apportioned_candidates(self):
        app_tic_candidates = []
        app_tvc_candidates = []
        app_spt_candidates = []
        for i, app in enumerate(self.apportionments):
            tic = self.optimal_candidates[i][1]
            tvc = self.optimal_candidates[i][2]
            spt = self.optimal_candidates[i][3]
            for _ in range(int(app)):
                app_tic_candidates.append(tic)
                app_tvc_candidates.append(tvc)
                app_spt_candidates.append(spt)
        app_tic_candidates = np.array(app_tic_candidates)
        app_tvc_candidates = np.array(app_tvc_candidates)
        app_spt_candidates = np.array(app_spt_candidates)
        return app_tic_candidates, app_tvc_candidates, app_spt_candidates

    def solve_cvar_problem(self, criterion, beta, n_spt=None, n_exp=None,
                           optimize_sampling_times=False, solver="ipopt",
                           solver_options=None, e0=None, write=False,
                           save_sensitivities=False, trim_fim=False,
                           pseudo_bayesian_type=None, regularize_fim=False,
                           reso=5, plot=False, n_bins=20, tol=1e-4, dpi=360,
                           **kwargs):
        """
        Solve the bi-objective average-CVaR experimental design problem via the
        epsilon-constraint method using Pyomo.
        """
        self._current_criterion = criterion.__name__

        if "cvar" not in self._current_criterion:
            raise SyntaxError(
                "Please pass in a valid cvar criterion e.g., cvar_d_opt_criterion."
            )

        # computing number of parameter scenarios that will be considered in CVaR
        self.beta = beta
        self.n_cvar_scr = (1 - self.beta) * self.n_scr
        if self.n_cvar_scr < 1:
            print(
                "[WARNING]: "
                "given n_scr * beta given is smaller than 1, this yields a maximin "
                "design. Please provide a larger number of n_scr if a CVaR design "
                "was desired."
            )
            self.n_cvar_scr = np.ceil(self.n_cvar_scr).astype(int)
        else:
            self.n_cvar_scr = np.floor(self.n_cvar_scr).astype(int)

        if reso < 3:
            print(
                f"The input reso is given as {reso}; the minimum value of reso is 3. "
                "Continuing with reso = 3."
            )
            reso = 3

        # initializing result lists
        self.cvar_optimal_candidates = []
        self.cvar_solution_times = []
        self._biobjective_values = np.empty((reso, 2))
        if plot:
            figs = []

            def add_fig(cdf, pdf):
                figs.append([cdf, pdf])

        def _common_kwargs():
            return dict(
                n_spt=n_spt,
                n_exp=n_exp,
                optimize_sampling_times=optimize_sampling_times,
                solver=solver,
                solver_options=solver_options,
                e0=e0,
                write=False,
                trim_fim=trim_fim,
                pseudo_bayesian_type=pseudo_bayesian_type,
                regularize_fim=regularize_fim,
                **kwargs,
            )

        def _phi_values():
            """Per-scenario info values from the last solve, for CDF plotting."""
            if self.pb_atomic_fims is None or self.efforts is None:
                return np.zeros(self.n_scr)
            e_flat = np.asarray(self.efforts).flatten()
            phis = []
            for j in range(self.n_scr):
                atoms_j = self.pb_atomic_fims[j]
                M_j = np.einsum('i,imn->mn', e_flat, atoms_j)
                cv = criterion(M_j)
                if isinstance(cv, tuple): cv = cv[0]
                phis.append(-float(cv))
            return np.array(phis)

        """ Iteration 1: Maximal (Type 1) Mean Design """
        if self._verbose >= 1:
            print(f" CVaR Problem ".center(100, "*"))
            print(f"")
            print(f"[Iteration 1/{reso}]".center(100, "="))
            print(f"Computing the maximal mean design, obtaining the mean UB and CVaR LB"
                  f" in the Pareto Frontier.")
            print(f"")
        self.design_experiment(criterion, beta=0.00, **_common_kwargs())
        self.beta = beta
        self.get_optimal_candidates()
        if self._verbose >= 1:
            self.print_optimal_candidates(tol=tol)
        iter_1_efforts = np.copy(self.efforts) / np.sum(self.efforts)
        mean_ub = self._criterion_value
        iter_1_phi = _phi_values()
        self._cvar_phi = iter_1_phi          # store for plot methods
        if self._verbose >= 1:
            print("")
            print("Computing CVaR of Iteration 1's Solution")

        # computing CVaR of Maximal Type 1 Mean Design
        self.design_experiment(criterion, beta=self.beta,
                               fix_effort=iter_1_efforts,
                               save_sensitivities=False, **_common_kwargs())
        cvar_lb = self._criterion_value
        if self._verbose >= 2:
            print(f"Time elapsed: {self._sensitivity_analysis_time:.2f} seconds.")

        self.cvar_optimal_candidates.append(self.optimal_candidates)
        self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
        self._biobjective_values[0, :] = np.array([mean_ub, cvar_lb])
        if self._verbose >= 1:
            print(f"CVaR LB: {cvar_lb}")
            print(f"Mean UB: {mean_ub}")
            print(f"[Iteration 1/{reso} Completed]".center(100, "="))
            print(f"")
        if plot:
            self._cvar_phi = iter_1_phi
            self._cvar_V   = float(np.percentile(iter_1_phi, (1 - beta) * 100))
            add_fig(
                self.plot_criterion_cdf(write=False, iteration=1),
                self.plot_criterion_pdf(write=False, iteration=1),
            )

        """ Iteration 2: Maximal CVaR_beta Design """
        if self._verbose >= 1:
            print(f"[Iteration 2/{reso}]".center(100, "="))
            print(f"Computing the maximal CVaR design, obtaining the CVaR UB, and mean "
                  f"LB in the Pareto Frontier.")
            print(f"")
        self.design_experiment(criterion, beta=self.beta,
                               save_sensitivities=False, **_common_kwargs())
        self.get_optimal_candidates()
        iter_2_efforts = np.copy(self.efforts) / np.sum(self.efforts)
        iter_2_phi = _phi_values()
        iter2_V    = float(np.percentile(iter_2_phi, (1 - beta) * 100))
        cvar_ub    = self._criterion_value
        if self._verbose >= 1:
            self.print_optimal_candidates(tol=tol)
            print("")
            print("Computing Mean of Iteration 2's Solution")

        self.design_experiment(criterion, beta=0.00,
                               fix_effort=iter_2_efforts,
                               save_sensitivities=False, **_common_kwargs())
        self.beta = beta
        mean_lb = self._criterion_value
        if self._verbose >= 2:
            print(f"Time elapsed: {self._sensitivity_analysis_time:.2f} seconds.")

        self.cvar_optimal_candidates.append(self.optimal_candidates)
        self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
        self._biobjective_values[1, :] = np.array([mean_lb, cvar_ub])
        if self._verbose >= 1:
            print(f"CVaR UB: {cvar_ub}")
            print(f"MEAN LB: {mean_lb}")
            print(f"[Iteration 2/{reso} Completed]".center(100, "="))
            print(f"")
        if plot:
            self._cvar_phi = iter_2_phi
            self._cvar_V   = iter2_V
            add_fig(
                self.plot_criterion_cdf(write=False, iteration=2),
                self.plot_criterion_pdf(write=False, iteration=2),
            )

        """ Iterations 3+: Intermediate Points """
        mean_values = np.linspace(mean_lb, mean_ub, reso)[1:-1]

        for i, mean in enumerate(mean_values):
            if self._verbose >= 1:
                print(f"[Iteration {i + 3}/{reso}]".center(100, "="))
            self.design_experiment(
                criterion, beta=self.beta,
                min_expected_value=mean,
                save_sensitivities=False,
                **_common_kwargs(),
            )
            self.get_optimal_candidates()
            iter_phi = _phi_values()
            self._cvar_phi = iter_phi
            self._cvar_V   = float(np.percentile(iter_phi, (1 - beta) * 100))
            self.cvar_optimal_candidates.append(self.optimal_candidates)
            self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
            self._biobjective_values[i + 2, :] = np.array([mean, self._criterion_value])

            if plot:
                add_fig(
                    self.plot_criterion_cdf(write=False, iteration=i+3),
                    self.plot_criterion_pdf(write=False, iteration=i+3),
                )
            if self._verbose >= 1:
                self.print_optimal_candidates(tol=tol)
                print(f"CVaR: {self._criterion_value}")
                print(f"MEAN: {iter_phi.mean():.6f}")
                print(f"[Iteration {i + 3}/{reso} Completed]".center(100, "="))
                print(f"")

        # use the same axes.xlim for all plotted cdfs and pdfs
        if plot:
            xlims = []
            for i, fig in enumerate(figs):
                cdf, pdf = fig[0], fig[1]
                xlims.append(cdf.axes[0].get_xlim())
            xlims = np.asarray(xlims)
            for i, fig in enumerate(figs):
                cdf, pdf = fig[0], fig[1]
                cdf.axes[0].set_xlim(xlims[:, 0].min(), xlims[:, 1].max())
                pdf.axes[0].set_xlim(xlims[:, 0].min(), xlims[:, 1].max())
                _safe_tight_layout(cdf)
                _safe_tight_layout(pdf)
                if write:
                    fn_cdf = f"iter_{i + 1}_cdf_{self.beta}_beta_{self.n_scr}_scr"
                    fp_cdf = self._generate_result_path(fn_cdf, "png")
                    fn_pdf = f"iter_{i + 1}_pdf_{self.beta}_beta_{self.n_scr}_scr"
                    fp_pdf = self._generate_result_path(fn_pdf, "png")
                    cdf.savefig(fp_cdf, dpi=dpi)
                    pdf.savefig(fp_pdf, dpi=dpi)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Unified Pyomo solver back-ends
    # ------------------------------------------------------------------

    def _make_pyomo_solver(self, solver_options=None):
        """
        Build a configured Pyomo SolverFactory from self._solver and solver_options.

        For standard AMPL solvers (ipopt, bonmin, cbc, …) options are forwarded
        via ``slvr.options``.

        For ``solver="gams"``, GAMS-specific arguments (``io_options``,
        ``add_options``) are handled at solve-time in ``_solve_pyomo``, not here.
        ``solver_options`` keys that start with ``"gams_"`` are stripped and not
        forwarded since they have no meaning as solver options.

        Special keys (extracted, not forwarded as numeric options):
            ``executable``  : full path to solver binary (AMPL solvers only).
        """
        solver_options = dict(solver_options or {})
        executable = solver_options.pop("executable", None)

        is_gams = (self._solver.lower() == "gams")

        if is_gams:
            slvr = _pyo.SolverFactory("gams")
        elif executable is not None:
            slvr = _pyo.SolverFactory(self._solver, executable=executable)
        else:
            slvr = _pyo.SolverFactory(self._solver)

        if not is_gams:
            defaults = {
                "max_iter"      : 3000,
                "tol"           : 1e-8,
                "acceptable_tol": 1e-6,
            }
            if self._verbose < 2:
                defaults["print_level"] = 0
            else:
                defaults["print_level"] = 5
            merged = {**defaults, **solver_options}
            for key, val in merged.items():
                slvr.options[key] = val

        return slvr

    def _pyomo_solve_kwargs(self, solver_options):
        """
        Extract GAMS solve-time kwargs (io_options, add_options) from
        solver_options when solver="gams".

        For all other solvers returns an empty dict.

        GAMS usage example::

            d.design_experiment(
                criterion      = d.d_opt_criterion,
                solver         = "gams",
                solver_options = {
                    "io_options"  : {"solver": "baron"},
                    "add_options" : [
                        "GAMS_MODEL.optfile = 1;",
                        "$onecho > baron.opt",
                        "MaxTime 1000",
                        "AbsConTol 1e-6",
                        "$offecho",
                    ],
                },
                min_effort = 0.05,
            )
        """
        if self._solver.lower() != "gams":
            return {}
        opts = dict(solver_options or {})
        kwargs = {}
        if "io_options" in opts:
            kwargs["io_options"] = opts["io_options"]
        if "add_options" in opts:
            kwargs["add_options"] = opts["add_options"]
        return kwargs

    def _solve_pyomo(self, criterion, e0, fix_effort, solver_options, **kwargs):
        """
        Solve the continuous-effort design NLP via native Pyomo expressions.

        The FIM is expressed as a linear combination of precomputed atomic FIMs:

            FIM(e) = Σᵢ eᵢ · Aᵢ   (linear in e, Aᵢ are numpy constants)

        For D-optimal, A-optimal, E-optimal and V-optimal criteria the objective
        is expressed entirely as native Pyomo expressions — no ExternalFunction
        or Python callbacks — so the model writes cleanly to a .nl file and works
        with any AMPL-compatible solver (IPOPT, Bonmin, SHOT, etc.).

        For unknown criteria (user-defined) we fall back to a scipy.optimize
        SLSQP solve using the criterion callable directly.

        For MINLP sparsity (min_effort > 0) binary variables are added and
        the problem is handed to a MINLP solver (Bonmin, Couenne, etc.).
        """
        import pyomo.environ as pyo

        n_e     = e0.size
        e0_flat = e0.flatten()
        min_eff = getattr(self, '_min_effort', 0.0) or 0.0
        use_minlp = (min_eff > 0.0)

        # Identify criterion type. This must happen BEFORE the atomic-FIM
        # guard below, because the pseudo-Bayesian type-0 path draws its
        # atomics from a different array.
        crit_name = getattr(criterion, '__name__', '')
        is_ds  = 'ds_opt' in crit_name and 'pb' not in crit_name
        is_d   = 'd_opt'  in crit_name and 'pb' not in crit_name and not is_ds
        is_a   = 'a_opt'  in crit_name and 'pb' not in crit_name
        is_e   = 'e_opt'  in crit_name and 'pb' not in crit_name
        is_v   = 'v_opt'  in crit_name
        is_b   = 'b_opt'  in crit_name   # Bracketing-optimal (Chen et al. 2018)
        is_pb  = self._pseudo_bayesian   # set by design_experiment() before we get here

        # --- Bracketing-optimal (b_opt) is FULLY ISOLATED from everything
        # below: it is dispatched to its own dedicated method before the
        # atomic-FIM computation, the structural-singularity gate, and the
        # Ds feasibility pre-check ever run. None of those are meaningful
        # for b_opt (it designs directly on input-factor values and
        # predicted responses, not on parameter sensitivities), and this
        # early return guarantees the new code path can never interact with
        # -- or be broken by future changes to -- the D/A/E/V/Ds machinery,
        # and vice versa.
        if is_b:
            if is_pb:
                raise ValueError(
                    "b_opt_criterion (Bracketing-optimal) has no "
                    "pseudo-Bayesian counterpart: it designs directly on "
                    "the input-factor space and predicted responses, not "
                    "on parameter sensitivities, so there is no meaningful "
                    "way to average it over parameter scenarios. Use a "
                    "fixed (non-scenario) model_parameters array."
                )
            if self._dynamic_system:
                raise NotImplementedError(
                    "b_opt_criterion currently supports only static "
                    "(non-dynamic) systems -- i.e. designer.simulate() with "
                    "no time dimension, one response per candidate. "
                    "Combining candidate AND sampling-time subset selection "
                    "is not yet implemented."
                )
            return self._solve_pyomo_b_opt(
                e0, fix_effort, solver_options,
                n_exp=kwargs.get('n_exp'),
                output_weight=kwargs.get('output_weight', 0.5),
            )

        # Pseudo-Bayesian "average information" (type 0) evaluates the criterion
        # at the SCENARIO-AVERAGED information matrix, f(meanₛ FIMₛ(e)). Since
        # every scenario FIM is linear in the efforts,
        #
        #     meanₛ FIMₛ(e) = meanₛ Σᵢ eᵢ·Aᵢ⁽ˢ⁾ = Σᵢ eᵢ·(meanₛ Aᵢ⁽ˢ⁾)
        #
        # the averaged FIM has exactly the same linear-in-e structure as a local
        # FIM assembled from scenario-averaged atomic FIMs. The existing native
        # formulations therefore apply verbatim — no per-scenario Cholesky
        # blocks, no extra variables — so type-0 can be solved symbolically
        # instead of by finite-difference SLSQP. Verified numerically for
        # d/a/e/ds: the type-0 criterion value and the corresponding local
        # criterion on averaged atomics agree to machine precision.
        #
        # Type 1 ("average criterion") is meanₛ f(FIMₛ) and does NOT reduce this
        # way — it needs one lifted block per scenario — so it keeps the SLSQP
        # fallback.
        #
        # V-optimality is excluded deliberately: v_opt_criterion has no
        # pseudo-Bayesian branch at all (there is no _pb_v_opt_criterion), so
        # there is no reference behaviour to validate a native pb-V solve
        # against, and the semantics of the W matrix under scenario averaging
        # are unspecified.
        #
        # CVaR reaches _solve_pyomo_cvar before this point, but is excluded
        # explicitly so this stays correct if that dispatch ever changes.
        _pb_type0 = (
            is_pb
            and not getattr(self, '_cvar_problem', False)
            and self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]
            and (is_d or is_a or is_e or is_ds)
        )

        # Any criterion not recognised as a native Pyomo type — and any
        # pseudo-Bayesian problem other than the type-0 case above — falls back
        # to scipy SLSQP on the criterion callable.
        is_native = (
            (is_d or is_a or is_e or is_v or is_ds) and (not is_pb or _pb_type0)
        )

        # For non-native criteria fall back to scipy SLSQP
        if not is_native:
            return self._solve_scipy_slsqp(
                criterion, e0, fix_effort, solver_options, **kwargs
            )

        # Ensure atomic FIMs are available and correctly sized for the current
        # effort vector.  The cache may be stale when n_spt has changed since
        # the last design_experiment call (e.g. n_spt=1 followed by n_spt=2):
        # n_e = e0.size reflects the new n_spt_comb but atomic_fims still has
        # the old count, causing an IndexError at A[i,j,k] below.
        if _pb_type0:
            _pb_atoms = self.pb_atomic_fims
            _stale = (
                _pb_atoms is None
                or np.asarray(_pb_atoms).ndim != 4
                or np.asarray(_pb_atoms).shape[0] != self.n_scr
                or np.asarray(_pb_atoms).shape[1] != n_e
            )
            if _stale:
                self._fd_jac = True
                self._compute_pb_atomics = True
                self.eval_fim(e0)
                _pb_atoms = self.pb_atomic_fims
            if _pb_atoms is None:
                # atomics unavailable (e.g. large-memory mode) — the symbolic
                # model cannot be built, so use the callable-based fallback
                return self._solve_scipy_slsqp(
                    criterion, e0, fix_effort, solver_options, **kwargs
                )
            # (n_scr, n_e, n_mp, n_mp) -> (n_e, n_mp, n_mp)
            A = np.asarray(_pb_atoms, dtype=float).mean(axis=0)
            if self._verbose >= 2:
                print(
                    f"[_solve_pyomo] pseudo-Bayesian type 0: averaged "
                    f"{self.n_scr} scenario atomic FIMs; solving natively "
                    f"with {crit_name} instead of the SLSQP fallback."
                )
        else:
            if self.atomic_fims is None or len(self.atomic_fims) != n_e:
                self._fd_jac = True
                self._compute_atomics = True
                self.eval_fim(e0)
            A = np.asarray(self.atomic_fims)   # (n_e, n_mp, n_mp)
        n_mp = self.n_mp

        # ── structural-singularity gate ───────────────────────────────────────
        # A FIM that is rank-deficient at the FULLY-SUPPORTED design is singular
        # for every admissible design. The solve may still "succeed" -- the
        # Cholesky-lifted log-det has a floored diagonal, so an interior-point
        # method can report optimality at a point propped up by that floor --
        # but the resulting criterion value rests on information the data does
        # not contain. Downstream symptoms are apportion() reporting the rounded
        # design as a fraction of a percent as informative as the continuous
        # one, with a Kiefer bound of 0.00%.
        #
        # Ds-optimality is the exception: it is DESIGNED for this situation and
        # is well posed as long as the deficiency stays inside the nuisance
        # block, which _ds_eval_schur() checks directly. So it is exempt here.
        if not getattr(self, "_allow_singular_fim", False) and not is_ds:
            _diag = self.diagnose_fim_structure(report=False)
            if _diag["singular"]:
                self.diagnose_fim_structure(report=True)
                raise ValueError(
                    f"The Fisher information matrix is STRUCTURALLY singular: "
                    f"rank {_diag['rank']} of {_diag['n_mp']} even at the "
                    f"fully-supported design, so no allocation of effort can "
                    f"repair it. Parameters implicated: {_diag['culprits']}. "
                    f"See the table above for the unidentifiable direction(s).\n\n"
                    f"This criterion asks for the precision of ALL parameters, "
                    f"which this data cannot deliver. Options:\n"
                    f"  * reparameterise so the unidentifiable combination "
                    f"becomes one parameter;\n"
                    f"  * add measurements that inform the direction;\n"
                    f"  * fix the offending parameter(s);\n"
                    f"  * design for a SUBSET instead: set "
                    f"designer.interest_parameters = [...] (excluding "
                    f"{_diag['culprits']}) and use designer.ds_opt_criterion;\n"
                    f"  * or, to proceed anyway and accept a criterion value "
                    f"that depends on the Cholesky floor, pass "
                    f"design_experiment(..., allow_singular_fim=True)."
                )

        # --- Ds-optimal feasibility pre-check ------------------------------
        # The native Ds formulation lifts BOTH log-det(FIM) and log-det(M_nn)
        # through Cholesky factors, which requires FIM ≻ 0 and M_nn ≻ 0 at
        # every iterate. If the nuisance block is rank-deficient even at the
        # most informative attainable design (all efforts on, i.e. the sum of
        # every atomic FIM plus any prior), then no feasible effort vector can
        # make M_nn positive definite and the Pyomo model is infeasible by
        # construction — IPOPT would report "restoration failed" rather than
        # anything interpretable.
        #
        # This is not a pathological corner: an unidentifiable nuisance
        # parameter is a normal and well-posed situation for Ds-optimal design,
        # because the criterion only needs the Schur complement S to be
        # positive definite, not M_nn. The numpy criterion path handles it via
        # the generalised (minimum-norm) Schur complement, so fall back to
        # SLSQP driving that path instead of emitting an infeasible NLP.
        if is_ds:
            _idx_s, _idx_n = self._resolve_ds_idx()
            if len(_idx_n) > 0:
                _fim_max = np.asarray(A).sum(axis=0)
                if self._prior_fim is not None:
                    _fim_max = _fim_max + np.asarray(self._prior_fim)
                if self._regularize_fim:
                    _fim_max = _fim_max + self._eps * np.identity(n_mp)
                _m_nn_max = _fim_max[np.ix_(_idx_n, _idx_n)]
                _eig = np.linalg.eigvalsh(0.5 * (_m_nn_max + _m_nn_max.T))
                _tol = max(1.0, float(np.abs(_eig).max())) * 1e-12
                if _eig.min() <= _tol:
                    if self._verbose >= 1:
                        print(
                            f"[INFO][ds_opt] nuisance block is rank-deficient "
                            f"even for the fully-supported design "
                            f"(lambda_min = {_eig.min():.3e}). The native "
                            f"Cholesky-lifted Pyomo formulation cannot be "
                            f"feasible, so falling back to SLSQP on the "
                            f"generalised Schur complement. This is a normal "
                            f"situation for Ds-optimal design and the result "
                            f"remains valid IF the Schur complement is "
                            f"positive definite at the starting design. If it "
                            f"is not, SLSQP has an infinite objective at the "
                            f"initial point and no gradient with which to "
                            f"escape it, so the design will not move and the "
                            f"criterion will stay infinite -- in that case "
                            f"enable regularize_fim, which keeps the solve on "
                            f"the native Cholesky-lifted path where IPOPT can "
                            f"make progress from an infeasible start. Note the "
                            f"resulting criterion value then depends on _eps."
                        )
                    return self._solve_scipy_slsqp(
                        criterion, e0, fix_effort, solver_options, **kwargs
                    )

        # --- build Pyomo model ---
        m = pyo.ConcreteModel()
        m.E   = pyo.RangeSet(0, n_e - 1)
        m.P   = pyo.RangeSet(0, n_mp - 1)

        if use_minlp:
            m.b = pyo.Var(m.E, domain=pyo.Binary)
            m.e = pyo.Var(m.E, domain=pyo.NonNegativeReals, bounds=(0, 1))
            m.sparsity_lb = pyo.Constraint(
                m.E, rule=lambda m, i: m.e[i] >= min_eff * m.b[i])
            m.sparsity_ub = pyo.Constraint(
                m.E, rule=lambda m, i: m.e[i] <= m.b[i])
        else:
            m.e = pyo.Var(m.E, domain=pyo.NonNegativeReals, bounds=(0, 1))

        if fix_effort is not None:
            fixed = (fix_effort / fix_effort.sum()).flatten()
            for i in m.E:
                m.e[i].fix(float(fixed[i]))

        m.sum_con = pyo.Constraint(expr=sum(m.e[i] for i in m.E) == 1.0)

        for i in m.E:
            m.e[i].set_value(float(e0_flat[i]))

        # FIM[j,k] = Σᵢ e[i] * A[i,j,k]  — linear Pyomo expression
        # Store as a dict for reuse in multiple criterion formulations
        fim_expr = {}
        for j in range(n_mp):
            for k in range(n_mp):
                fim_expr[j, k] = sum(
                    float(A[i, j, k]) * m.e[i] for i in m.E
                    if abs(A[i, j, k]) > 1e-30
                )

        # add prior FIM if registered
        if self._prior_fim is not None:
            prior = self._prior_fim.copy()
            if self._current_scr_mp is not None and self._prior_fim_mp is not None:
                if not np.allclose(self._current_scr_mp, self._prior_fim_mp, rtol=1e-10):
                    scale   = self._current_scr_mp / self._prior_fim_mp
                    rescale = np.outer(scale, scale)
                    prior   = prior * rescale
            for j in range(n_mp):
                for k in range(n_mp):
                    if abs(prior[j, k]) > 1e-30:
                        fim_expr[j, k] = fim_expr[j, k] + float(prior[j, k])

        # add Tikhonov regularization eps*I to FIM if requested
        # This mirrors the same regularization applied in eval_fim() and ensures
        # the native Pyomo/IPOPT solve uses the same FIM as the numpy callback path.
        if self._regularize_fim:
            for j in range(n_mp):
                fim_expr[j, j] = fim_expr[j, j] + float(self._eps)

        if is_d:
            # D-optimal: maximise log-det(FIM)
            # Expressed via auxiliary lower-triangular Cholesky factor L:
            #   FIM = L @ L.T,   log-det(FIM) = 2 * Σⱼ log(L[j,j])
            # This is a standard SDP-representable formulation that IPOPT handles
            # natively without any Python callbacks.
            m.L = pyo.Var(m.P, m.P, initialize=0.0)
            # fix upper triangle to zero
            for j in range(n_mp):
                for k in range(j + 1, n_mp):
                    m.L[j, k].fix(0.0)
            # diagonal must be positive
            for j in range(n_mp):
                m.L[j, j].setlb(1e-8)

            # Cholesky constraints: FIM[j,k] = Σ_r L[j,r]*L[k,r]  for k<=j
            def chol_rule(m, j, k):
                if k > j:
                    return pyo.Constraint.Skip
                lhs = fim_expr[j, k]
                rhs = sum(m.L[j, r] * m.L[k, r] for r in range(k + 1))
                return lhs == rhs
            m.chol_con = pyo.Constraint(m.P, m.P, rule=chol_rule)

            # objective: minimise -2*Σⱼ log(L[j,j])
            m.obj = pyo.Objective(
                expr=-2.0 * sum(pyo.log(m.L[j, j]) for j in m.P),
                sense=pyo.minimize,
            )

            # warm-start L from Cholesky of initial FIM
            try:
                FIM0 = sum(float(e0_flat[i]) * A[i] for i in range(n_e))
                if self._prior_fim is not None:
                    FIM0 = FIM0 + prior
                L0 = np.linalg.cholesky(FIM0 + 1e-6 * np.eye(n_mp))
                for j in range(n_mp):
                    for k in range(j + 1):
                        m.L[j, k].set_value(float(L0[j, k]))
            except np.linalg.LinAlgError:
                for j in range(n_mp):
                    m.L[j, j].set_value(1.0)

        elif is_ds:
            # Ds-optimal: maximise log-det(Schur complement of the
            # nuisance-parameter block), i.e. D-optimality restricted to a
            # subset of "interest" parameters (indices idx_s) while
            # marginalising out the "nuisance" parameters (indices idx_n):
            #
            #   log-det(S) = log-det(FIM) - log-det(M_nn)
            #
            # where M_nn is the FIM sub-block over idx_n and S is the Schur
            # complement of M_nn in the FIM (Schur determinant identity).
            #
            # Both log-dets are lifted via their own Cholesky factors (L for
            # the full FIM, Ln for M_nn), giving an exact, symbolic Pyomo
            # formulation — IPOPT differentiates it natively, no finite
            # differences and no Python callbacks are involved. When there
            # are no nuisance parameters this reduces exactly to the D-optimal
            # formulation above.
            #
            # The determinant identity is safe to use HERE (unlike in the
            # numpy criterion path) precisely because the pre-check above has
            # already established that M_nn can be made positive definite, so
            # the 0/0 indeterminacy that motivates the direct Schur route in
            # _ds_eval_schur() cannot arise on this branch.
            #
            # Note on convexity: -log-det(S) is convex in the efforts (verified
            # numerically; S is matrix-concave in the FIM and the FIM is linear
            # in e), so the underlying problem is convex and the optimum is
            # global. The lifted form below presents it to IPOPT as convex plus
            # concave (-2Σlog L_jj + 2Σlog Ln_jj), which can slow convergence
            # but does not introduce spurious local optima in the design.
            idx_s, idx_n = self._resolve_ds_idx()
            n_n = len(idx_n)

            # Scale-aware floor on the Cholesky diagonals. A hard-coded 1e-8
            # is not scale-free: for a badly scaled FIM whose true Cholesky
            # diagonal entries are legitimately smaller than the floor, the
            # bound makes the NLP INFEASIBLE rather than merely regularised.
            # Referencing the floor to the magnitude of the FIM keeps it a
            # strictly-positive guard without ever excluding the true solution.
            _fim_scale = float(np.abs(np.asarray(A).sum(axis=0)).max())
            _chol_lb = max(1e-12, 1e-10 * np.sqrt(max(_fim_scale, 1e-30)))

            m.L = pyo.Var(m.P, m.P, initialize=0.0)
            for j in range(n_mp):
                for k in range(j + 1, n_mp):
                    m.L[j, k].fix(0.0)
                m.L[j, j].setlb(_chol_lb)

            def chol_rule(m, j, k):
                if k > j:
                    return pyo.Constraint.Skip
                lhs = fim_expr[j, k]
                rhs = sum(m.L[j, r] * m.L[k, r] for r in range(k + 1))
                return lhs == rhs
            m.chol_con = pyo.Constraint(m.P, m.P, rule=chol_rule)

            if n_n > 0:
                m.PN = pyo.RangeSet(0, n_n - 1)
                m.Ln = pyo.Var(m.PN, m.PN, initialize=0.0)
                for j in range(n_n):
                    for k in range(j + 1, n_n):
                        m.Ln[j, k].fix(0.0)
                    m.Ln[j, j].setlb(_chol_lb)

                def chol_n_rule(m, j, k):
                    if k > j:
                        return pyo.Constraint.Skip
                    jj, kk = int(idx_n[j]), int(idx_n[k])
                    lhs = fim_expr[jj, kk]
                    rhs = sum(m.Ln[j, r] * m.Ln[k, r] for r in range(k + 1))
                    return lhs == rhs
                m.chol_n_con = pyo.Constraint(m.PN, m.PN, rule=chol_n_rule)

                # minimise -[2*Σ log(L_jj) - 2*Σ log(Ln_jj)]
                #        = -2*Σ log(L_jj) + 2*Σ log(Ln_jj)
                m.obj = pyo.Objective(
                    expr=-2.0 * sum(pyo.log(m.L[j, j]) for j in m.P)
                         + 2.0 * sum(pyo.log(m.Ln[j, j]) for j in m.PN),
                    sense=pyo.minimize,
                )
            else:
                # no nuisance parameters: Ds reduces to D-optimality
                m.obj = pyo.Objective(
                    expr=-2.0 * sum(pyo.log(m.L[j, j]) for j in m.P),
                    sense=pyo.minimize,
                )

            # warm-start L (and Ln) from Cholesky of the initial FIM
            try:
                FIM0 = sum(float(e0_flat[i]) * A[i] for i in range(n_e))
                if self._prior_fim is not None:
                    FIM0 = FIM0 + prior
                L0 = np.linalg.cholesky(FIM0 + 1e-6 * np.eye(n_mp))
                for j in range(n_mp):
                    for k in range(j + 1):
                        m.L[j, k].set_value(float(L0[j, k]))
                if n_n > 0:
                    M_nn0 = FIM0[np.ix_(idx_n, idx_n)]
                    Ln0 = np.linalg.cholesky(M_nn0 + 1e-6 * np.eye(n_n))
                    for j in range(n_n):
                        for k in range(j + 1):
                            m.Ln[j, k].set_value(float(Ln0[j, k]))
            except np.linalg.LinAlgError:
                for j in range(n_mp):
                    m.L[j, j].set_value(1.0)
                if n_n > 0:
                    for j in range(n_n):
                        m.Ln[j, j].set_value(1.0)

        elif is_a:
            # A-optimal: minimise trace(FIM⁻¹)
            # Via Schur complement: FIM⁻¹[j,j] = (FIM \ eⱼ)ⱼ
            # Lifted form: minimise Σⱼ t[j]  s.t. [FIM  I; I  diag(t)] >= 0
            # IPOPT-friendly form: auxiliary variables z[j] with constraints
            #   FIM @ z[j] = eⱼ,  t[j] >= z[j][j]
            m.Z = pyo.Var(m.P, m.P, initialize=0.0)  # Z[:,j] = FIM^{-1} e_j
            m.t = pyo.Var(m.P, domain=pyo.NonNegativeReals, initialize=1.0)

            # FIM @ Z[:,j] = I[:,j]  i.e. Σ_k FIM[i,k]*Z[k,j] = delta_{i,j}
            def fz_rule(m, i, j):
                lhs = sum(fim_expr[i, k] * m.Z[k, j] for k in range(n_mp))
                rhs = 1.0 if i == j else 0.0
                return lhs == rhs
            m.fz_con = pyo.Constraint(m.P, m.P, rule=fz_rule)

            # t[j] >= Z[j,j]  (diagonal of FIM^{-1})
            m.t_con = pyo.Constraint(
                m.P, rule=lambda m, j: m.t[j] >= m.Z[j, j]
            )

            m.obj = pyo.Objective(
                expr=sum(m.t[j] for j in m.P),
                sense=pyo.minimize,
            )

            # warm-start
            try:
                FIM0 = sum(float(e0_flat[i]) * A[i] for i in range(n_e))
                if self._prior_fim is not None:
                    FIM0 = FIM0 + prior
                Z0 = np.linalg.inv(FIM0 + 1e-6 * np.eye(n_mp))
                for j in range(n_mp):
                    for k in range(n_mp):
                        m.Z[j, k].set_value(float(Z0[j, k]))
                    m.t[j].set_value(float(Z0[j, j]))
            except np.linalg.LinAlgError:
                pass

        elif is_e:
            # E-optimal: maximise lambda_min(FIM)
            # Lifted form: maximise γ  s.t.  FIM - γ*I >= 0
            # IPOPT-friendly via Cholesky of (FIM - γ*I)
            m.gamma = pyo.Var(domain=pyo.Reals, initialize=0.1)
            m.gamma.setlb(0.0)
            m.L = pyo.Var(m.P, m.P, initialize=0.0)
            for j in range(n_mp):
                for k in range(j + 1, n_mp):
                    m.L[j, k].fix(0.0)
            for j in range(n_mp):
                m.L[j, j].setlb(1e-8)

            # Cholesky of (FIM - gamma*I)
            def echol_rule(m, j, k):
                if k > j:
                    return pyo.Constraint.Skip
                diag_adj = float(-1.0) * m.gamma if j == k else 0.0
                lhs = fim_expr[j, k] + (diag_adj if j == k else 0.0)
                rhs = sum(m.L[j, r] * m.L[k, r] for r in range(k + 1))
                return lhs == rhs
            m.echol_con = pyo.Constraint(m.P, m.P, rule=echol_rule)

            m.obj = pyo.Objective(expr=-m.gamma, sense=pyo.minimize)

        elif is_v:
            # V-optimal: minimise trace(W @ FIM^{-1} @ W.T)
            # Same lifted form as A-optimal but with W weighting
            if self.W is None:
                raise RuntimeError(
                    "V-optimal criterion requires W matrix. "
                    "Assign designer.dw_tic (and designer.dw_spt), or call "
                    "find_optimal_operating_point() first."
                )
            W = np.asarray(self.W)   # (n_pred, n_mp)
            n_pred = W.shape[0]
            m.PRED = pyo.RangeSet(0, n_pred - 1)

            # FIM @ Z = W.T  i.e. solve for Z = FIM^{-1} @ W.T
            m.Z = pyo.Var(m.P, m.PRED, initialize=0.0)
            m.t = pyo.Var(m.PRED, domain=pyo.NonNegativeReals, initialize=1.0)

            def vfz_rule(m, i, q):
                lhs = sum(fim_expr[i, k] * m.Z[k, q] for k in range(n_mp))
                rhs = float(W[q, i])
                return lhs == rhs
            m.vfz_con = pyo.Constraint(m.P, m.PRED, rule=vfz_rule)

            # trace(W @ FIM^{-1} @ W.T) = trace(W @ Z) = Σ_q (W @ Z)_{q,q}
            # = Σ_q Σ_k W[q,k] * Z[k,q]
            m.t_con = pyo.Constraint(
                m.PRED,
                rule=lambda m, q: m.t[q] >= sum(
                    float(W[q, k]) * m.Z[k, q] for k in range(n_mp)
                )
            )

            m.obj = pyo.Objective(
                expr=sum(m.t[q] for q in m.PRED),
                sense=pyo.minimize,
            )

        slvr = self._make_pyomo_solver(solver_options)
        gams_kwargs = self._pyomo_solve_kwargs(solver_options)
        result = slvr.solve(m, tee=(self._verbose >= 2), **gams_kwargs)

        tc = result.solver.termination_condition
        ok_conditions = {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible,
        }
        if tc not in ok_conditions:
            if self._verbose >= 0:
                print(f"[WARNING] Solver termination: {tc}. "
                      f"Results may be suboptimal.")

        e_opt = np.array([pyo.value(m.e[i]) for i in m.E])
        if self._specified_n_spt:
            self.efforts = e_opt.reshape((self.n_c, self.n_spt_comb))
        else:
            self.efforts = e_opt.reshape((self.n_c, self.n_spt))
        self._efforts_transformed = False

        obj_val = float(pyo.value(m.obj))
        return -obj_val

    def _solve_pyomo_b_opt(self, e0, fix_effort, solver_options,
                            n_exp=None, output_weight=0.5):
        """
        Bracketing-optimal (b_opt) design -- Chen, Paulavicius & Adjiman
        (2018), AIChE J. 64:3944-3957. Combines two objectives via
        weighted-sum scalarization (their Eq. 24):

          (1) INPUT-SPACE bracketing: D-optimality applied DIRECTLY to the
              (scaled) input-factor values themselves -- NOT to parameter
              sensitivities. Uses self.ti_controls_candidates directly; has
              nothing to do with self.atomic_fims / eval_fim().

          (2) OUTPUT-SPACE coverage: maximise the volume of the ellipsoid
              spanned by the selected candidates' PREDICTED RESPONSES
              (self.response), via a centered-covariance log-det.

        Both log-dets are lifted through their OWN Cholesky factor,
        mirroring the is_d branch in _solve_pyomo exactly: FIM-like matrix
        M = L @ L.T, log-det(M) = 2*sum(log(L[j,j])). This ONLY EVER
        introduces pairwise (bilinear) products L[j,r]*L[k,r] -- deliberately
        NOT an LDL/division recursion, which for 3+ input factors produces
        genuine trilinear monomials. That distinction matters: an LDL-based
        version of this exact model was empirically found to let BARON
        (via GAMS) report an incorrect zero optimality gap on a design that
        a verified true-determinant check showed was NOT optimal, at
        n_exp=4 with only 3 input factors, on the tablet-coater case study
        from the same paper. Bonmin (also tested) never produced an
        incorrect certificate across ~10 independent test cases against the
        same ground truth. Cholesky lifting removes the specific mechanism
        (trilinear terms) identified as the likely cause, so BARON may well
        be reliable on THIS formulation, but bonmin is the recommended
        default until that is independently re-verified.

        This method is FULLY ISOLATED from the rest of _solve_pyomo -- it
        is called from an early-return dispatch before any FIM/atomic_fims/
        singularity-gate code runs, is never reached by any other
        criterion, and touches no state (self.atomic_fims, self.fim, etc.)
        that any other criterion depends on.

        n_exp is the SAME n_exp already accepted by design_experiment() for
        every other criterion (used there only for apportion()) -- for
        b_opt it additionally becomes a hard cardinality constraint driving
        the solve itself, rather than a separate new argument.
        """
        import pyomo.environ as pyo

        if n_exp is None:
            raise ValueError(
                "b_opt_criterion requires an exact number of experiments: "
                "call design_experiment(designer.b_opt_criterion, n_exp=<int>, "
                "...)."
            )
        if not isinstance(n_exp, int) or n_exp < 2:
            raise ValueError(
                f"n_exp must be an integer >= 2 for b_opt_criterion "
                f"(covariance needs at least 2 selected candidates); got "
                f"{n_exp!r}."
            )
        if self.response is None:
            raise RuntimeError(
                "b_opt_criterion needs predicted responses at every "
                "candidate. Call designer.simulate_candidates() before "
                "design_experiment()."
            )
        if not (0.0 <= output_weight <= 1.0):
            raise ValueError(
                f"output_weight must be in [0, 1]; got {output_weight!r}."
            )

        U_raw = np.asarray(self.ti_controls_candidates, dtype=float)
        n_c, phi = U_raw.shape
        if n_c != self.n_c:
            raise RuntimeError(
                "Internal inconsistency: ti_controls_candidates row count "
                f"({n_c}) != self.n_c ({self.n_c})."
            )
        if n_exp > n_c:
            raise ValueError(
                f"n_exp ({n_exp}) cannot exceed the number of candidates "
                f"({n_c})."
            )

        lb, ub = U_raw.min(axis=0), U_raw.max(axis=0)
        span = np.where(ub > lb, ub - lb, 1.0)
        U = 2.0 * (U_raw - lb) / span - 1.0   # scaled to [-1, 1], per factor

        Y_raw = np.asarray(self.response, dtype=float).reshape(n_c, -1)
        n_resp = Y_raw.shape[1]

        # ---- rank feasibility of the two Cholesky lifts -------------------
        # Both lifts below are built UNCONDITIONALLY, whatever output_weight
        # is, and both floor their Cholesky diagonal at 1e-8. That floor is
        # what turns an under-sized n_exp into a HARD INFEASIBILITY rather
        # than a merely ill-conditioned solve:
        #
        #   input  M_in  is phi x phi,      rank <= n_exp      -> n_exp >= phi
        #   output M_out is n_resp x n_resp and CENTERED, so
        #                                   rank <= n_exp - 1
        #
        # For M_out the ALGEBRAIC bound is n_exp >= n_resp + 1, but that is not
        # sufficient in practice and the difference is measured, not guessed.
        # At n_exp == n_resp + 1 the centered covariance is rank-EXACTLY-n_resp
        # with no margin, and the Cholesky floor demands strict positive
        # definiteness with room to spare; bonmin then reports the problem
        # INFEASIBLE whenever the output term carries weight. Observed on a
        # 10-candidate phi=2/n_resp=2 pool: n_exp=3 infeasible at
        # output_weight >= 0.5, n_exp=4 solving immediately. Hence + 2.
        #
        # Do not "correct" this back to + 1 on the strength of the algebra.
        #
        # M == L L^T with every L[j,j] >= 1e-8 forces det(M) >= 1e-8**(2*dim),
        # so a rank-deficient M cannot be represented at all.
        #
        # Caught HERE rather than left to the solver, because proving
        # infeasibility of a nonconvex MINLP is the expensive direction:
        # measured on a 70-candidate phi=6 pool, n_exp=5 ran for over 17
        # minutes of bonmin CPU without terminating, while n_exp=6 solved in
        # seconds. Without this guard a mistyped n_exp costs an unbounded
        # hang and prints no diagnosis at all.
        n_exp_min = max(phi, n_resp + 2)
        if n_exp < n_exp_min:
            reasons = []
            if n_exp < phi:
                reasons.append(
                    f"the input-space matrix is {phi}x{phi}, so it needs "
                    f"n_exp >= {phi}"
                )
            if n_exp < n_resp + 2:
                reasons.append(
                    f"the centered output covariance is {n_resp}x{n_resp}, "
                    f"and needs a rank margin above the Cholesky floor, so it "
                    f"needs n_exp >= {n_resp + 2}"
                )
            raise ValueError(
                f"n_exp ({n_exp}) is too small for b_opt_criterion: "
                + "; ".join(reasons)
                + f". With {phi} input factor(s) and {n_resp} response(s), "
                f"use n_exp >= {n_exp_min}. Below that the Cholesky lift "
                f"(diagonal floored at 1e-8) makes the problem strictly "
                f"infeasible, and an MINLP solver will not report that "
                f"quickly -- it will appear to hang."
            )
        Y_mean, Y_std = Y_raw.mean(axis=0), Y_raw.std(axis=0)
        Y_std = np.where(Y_std > 0, Y_std, 1.0)
        Y = (Y_raw - Y_mean) / Y_std   # z-scored responses

        win, wout = 1.0 - float(output_weight), float(output_weight)
        self._b_opt_output_weight = float(output_weight)   # for _b_opt_criterion()

        m = pyo.ConcreteModel()
        m.E = pyo.RangeSet(0, n_c - 1)
        m.b = pyo.Var(m.E, domain=pyo.Binary)
        m.e = pyo.Var(m.E, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # exact cardinality + equal weight -- b_opt ONLY; the existing
        # sparsity (min_effort) constraints in _solve_pyomo are never
        # touched by this method and this method never touches them.
        m.cardinality_con = pyo.Constraint(expr=sum(m.b[i] for i in m.E) == n_exp)
        m.equal_weight_con = pyo.Constraint(
            m.E, rule=lambda m, i: m.e[i] == m.b[i] / n_exp)

        e0_flat = np.asarray(e0, dtype=float).flatten()
        if fix_effort is not None:
            fixed = (fix_effort / fix_effort.sum()).flatten()
            for i in m.E:
                m.b[i].fix(1.0 if fixed[i] > 1e-9 else 0.0)
        for i in m.E:
            if not m.b[i].fixed:
                m.b[i].set_value(1.0 if e0_flat[i] > 1e-9 else 0.0)

        # ---- input-space bracketing: M_in[p,q] = sum_i b[i]*U[i,p]*U[i,q] ----
        m.PHI = pyo.RangeSet(0, phi - 1)
        M_in = {(p, q): sum(m.b[i] * float(U[i, p] * U[i, q]) for i in m.E)
                for p in range(phi) for q in range(phi)}

        m.L = pyo.Var(m.PHI, m.PHI, initialize=0.0)
        for p in range(phi):
            for q in range(p + 1, phi):
                m.L[p, q].fix(0.0)
            m.L[p, p].setlb(1e-8)

        def chol_in_rule(m, p, q):
            if q > p:
                return pyo.Constraint.Skip
            return M_in[p, q] == sum(m.L[p, r] * m.L[q, r] for r in range(q + 1))
        m.chol_in_con = pyo.Constraint(m.PHI, m.PHI, rule=chol_in_rule)

        # ---- output-space coverage: centered covariance via a collapsed
        # identity: sum_i b_i*(y_i-yc)(y_i-yc)^T = sum_i b_i*y_i*y_i^T - n*yc*yc^T
        # -- only n_resp^2 bilinear (yc*yc) terms, not O(n_c^2) ----
        m.RESP = pyo.RangeSet(0, n_resp - 1)
        yc_vars = {}
        for c in range(n_resp):
            v = pyo.Var(initialize=0.0, bounds=(-10, 10))
            m.add_component(f"b_opt_yc_{c}", v)
            con = pyo.Constraint(
                expr=n_exp * v == sum(m.b[i] * float(Y[i, c]) for i in m.E))
            m.add_component(f"b_opt_yc_con_{c}", con)
            yc_vars[c] = v

        P = {(c, d): Y[:, c] * Y[:, d] for c in range(n_resp) for d in range(n_resp)}
        M_out = {}
        for c in range(n_resp):
            for d in range(n_resp):
                lin = sum(m.b[i] * float(P[c, d][i]) for i in m.E)
                M_out[c, d] = (lin - n_exp * yc_vars[c] * yc_vars[d]) / max(n_exp - 1, 1)

        m.Lo = pyo.Var(m.RESP, m.RESP, initialize=0.0)
        for c in range(n_resp):
            for d in range(c + 1, n_resp):
                m.Lo[c, d].fix(0.0)
            m.Lo[c, c].setlb(1e-8)

        def chol_out_rule(m, c, d):
            if d > c:
                return pyo.Constraint.Skip
            return M_out[c, d] == sum(m.Lo[c, r] * m.Lo[d, r] for r in range(d + 1))
        m.chol_out_con = pyo.Constraint(m.RESP, m.RESP, rule=chol_out_rule)

        # ---- anti-clustering: mutual exclusion on near-duplicate response
        # candidates. Candidates are FIXED, so pairwise response distances
        # are precomputable constants -- a discrete-problem simplification
        # of Chen et al.'s continuous log-barrier penalty (mu). Off by
        # default; opt in via designer._b_opt_min_sep_frac. ----
        min_sep_frac = getattr(self, "_b_opt_min_sep_frac", 0.0)
        if min_sep_frac > 0:
            diffs = Y[:, None, :] - Y[None, :, :]
            dist2 = np.sum(diffs ** 2, axis=-1)
            thresh = min_sep_frac * dist2.max()
            m.b_opt_excl = pyo.ConstraintList()
            for i in range(n_c):
                for j in range(i + 1, n_c):
                    if dist2[i, j] < thresh:
                        m.b_opt_excl.add(m.b[i] + m.b[j] <= 1)

        # ---- combined weighted-sum objective (Chen et al. Eq. 24) ----
        eps = 1e-9
        fin_log = 2.0 * sum(pyo.log(m.L[p, p] + eps) for p in range(phi))
        fout_log = 2.0 * sum(pyo.log(m.Lo[c, c] + eps) for c in range(n_resp))
        m.obj = pyo.Objective(
            expr=-(win * fin_log + wout * fout_log), sense=pyo.minimize)

        slvr = self._make_pyomo_solver(solver_options)
        gams_kwargs = self._pyomo_solve_kwargs(solver_options)
        result = slvr.solve(m, tee=(self._verbose >= 2), **gams_kwargs)

        tc = result.solver.termination_condition
        self._b_opt_termination = str(tc)
        ok_conditions = {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible,
        }
        # A FAILED solve must not yield a design. When bonmin reports
        # `infeasible`, pyo.value(m.b[i]) returns 1.0 for EVERY candidate, so
        # the old warn-and-continue produced e_opt = 1/n_exp across the whole
        # pool: a "design" that selects every candidate, breaches the
        # cardinality constraint outright, and has efforts summing to n_c/n_exp
        # instead of 1. That is not a suboptimal answer, it is not an answer,
        # and printing a warning while returning it means anyone not reading
        # stderr treats it as a result.
        _hard_failures = {
            pyo.TerminationCondition.infeasible,
            pyo.TerminationCondition.infeasibleOrUnbounded,
            pyo.TerminationCondition.unbounded,
            pyo.TerminationCondition.error,
            pyo.TerminationCondition.internalSolverError,
        }
        if tc in _hard_failures:
            raise RuntimeError(
                f"b_opt_criterion: the MINLP solver returned '{tc}', so there "
                f"is no design to report. Most often this means n_exp is too "
                f"close to the rank bound -- with {phi} input factor(s) and "
                f"{n_resp} response(s), try n_exp >= {max(phi, n_resp + 2)} "
                f"(currently {n_exp}); the centered output covariance needs a "
                f"margin above the Cholesky floor, not merely full rank. "
                f"Other causes: a candidate pool with duplicate or collinear "
                f"rows, or an over-tight _b_opt_min_sep_frac."
            )
        # Limit conditions are different: there IS an incumbent, it is just not
        # proven optimal. Report it, and record the fact so a caller (or an
        # example printing a table) can distinguish a proven optimum from a
        # best-so-far. Silently mixing the two is how a solver artefact gets
        # read as a non-monotonic Pareto front.
        self._b_opt_proven_optimal = (tc == pyo.TerminationCondition.optimal)
        if tc not in ok_conditions:
            if self._verbose >= 0:
                print(f"[WARNING] Solver termination: {tc}. The design below "
                      f"is the solver's best incumbent, NOT a proven optimum.")

        b_opt_val = np.array([pyo.value(m.b[i]) for i in m.E])

        # Validate independently of what the solver reported. This catches the
        # failure mode above even if a future solver returns a status not in
        # _hard_failures, and it is cheap: the selection is binary by
        # construction, so anything else means the values did not come from a
        # completed solve.
        _n_sel = int(np.sum(b_opt_val > 0.5))
        _binary_ok = np.all((b_opt_val < 1e-4) | (b_opt_val > 1.0 - 1e-4))
        if _n_sel != n_exp or not _binary_ok:
            raise RuntimeError(
                f"b_opt_criterion: the solver reported '{tc}' but the returned "
                f"selection is not a valid design -- {_n_sel} candidate(s) "
                f"selected where n_exp={n_exp}"
                + ("" if _binary_ok else ", and the selection variables are not "
                   "binary")
                + ". Refusing to report it. Please raise this as an issue "
                  "with the candidate pool and arguments used."
            )

        e_opt = b_opt_val / n_exp
        self.efforts = e_opt.reshape((self.n_c, self.n_spt))
        self._efforts_transformed = False
        self._b_opt_selected_idx = np.where(b_opt_val > 0.5)[0]   # convenience
        self._b_opt_apportion_redundant = True   # design is already exact/equal-weight

        obj_val = float(pyo.value(m.obj))
        return -obj_val

    def _solve_scipy_slsqp(self, criterion, e0, fix_effort, solver_options, **kwargs):
        """
        Fallback solver for criteria that cannot be expressed as native Pyomo
        expressions (e.g. pseudo-Bayesian, user-defined criteria).
        Uses scipy.optimize.minimize with method='SLSQP'.
        """
        from scipy.optimize import minimize as _sp_minimize

        n_e     = e0.size
        e0_flat = e0.flatten()

        bounds = [(0.0, 1.0)] * n_e
        if fix_effort is not None:
            fixed = (fix_effort / fix_effort.sum()).flatten()
            bounds = [(float(f), float(f)) for f in fixed]

        constraints = [{"type": "eq", "fun": lambda e: np.sum(e) - 1.0}]

        opts = {"ftol": 1e-9, "maxiter": 5000, "disp": self._verbose >= 2}
        if solver_options:
            opts.update({k: v for k, v in solver_options.items()
                         if k in ("ftol", "maxiter", "disp")})

        self._fd_jac = True
        res = _sp_minimize(
            fun=criterion,
            x0=e0_flat,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=opts,
        )

        if not res.success and self._verbose >= 1:
            print(f"[WARNING] SLSQP: {res.message}")

        e_opt = res.x
        if self._specified_n_spt:
            self.efforts = e_opt.reshape((self.n_c, self.n_spt_comb))
        else:
            self.efforts = e_opt.reshape((self.n_c, self.n_spt))
        self._efforts_transformed = False

        return -float(res.fun)

    def _solve_pyomo_operating_point(self, x0, lb_arr, ub_arr, solver_options):
        """
        Solve the operating-point optimisation via PyNumero + cyipopt.

        PyNumero's ExternalGreyBoxBlock allows Python callables to be embedded
        in a Pyomo model without requiring pyomo_ampl.so. libpynumero_ASL.dylib
        is present in the IDAES solver package and supports this path.
        Falls back to scipy SLSQP if PyNumero is unavailable.
        """
        try:
            return self._solve_operating_point_pynumero(
                x0, lb_arr, ub_arr, solver_options
            )
        except Exception:
            return self._solve_operating_point_scipy(
                x0, lb_arr, ub_arr, solver_options
            )

    def _solve_operating_point_scipy(self, x0, lb_arr, ub_arr, solver_options):
        """Scipy SLSQP fallback for operating point optimisation."""
        from scipy.optimize import minimize as _sp_min

        n_tic = self.n_tic if self._invariant_controls else 0
        sign  = -1.0 if self.dw_sense == "maximize" else 1.0
        dr    = self

        def obj(x):
            return sign * float(dr.process_objective(
                x[:n_tic], x[n_tic:], dr.model_parameters))

        raw = []
        if dr.process_constraints is not None:
            raw = dr.process_constraints(x0[:n_tic], x0[n_tic:], dr.model_parameters)

        sp_cons = []
        for c in raw:
            f = c["fun"]
            sp_cons.append({
                "type": c["type"],
                "fun" : lambda x, _f=f: float(
                    _f(x[:n_tic], x[n_tic:], dr.model_parameters))
            })

        bounds = list(zip(
            [float(v) if np.isfinite(v) else None for v in lb_arr],
            [float(v) if np.isfinite(v) else None for v in ub_arr],
        ))

        opts = {"ftol": 1e-8, "maxiter": 3000, "disp": self._verbose >= 2}
        if solver_options:
            opts.update({k: v for k, v in solver_options.items()
                         if k in ("ftol", "maxiter", "disp")})

        res = _sp_min(obj, x0, method="SLSQP",
                      bounds=bounds, constraints=sp_cons, options=opts)

        obj_val = sign * float(res.fun)
        return res.x, obj_val

    def _solve_operating_point_pynumero(self, x0, lb_arr, ub_arr, solver_options):
        """
        Operating point optimisation via PyNumero ExternalGreyBoxBlock + cyipopt.
        This uses libpynumero_ASL.dylib (present in IDAES) rather than pyomo_ampl.so.
        """
        from pyomo.contrib.pynumero.interfaces.external_grey_box import (
            ExternalGreyBoxModel, ExternalGreyBoxBlock,
        )
        from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
            CyIpoptSolver, CyIpoptNLP,
        )
        import pyomo.environ as pyo

        n_tic = self.n_tic if self._invariant_controls else 0
        n_tvc = self.n_tvc if self._dynamic_controls else 0
        n_x   = n_tic + n_tvc
        sign  = -1.0 if self.dw_sense == "maximize" else 1.0
        dr    = self
        h_fd  = np.sqrt(np.finfo(float).eps)

        raw_cons = []
        if dr.process_constraints is not None:
            raw_cons = dr.process_constraints(
                x0[:n_tic], x0[n_tic:], dr.model_parameters
            )
        n_eq  = sum(1 for c in raw_cons if c["type"] == "eq")
        n_ineq = sum(1 for c in raw_cons if c["type"] == "ineq")

        class _OpModel(ExternalGreyBoxModel):
            def input_names(self):
                """list of str: Primal variable names, for the cyipopt interface."""
                return [f"x{i}" for i in range(n_x)]
            def equality_constraint_names(self):
                """list of str: Equality constraint names, for cyipopt."""
                return [f"eq{k}" for k in range(n_eq)]
            def output_names(self):
                """list of str: Output names, for cyipopt."""
                return []
            def set_input_values(self_, x):
                """Set the primal values at which the callbacks are evaluated."""
                self_._x = np.array(x)
            def evaluate_equality_constraints(self_):
                """Return the equality constraint residuals at the current point."""
                eq_vals = [float(c["fun"](
                    self_._x[:n_tic], self_._x[n_tic:], dr.model_parameters
                )) for c in raw_cons if c["type"] == "eq"]
                return np.array(eq_vals)
            def evaluate_jacobian_equality_constraints(self_):
                """Return the Jacobian of the equality constraints, sparse."""
                import scipy.sparse as sp
                rows, cols, vals = [], [], []
                eq_idx = 0
                for c in raw_cons:
                    if c["type"] != "eq":
                        continue
                    f  = c["fun"]
                    f0 = float(f(self_._x[:n_tic], self_._x[n_tic:], dr.model_parameters))
                    for j in range(n_x):
                        xp = self_._x.copy(); xp[j] += h_fd
                        fp = float(f(xp[:n_tic], xp[n_tic:], dr.model_parameters))
                        rows.append(eq_idx); cols.append(j); vals.append((fp-f0)/h_fd)
                    eq_idx += 1
                return sp.coo_matrix((vals, (rows, cols)), shape=(n_eq, n_x))

        m = pyo.ConcreteModel()
        m.ex = ExternalGreyBoxBlock()
        m.ex.set_external_model(_OpModel())
        m.x = m.ex.inputs

        # objective
        def _obj_expr():
            xv = np.array([pyo.value(m.x[f"x{i}"]) for i in range(n_x)])
            return sign * float(dr.process_objective(
                xv[:n_tic], xv[n_tic:], dr.model_parameters))

        # inequality constraints as regular Pyomo constraints
        for k, c in enumerate(raw_cons):
            if c["type"] == "ineq":
                f = c["fun"]
                def _ineq(m, _f=f):
                    xv = np.array([pyo.value(m.x[f"x{i}"]) for i in range(n_x)])
                    return float(_f(xv[:n_tic], xv[n_tic:], dr.model_parameters)) >= 0
                setattr(m, f"ineq_{k}", pyo.Constraint(rule=_ineq))

        # bounds
        for i in range(n_x):
            v = m.x[f"x{i}"]
            v.set_value(float(x0[i]))
            if np.isfinite(lb_arr[i]): v.setlb(float(lb_arr[i]))
            if np.isfinite(ub_arr[i]): v.setub(float(ub_arr[i]))

        # fall through to scipy if this gets too complex
        raise NotImplementedError("PyNumero path not fully implemented; using scipy.")

    def _solve_pyomo_cvar(self, criterion, beta, e0, min_expected_value,
                          solver_options, **kwargs):
        """
        Solve the CVaR experimental design problem via scipy SLSQP.

        The CVaR objective involves per-scenario FIM evaluations that cannot
        be expressed as native Pyomo expressions (they depend on the criterion
        callable). SLSQP handles this efficiently for moderate n_scr.

        Augmented decision vector: x = [e (n_e),  V (1),  delta (n_scr)]

        Objective (minimise):
            -V + 1/(n_scr*(1-beta)) * sum(delta)

        Constraints:
            sum(e) == 1
            delta_j >= V - phi_j(e)   for j = 0..n_scr-1
            (optional) mean(phi_j) >= min_expected_value
        """
        from scipy.optimize import minimize as _sp_min

        if self._large_memory_requirement:
            raise NotImplementedError(
                "The CVaR solver requires pb_atomic_fims to be stored in memory."
            )

        self.efforts = e0
        self.eval_fim(e0)

        n_e    = e0.size
        n_scr  = self.n_scr
        pb_atomics = self.pb_atomic_fims   # (n_scr, n_e, n_mp, n_mp)

        def _phi(p_flat, scr_idx):
            atoms_j = pb_atomics[scr_idx]
            M_j = np.einsum('i,imn->mn', p_flat, atoms_j)
            cv  = criterion(M_j)
            if isinstance(cv, tuple): cv = cv[0]
            return -float(cv)

        e0_flat = e0.flatten()
        phis0   = np.array([_phi(e0_flat, j) for j in range(n_scr)])
        V0      = float(np.percentile(phis0, (1 - beta) * 100))
        d0      = np.maximum(0.0, V0 - phis0)
        x0_aug  = np.concatenate([e0_flat, [V0], d0])

        coeff = 1.0 / (n_scr * (1.0 - beta))

        def obj(x):
            V     = x[n_e]
            delta = x[n_e + 1:]
            return -V + coeff * np.sum(delta)

        def grad_obj(x):
            g = np.zeros_like(x)
            g[n_e]       = -1.0
            g[n_e + 1:]  =  coeff
            return g

        constraints = []
        # sum(e) == 1
        constraints.append({
            "type": "eq",
            "fun" : lambda x: np.sum(x[:n_e]) - 1.0,
            "jac" : lambda x: np.concatenate([np.ones(n_e), np.zeros(1 + n_scr)]),
        })
        # delta[j] - V + phi_j(e) >= 0
        for j in range(n_scr):
            _j = j
            def _cj(x, __j=_j):
                return x[n_e + 1 + __j] - x[n_e] + _phi(x[:n_e], __j)
            constraints.append({"type": "ineq", "fun": _cj})

        if min_expected_value is not None:
            def _mean_phi(x):
                return np.mean([_phi(x[:n_e], j) for j in range(n_scr)]) - min_expected_value
            constraints.append({"type": "ineq", "fun": _mean_phi})

        lb = np.concatenate([np.zeros(n_e),       [-np.inf],  np.zeros(n_scr)])
        ub = np.concatenate([np.ones(n_e),         [np.inf], np.full(n_scr, np.inf)])
        bounds = list(zip(lb, ub))

        opts = {"ftol": 1e-8, "maxiter": 5000, "disp": self._verbose >= 2}
        if solver_options:
            opts.update({k: v for k, v in solver_options.items()
                         if k in ("ftol", "maxiter", "disp")})

        res = _sp_min(obj, x0_aug, jac=grad_obj,
                      method="SLSQP", bounds=bounds,
                      constraints=constraints, options=opts)

        if not res.success and self._verbose >= 1:
            print(f"[WARNING] CVaR SLSQP: {res.message}")

        e_opt = res.x[:n_e]
        if self._specified_n_spt:
            self.efforts = e_opt.reshape((self.n_c, self.n_spt_comb))
        else:
            self.efforts = e_opt.reshape((self.n_c, self.n_spt))
        self._efforts_transformed = False

        # store CVaR stats for plotting
        V_opt    = res.x[n_e]
        self._cvar_V   = float(V_opt)
        self._cvar_phi = np.array([_phi(e_opt, j) for j in range(n_scr)])

        return -float(res.fun)



    # kept for internal compatibility — now delegates to _solve_pyomo
    def _solve_ipopt(self, criterion, e0, fix_effort, opt_options, **kwargs):
        """Delegate to unified Pyomo solver (kept for internal compatibility)."""
        return self._solve_pyomo(criterion, e0, fix_effort, opt_options, **kwargs)

    def find_optimal_operating_point(self, init_guess, solver="ipopt",
                                      solver_options=None, n_starts=1):
        """
        Stage 1 of V-optimal MBDoE: find the process operating condition(s)
        dw at which the model needs to be most accurate.

        Solves a nonlinear constrained optimisation over the ti_controls and
        tv_controls space via Pyomo.  The objective and constraints are
        user-defined via ``process_objective`` and ``process_constraints``.
        The result is stored in ``dw_tic`` and ``dw_tvc`` and fixed for the
        remainder of the workflow — Stage 2 (design_v_optimal) will use these
        to build the W matrix and target the FIM inversion accordingly.

        This function must be called before ``design_v_optimal()``.

        Parameters
        ----------
        init_guess : array-like, shape (n_x,) or (r_w, n_x)
            Initial guess(es) for [tic | tvc].  If 2-D, each row is solved
            independently and all solutions are stored.

        solver : str
            Pyomo solver name (default ``"ipopt"``).  Any solver registered
            with ``pyo.SolverFactory`` may be used.

        solver_options : dict, optional
            Options forwarded to the solver.  For IPOPT use keys such as
            ``"tol"``, ``"max_iter"``, ``"linear_solver"`` (e.g. ``"ma57"``).

        n_starts : int
            Number of random restarts per operating point (default 1).

        Returns
        -------
        dw_tic : np.ndarray, shape (r_w, n_tic)
        dw_tvc : np.ndarray, shape (r_w, n_tvc)

        Examples
        --------
        >>> designer.find_optimal_operating_point(
        ...     init_guess    = np.array([[T0_guess, Tj_guess, cat_guess]]),
        ...     solver        = "ipopt",
        ...     solver_options = {"tol": 1e-8, "linear_solver": "ma57"},
        ... )
        """
        # --- guards ---
        if self._status != 'ready':
            raise SyntaxError(
                "Designer must be initialized before calling "
                "find_optimal_operating_point(). Call designer.initialize() first."
            )
        if self.process_objective is None:
            raise SyntaxError(
                "process_objective must be set before calling "
                "find_optimal_operating_point()."
            )

        n_tic = self.n_tic if self._invariant_controls else 0
        n_tvc = self.n_tvc if self._dynamic_controls   else 0
        n_x   = n_tic + n_tvc

        if n_x == 0:
            raise SyntaxError(
                "No decision variables found. Ensure ti_controls_candidates and/or "
                "tv_controls_candidates are set and designer is initialized."
            )

        # --- build bound arrays ---
        bounds_tic = self.dw_bounds_tic if self.dw_bounds_tic is not None \
            else [(-np.inf, np.inf)] * n_tic
        bounds_tvc = self.dw_bounds_tvc if self.dw_bounds_tvc is not None \
            else [(-np.inf, np.inf)] * n_tvc
        all_bounds = list(bounds_tic) + list(bounds_tvc)
        lb_arr = np.array([b[0] for b in all_bounds], dtype=float)
        ub_arr = np.array([b[1] for b in all_bounds], dtype=float)

        # --- normalise init_guess to 2-D ---
        init_guess = np.atleast_2d(init_guess)   # shape (r_w, n_x)
        r_w = init_guess.shape[0]

        if init_guess.shape[1] != n_x:
            raise SyntaxError(
                f"init_guess has {init_guess.shape[1]} columns but "
                f"n_tic + n_tvc = {n_x}. Each row must be [tic | tvc]."
            )

        # store solver choice
        old_solver      = self._solver
        self._solver    = solver

        results_tic = []
        results_tvc = []
        results_obj = []

        try:
          for w in range(r_w):
            best_x   = None
            best_obj = np.inf

            for start in range(n_starts):
                if start == 0:
                    x0 = init_guess[w].copy()
                else:
                    lo = np.where(np.isfinite(lb_arr), lb_arr, -1e6)
                    hi = np.where(np.isfinite(ub_arr), ub_arr,  1e6)
                    x0 = np.random.uniform(lo, hi)

                if self._verbose >= 1:
                    tag = f"point {w+1}/{r_w}, start {start+1}/{n_starts}"
                    print(f"[find_optimal_operating_point] Solving {tag} ...")

                try:
                    x_opt, obj_val = self._solve_pyomo_operating_point(
                        x0, lb_arr, ub_arr, solver_options
                    )
                except Exception as exc:
                    if self._verbose >= 1:
                        print(f"  Warning: solver failed ({exc}), skipping this start.")
                    continue

                cmp = obj_val if self.dw_sense == "minimize" else -obj_val
                if cmp < best_obj:
                    best_obj = cmp
                    best_x   = x_opt

                if self._verbose >= 1:
                    print(f"  Objective ({self.dw_sense}): {obj_val:.6g}")

            if best_x is None:
                raise RuntimeError(
                    f"All {n_starts} start(s) failed for operating point "
                    f"{w+1}/{r_w} (solver='{solver}'). Check bounds, initial guess, and constraints."
                )

            results_tic.append(best_x[:n_tic])
            results_tvc.append(best_x[n_tic:])
            results_obj.append(
                -best_obj if self.dw_sense == "maximize" else best_obj
            )

            if self._verbose >= 1:
                print(f"  dw_tic[{w}] = {best_x[:n_tic]}")
                print(f"  dw_tvc[{w}] = {best_x[n_tic:]}")

          self.dw_tic       = np.array(results_tic)   # (r_w, n_tic)  — also sets _dw_fixed
          self.dw_tvc       = np.array(results_tvc)   # (r_w, n_tvc)
          self._dw_obj_vals = np.array(results_obj)   # (r_w,) objective at each point

        finally:
            self._solver = old_solver

        if self._verbose >= 1:
            print(f"[find_optimal_operating_point] Done. "
                  f"{r_w} operating point(s) fixed.")

        return self.dw_tic, self.dw_tvc

    def _solve_cvar_ipopt(self, criterion, beta, e0, min_expected_value,
                          solver_options, **kwargs):
        """Delegate to unified Pyomo CVaR solver (kept for internal compatibility)."""
        return self._solve_pyomo_cvar(
            criterion, beta, e0, min_expected_value, solver_options, **kwargs
        )

    def _formulate_cvar_problem(self, criterion, beta, p_cons, min_expected_value=None):
        """Legacy cvxpy formulation — no longer used. CVaR is handled by _solve_pyomo_cvar."""
        raise NotImplementedError(
            "_formulate_cvar_problem is a legacy cvxpy method. "
            "CVaR problems are now solved via _solve_pyomo_cvar."
        )

    def solve_cvar_problem_alt(self, criterion, beta, n_spt=None, n_exp=None,
                           optimize_sampling_times=False, solver="ipopt",
                           solver_options=None, e0=None, write=True,
                           save_sensitivities=False, trim_fim=False,
                           pseudo_bayesian_type=None, regularize_fim=False,
                           reso=5, plot=False, n_bins=20, tol=1e-4, **kwargs):
        """
        Alternative formulation of the bi-objective average-CVaR design problem
        using Pyomo (maximize mean subject to CVaR constraint).
        """
        self._current_criterion = criterion.__name__

        if "cvar" not in self._current_criterion:
            raise SyntaxError(
                "Please pass in a valid cvar criterion e.g., cvar_d_opt_criterion."
            )

        self.n_cvar_scr = (1 - beta) * self.n_scr
        if self.n_cvar_scr < 1:
            print(
                "[WARNING]: "
                "given n_scr * beta given is smaller than 1, this yields a maximin "
                "design. Please provide a larger number of n_scr if a CVaR design "
                "was desired."
            )
            self.n_cvar_scr = np.ceil(self.n_cvar_scr).astype(int)
        else:
            self.n_cvar_scr = np.floor(self.n_cvar_scr).astype(int)

        if reso < 3:
            print(
                f"The input reso is given as {reso}; the minimum value of reso is 3. "
                "Continuing with reso = 3."
            )
            reso = 3

        self.cvar_optimal_candidates = []
        self.cvar_solution_times = []
        self._biobjective_values = np.empty((reso, 2))
        if plot:
            figs = []

            def add_fig(cdf, pdf):
                figs.append([cdf, pdf])

        self._alt_cvar = True

        def _common_kwargs():
            return dict(
                n_spt=n_spt,
                n_exp=n_exp,
                optimize_sampling_times=optimize_sampling_times,
                solver=solver,
                solver_options=solver_options,
                e0=e0,
                write=False,
                trim_fim=trim_fim,
                pseudo_bayesian_type=pseudo_bayesian_type,
                regularize_fim=regularize_fim,
                **kwargs,
            )

        def _phi_values():
            if self.pb_atomic_fims is None or self.efforts is None:
                return np.zeros(self.n_scr)
            e_flat = np.asarray(self.efforts).flatten()
            phis = []
            for j in range(self.n_scr):
                atoms_j = self.pb_atomic_fims[j]
                M_j = np.einsum('i,imn->mn', e_flat, atoms_j)
                cv = criterion(M_j)
                if isinstance(cv, tuple): cv = cv[0]
                phis.append(-float(cv))
            return np.array(phis)

        """ Iteration 1: Maximal Mean Design """
        if self._verbose >= 1:
            print(f" CVaR Problem (Alt) ".center(100, "*"))
            print(f"[Iteration 1/{reso}]".center(100, "="))
        self.design_experiment(criterion, min_expected_value=-1000, **_common_kwargs())
        self.get_optimal_candidates()
        if self._verbose >= 1:
            self.print_optimal_candidates(tol=tol, write=False)
        iter_1_efforts = np.copy(self.efforts)
        mean_ub = self._criterion_value
        iter_1_phi = _phi_values()
        self._cvar_phi = iter_1_phi
        self._cvar_V   = float(np.percentile(iter_1_phi, (1 - beta) * 100))
        # CVaR at iter-1 solution
        self.design_experiment(criterion, beta=beta,
                               fix_effort=iter_1_efforts / np.sum(iter_1_efforts),
                               save_sensitivities=False, **_common_kwargs())
        cvar_lb = self._criterion_value

        self.cvar_optimal_candidates.append(self.optimal_candidates)
        self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
        self._biobjective_values[0, :] = np.array([mean_ub, cvar_lb])
        if self._verbose >= 1:
            print(f"CVaR LB: {cvar_lb}  Mean UB: {mean_ub}")
            print(f"[Iteration 1/{reso} Completed]".center(100, "="))
        if plot:
            add_fig(
                self.plot_criterion_cdf(write=False, iteration=1),
                self.plot_criterion_pdf(write=False, iteration=1),
            )

        """ Iteration 2: Maximal CVaR Design """
        if self._verbose >= 1:
            print(f"[Iteration 2/{reso}]".center(100, "="))
        self.design_experiment(criterion, beta=beta,
                               save_sensitivities=False, **_common_kwargs())
        self.get_optimal_candidates()
        iter_2_efforts = np.copy(self.efforts)
        iter_2_phi = _phi_values()
        iter2_V    = float(np.percentile(iter_2_phi, (1 - beta) * 100))
        cvar_ub    = self._criterion_value
        if self._verbose >= 1:
            self.print_optimal_candidates(tol=tol, write=False)
        self.design_experiment(criterion, beta=0.00,
                               fix_effort=iter_2_efforts / np.sum(iter_2_efforts),
                               save_sensitivities=False, **_common_kwargs())
        mean_lb = self._criterion_value

        self.cvar_optimal_candidates.append(self.optimal_candidates)
        self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
        self._biobjective_values[1, :] = np.array([mean_lb, cvar_ub])
        if self._verbose >= 1:
            print(f"CVaR UB: {cvar_ub}  Mean LB: {mean_lb}")
            print(f"[Iteration 2/{reso} Completed]".center(100, "="))
        if plot:
            self._cvar_phi = iter_2_phi
            self._cvar_V   = iter2_V
            add_fig(
                self.plot_criterion_cdf(write=False, iteration=2),
                self.plot_criterion_pdf(write=False, iteration=2),
            )

        """ Iterations 3+: Intermediate Points """
        cvar_values = np.linspace(cvar_lb, cvar_ub, reso)[1:-1]

        for i, cvar_min in enumerate(cvar_values):
            if self._verbose >= 1:
                print(f"[Iteration {i + 3}/{reso}]".center(100, "="))
            self.design_experiment(
                criterion, beta=beta,
                min_expected_value=cvar_min,
                save_sensitivities=False,
                **_common_kwargs(),
            )
            self.get_optimal_candidates()
            iter_phi = _phi_values()
            self._cvar_phi = iter_phi
            self._cvar_V   = float(np.percentile(iter_phi, (1 - beta) * 100))
            self.cvar_optimal_candidates.append(self.optimal_candidates)
            self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
            self._biobjective_values[i + 2, :] = np.array([self._criterion_value, cvar_min])
            if plot:
                add_fig(
                    self.plot_criterion_cdf(write=False, iteration=i+3),
                    self.plot_criterion_pdf(write=False, iteration=i+3),
                )
            if self._verbose >= 1:
                self.print_optimal_candidates(tol=tol, write=False)
                print(f"Mean: {self._criterion_value:.6f}  CVaR constraint: {cvar_min:.6f}")
                print(f"[Iteration {i + 3}/{reso} Completed]".center(100, "="))

        if plot:
            xlims = []
            for i, fig in enumerate(figs):
                cdf, pdf = fig[0], fig[1]
                xlims.append(cdf.axes[0].get_xlim())
            xlims = np.asarray(xlims)
            for i, fig in enumerate(figs):
                cdf, pdf = fig[0], fig[1]
                cdf.axes[0].set_xlim(xlims[:, 0].min(), xlims[:, 1].max())
                pdf.axes[0].set_xlim(xlims[:, 0].min(), xlims[:, 1].max())

    def _formulate_cvar_problem_alt(self, criterion, beta, p_cons, min_cvar_value=None):
        """Legacy cvxpy formulation — no longer used. CVaR is handled by _solve_pyomo_cvar."""
        raise NotImplementedError(
            "_formulate_cvar_problem_alt is a legacy cvxpy method. "
            "CVaR problems are now solved via _solve_pyomo_cvar."
        )

    def design_experiment(self, criterion, n_spt=None, n_exp=None,
                          optimize_sampling_times=False, solver="ipopt",
                          solver_options=None, e0=None, write=False,
                          save_sensitivities=False, trim_fim=False,
                          pseudo_bayesian_type=None, regularize_fim=False, beta=0.90,
                          min_expected_value=None, fix_effort=None, save_atomics=False,
                          min_effort=None, allow_singular_fim=False, output_weight=0.5,
                          **kwargs):
        """Solve for the optimal continuous experimental design.

        The central method of the package. Allocates a unit budget of
        experimental effort across the candidate grid so as to optimise
        ``criterion``, returning a CONTINUOUS design — a weight in [0, 1] per
        candidate, summing to one. Use :meth:`apportion` to turn that into whole
        experimental runs.

        Which solver actually runs depends on the criterion. D, A, E, V, Ds and
        pseudo-Bayesian type 0 are built as symbolic Pyomo programs and passed to
        ``solver``. Everything else — the generalized/individual family, CVaR,
        and pseudo-Bayesian type 1 — is handed to scipy's SLSQP as a black box,
        in which case ``solver`` is IGNORED and ``solver_options`` is filtered to
        ``ftol``/``maxiter``/``disp``. See the class docstring, "Which solver
        actually runs".

        Args:
            criterion (callable): A bound criterion method such as
                ``designer.d_opt_criterion``, or any callable taking the effort
                vector and returning a scalar to MINIMISE.
            n_spt (int, optional): Restrict each experiment to exactly this many
                samples. Requires ``optimize_sampling_times=True``. Clear
                :attr:`atomic_fims` first if changing it between runs.
            n_exp (int, optional): Solve for a discrete design of this many
                experiments instead of a continuous one.
            optimize_sampling_times (bool): Also choose WHEN to sample, rather
                than measuring every listed time.
            solver (str): NLP solver for the native Pyomo path — any
                AMPL-compatible solver. Ignored on the SLSQP path.
            solver_options (dict, optional): Passed to the solver. On the SLSQP
                path all but ``ftol``/``maxiter``/``disp`` are silently dropped.
            e0 (numpy.ndarray, optional): Initial effort vector.
            write (bool): Write the result to the result directory.
            save_sensitivities (bool): Cache sensitivities to disk.
            trim_fim (bool): Drop uninformative rows/columns from the FIM.
            pseudo_bayesian_type (int or str, optional): 0 or ``'avg_inf'`` to
                average the information matrices, 1 or ``'avg_crit'`` to average
                the criterion. Required when :attr:`model_parameters` is a
                scenario array.
            regularize_fim (bool): Add ``_eps * I`` to the FIM. NOTE this
                OVERWRITES ``self._regularize_fim`` from the keyword, so setting
                that attribute directly has no effect.
            beta (float): CVaR confidence level.
            min_expected_value (float, optional): CVaR constraint.
            fix_effort (numpy.ndarray, optional): Fix part of the effort vector.
            save_atomics (bool): Cache the atomic FIMs to disk.
            min_effort (float, optional): Enforce a minimum non-zero effort,
                turning the problem into an MINLP and giving a sparser design.
            allow_singular_fim (bool): Proceed even when the FIM is
                structurally singular. By default the solve is refused with a
                diagnosis naming the responsible parameters, because a criterion
                value obtained from a rank-deficient FIM rests on the Cholesky
                floor rather than on information the data contains.
            output_weight (float): ONLY used by ``b_opt_criterion``
                (Bracketing-optimal). Weight in [0, 1] on the output-space
                coverage objective; the input-space bracketing objective gets
                weight ``1 - output_weight``. See Chen, Paulavicius & Adjiman
                (2018), AIChE J. 64:3944-3957, Eq. 24. Ignored by every other
                criterion.
            **kwargs: Forwarded to the solver interface.

        Returns:
            The optimisation result object. The design itself is on the
            designer: :attr:`efforts`, and ``_criterion_value``.

        Raises:
            ValueError: If the FIM is structurally singular and
                ``allow_singular_fim`` is False.

        Note:
            Ds-optimality is exempt from the singularity gate — it is designed
            for that case and checks its own Schur complement instead.
        """
        # storing user choices
        self._regularize_fim = regularize_fim
        # Clear the latched det/pseudo-det decision for dg/di/vdi so each design
        # run re-decides from scratch (the choice must be fixed WITHIN a run, but
        # not carried across runs whose sensitivities may differ).
        self.reset_pvar_logdet_mode()
        self._allow_singular_fim = bool(allow_singular_fim)
        self._solver         = solver
        self._fd_jac         = True          # always True; gradient strategy is internal
        self._unconstrained_form = False     # no longer a user concern
        self._opt_sampling_times = optimize_sampling_times
        self._save_sensitivities = save_sensitivities
        self._current_criterion  = criterion.__name__
        self._trim_fim           = trim_fim
        self._save_atomics       = save_atomics
        self._min_effort         = min_effort  # sparsity threshold (MINLP when set)

        """ checking if CVaR problem """
        if "cvar" in self._current_criterion:
            self._cvar_problem = True
            self.beta = beta
        else:
            self._cvar_problem = False

        """ resetting optimal candidates """
        self.optimal_candidates = None
        self._b_opt_apportion_redundant = False   # reset each call; set True inside
                                                    # _solve_pyomo_b_opt only on success

        """ setting verbal behaviour """
        if self._verbose >= 2:
            opt_verbose = True
        else:
            opt_verbose = False

        """ handling problems with defined n_spt """
        if n_spt is not None:
            if not self._dynamic_system:
                raise SyntaxError(
                    f"n_spt specified for a non-dynamic system."
                )
            if not self._opt_sampling_times:
                print(
                    f"[Warning]: n_spt specified, but "
                    f"optimize_sampling_times = False. "
                    f"Overriding, and setting optimize_sampling_times = True."
                )
            self._opt_sampling_times = True
            self._n_spt_spec = n_spt
            if not isinstance(n_spt, int):
                raise SyntaxError(
                    f"Supplied n_spt is a {type(n_exp)}, "
                    f"but \"n_spt\" must be an integer."
                )
            self._specified_n_spt = True
            self.spt_candidates_combs = []
            for spt in self.sampling_times_candidates:
                spt_idx = np.arange(0, len(spt))
                self.spt_candidates_combs.append(
                    list(itertools.combinations(spt_idx, n_spt))
                )
            self.spt_candidates_combs = np.asarray(
                self.spt_candidates_combs
            )
            _, self.n_spt_comb, _ = self.spt_candidates_combs.shape
        else:
            self._specified_n_spt = False
            self._n_spt_spec = 1

        """ determining if discrete design problem """
        if n_exp is not None:
            self._discrete_design = True
            if not isinstance(n_exp, int):
                raise SyntaxError(
                    f"Supplied n_exp is a {type(n_exp)}, "
                    f"but \"n_exp\" must be an integer."
                )
        else:
            self._discrete_design = False

        """ re-check local vs pseudo-Bayesian based on current model_parameters
            (user may have set a 2D scenarios array after initialize() was called
            with a 1D array, so _pseudo_bayesian, n_scr, and n_mp must be refreshed) """
        self._check_stats_framework()
        if self._pseudo_bayesian:
            self.n_scr, self.n_mp = self.model_parameters.shape
            self._current_scr_mp = self.model_parameters[0]
        else:
            self.n_mp = self.model_parameters.shape[0]
            self._current_scr_mp = self.model_parameters

        """ setting default semi-bayes behaviour """
        if self._pseudo_bayesian:
            if pseudo_bayesian_type is None:
                self._pseudo_bayesian_type = 0
            else:
                valid_types = [
                    0, 1,
                    "avg_inf", "avg_crit",
                    "average_information", "average_criterion"
                ]
                if pseudo_bayesian_type in valid_types:
                    self._pseudo_bayesian_type = pseudo_bayesian_type
                else:
                    raise SyntaxError(
                        "Unrecognized pseudo_bayesian criterion type. Valid types: '0' "
                        "for average information, '1' for average criterion."
                    )

        """ force fd_jac for large problems """
        if self._large_memory_requirement and not self._fd_jac:
            print("Warning: analytic Jacobian is specified on a large problem."
                  "Overwriting and continuing with finite differences.")
            self._fd_jac = True

        """ main codes """
        if self._verbose >= 1:
            print(" Computing Optimal Experiment Design ".center(100, "#"))
        if self._verbose >= 2:
            print(f"{'Started on':<40}: {datetime.now()}")
            print(f"{'Criterion':<40}: {self._current_criterion}")
            print(f"{'Pseudo-bayesian':<40}: {self._pseudo_bayesian}")
            if self._pseudo_bayesian:
                print(f"{'Pseudo-bayesian Criterion Type':<40}: {self._pseudo_bayesian_type}")
            print(f"{'Dynamic':<40}: {self._dynamic_system}")
            print(f"{'Time-invariant Controls':<40}: {self._invariant_controls}")
            print(f"{'Time-varying Controls':<40}: {self._dynamic_controls}")
            print(f"{'Number of Candidates':<40}: {self.n_c}")
            if self._dynamic_system:
                print(f"{'Number of Sampling Time Choices':<40}: {self.n_spt}")
                print(f"{'Sampling Times Optimized':<40}: {self._opt_sampling_times}")
            if self._pseudo_bayesian:
                print(f"{'Number of Scenarios':<40}: {self.n_scr}")
            print(f"{'Solver':<40}: {self._solver}")
            if min_effort is not None:
                print(f"{'Min. effort (sparsity)':<40}: {min_effort}")
            if self._prior_fim is not None:
                print(f"{'Prior FIM':<40}: registered  "
                      f"({self._prior_n_exp} prior experiment(s))")
            else:
                print(f"{'Prior FIM':<40}: none")
        """
        set initial guess for optimal experimental efforts, if none given, equal
        efforts for all candidates
        """
        if e0 is None:
            if self._specified_n_spt:
                e0 = np.ones((self.n_c, self.n_spt_comb)) / (self.n_c * self.n_spt_comb)
            else:
                e0 = np.ones((self.n_c, self.n_spt)) / (self.n_c * self.n_spt)
        else:
            msg = 'Initial guess for effort must be a 2D numpy array.'
            if not isinstance(e0, np.ndarray):
                raise SyntaxError(msg)
            elif e0.ndim != 2:
                raise SyntaxError(msg)
            elif e0.shape[0] != self.n_c:
                raise SyntaxError(
                    f"Error: inconsistent number of candidates provided;"
                    f"number of candidates in e0: {e0.shape[0]},"
                    f"number of candidates from initialization: {self.n_c}."
                )
            if self._specified_n_spt:
                if e0.shape[1] != self.n_spt_comb:
                    raise SyntaxError(
                        f"Error: second dimension of e0 must be {self.n_spt_comb} "
                        f"long, corresponding to n_spt_combs; given is {e0.shape[1]}."
                    )
            else:
                if e0.shape[1] != self.n_spt:
                    raise SyntaxError(
                        f"Error: inconsistent number of sampling times provided;"
                        f"number of sampling times in e0: {e0.shape[1]},"
                        f"number of candidates from initialization: {self.n_spt}."
                    )

        # declare and solve optimization problem
        self._sensitivity_analysis_time = 0
        start = time()

        # single unified Pyomo dispatch
        # n_exp/output_weight are forwarded via kwargs -- they are read only
        # by the b_opt_criterion early-dispatch branch inside _solve_pyomo;
        # every other criterion ignores unknown kwargs exactly as before.
        if self._cvar_problem:
            opt_fun = self._solve_pyomo_cvar(
                criterion, beta, e0, min_expected_value, solver_options, **kwargs
            )
        else:
            opt_fun = self._solve_pyomo(
                criterion, e0, fix_effort, solver_options,
                n_exp=n_exp, output_weight=output_weight, **kwargs
            )

        finish = time()

        """ report status and performance """
        self._optimization_time = finish - start - self._sensitivity_analysis_time
        if self._verbose >= 2:
            print(
                f"[Optimization Complete in {self._optimization_time:.2f} s]".center(100, "-")
            )
        if self._verbose >= 1:
            print(
                f"Complete: \n"
                f" ~ sensitivity analysis took {self._sensitivity_analysis_time:.2f} "
                f"CPU seconds.\n"
                f" ~ optimization with {self._solver} took "
                f"{self._optimization_time:.2f} CPU seconds."
            )
            print("".center(100, "#"))

        """ storing and writing result """
        self._criterion_value = opt_fun
        self.oed_result = {
            "solution_time": finish - start,
            "optimization_time": self._optimization_time,
            "sensitivity_analysis_time": self._sensitivity_analysis_time,
            "optimality_criterion": criterion.__name__,
            "ti_controls_candidates": self.ti_controls_candidates,
            "tv_controls_candidates": self.tv_controls_candidates,
            "model_parameters": self.model_parameters,
            "sampling_times_candidates": self.sampling_times_candidates,
            "optimal_efforts": self.efforts,
            "criterion_value": self._criterion_value,
            "solver": self._solver,
            "pseudo_bayesian": self._pseudo_bayesian,
            "pseudo_bayesian_type": self._pseudo_bayesian_type,
            "optimize_sampling_times": self._opt_sampling_times,
            "regularized": self._regularize_fim,
            "n_spt_spec": self._n_spt_spec,
            "prior_fim": self._prior_fim,
            "prior_fim_mp": self._prior_fim_mp,
            "prior_n_exp": self._prior_n_exp,
        }
        if write:
            self.write_oed_result()

        return self.oed_result

    def plot_criterion_cdf(self, write=False, iteration=None, dpi=360, figsize=(4.5, 3.5), annotate=False, minor_ticks=False, legend=False, grid=False):
        """Plot the cumulative distribution of the criterion across scenarios.

        Pseudo-Bayesian only. Shows how the design performs over the parameter
        prior rather than at a single nominal — what a CVaR design targets.

        Args:
            write (bool): Save to the result directory.
            iteration (int, optional): Label for a CVaR iteration.
            dpi (int): Resolution when writing.
            figsize (tuple): Figure size.
            annotate (bool): Annotate the quantiles.
        """
        if not self._pseudo_bayesian or not self._cvar_problem:
            raise SyntaxError(
                "Plotting cumulative distribution function only valid for pseudo-"
                "bayesian and cvar problems."
            )

        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(111)
        if self._cvar_problem:
            phi_vals = getattr(self, '_cvar_phi', np.zeros(self.n_scr))
            V_val    = getattr(self, '_cvar_V',   float('nan'))
            x = np.sort(phi_vals)
            mean = phi_vals.mean()
            x = np.insert(x, 0, x[0])
            y = np.linspace(0, 1, x.size)
            axes.plot(x, y, "o--", alpha=0.3, c="#1f77b4")
            axes.plot(x, y, drawstyle="steps-post", c="#1f77b4")
            axes.axvline(
                x=V_val,
                ymin=0,
                ymax=1,
                c="tab:red",
                label=f"VaR {self.beta}",
            )
            axes.axvline(
                x=getattr(self, "_criterion_value", float("nan")),
                ymin=0,
                ymax=1,
                c="tab:green",
                label=f"CVaR {self.beta}",
            )
            axes.axvline(
                x=mean,
                ymin=0,
                ymax=1,
                c="tab:blue",
                label=f"Mean",
            )
            axes.set_xlabel(f"{self._current_criterion}")
            axes.set_ylim(0, 1)
            axes.set_ylabel("Cumulative Probability")

            if legend:
                axes.legend()

            if minor_ticks:
                axes.xaxis.set_minor_locator(AutoMinorLocator(5))
                axes.yaxis.set_minor_locator(AutoMinorLocator(5))

            if grid:
                axes.grid(visible=False, which="both")

            if annotate:
                axes.axhline(
                    y=1-self.beta,
                    ls="--",
                    c="tab:red",
                )
                axes.annotate(
                    rf"$(1 - \beta) = {1 - self.beta:.2f}$",
                    xy=(0.20, 1 - self.beta),
                    xytext=(0.50, 1 - self.beta + 0.25),
                    xycoords="axes fraction",
                    arrowprops={
                        "width": 5,
                        "shrink": 0.05,
                        "facecolor": "tab:red",
                        "edgecolor": "k",
                    },
                )
                axes.annotate(
                    "VaR",
                    xy=(V_val, 0.80),
                    xytext=(V_val + 0.2 * np.abs(V_val), 0.80),
                    arrowprops={
                        "width": 5,
                        "shrink": 0.05,
                        "facecolor": "tab:red",
                        "edgecolor": "k",
                    },
                )
                cvar = getattr(self, "_criterion_value", float("nan"))
                axes.annotate(
                    "CVaR",
                    xy=(cvar, 0.50),
                    xytext=(cvar + 0.2 * np.abs(cvar), 0.50),
                    arrowprops={
                        "width": 5,
                        "shrink": 0.05,
                        "facecolor": "tab:green",
                        "edgecolor": "k",
                    },
                )
                axes.annotate(
                    "Mean",
                    xy=(mean, 0.10),
                    xytext=(mean + 0.2 * np.abs(mean), 0.10),
                    arrowprops={
                        "width": 5,
                        "shrink": 0.05,
                        "facecolor": "tab:blue",
                        "edgecolor": "k",
                    },
                )
            _safe_tight_layout(fig)
        else:
            raise NotImplementedError(
                "Plotting cumulative distribution function not implemented for pseudo-"
                "bayesian problems."
            )

        if write:
            fn = f"cdf_{self.beta*100}_beta_{self.n_scr}_scr"
            fp = self._generate_result_path(fn, "png", iteration=iteration)
            fig.savefig(fname=fp, dpi=dpi)

        return fig

    def plot_criterion_pdf(self, n_bins=20, write=False, iteration=None, dpi=360):
        """Plot the criterion distribution across scenarios as a histogram.

        Pseudo-Bayesian counterpart of :meth:`plot_criterion_cdf`.

        Args:
            n_bins (int): Histogram bins.
            write (bool): Save to the result directory.
            iteration (int, optional): Label for a CVaR iteration.
            dpi (int): Resolution when writing.

        Raises:
            SyntaxError: For non-pseudo-Bayesian designs.
        """
        if not self._pseudo_bayesian or not self._cvar_problem:
            raise SyntaxError(
                "Plotting probability density function only valid for pseudo-"
                "bayesian and cvar problems."
            )

        fig = plt.figure()
        axes = fig.add_subplot(111)
        if self._cvar_problem:
            x     = getattr(self, '_cvar_phi', np.zeros(self.n_scr))
            V_val = getattr(self, '_cvar_V',   float('nan'))
            axes.hist(x, bins=n_bins)
            axes.axvline(V_val, 0, 1, c="tab:red",   label=f"VaR {self.beta}")
            axes.axvline(self._criterion_value, 0, 1, c="tab:green", label=f"CVaR {self.beta}")
            axes.set_xlabel(f"{self._current_criterion}")
            axes.set_ylabel("Frequency")
            axes.legend()
            _safe_tight_layout(fig)
        else:
            raise NotImplementedError(
                "Plotting probability density function not implemented for pseudo-"
                "bayesian problems."
            )

        if write:
            fn = f"pdf_{self.beta*100}_beta_{self.n_scr}_scr"
            fp = self._generate_result_path(fn, "png", iteration=iteration)
            fig.savefig(fname=fp, dpi=dpi)

        return fig

    def compute_criterion_value(self, criterion, decimal_places=3):
        """Evaluate a criterion at the design currently held in :attr:`efforts`.

        Useful for scoring a design under a criterion other than the one it was
        optimised for, or for scoring a hand-specified effort vector.

        Args:
            criterion (callable): A bound criterion method.
            decimal_places (int): Rounding for the printed value.

        Returns:
            float: The criterion value. Note criteria are MINIMISED, so lower is
            better and ``+inf`` marks an unusable information matrix.
        """
        crit_val = criterion(self.efforts)
        if isinstance(crit_val, tuple):
            crit_val = crit_val[0]
        crit_val = float(np.squeeze(crit_val))
        if self._verbose >= 1:
            print(f"{criterion.__name__}: {crit_val:.{decimal_places}E}")
        return crit_val

    def apportion(self, n_exp, method="adams", trimmed=True, compute_actual_efficiency=True):
        """Round a continuous design into a whole number of experimental runs.

        A continuous design gives fractional effort per candidate, which cannot
        be performed. This converts it into integer run counts totalling
        ``n_exp``, and reports how much information the rounding costs.

        Two rules are used depending on the budget. When ``n_exp`` exceeds the
        number of supports every support gets at least one run and the remainder
        is shared proportionally (Adams divisor method); when it does not, the
        ``n_exp`` largest-effort supports are selected and run once each.

        Args:
            n_exp (int): Number of experimental runs to allocate.
            method (str): Divisor method for the proportional case.
            trimmed (bool): Drop zero-effort candidates before apportioning.
            compute_actual_efficiency (bool): Also evaluate the criterion at the
                rounded design and report it as a percentage of the continuous
                one.

        Note:
            Read the reported efficiency, not just the run counts. A rounded
            design worth a fraction of a percent of the continuous one usually
            means the continuous optimum rests on a near-singular direction that
            rounding destroys — check :meth:`diagnose_fim_structure` if so.
        """
        if getattr(self, "_b_opt_apportion_redundant", False):
            print(
                "[Warning] The current design was produced by "
                "b_opt_criterion, which already solves for an EXACT, "
                "equal-weighted subset of exactly n_exp candidates -- "
                "there is no continuous design left to round. Proceeding, "
                "but apportion() is redundant here; the design in "
                "designer.efforts is already the final discrete design."
            )
        self.n_exp = n_exp

        _original_save_atomics = np.copy(self._save_atomics)
        self._save_atomics = False

        if self._dynamic_system and self._specified_n_spt:
            # Adams apportionment over the flat (candidate × variant) effort vector.
            # Each optimal candidate has n_variants = len(opt_cand[4]) sampling time
            # combinations; we flatten all (candidate, variant) pairs into a single
            # effort vector, run Adams rounding on it, then scatter the integer counts
            # back per candidate for reporting.
            self.get_optimal_candidates()

            # Build flat effort vector and index map
            flat_efforts = []
            flat_index   = []   # (candidate_pos, variant_pos) in optimal_candidates
            for i, opt_cand in enumerate(self.optimal_candidates):
                variant_efforts = opt_cand[4]   # list of arrays, one per variant
                for j, e in enumerate(variant_efforts):
                    # e may be a scalar or a small array — use nansum to collapse
                    val = float(np.nansum(e))
                    flat_efforts.append(val)
                    flat_index.append((i, j))
            flat_efforts = np.array(flat_efforts)

            # Normalise (should already sum to ~1 but guard against floating point)
            total = flat_efforts.sum()
            if total > 0:
                flat_efforts = flat_efforts / total

            # Choose the rounding rule by comparing the budget with the number
            # of supports.
            #
            # The condition here used to be  len(flat_efforts) < n_exp,  which is
            # backwards and under-allocated the budget badly. Worked example:
            # 2 optimal candidates x 2 sampling schedules = 4 supports with
            # apportion(12) took the greatest-effort branch, and that routine
            # assigns AT MOST ONE run per support (it zeroes each chosen effort
            # and writes  = 1  rather than  += 1). The result was 4 experiments
            # allocated out of 12 requested, reported as "Run 2/12" per
            # candidate, with 8 runs silently dropped.
            #
            #   n_exp <= n_supports : cannot run every support even once, so pick
            #                         the n_exp largest and run each once
            #                         -> greatest-effort selection
            #   n_exp >  n_supports : every support gets at least one run and the
            #                         remainder must be shared proportionally
            #                         -> Adams (divisor-method) apportionment
            if n_exp <= len(flat_efforts):
                app_flat = self._greatest_effort_apportionment(flat_efforts, n_exp)
            else:
                app_flat = self._adams_apportionment(flat_efforts, n_exp)

            # Scatter back into per-candidate apportionment arrays
            self.apportionments = []
            for i, opt_cand in enumerate(self.optimal_candidates):
                n_variants = len(opt_cand[4])
                cand_app = np.zeros(n_variants)
                for j in range(n_variants):
                    flat_pos = flat_index.index((i, j))
                    cand_app[j] = app_flat[flat_pos]
                self.apportionments.append(cand_app)
            self.apportionments = np.array(self.apportionments, dtype=object)

            # Report
            if self._verbose >= 1:
                print(f" Optimal Experiment for {n_exp:d} Runs ".center(100, "#"))
                print(f"{'Obtained on':<40}: {datetime.now()}")
                print(f"{'Criterion':<40}: {self._current_criterion}")
                print(f"{'Criterion Value':<40}: {self._criterion_value}")
                print(f"{'Dynamic':<40}: {self._dynamic_system}")
                print(f"{'Number of Candidates':<40}: {self.n_c}")
                print(f"{'Number of Optimal Candidates':<40}: {self.n_opt_c}")
                print(f"{'Sampling Times Optimized':<40}: {self._opt_sampling_times}")
                print(f"{'Number of Samples Per Experiment':<40}: {self._n_spt_spec}")
                for i, (app_eff, opt_cand) in enumerate(zip(self.apportionments, self.optimal_candidates)):
                    print(f"{f'[Candidate {opt_cand[0] + 1:d}]':-^100}")
                    print(f"{f'Recommended Apportionment: Run {np.nansum(app_eff):.0f}/{n_exp:d} Experiments':^100}")
                    if self._invariant_controls:
                        print("Time-invariant Controls:")
                        print(opt_cand[1])
                    print("Sampling Schedules  (same experimental conditions, different sampling times):")
                    for comb, (spt_comb, app) in enumerate(zip(opt_cand[3], app_eff)):
                        print(f"  Schedule {comb + 1} ~ [", end='')
                        for sp_time in spt_comb:
                            print(f"{f'{sp_time:.2f}':>10}", end='')
                        print(f"]: Run {f'{app:.0f}/{np.nansum(app_eff):.0f}':>6} experiments, "
                              f"collecting {self._n_spt_spec} samples at given times")
                print(f"".center(100, "-"))
            self._save_atomics = _original_save_atomics
            return
        self.get_optimal_candidates()

        """ Initialize opt_eff shape """
        if self._opt_sampling_times:
            self.opt_eff = np.empty((len(self.optimal_candidates), self.max_n_opt_spt))
        else:
            self.opt_eff = np.empty((len(self.optimal_candidates)))
        self.opt_eff[:] = np.nan

        """ Get the optimal efforts from optimal_candidates """
        for i, opt_cand in enumerate(self.optimal_candidates):
            if self._opt_sampling_times:
                for j, spt in enumerate(opt_cand[4]):
                    if self._specified_n_spt:
                        self.opt_eff[i, j] = np.nansum(spt)
                    else:
                        self.opt_eff[i, j] = spt
            else:
                self.opt_eff[i] = np.nansum(opt_cand[4])

        """ do the apportionment """
        if method == "adams":
            if n_exp < self.n_factor_sups:
                self.apportionments = self._greatest_effort_apportionment(self.opt_eff, n_exp)
            else:
                self.apportionments = self._adams_apportionment(self.opt_eff, n_exp)
        else:
            raise NotImplementedError(
                "At the moment, the only method implemented is 'adams', please use it. "
                "More apportionment methods will be implemented, but there is proof "
                "that Adam's method is the most efficient amongst other popular "
                "methods used in electoral college apportionments."
            )

        """ Report the obtained apportionment """
        if self._verbose >= 1:
            print(f" Optimal Experiment for {n_exp:d} Runs ".center(100, "#"))
            print(f"{'Obtained on':<40}: {datetime.now()}")
            print(f"{'Criterion':<40}: {self._current_criterion}")
            print(f"{'Criterion Value':<40}: {self._criterion_value}")
            print(f"{'Pseudo-bayesian':<40}: {self._pseudo_bayesian}")
            if self._pseudo_bayesian:
                print(f"{'Pseudo-bayesian Criterion Type':<40}: {self._pseudo_bayesian_type}")
            print(f"{'CVaR Problem':<40}: {self._cvar_problem}")
            if self._cvar_problem:
                print(f"{'Beta':<40}: {self.beta}")
                print(f"{'Constrained Problem':<40}: {self._constrained_cvar}")
                if self._constrained_cvar:
                    print(f"{'Min. Mean Value':<40}: {getattr(self, '_cvar_mean_phi', float('nan')):.6f}")
            print(f"{'Dynamic':<40}: {self._dynamic_system}")
            print(f"{'Time-invariant Controls':<40}: {self._invariant_controls}")
            print(f"{'Time-varying Controls':<40}: {self._dynamic_controls}")
            print(f"{'Number of Candidates':<40}: {self.n_c}")
            print(f"{'Number of Optimal Candidates':<40}: {self.n_opt_c}")
            if self._dynamic_system:
                print(f"{'Number of Sampling Time Choices':<40}: {self.n_spt}")
                print(f"{'Sampling Times Optimized':<40}: {self._opt_sampling_times}")
                if self._opt_sampling_times:
                    print(f"{'Number of Samples Per Experiment':<40}: {self._n_spt_spec}")
            if self._pseudo_bayesian:
                print(f"{'Number of Scenarios':<40}: {self.n_scr}")

            for i, (app_eff, opt_cand) in enumerate(zip(self.apportionments, self.optimal_candidates)):
                print(f"{f'[Candidate {opt_cand[0] + 1:d}]':-^100}")
                print(
                    f"{f'Recommended Apportionment: Run {np.nansum(app_eff):.0f}/{n_exp:d} Experiments':^100}")
                if self._invariant_controls:
                    print("Time-invariant Controls:")
                    print(opt_cand[1])
                if self._dynamic_controls:
                    print("Time-varying Controls:")
                    print(opt_cand[2])
                if self._dynamic_system:
                    if self._opt_sampling_times:
                        if self._specified_n_spt:
                            print("Sampling Schedules  (same experimental conditions, different sampling times):")
                            for comb, spt_comb in enumerate(opt_cand[3]):
                                print(f"  Schedule {comb + 1} ~ [", end='')
                                for j, sp_time in enumerate(spt_comb):
                                    print(f"{f'{sp_time:.2f}':>10}", end='')
                                print("]: ", end='')
                                print(
                                    f'Run {f"{app_eff[comb]:.0f}/{np.nansum(app_eff):.0f}":>6} experiments, collecting {self._n_spt_spec} samples at given times')
                        else:
                            print("Sampling Times:")
                            for j, sp_time in enumerate(opt_cand[3]):
                                print(f"[{f'{sp_time:.2f}':>10}]: "
                                      f"Run {f'{app_eff[j]:.0f}/{np.nansum(app_eff):.0f}':>6} experiments, sampling at given time")
                    else:
                        print("Sampling Times:")
                        print(self.sampling_times_candidates[i])

            """ Computing and Reporting Rounding Efficiency """
            self.epsilon = self._eval_efficiency_bound(
                self.apportionments / n_exp,
                self.opt_eff,
            )

            """ 
            =============================================================================
            Computing actual efficiency 
            =============================================================================
            the rounding efficiency above is computed using efforts that excludes
            experimental candidates with non-zero efforts i.e., only supports
            to compute actual efficiency, non_trimmed_apportionment is required
            i.e., need candidates with zero efforts too.
            """
            # initialize the non_trimmed_apportionments
            self.non_trimmed_apportionments = np.zeros_like(self.efforts)
            for opt_c, app_c in zip(self.optimal_candidates, self.apportionments):
                opt_idx = opt_c[0]
                opt_spt = opt_c[5]
                if isinstance(app_c, float):
                    self.non_trimmed_apportionments[opt_idx, opt_spt] = app_c
                else:
                    for spt, app in zip(opt_spt, app_c):
                        self.non_trimmed_apportionments[opt_idx, spt] = app
            # normalized to non_trimmed_rounded_efforts
            non_trimmed_rounded_efforts = self.non_trimmed_apportionments / np.sum(self.non_trimmed_apportionments)
            if compute_actual_efficiency:
                _original_efforts = np.copy(self.efforts)
                try:
                    rounded_criterion_value = getattr(self, self._current_criterion)(non_trimmed_rounded_efforts).value
                except AttributeError:
                    rounded_criterion_value = getattr(self, self._current_criterion)(non_trimmed_rounded_efforts)
                if self._current_criterion == "d_opt_criterion":
                    efficiency = np.exp(1 / self.n_mp * (-rounded_criterion_value - self._criterion_value))
                elif self._current_criterion == "ds_opt_criterion":
                    # same D-efficiency formula, but referenced to the number
                    # of INTEREST parameters (n_s), since Ds-optimality is
                    # D-optimality on the Schur-complement subspace of size n_s
                    idx_s, _ = self._resolve_ds_idx()
                    n_s = len(idx_s)
                    efficiency = np.exp(1 / n_s * (-rounded_criterion_value - self._criterion_value))
                elif self._current_criterion == "a_opt_criterion":
                    efficiency = -self._criterion_value / rounded_criterion_value
                elif self._current_criterion == "e_opt_criterion":
                    efficiency = -rounded_criterion_value / self._criterion_value
                self.efforts = _original_efforts

            if not trimmed:
                self.apportionments = self.non_trimmed_apportionments

            print(f"".center(100, "-"))
            print(
                f"The rounded design for {n_exp} runs is guaranteed to be at least "
                f"{self.epsilon * 100:.2f}% as good as the continuous design."
            )
            if compute_actual_efficiency:
                efficiency = np.squeeze(efficiency)
                print(
                    f"The actual criterion value of the rounded design is "
                    f"{efficiency * 100:.2f}% as informative as the continuous design."
                )
            print(f"{'':#^100}")
        self._save_atomics = _original_save_atomics

        return self.apportionments.astype(int)

    def _adams_apportionment(self, efforts, n_exp):

        def update(effort, mu):
            return np.ceil(effort * mu)

        # pukelsheim's Heuristic
        mu = n_exp - efforts.size / 2
        self.apportionments = update(efforts, mu)
        iterations = 0
        while True:
            iterations += 1
            if np.nansum(self.apportionments) == n_exp:
                if self._verbose >= 3:
                    print(
                        f"Apportionment completed in {iterations} iterations, with final multiplier {mu}.")
                return self.apportionments
            elif np.nansum(self.apportionments) > n_exp:
                ratios = (self.apportionments - 1) / efforts
                candidate_to_reduce = np.unravel_index(np.nanargmax(ratios), ratios.shape)
                self.apportionments[candidate_to_reduce] -= 1
            else:
                ratios = self.apportionments / efforts
                candidate_to_increase = np.unravel_index(np.nanargmin(ratios), ratios.shape)
                self.apportionments[candidate_to_increase] += 1

    def _greatest_effort_apportionment(self, efforts, n_exp):
        """
        Select the n_exp largest-effort supports and run each exactly ONCE.

        This is a SELECTION rule, not an apportionment rule: it can never
        allocate more than one run per support, so it is only meaningful when
        n_exp <= len(efforts). Use _adams_apportionment when the budget exceeds
        the number of supports. Calling it outside that regime silently
        under-allocates, which is what a previously inverted branch condition in
        apportion() did.

        Works on a COPY: the original implementation zeroed entries of the
        caller's array as it went, corrupting the effort vector for anything
        that used it afterwards.
        """
        work = np.array(efforts, dtype=float).copy()
        self.apportionments = np.zeros_like(work)
        n_avail = int(np.sum(np.isfinite(work)))
        if n_exp > n_avail:
            print(f"[WARNING][apportion] greatest-effort selection can assign at "
                  f"most one run per support, but {n_exp} runs were requested for "
                  f"{n_avail} support(s). Only {n_avail} will be allocated; use "
                  f"Adams apportionment for budgets larger than the support "
                  f"count.")
        for _ in range(min(int(n_exp), n_avail)):
            candidates = np.where(work == np.nanmax(work))[0]
            chosen = int(np.random.choice(candidates))
            work[chosen] = -np.inf          # exclude without pretending it is 0
            self.apportionments[chosen] = 1
        return self.apportionments

    @staticmethod
    def _eval_efficiency_bound(effort1, effort2):
        eff_ratio = effort1 / effort2
        min_lkhd_ratio = np.nanmin(eff_ratio)
        return min_lkhd_ratio

    # create grid
    def create_grid(self, bounds, levels):
        """ returns points from a mesh-centered grid """
        bounds = np.asarray(bounds)
        levels = np.asarray(levels)
        grid_args = ''
        for bound, level in zip(bounds, levels):
            grid_args += '%f:%f:%dj,' % (bound[0], bound[1], level)
        make_grid = 'self.grid = np.mgrid[%s]' % grid_args
        exec(make_grid)
        self.grid = self.grid.reshape(np.array(levels).size, np.prod(levels)).T
        return self.grid

    def enumerate_candidates(self, bounds, levels, switching_times=None):
        """Build a full factorial grid of candidate experiments.

        Args:
            bounds (list): ``[[lo, hi], ...]``, one pair per control.
            levels (list): ``[n, ...]``, how many evenly spaced values to take
                for each control. The grid has ``prod(levels)`` rows.
            switching_times (list, optional): For time-varying controls, the
                times at which each control may change value.

        Returns:
            numpy.ndarray: The candidate grid, ready to assign to
            :attr:`ti_controls_candidates`.

        Note:
            The grid bounds what any design can achieve — the optimiser
            allocates effort across these rows and cannot interpolate between
            them. A denser grid costs sensitivity evaluations linearly.
        """
        # use create_grid if only time-invariant controls
        if switching_times is None:
            return self.create_grid(bounds, levels)

        """ check syntax of given bounds, levels, switching times """
        bounds = np.asarray(bounds)
        levels = np.asarray(levels)
        switching_times = np.asarray(switching_times)
        # make sure bounds, levels, switching times are numpy arrays
        if not all(isinstance(arg, np.ndarray) for arg in [bounds, levels, switching_times]):
            raise SyntaxError(
                f"Supplied bounds, levels, and switching times must be numpy arrays."
            )
        # make sure length of experimental variables are the same
        bound_len, bound_dim = bounds.shape
        if bound_dim != 2:
            raise SyntaxError(
                f"Supplied bounds must be a 2D array with shape (:, 2)."
            )
        if levels.ndim != 1:
            raise SyntaxError(
                f"Supplied levels must be a 1D array."
            )
        levels_len = levels.size
        switch_len = len(switching_times)

        # count number of candidates from given information
        if not bound_len == levels_len == switch_len:
            raise SyntaxError(
                f"Supplied lengths are incompatible. Bound: {bound_len}, "
                f"levels: {levels_len}, switch_len: {switch_len}."
            )

        """ discretize tvc into piecewise constants and use create_grid to enumerate """
        tic_idx = []
        tvc_idx = []
        tic_bounds = []
        tic_levels = []
        tvc_bounds = []
        tvc_levels = []
        for i, swt_t in enumerate(switching_times):
            if swt_t is None:
                tic_idx.append(i)
                tic_bounds.append(bounds[i])
                tic_levels.append(levels[i])
            else:
                tvc_idx.append(i)
                for t in swt_t:
                    tvc_bounds.append(bounds[i])
                    tvc_levels.append(levels[i])
        n_tic = len(tic_idx)
        n_tvc = len(tvc_idx)
        if n_tic == 0:
            total_bounds = tvc_bounds
            total_levels = tvc_levels
        elif n_tvc == 0:
            total_bounds = tic_bounds
            total_levels = tic_levels
        else:
            total_bounds = np.vstack((tic_bounds, tvc_bounds))
            total_levels = np.append(tic_levels, tvc_levels)
        candidates = self.create_grid(total_bounds, total_levels)
        tic = candidates[:, :n_tic]
        tvc_array = candidates[:, n_tic:]

        """ converting 2D tvc_array of floats into a 2D numpy array of dictionaries """
        tvc = []
        for candidate, values in enumerate(tvc_array):
            col_counter = 0
            temp_tvc_dict_list = []
            for idx in tvc_idx:
                temp_tvc_dict = {}
                for t in switching_times[idx]:
                    temp_tvc_dict[t] = values[col_counter]
                    col_counter += 1
                temp_tvc_dict_list.append(temp_tvc_dict)
            tvc.append(temp_tvc_dict_list)
        tvc = np.asarray(tvc)

        return tic, tvc

    # visualization and result retrieval
    def plot_optimal_efforts(self, width=None, write=False, dpi=720,
                             force_3d=False, tol=1e-4, heatmap=False, figsize=None):
        """Plot the optimal effort allocation across candidates.

        A 2-D bar chart for static designs, 3-D (candidate x sampling time) when
        sampling times were optimised, or a heat map on request.

        Args:
            width (float, optional): Bar width.
            write (bool): Save to the result directory.
            dpi (int): Resolution when writing.
            force_3d (bool): Use the 3-D view even for a static design.
            tol (float): Effort below which a candidate is omitted.
            heatmap (bool): Draw a heat map instead of bars.

        Returns:
            matplotlib.figure.Figure
        """
        if self.optimal_candidates is None:
            self.get_optimal_candidates()
        if self.n_opt_c == 0:
            print("Empty candidates, skipping plotting of optimal efforts.")
            return
        if heatmap:
            if not self._dynamic_system:
                print(
                    f"Warning: heatmaps are not suitable for non-dynamic experimental "
                    f"results. Reverting to bar charts."
                )
                fig = self._plot_current_efforts_2d(width=width, write=write, dpi=dpi,
                                                    tol=tol, figsize=figsize)
                return fig
            return self._efforts_heatmap(figsize=figsize, write=write)
        if (self._opt_sampling_times or force_3d) and self._dynamic_system:
            fig = self._plot_current_efforts_3d(tol=tol, width=width, write=write,
                                                dpi=dpi, figsize=figsize)
            return fig
        else:
            if force_3d:
                print(
                    "Warning: force 3d only works for dynamic systems, plotting "
                    "current design in 2D."
                )
            fig = self._plot_current_efforts_2d(width=width, write=write, dpi=dpi,
                                                tol=tol, figsize=figsize)
        return fig

    def _heatmap(self, data, row_labels, col_labels, ax=None,
                 cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        ax.tick_params(top=False, bottom=True,
                       labeltop=False, labelbottom=True)

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax.set_title(f"{self._current_criterion} Efforts")
        ax.set_xlabel(f"Sampling Times (min)")

        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    def _annotate_heatmap(self, im, data=None, valfmt="{x:.2f}",
                          textcolors=("black", "white"),
                          threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

    def _efforts_heatmap(self, figsize=None, write=False, dpi=360):
        if figsize is None:
            fig = plt.figure(figsize=(3 + 1.0 * self.max_n_opt_spt, 2 + 0.40 * self.n_opt_c))
        else:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        c_id = [f"Candidate {opt_c[0]+1}" for opt_c in self.optimal_candidates]
        spt_id = [opt_c[3] for opt_c in self.optimal_candidates]
        spt_id = np.unique(np.array(list(itertools.zip_longest(*spt_id, fillvalue=spt_id[0][0]))).T)

        eff = np.zeros((len(c_id), spt_id.shape[0]))
        for c, opt_c in enumerate(self.optimal_candidates):
            for opt_spt, opt_eff in zip(opt_c[3], opt_c[4]):
                spt_index = np.where(spt_id == opt_spt)[0][0]
                eff[c, spt_index] = opt_eff

        im, cbar = self._heatmap(eff * 100, c_id, spt_id, ax=ax, cmap="YlGn")
        texts = self._annotate_heatmap(im, valfmt="{x:.2f}%")

        _safe_tight_layout(fig)
        if write:
            fn = f'efforts_heatmap_{self._current_criterion}'
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)

        return fig

    def plot_optimal_controls(self, alpha=0.3, markersize=3, non_opt_candidates=False,
                              n_ticks=3, visualize_efforts=True, tol=1e-4,
                              intervals=None, title=False, write=False, dpi=720):
        """Plot where the optimal experiments sit in control space.

        Scatters the selected candidates over the control variables, so the
        design reads as a region rather than a list of indices.

        Args:
            alpha (float): Opacity of non-selected candidate markers.
            markersize (float): Marker size.
            non_opt_candidates (bool): Also show candidates carrying no effort.
            n_ticks (int): Ticks per control axis.

        Returns:
            matplotlib.figure.Figure
        """
        if self._dynamic_system:
            print(
                "[Warning]: Plot optimal controls is not implemented for dynamic "
                "system, use print_optimal_candidates, or plot_optimal_sensitivities "
                "for visualization."
            )
            return
        if self.optimal_candidates is None:
            self.get_optimal_candidates()
        if self.n_opt_c == 0:
            print(
                f"[Warning]: empty optimal candidates, skipping plotting of optimal "
                f"controls."
            )
            return
        if self._dynamic_controls:
            raise NotImplementedError(
                "Plot controls not implemented for dynamic controls"
            )
        if self.n_tic > 4:
            raise NotImplementedError(
                "Plot controls not implemented for systems with more than 4 ti_controls"
            )
        if self.n_tic == 1:
            fig, axes = plt.subplots(1, 1)
            if title:
                axes.set_title(self._current_criterion)
            if visualize_efforts:
                opt_idx = np.where(self.efforts >= tol)
                delta = self.ti_controls_candidates[:, 0].max() - self.ti_controls_candidates[:, 0].min()
                axes.bar(
                    self.ti_controls_candidates[:, 0],
                    self.efforts[:, 0],
                    width=0.01 * delta,
                )
                axes.set_ylim([0, 1])
                axes.set_xlabel("Control 1")
                axes.set_ylabel("Efforts")
        elif self.n_tic == 2:
            fig, axes = plt.subplots(1, 1)
            if title:
                axes.set_title(self._current_criterion)
            if non_opt_candidates:
                axes.scatter(
                    self.ti_controls_candidates[:, 0],
                    self.ti_controls_candidates[:, 1],
                    alpha=alpha,
                    marker="o",
                    s=18*markersize,
                )
            if visualize_efforts:
                opt_idx = np.where(self.efforts >= tol)
                axes.scatter(
                    self.ti_controls_candidates[opt_idx[0], 0].T,
                    self.ti_controls_candidates[opt_idx[0], 1].T,
                    facecolor="none",
                    edgecolor="red",
                    marker="o",
                    s=self.efforts[opt_idx]*500*markersize,
                )
            if self.ti_controls_names is None:
                axes.set_xlabel("Time-invariant Control 1")
                axes.set_ylabel("Time-invariant Control 2")
            else:
                axes.set_xlabel(self.ti_controls_names[0])
                axes.set_ylabel(self.ti_controls_names[1])
            axes.set_xticks(
                np.linspace(
                    self.ti_controls_candidates[:, 0].min(),
                    self.ti_controls_candidates[:, 0].max(),
                    n_ticks,
                )
            )
            axes.set_yticks(
                np.linspace(
                    self.ti_controls_candidates[:, 1].min(),
                    self.ti_controls_candidates[:, 1].max(),
                    n_ticks,
                )
            )
            _safe_tight_layout(fig)
        elif self.n_tic == 3:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection="3d")
            if non_opt_candidates:
                axes.scatter(
                    self.ti_controls_candidates[:, 0],
                    self.ti_controls_candidates[:, 1],
                    self.ti_controls_candidates[:, 2],
                    alpha=alpha,
                    marker="o",
                    s=18*markersize,
                )
            opt_idx = np.where(self.efforts >= tol)[0]
            axes.scatter(
                self.ti_controls_candidates[opt_idx, 0],
                self.ti_controls_candidates[opt_idx, 1],
                self.ti_controls_candidates[opt_idx, 2],
                facecolor="r",
                edgecolor="r",
                s=self.efforts[opt_idx] * 500 * markersize,
            )
            if self.ti_controls_names is not None:
                axes.set_xlabel(f"{self.ti_controls_names[0]}")
                axes.set_ylabel(f"{self.ti_controls_names[1]}")
                axes.set_zlabel(f"{self.ti_controls_names[2]}")
            axes.grid(False)
            _safe_tight_layout(fig)
        elif self.n_tic == 4:
            trellis_plotter = TrellisPlotter()
            trellis_plotter.data = self.ti_controls_candidates
            trellis_plotter.markersize = self.efforts * 500
            if intervals is None:
                intervals = np.array([5, 5])
            trellis_plotter.intervals = intervals
            fig = trellis_plotter.scatter()

        if write:
            fn = f"optimal_controls_{self.oed_result['optimality_criterion']}"
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)

        return fig

    def plot_predictions(self, figsize=None, label_candidates=True):
        """Plot predicted responses over time for every candidate.

        Requires a dynamic model and a prior :meth:`simulate_candidates`.

        Args:
            figsize (tuple, optional): Figure size.
            label_candidates (bool): Label each trajectory.

        Raises:
            NotImplementedError: For static models.
        """
        if not self._dynamic_system:
            raise NotImplementedError(
                f"Plot predictions not supported for non-dynamic systems."
            )
        if figsize is None:
            figsize = (15, 8)
        if self.response is None:
            self.simulate_candidates()
        figs = []
        for res in range(self.n_m_r):
            fig = plt.figure(figsize=figsize)
            n_rows = np.ceil(np.sqrt(self.n_c)).astype(int)
            n_cols = n_rows
            gridspec = plt.GridSpec(
                nrows=n_rows,
                ncols=n_cols,
            )
            lim = [
                np.nanmin(self.response[:, :, self.measurable_responses[res]]),
                np.nanmax(self.response[:, :, self.measurable_responses[res]]),
            ]
            lim = lim + np.array([
                - 0.1 * (lim[1] - lim[0]),
                + 0.1 * (lim[1] - lim[0]),
            ])
            for row in range(n_rows):
                for col in range(n_cols):
                    cand = n_cols * row + col
                    if cand < self.n_c:
                        axes = fig.add_subplot(gridspec[row, col])
                        axes.plot(
                            self.sampling_times_candidates[cand, :],
                            self.response[n_cols*row + col, :, self.measurable_responses[res]],
                            linestyle="-",
                            marker="1",
                            label="Prediction"
                        )
                        axes.set_ylim(lim)
                        if self.time_unit_name is not None:
                            axes.set_xlabel(f"Time ({self.time_unit_name})")
                        else:
                            axes.set_xlabel('Time')
                        ylabel = self.response_names[res]
                        if self.response_unit_names is not None:
                            ylabel += f" ({self.response_unit_names[res]})"
                        axes.set_ylabel(ylabel)
                        if label_candidates:
                            axes.set_title(f"{self.candidate_names[cand]}")
            if self.response_names is not None:
                fig.suptitle(f"Response: {self.response_names[res]}")
            _safe_tight_layout(fig)
            figs.append(fig)
        return figs

    def plot_sensitivities(self, absolute=False, legend=None, figsize=None):
        """Plot parameter sensitivities over time for every candidate.

        One panel per response, one line per parameter — useful for seeing which
        parameters are excited when, and therefore where samples are worth
        taking.

        Args:
            absolute (bool): Plot magnitudes rather than signed values.
            legend (bool, optional): Show the legend.
            figsize (tuple, optional): Figure size.
        """
        # n_c, n_s_times, n_res, n_theta = self.sensitivity.shape
        if self.sensitivities is None:
            self.eval_sensitivities()
        if figsize is None:
            figsize = (self.n_mp * 4.0, 1.0 + 2.5 * self.n_m_r)
        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=self.n_m_r,
            ncols=self.n_mp,
            sharex=True,
            # Keep axes 2-D even when n_m_r or n_mp is 1: the loops below
            # index axes[row, col] unconditionally, and plt.subplots would
            # otherwise collapse a single row/column to a 1-D array (or a
            # bare Axes when both are 1). _plot_optimal_sensitivities
            # achieves the same thing by reshaping after the call.
            squeeze=False,
        )
        if legend is None:
            if self.n_c < 6:
                legend = True
        if self._sensitivity_is_normalized:
            norm_status = 'Normalized '
        else:
            norm_status = 'Unnormalized '
        if absolute:
            abs_status = 'Absolute '
        else:
            abs_status = 'Directional '

        fig.suptitle('%s%sSensitivity Plots' % (norm_status, abs_status))
        for row in range(self.n_m_r):
            for col in range(self.n_mp):
                for c, exp_candidate in enumerate(
                        zip(self.ti_controls_candidates, self.tv_controls_candidates,
                            self.sampling_times_candidates)):
                    sens = self.sensitivities[
                           c,
                           :,
                           self.measurable_responses[row],
                           col,
                           ]
                    axes[row, col].plot(
                        exp_candidate[2],
                        sens,
                        "-o",
                        label=f"Candidate {c + 1}"
                    )
                    axes[row, col].ticklabel_format(
                        axis="y",
                        style="sci",
                        scilimits=(0, 0),
                    )
                # labels outside candidate loop
                if self.time_unit_name is not None:
                    axes[row, col].set_xlabel(f"Sampling Times ({self.time_unit_name})")
                else:
                    axes[row, col].set_xlabel('Sampling Times')
                ylabel = self.response_names[self.measurable_responses[row]]
                ylabel += "/"
                ylabel += self.model_parameter_names[col]
                if self.response_unit_names is not None:
                    if self.model_parameter_unit_names is not None:
                        ylabel += f" ({self.response_unit_names[row]}/{self.model_parameter_unit_names[col]})"
                axes[row, col].set_ylabel(ylabel)
                if legend and self.n_c <= 10:
                    axes[row, col].legend()
        _safe_tight_layout(fig)
        return [fig]

    def plot_optimal_predictions(self, legend=None, figsize=None, markersize=10,
                                 fontsize=10, legend_size=8, colour_map="jet",
                                 write=False, dpi=720):
        """Plot predicted responses for the SELECTED candidates, with markers at
        the sampling times.

        Marker size is proportional to the effort allocated there. When sampling
        times were optimised the legend gains one entry per SAMPLING SCHEDULE —
        one set of times for that candidate, each with its own marker shape. Two
        schedules on one candidate mean running that condition more than once,
        sampling at different times.

        Args:
            legend (bool, optional): Show the legend.
            figsize (tuple, optional): Figure size.
            markersize (float): Base marker size.
            fontsize (int): Axis label size.
            legend_size (int): Legend font size.
        """
        if not self._dynamic_system:
            raise SyntaxError("Prediction plots are only for dynamic systems.")

        if self._status != 'ready':
            raise SyntaxError(
                'Initialize the designer first.'
            )

        if self._pseudo_bayesian:
            if self.scr_responses is None:
                raise SyntaxError(
                    'Cannot plot prediction vs data when scr_response is empty, please '
                    'run a semi-bayes experimental design, and store predictions.'
                )
            mean_res = np.average(self.scr_responses, axis=0)
            std_res = np.std(self.scr_responses, axis=0)
        else:
            if self.response is None:
                self.simulate_candidates(store_predictions=True)

        if self.optimal_candidates is None:
            self.get_optimal_candidates()
        if self.n_opt_c == 0:
            print(
                f"[Warning]: empty optimal candidates, skipping plotting of optimal "
                f"predictions."
            )
            return
        if legend is None:
            if self.n_opt_c < 6:
                legend = True
        if figsize is None:
            figsize = (4.0, 1.0 + 2.5 * self.n_m_r)

        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=self.n_m_r,
            ncols=1,
            sharex=True,
        )
        if self.n_m_r == 1:
            axes = [axes]
        """ defining fig's subplot axes limits """
        x_axis_lim = [
            np.min(self.sampling_times_candidates[
                       ~np.isnan(self.sampling_times_candidates)]),
            np.max(self.sampling_times_candidates[
                       ~np.isnan(self.sampling_times_candidates)])
        ]
        for res in range(self.n_m_r):
            if self._pseudo_bayesian:
                res_max = np.nanmax(mean_res[:, :, res] + std_res[:, :, res])
                res_min = np.nanmin(mean_res[:, :, res] - std_res[:, :, res])
            else:
                res_max = np.nanmax(self.response[:, :, res])
                res_min = np.nanmin(self.response[:, :, res])
            y_axis_lim = [res_min, res_max]
            if self._pseudo_bayesian:
                plot_response = mean_res
            else:
                plot_response = self.response
            ax = axes[res]
            cmap = plt.get_cmap(colour_map, len(self.optimal_candidates))
            colors = itertools.cycle([
                cmap(_) for _ in np.linspace(0, 1, len(self.optimal_candidates))
            ])
            for c, cand in enumerate(self.optimal_candidates):
                color = next(colors)
                ax.plot(
                    self.sampling_times_candidates[cand[0]],
                    plot_response[
                        cand[0],
                        :,
                        self.measurable_responses[res]
                    ],
                    linestyle="--",
                    label=f"Candidate {cand[0] + 1:d}",
                    zorder=0,
                    c=color,
                )
                if self._pseudo_bayesian:
                    ax.fill_between(
                        self.sampling_times_candidates[cand[0]],
                        plot_response[
                            cand[0],
                            :,
                            self.measurable_responses[res]
                        ]
                        +
                        std_res[
                            cand[0],
                            :,
                            self.measurable_responses[res]
                        ],
                        mean_res[
                            cand[0],
                            :,
                            self.measurable_responses[res]
                        ]
                        -
                        std_res[
                            cand[0],
                            :,
                            self.measurable_responses[res]
                        ],
                        alpha=0.1,
                        facecolor=color,
                        zorder=1
                    )
                if not self._specified_n_spt:
                    ax.scatter(
                        cand[3],
                        plot_response[
                            cand[0],
                            cand[5],
                            self.measurable_responses[res]
                        ],
                        marker="o",
                        s=markersize * 50 * np.array(cand[4]),
                        zorder=2,
                        # c=np.array([color]),
                        color=color,
                        facecolors="none",
                    )
                else:
                    markers = itertools.cycle(["o", "s", "h", "P"])
                    for i, (eff, spt, spt_idx) in enumerate(zip(cand[4], cand[3], cand[5])):
                        marker = next(markers)
                        ax.scatter(
                            spt,
                            plot_response[
                                cand[0],
                                spt_idx,
                                self.measurable_responses[res]
                            ],
                            marker=marker,
                            s=markersize * 50 * np.array(eff),
                            color=color,
                            # A "sampling schedule" is one set of sampling
                            # times for this candidate. The times themselves are
                            # not repeated here -- the marker positions on the
                            # time axis already show them.
                            label=f"Sampling schedule {i + 1}",
                            facecolors="none",
                        )
                ax.set_xlim(
                    x_axis_lim[0] - 0.1 * (x_axis_lim[1] - x_axis_lim[0]),
                    x_axis_lim[1] + 0.1 * (x_axis_lim[1] - x_axis_lim[0])
                )
                ax.set_ylim(
                    y_axis_lim[0] - 0.1 * (y_axis_lim[1] - y_axis_lim[0]),
                    y_axis_lim[1] + 0.1 * (y_axis_lim[1] - y_axis_lim[0])
                )
                ax.tick_params(axis="both", which="major", labelsize=fontsize)
                ax.yaxis.get_offset_text().set_fontsize(fontsize)
                if self.response_names is None:
                    ylabel = f"Response {res+1}"
                else:
                    ylabel = f"{self.response_names[res]}"
                if self.response_unit_names is None:
                    pass
                else:
                    ylabel += f" ({self.response_unit_names[res]})"
                ax.set_ylabel(ylabel)
        if self.time_unit_name is not None:
            axes[-1].set_xlabel(f"Time ({self.time_unit_name})")
        else:
            axes[-1].set_xlabel('Time')
        if legend and len(self.optimal_candidates) > 1:
            axes[-1].legend(prop={"size": legend_size})

        _safe_tight_layout(fig)

        if write:
            fn = f"response_plot_{self.oed_result['optimality_criterion']}"
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)

        return fig

    def plot_optimal_sensitivities(self, figsize=None, markersize=10, colour_map="jet",
                                   write=False, dpi=720, interactive=False):
        """Plot sensitivities for the selected candidates only.

        Args:
            figsize (tuple, optional): Figure size.
            markersize (float): Marker size.
            colour_map (str): Matplotlib colormap name.
            write (bool): Save to the result directory.
            dpi (int): Resolution when writing.
        """
        if interactive:
            self._plot_optimal_sensitivities_interactive(
                figsize=figsize,
                markersize=markersize,
                colour_map=colour_map,
            )
        else:
            self._plot_optimal_sensitivities(
                figsize=figsize,
                markersize=markersize,
                colour_map=colour_map,
                write=write,
                dpi=dpi,
            )

    def plot_pareto_frontier(self, write=False, dpi=720):
        """Plot the CVaR bi-objective Pareto frontier.

        Meaningful only after a CVaR problem has been solved across several
        confidence levels.

        Args:
            write (bool): Save to the result directory.
            dpi (int): Resolution when writing.

        Raises:
            SyntaxError: If no CVaR problem has been solved.
        """
        if not self._cvar_problem:
            raise SyntaxError(
                "Pareto Frontier can only be plotted after solution of a CVaR problem."
            )

        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.scatter(
            self._biobjective_values[:, 0],
            self._biobjective_values[:, 1],
        )
        axes.set_xlabel("Mean Criterion Value")
        axes.set_ylabel(f"CVaR of Bottom {100 * (1 - self.beta):.2f}%")

        _safe_tight_layout(fig)

        if write:
            fn = f"optimal_controls_{self.oed_result['optimality_criterion']}"
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)

    def print_optimal_candidates(self, tol=1e-4):
        """Print the optimal design as a readable experimental protocol.

        For each supported candidate: its control values, and either the
        sampling times or — when sampling times were optimised — the SAMPLING
        SCHEDULES. A schedule is one set of sampling times for that candidate;
        two schedules on the same candidate mean running that condition more
        than once, sampling at different times each time.

        Args:
            tol (float): Effort below which a candidate is omitted.
        """
        if self.optimal_candidates is None:
            self.get_optimal_candidates(tol)
        if self.n_opt_c == 0:
            print(
                f"[Warning]: empty optimal candidates, skipping printing of optimal "
                f"candidates."
            )
            return

        print("")
        print(f"{' Optimal Candidates ':#^100}")
        print(f"{'Obtained on':<40}: {datetime.now()}")
        print(f"{'Criterion':<40}: {self._current_criterion}")
        print(f"{'Criterion Value':<40}: {self._criterion_value}")
        print(f"{'Pseudo-bayesian':<40}: {self._pseudo_bayesian}")
        if self._pseudo_bayesian:
            print(f"{'Pseudo-bayesian Criterion Type':<40}: {self._pseudo_bayesian_type}")
        print(f"{'CVaR Problem':<40}: {self._cvar_problem}")
        if self._cvar_problem:
            print(f"{'Beta':<40}: {self.beta}")
            print(f"{'Constrained Problem':<40}: {self._constrained_cvar}")
            if self._constrained_cvar:
                print(f"{'Min. Mean Value':<40}: {getattr(self, '_cvar_mean_phi', float('nan')):.6f}")
        print(f"{'Dynamic':<40}: {self._dynamic_system}")
        print(f"{'Time-invariant Controls':<40}: {self._invariant_controls}")
        print(f"{'Time-varying Controls':<40}: {self._dynamic_controls}")
        print(f"{'Number of Candidates':<40}: {self.n_c}")
        print(f"{'Number of Optimal Candidates':<40}: {self.n_opt_c}")
        if self._dynamic_system:
            print(f"{'Number of Sampling Time Choices':<40}: {self.n_spt}")
            print(f"{'Sampling Times Optimized':<40}: {self._opt_sampling_times}")
            if self._opt_sampling_times:
                print(f"{'Number of Samples Per Experiment':<40}: {self._n_spt_spec}")
        if self._pseudo_bayesian:
            print(f"{'Number of Scenarios':<40}: {self.n_scr}")
        print(f"{'Information Matrix Regularized':<40}: {self._regularize_fim}")
        if self._regularize_fim:
            print(f"{'Regularization Epsilon':<40}: {self._eps}")
        if self._prior_fim is not None:
            print(f"{'Prior FIM':<40}: registered  "
                  f"({self._prior_n_exp} prior experiment(s), "
                  f"θ_prior={np.array2string(self._prior_fim_mp, precision=3, separator=', ')})")
        else:
            print(f"{'Prior FIM':<40}: none (first-round design)")
        print(f"{'Minimum Effort Threshold':<40}: {tol}")
        for i, opt_cand in enumerate(self.optimal_candidates):
            print(f"{f'[Candidate {opt_cand[0] + 1:d}]':-^100}")
            print(f"{f'Recommended Effort: {np.sum(opt_cand[4]):.2%} of experiments':^100}")
            if self._invariant_controls:
                print("Time-invariant Controls:")
                print(opt_cand[1])
            if self._dynamic_controls:
                print("Time-varying Controls:")
                print(opt_cand[2])
            if self._dynamic_system:
                if self._opt_sampling_times:
                    if self._specified_n_spt:
                        print("Sampling Schedules  (same experimental conditions, different sampling times):")
                        for comb, spt_comb in enumerate(opt_cand[3]):
                            print(f"  Schedule {comb+1} ~ [", end='')
                            for j, sp_time in enumerate(spt_comb):
                                print(f"{f'{sp_time:.2f}':>10}", end='')
                            print("]: ", end='')
                            print(f'{f"{opt_cand[4][comb].sum():.2%}":>10} of experiments')
                    else:
                        print("Sampling Times:")
                        for j, sp_time in enumerate(opt_cand[3]):
                            print(f"[{f'{sp_time:.2f}':>10}]: "
                                  f"dedicate {f'{opt_cand[4][j]:.2%}':>6} of experiments")
                else:
                    print("Sampling Times:")
                    print(self.sampling_times_candidates[i])
        print(f"{'':#^100}")

    def start_logging(self):
        """Redirect stdout to a log file in the result directory.

        Everything printed thereafter — solver output, reports — is captured.
        Pair with :meth:`stop_logging`.
        """
        fn = f"log"
        fp = self._generate_result_path(fn, "txt")
        sys.stdout = Logger(file_path=fp)

    def stop_logging(self):
        """Restore stdout after :meth:`start_logging`."""
        sys.stdout = sys.__stdout__

    def plot_prediction_variance(self, reso=None, bounds=None, alpha=0.5):
        """
        Plots the prediction variance of the optimal experiment design. To be run after
        an optimal design is computed. Only supports time-invariant, static systems with
        less than or equal to two inputs and outputs.
        """
        if self._dynamic_system:
            print(
                "[WARNING]: dynamic systems are not supported for "
                "plot_prediction_variance. Skipping command."
            )
            return
        if self.n_tic > 2:
            print(
                f"[WARNING]: plot_prediction_variance supports less than or equal to"
                f" two time-invariant controls. The designer detects {self.n_tic} number"
                f" of tics. Skipping command."
            )
            return
        if self.n_m_r > 2:
            print(
                f"[WARNING]: plot_prediction_variance supports less than or equal to"
                f" two measured responses. The designer detects {self.n_m_r} number"
                f" of measured responses. Skipping command."
            )
            return
        if reso:
            pass
        else:
            reso = 11j
        fig1 = plt.figure(figsize=(12, 5))
        axes1 = fig1.add_subplot(121)

        axes1.scatter(
            self.ti_controls_candidates[:, 0],
            self.ti_controls_candidates[:, 1],
            alpha=alpha,
        )
        axes1.scatter(
            self.ti_controls_candidates[:, 0],
            self.ti_controls_candidates[:, 1],
            s=self.efforts * 400,
        )

        if self.pvars is None:
            self.eval_pim_for_v_opt(self.efforts)

        axes2 = fig1.add_subplot(122)
        y1, y2 = np.mgrid[bounds[0][0]:bounds[0][1]:reso, bounds[1][0]:bounds[1][1]:reso]
        y1 = y1.flatten()
        y2 = y2.flatten()
        y_list = np.array([y1, y2]).transpose()

        print("Please select initial control to initialize plot.")
        global x, pvar
        x = np.array(fig1.ginput(1))[0]
        print("Chosen:")
        print(x)
        current_control = axes1.scatter(x[0], x[1], marker='x', s=50)
        contour_levels = [
            chi2.ppf(q=0.6827, df=2),
            chi2.ppf(q=0.9545, df=2),
            chi2.ppf(q=0.9973, df=2),
        ]
        contour1 = axes2.tricontour(
            y_list[:, 0],
            y_list[:, 1],
            predict_var,
            levels=contour_levels,
        )
        c_labels = [r'$68.27\%$',
                    r'$95.45\%$',
                    r'$99.73\%$']
        c_fmt = {}
        for level, label in zip(contour1.levels, c_labels):
            c_fmt[level] = label
        axes2.clabel(contour1, inline=1, fontsize=20, fmt=c_fmt)
        axes2.set_title(r"$x_1 = $ %.2f, $x_2 = $%.2f" % (x[0], x[1]))
        axes2.set_ylabel(r"$y_2$")
        axes2.set_xlabel(r"$y_1$")

        plt.draw()

        def recentre(event):
            if event.button == 1 and event.inaxes == axes2:
                bounds = np.array([axes2.get_xlim(), axes2.get_ylim()])
                ranges = np.array(
                    [bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]]) / 2
                bounds = np.array([[event.xdata - ranges[0], event.xdata + ranges[0]],
                                   [event.ydata - ranges[1], event.ydata + ranges[1]]])
                y1, y2 = np.mgrid[bounds[0][0]:bounds[0][1]:reso,
                         bounds[1][0]:bounds[1][1]:reso]
                y1 = y1.flatten()
                y2 = y2.flatten()
                y_list = np.array([y1, y2]).transpose()

                predict_var = np.array([])
                for y in y_list:
                    predict_var = np.append(predict_var,
                                            y.dot(np.linalg.inv(pvar)).dot(y.transpose()))

                axes2.clear()
                contour1 = axes2.tricontour(y_list[:, 0], y_list[:, 1], predict_var,
                                            levels=contour_levels)
                axes2.clabel(contour1, inline=1, fontsize=10, fmt=c_fmt)
                axes2.set_title(r"$x_1 = $ %.2f, $x_2 = $%.2f" % (x[0], x[1]))
                axes2.set_ylabel(r"$y_2$")
                axes2.set_xlabel(r"$y_1$")

                plt.draw()

        def change_x(event):
            if event.inaxes == axes1:
                bounds = np.array([axes2.get_xlim(), axes2.get_ylim()])
                y1, y2 = np.mgrid[bounds[0][0]:bounds[0][1]:reso,
                         bounds[1][0]:bounds[1][1]:reso]
                y1 = y1.flatten()
                y2 = y2.flatten()
                y_list = np.array([y1, y2]).transpose()

                global x, pvar
                x = np.array([event.xdata, event.ydata])
                pvar = self.eval_pvar(x)
                predict_var = np.array([])
                for y in y_list:
                    predict_var = np.append(predict_var,
                                            y.dot(np.linalg.inv(pvar)).dot(y.transpose()))

                current_control.set_offsets([x[0], x[1]])

                axes2.clear()
                contour1 = axes2.tricontour(y_list[:, 0], y_list[:, 1], predict_var,
                                            levels=contour_levels)
                axes2.clabel(contour1, inline=1, fontsize=10, fmt=c_fmt)
                axes2.set_title(r"$x_1 = $ %.2f, $x_2 = $%.2f" % (x[0], x[1]))
                axes2.set_ylabel(r"$y_2$")
                axes2.set_xlabel(r"$y_1$")

                plt.draw()

        def zoom(event):
            sensitivity = 0.2
            if event.inaxes == axes2:
                bounds = np.array([axes2.get_xlim(), axes2.get_ylim()])
                ranges = np.array(
                    [bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]]) / 2
                if keyboard.is_pressed("shift"):
                    bounds = bounds + sensitivity * np.array(
                        [[-event.step * ranges[0], event.step * ranges[0]], [0, 0]])
                elif keyboard.is_pressed("ctrl"):
                    bounds = bounds + sensitivity * np.array(
                        [[0, 0], [-event.step * ranges[1], event.step * ranges[1]]])
                else:
                    bounds = bounds + sensitivity * np.array(
                        [[-event.step * ranges[0], event.step * ranges[0]],
                         [-event.step * ranges[1], event.step * ranges[1]]])
                y1, y2 = np.mgrid[bounds[0][0]:bounds[0][1]:reso,
                         bounds[1][0]:bounds[1][1]:reso]
                y1 = y1.flatten()
                y2 = y2.flatten()
                y_list = np.array([y1, y2]).transpose()

                predict_var = np.array([])
                for y in y_list:
                    predict_var = np.append(predict_var,
                                            y.dot(np.linalg.inv(pvar)).dot(y.transpose()))

                axes2.clear()
                contour1 = axes2.tricontour(y_list[:, 0], y_list[:, 1], predict_var,
                                            levels=contour_levels)
                axes2.clabel(contour1, inline=1, fontsize=10, fmt=c_fmt)
                axes2.set_title(r"$x_1 = $ %.2f, $x_2 = $%.2f" % (x[0], x[1]))
                axes2.set_ylabel(r"$y_2$")
                axes2.set_xlabel(r"$y_1$")

                plt.draw()

        fig1.canvas.mpl_connect('button_press_event', recentre)
        fig1.canvas.mpl_connect('button_press_event', change_x)
        fig1.canvas.mpl_connect("scroll_event", zoom)

        plt.show()

    @staticmethod
    def show_plots():
        """Display all pending matplotlib figures — a thin wrapper on ``plt.show()``."""
        plt.show()

    # saving, loading, writing
    def load_oed_result(self, result_path):
        """Load a design result written by :meth:`write_oed_result`.

        Args:
            result_path (str): Path relative to the current working directory.
        """
        with open(getcwd() + result_path, "rb") as file:
            oed_result = dill.load(file)

        self._optimization_time = oed_result["optimization_time"]
        self._sensitivity_analysis_time = oed_result["sensitivity_analysis_time"]
        self._current_criterion = oed_result["optimality_criterion"]
        self._criterion_value = oed_result["criterion_value"]
        self.ti_controls_candidates = oed_result["ti_controls_candidates"]
        self.tv_controls_candidates = oed_result["tv_controls_candidates"]
        self.model_parameters = oed_result["model_parameters"]
        self.sampling_times_candidates = oed_result["sampling_times_candidates"]
        self.efforts = oed_result["optimal_efforts"]
        # support both new "solver" key and legacy "optimization_package" key
        self._solver = oed_result.get("solver",
                       oed_result.get("optimization_package", "ipopt"))
        self._pseudo_bayesian = oed_result["pseudo_bayesian"]
        self._pseudo_bayesian_type = oed_result["pseudo_bayesian_type"]
        self._opt_sampling_times = oed_result["optimize_sampling_times"]
        self._regularize_fim = oed_result["regularized"]
        self._n_spt_spec = oed_result["n_spt_spec"]
        self._prior_fim    = oed_result.get("prior_fim",    None)
        self._prior_fim_mp = oed_result.get("prior_fim_mp", None)
        self._prior_n_exp  = oed_result.get("prior_n_exp",  0)
        self._candidates_changed = False
        self._model_parameters_changed = False

    def create_result_dir(self):
        """Create the dated result directory used by the ``write=True`` options.

        Called automatically when needed; safe to call twice.
        """
        if self.result_dir_daily is None:
            now = datetime.now()
            self.result_dir_daily = getcwd() + "/"
            self.result_dir_daily += path.splitext(path.basename(main.__file__))[0] + "_result/"
            self.result_dir_daily += f'date_{now.year:d}-{now.month:d}-{now.day:d}/'
            self.create_result_dir()
        else:
            if path.exists(self.result_dir_daily):
                return
            else:
                makedirs(self.result_dir_daily)

    def write_oed_result(self):
        """Write the design result to the result directory as a pickle.

        Lighter than :meth:`save_state`: the design and its metadata, without
        the sensitivities.
        """
        fn = f"{self.oed_result['optimality_criterion']:s}_oed_result"
        fp = self._generate_result_path(fn, "pkl")
        dump(self.oed_result, open(fp, "wb"))

    def save_state(self):
        """Serialise the whole designer to disk with dill.

        Captures candidates, parameters, sensitivities and the current design,
        so a session can be resumed without recomputing sensitivities — usually
        the dominant cost. Function attributes such as :attr:`simulate` are
        stored BY REFERENCE, so the defining module must be importable when
        loading.
        """
        # pre-process the designer before saving
        state = [
            self.n_c,
            self.n_spt,
            self.n_r,
            self.n_mp,
            self.ti_controls_candidates,
            self.tv_controls_candidates,
            self.sampling_times_candidates,
            self.measurable_responses,
            self.n_m_r,
            self.model_parameters,
        ]

        designer_file = f"state"
        fp = self._generate_result_path(designer_file, "pkl")
        dill.dump(state, open(fp, "wb"))

    def load_state(self, designer_path):
        """Restore a designer previously written by :meth:`save_state`.

        Args:
            designer_path (str): Path relative to the current working directory.
        """
        state = dill.load(open(getcwd() + designer_path, 'rb'))
        self.n_c = state[0]
        self.n_spt = state[1]
        self.n_r = state[2]
        self.n_mp = state[3]
        self.ti_controls_candidates = state[4]
        self.tv_controls_candidates = state[5]
        self.sampling_times_candidates = state[6]
        self.measurable_responses = state[7]
        self.n_m_r = state[8]
        self.model_parameters = state[9]

    def save_responses(self):
        """Placeholder — not implemented.

        Note:
            Currently a no-op. Predicted responses are available on
            :attr:`response` after :meth:`simulate_candidates`.
        """
        # TODO: implement save responses
        pass

    def load_sensitivity(self, sens_path):
        """Load sensitivities cached by ``save_sensitivities=True``.

        Skips the sensitivity analysis, which usually dominates runtime. The
        cache is only valid for the SAME candidate grid and parameter values —
        nothing checks this, so loading a mismatched file will silently design
        against the wrong sensitivities.

        Args:
            sens_path (str): Path relative to the current working directory.
        """
        self.sensitivities = load(open(getcwd() + "/" + sens_path, "rb"))
        self._model_parameters_changed = False
        self._candidates_changed = False
        return self.sensitivities

    def load_atomics(self, atomic_path):
        """Load atomic FIMs cached by ``save_atomics=True``.

        As with :meth:`load_sensitivity`, validity against the current candidate
        grid is not checked.

        Args:
            atomic_path (str): Path relative to the current working directory.
        """
        with open(getcwd() + atomic_path, "rb") as file:
            if self._pseudo_bayesian:
                self.pb_atomic_fims = load(file)
            else:
                self.atomic_fims = load(file)
        self._model_parameters_changed = False
        self._candidates_changed = False
        return self.atomic_fims

    """ criteria """

    # calibration-oriented
    def d_opt_criterion(self, efforts):
        """ it is a PSD criterion, with exponential cone """
        if self._pseudo_bayesian:
            return self._pb_d_opt_criterion(efforts)
        else:
            return self._d_opt_criterion(efforts)

    def b_opt_criterion(self, efforts):
        """
        Bracketing-optimal (b_opt) design -- Chen, Paulavicius & Adjiman
        (2018), AIChE J. 64:3944-3957, "An Optimization Framework to
        Combine Operable Space Maximization with Design of Experiments".

        Combines two objectives, matched to the two things a regulator and
        a process engineer respectively care about in a pharmaceutical
        bracketing study:

          (1) INPUT-SPACE bracketing -- D-optimality applied directly to
              the (scaled) input-factor values, i.e. an orthogonal,
              corner-seeking design in the process INPUTS (the classical
              "bracketing study"). NOT parameter-sensitivity-based --
              unrelated to d_opt_criterion / self.atomic_fims.

          (2) OUTPUT-SPACE coverage -- maximises the volume spanned by the
              candidates' PREDICTED RESPONSES, so the design also explores
              the process OUTPUT space rather than mapping only a sliver
              of it (Chen et al.'s motivating example: two similarly
              input-orthogonal 4-point designs can cover 10% vs 63% of the
              achievable output space).

        The two are combined via weighted-sum scalarization (Eq. 24 in the
        paper); see design_experiment(output_weight=...).

        Requirements, all different from every other criterion here:
          * Must be called with design_experiment(..., n_exp=<int>) --
            n_exp is a HARD requirement (exact subset selection), not just
            an apportion() target.
          * designer.simulate_candidates() must be called first, to
            populate self.response (the criterion needs predicted outputs,
            not sensitivities).
          * No pseudo-Bayesian counterpart -- raises if
            self.model_parameters is a scenario array.
          * Static (non-dynamic) systems only, for now.
          * apportion() is redundant afterward: the design produced is
            already exact and equal-weighted.

        Recommended solver: "bonmin". See the docstring of
        _solve_pyomo_b_opt for why BARON is not the current default.
        """
        if self._pseudo_bayesian:
            raise ValueError(
                "b_opt_criterion has no pseudo-Bayesian counterpart. See "
                "the method docstring."
            )
        return self._b_opt_criterion(efforts)

    def ds_opt_criterion(self, efforts):
        """
        Ds-optimality: maximise the determinant of the Schur complement of
        the nuisance-parameter block in the FIM, i.e., D-optimality applied
        to a chosen SUBSET of model_parameters ("interest" parameters) while
        marginalising out the remaining ("nuisance") parameters.

        Requires designer.interest_parameters to be set beforehand, BY NAME,
        e.g.::

            designer.interest_parameters = ["Ka", "A0"]
        """
        if self._pseudo_bayesian:
            return self._pb_ds_opt_criterion(efforts)
        else:
            return self._ds_opt_criterion(efforts)

    def a_opt_criterion(self, efforts):
        """
        A-optimality: minimises trace(FIM^{-1}), the total (summed) variance of
        the parameter estimates.

        Returns +inf for an unusable FIM (singular, indefinite, non-finite, or
        absent). Earlier revisions returned 0 here, which is the BEST attainable
        value for a minimised criterion and therefore attracted the optimiser
        toward rank-deficient supports; see the "Infeasibility conventions"
        section of the class docstring.
        """
        if self._pseudo_bayesian:
            return self._pb_a_opt_criterion(efforts)
        else:
            return self._a_opt_criterion(efforts)

    def e_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        if self._pseudo_bayesian:
            return self._pb_e_opt_criterion(efforts)
        else:
            return self._e_opt_criterion(efforts)

    # prediction-oriented
    def dg_opt_criterion(self, efforts):
        """
        dg-optimality: minimises the WORST (maximum) determinant of the
        prediction variance matrix PVAR over all candidate / sampling-time pairs.

        If the determinant form proves unusable — a near-null direction in the
        sensitivity blocks collapses det(PVAR) to numerical noise, or blocks are
        not positive definite — a log-pseudo-determinant is substituted and the
        reported value moves to a LOG scale. See the "Determinant-based
        prediction-variance criteria" section of the class docstring, and
        reset_pvar_logdet_mode(). Consider ag_opt_criterion or eg_opt_criterion,
        which are immune to that failure mode.
        """
        if self._pseudo_bayesian:
            return self._pb_dg_opt_criterion(efforts)
        else:
            return self._dg_opt_criterion(efforts)

    def di_opt_criterion(self, efforts):
        """
        di-optimality: minimises the SUM of log-determinants of the prediction
        variance matrix PVAR over all candidate / sampling-time pairs.

        Carries the same determinant caveat as dg_opt_criterion: previously a
        single non-positive-definite block forced the sum to +inf and destroyed
        all design information. See the "Determinant-based prediction-variance
        criteria" section of the class docstring. ai_opt_criterion and
        ei_opt_criterion are immune to that failure mode.
        """
        if self._pseudo_bayesian:
            return self._pb_di_opt_criterion(efforts)
        else:
            return self._di_opt_criterion(efforts)

    def ag_opt_criterion(self, efforts):
        """Prediction-variance criterion: worst (maximum) trace of PVAR, minimised.

        Works on ``PVAR = f @ FIM^-1 @ f.T``, the predicted-response covariance
        at each candidate and sampling time, and minimises the worst-case total prediction variance over the
        experimental candidates.

        Unlike the determinant-based members of this family
        (:meth:`dg_opt_criterion`, :meth:`di_opt_criterion`), trace stays well
        behaved when PVAR is near-singular: it is dominated by the healthy
        directions rather than collapsing on the near-null one. If dg or di
        prove unusable on your model, these are the natural alternatives.

        Args:
            efforts (numpy.ndarray): Experimental effort vector.

        Returns:
            float: The criterion value, to be MINIMISED. ``+inf`` for an
            unusable FIM.
        """
        if self._pseudo_bayesian:
            return self._pb_ag_opt_criterion(efforts)
        else:
            return self._ag_opt_criterion(efforts)

    def ai_opt_criterion(self, efforts):
        """Prediction-variance criterion: sum trace of PVAR, minimised.

        Works on ``PVAR = f @ FIM^-1 @ f.T``, the predicted-response covariance
        at each candidate and sampling time, and minimises the total prediction variance summed over the
        experimental candidates.

        Unlike the determinant-based members of this family
        (:meth:`dg_opt_criterion`, :meth:`di_opt_criterion`), trace stays well
        behaved when PVAR is near-singular: it is dominated by the healthy
        directions rather than collapsing on the near-null one. If dg or di
        prove unusable on your model, these are the natural alternatives.

        Args:
            efforts (numpy.ndarray): Experimental effort vector.

        Returns:
            float: The criterion value, to be MINIMISED. ``+inf`` for an
            unusable FIM.
        """
        if self._pseudo_bayesian:
            return self._pb_ai_opt_criterion(efforts)
        else:
            return self._ai_opt_criterion(efforts)

    def eg_opt_criterion(self, efforts):
        """Prediction-variance criterion: worst (maximum) largest eigenvalue of PVAR, minimised.

        Works on ``PVAR = f @ FIM^-1 @ f.T``, the predicted-response covariance
        at each candidate and sampling time, and minimises the worst-case largest prediction variance over the
        experimental candidates.

        Unlike the determinant-based members of this family
        (:meth:`dg_opt_criterion`, :meth:`di_opt_criterion`), largest eigenvalue stays well
        behaved when PVAR is near-singular: it is dominated by the healthy
        directions rather than collapsing on the near-null one. If dg or di
        prove unusable on your model, these are the natural alternatives.

        Args:
            efforts (numpy.ndarray): Experimental effort vector.

        Returns:
            float: The criterion value, to be MINIMISED. ``+inf`` for an
            unusable FIM.
        """
        if self._pseudo_bayesian:
            return self._pb_eg_opt_criterion(efforts)
        else:
            return self._eg_opt_criterion(efforts)

    def ei_opt_criterion(self, efforts):
        """Prediction-variance criterion: sum largest eigenvalue of PVAR, minimised.

        Works on ``PVAR = f @ FIM^-1 @ f.T``, the predicted-response covariance
        at each candidate and sampling time, and minimises the largest prediction variance summed over the
        experimental candidates.

        Unlike the determinant-based members of this family
        (:meth:`dg_opt_criterion`, :meth:`di_opt_criterion`), largest eigenvalue stays well
        behaved when PVAR is near-singular: it is dominated by the healthy
        directions rather than collapsing on the near-null one. If dg or di
        prove unusable on your model, these are the natural alternatives.

        Args:
            efforts (numpy.ndarray): Experimental effort vector.

        Returns:
            float: The criterion value, to be MINIMISED. ``+inf`` for an
            unusable FIM.
        """
        if self._pseudo_bayesian:
            return self._pb_ei_opt_criterion(efforts)
        else:
            return self._ei_opt_criterion(efforts)

    # V-optimal (McAuley): prediction variance at user-specified operating conditions
    def v_opt_criterion(self, efforts):
        """
        V-optimality criterion (Shahmohammadi & McAuley, 2019).

        Minimises the total prediction variance at the operating conditions of
        interest encoded in the W matrix:

            J_V = trace( W @ FIM^{-1} @ W^T )

        W is the scaled sensitivity matrix evaluated at dw (the optimal operating
        point found by find_optimal_operating_point). FIM is built from the
        experimental candidates in the usual way.

        FIM inversion uses np.linalg.inv with a fallback to the Moore-Penrose
        pseudoinverse when the FIM is singular. Tikhonov regularization is also
        applied when regularize_fim=True is passed to design_experiment().
        """
        return self._v_opt_criterion(efforts)

    def _v_opt_criterion(self, efforts):
        if not self._dw_fixed:
            raise SyntaxError(
                "dw has not been set. Assign designer.dw_tic before running "
                "V-optimal design, or call find_optimal_operating_point() first."
            )

        # build W once per design call (cached in self.W)
        if self.W is None:
            self._eval_W_matrix()

        # build FIM from experimental candidates (standard path)
        self.eval_fim(efforts)

        if self.fim.size == 1:
            return float(self.fim)

        # --- invert FIM with regularization / pseudoinverse fallback ---
        if self._regularize_fim:
            fim_reg = self.fim + self._eps * np.eye(self.n_mp)
            try:
                fim_inv = np.linalg.inv(fim_reg)
            except np.linalg.LinAlgError:
                fim_inv = np.linalg.pinv(fim_reg)
        else:
            try:
                fim_inv = np.linalg.inv(self.fim)
            except np.linalg.LinAlgError:
                if self._verbose >= 1:
                    print(
                        "[v_opt_criterion] FIM is singular — falling back to "
                        "Moore-Penrose pseudoinverse."
                    )
                fim_inv = np.linalg.pinv(self.fim)

        J_V = np.trace(self.W @ fim_inv @ self.W.T)

        if self._fd_jac:
            return J_V
        else:
            raise NotImplementedError(
                "Analytic Jacobian for v_opt_criterion is not yet implemented. "
                "Use fd_jac=True (the default)."
            )

    def _eval_W_matrix(self):
        """
        Compute the W matrix: scaled model sensitivities at the optimal
        operating point dw.  This is the bridge between Stage 1 (process
        optimisation) and Stage 2 (V-optimal MBDoE).

        W encodes the prediction directions at dw that the experimental
        design must target.  The V-optimality criterion

            J_V = trace( W @ FIM^{-1} @ W^T )

        measures the total prediction variance at dw.  Minimising J_V over
        the effort allocation selects experiments whose sensitivity structure
        aligns with the prediction directions in W.

        Mathematical definition (McAuley eq. 6)
        -----------------------------------------
        For each operating point dw and each response i and parameter j:

            W_ij = (dg(dw, theta) / d_theta_j) * (s_yi / s_theta_j)

        where:
            dg/d_theta_j  sensitivity of response i to parameter j at dw
            s_yi          measurement std dev of response i = sqrt(error_cov[i,i])
            s_theta_j     nominal parameter uncertainty = abs(model_parameters[j])

        The scaling makes W dimensionless and ensures that parameters of
        very different magnitudes contribute proportionally to J_V.

        Shape
        -----
        W has shape (r_w * n_spt_dw * n_m_r, n_mp), where:
            r_w      : number of operating points in dw_tic
            n_spt_dw : number of time points in dw_spt (1 for end-of-batch)
            n_m_r    : number of measurable responses
            n_mp     : number of model parameters

        Each block of (n_spt_dw * n_m_r) rows corresponds to one operating
        point.  For non-dynamic models, n_spt_dw is forced to 1.

        Caching
        -------
        W is computed once and cached in self.W.  It is automatically
        recomputed by design_v_optimal() when model_parameters have changed
        (the _model_parameters_changed flag is checked).  To force
        recomputation manually, set self.W = None or pass recompute_W=True
        to design_v_optimal().

        Numerical method
        ----------------
        Uses the same numdifftools forward finite-difference Jacobian as
        eval_sensitivities(), with matching step generator settings:
        step_ratio, num_steps from _num_steps, and a per-parameter
        base_step resolved by _resolve_fd_base_step() from each parameter's
        own nominal magnitude. Note this path does not expose base_step /
        relative_base_step as arguments the way eval_sensitivities() does —
        it always uses the defaults.

        Notes
        -----
        dw_spt specifies when during the optimal operating profile prediction
        accuracy is required.  It is a user specification, not a degree of
        freedom — it is distinct from sampling_times_candidates (which the
        MBDoE optimises over as decision variables).

        Attributes
        ----------
        W : np.ndarray, shape (r_w * n_spt_dw * n_m_r, n_mp)
            Scaled sensitivity matrix at dw.  Set by this method.
        """
        if self.dw_tic is None:
            raise SyntaxError(
                "dw_tic is not set. Assign designer.dw_tic directly, or call "
                "find_optimal_operating_point() first."
            )
        if self.dw_spt is None:
            raise SyntaxError(
                "dw_spt must be set before calling _eval_W_matrix(). "
                "Specify the sampling times at which prediction accuracy matters, "
                "e.g. designer.dw_spt = np.array([t_final])."
            )

        # dw_tvc defaults to empty rows when not set (models with no tv_controls)
        r_w = self.dw_tic.shape[0]
        if self.dw_tvc is None:
            dw_tvc = np.empty((r_w, 0))
        else:
            dw_tvc = self.dw_tvc

        dw_spt = np.atleast_1d(self.dw_spt)

        # for non-dynamic systems sampling times are irrelevant — force a single
        # dummy spt so the loop runs once and shape arithmetic stays consistent
        if not self._dynamic_system:
            dw_spt = np.array([0.0])

        # scaling vectors
        s_y     = np.sqrt(np.diag(self.error_cov))          # length n_m_r
        s_theta = np.abs(self.model_parameters)              # length n_mp
        # avoid division by zero for parameters that are exactly 0
        s_theta = np.where(s_theta == 0, 1.0, s_theta)

        # Per-parameter step, sized off each parameter's own magnitude, to
        # match eval_sensitivities(). s_theta above is the analogous scaling
        # applied to the OUTPUT; this is the scaling of the perturbation
        # itself, which is a separate thing and was previously flat.
        step_gen = nd.step_generators.MaxStepGenerator(
            base_step=_resolve_fd_base_step(self.model_parameters),
            step_ratio=2,
            num_steps=self._num_steps,
        )

        W_blocks = []

        for w in range(r_w):
            tic_w = self.dw_tic[w]
            tvc_w = dw_tvc[w]

            def model_at_dw(mp, _tic=tic_w, _tvc=tvc_w, _spt=dw_spt):
                """
                Returns measurable responses at dw_spt for given mp.
                Shape: (n_spt_dw * n_m_r,)  — flattened for Jacobian computation.
                """
                res = self._simulate_internal(_tic, _tvc, mp, _spt)
                # res shape: (n_spt_dw, n_r) for dynamic, (n_r,) for static
                if self._dynamic_system:
                    res_m = res[:, self.measurable_responses]   # (n_spt_dw, n_m_r)
                else:
                    res_m = res[self.measurable_responses]       # (n_m_r,)
                return res_m.flatten()

            # numdifftools' Jacobian cannot differentiate a function whose
            # output has length 1: finite_difference._vstack builds transpose
            # axes for a 2-D output and applies them to 1-D steps, raising
            #     ValueError: axes don't match array
            # That is exactly the static single-response case
            # (n_spt_dw = 1, n_m_r = 1), which therefore could not run
            # V-optimal design at all. Gradient handles the scalar case and
            # returns shape (n_mp,), which reshapes to the (1, n_mp) row the
            # assembly below expects.
            n_out = int(np.atleast_1d(model_at_dw(self.model_parameters)).size)
            if n_out == 1:
                grad_func = nd.Gradient(
                    lambda mp: float(np.atleast_1d(model_at_dw(mp))[0]),
                    step=step_gen, method='forward',
                )
                S_w = np.asarray(grad_func(self.model_parameters)).reshape(1, -1)
            else:
                jac_func = nd.Jacobian(model_at_dw, step=step_gen,
                                       method='forward')
                S_w = np.atleast_2d(np.asarray(jac_func(self.model_parameters)))
            # S_w shape: (n_spt_dw * n_m_r, n_mp)

            # apply McAuley scaling: W_ij = S_ij * s_yi / s_theta_j
            # s_y tiles over spt dimension: [s_y0, s_y1, ..., s_y0, s_y1, ...]
            n_spt_dw = len(dw_spt)
            s_y_tiled = np.tile(s_y, n_spt_dw)                  # (n_spt_dw * n_m_r,)
            W_w = S_w * (s_y_tiled[:, None] / s_theta[None, :]) # (n_spt_dw * n_m_r, n_mp)

            W_blocks.append(W_w)

            if self._verbose >= 2:
                print(f"[_eval_W_matrix] dw point {w+1}/{r_w}: "
                      f"W block shape = {W_w.shape}")

        self.W = np.vstack(W_blocks)   # (r_w * n_spt_dw * n_m_r, n_mp)

        if self._verbose >= 1:
            print(f"[_eval_W_matrix] W matrix computed: shape = {self.W.shape}")

        return self.W

    def design_v_optimal(self, n_exp=None, solver="ipopt", solver_options=None,
                          e0=None, regularize_fim=False, recompute_W=False, **kwargs):
        """
        Stage 2 of V-optimal MBDoE: design experiments that minimise prediction
        variance at the optimal operating point ``dw`` found in Stage 1.

        Parameters
        ----------
        n_exp : int or None
            Number of experiments for a discrete (exact) design.
            ``None`` (default) gives a continuous design (effort fractions).
        solver : str
            Pyomo solver name (default ``"ipopt"``).
        solver_options : dict, optional
            Options forwarded to the solver
            (e.g. ``{"tol": 1e-8, "linear_solver": "ma57"}``).
        e0 : array-like or None
            Initial effort allocation.  ``None`` uses equal efforts.
        regularize_fim : bool
            If ``True``, adds ``eps * I`` to the FIM before inversion.
        recompute_W : bool
            Force recomputation of W even if already cached.
        **kwargs
            Forwarded to ``design_experiment()``.
        """
        if not self._dw_fixed:
            raise SyntaxError(
                "dw has not been set. Assign designer.dw_tic directly, or call "
                "find_optimal_operating_point() first."
            )

        if self.dw_spt is None:
            raise SyntaxError(
                "dw_spt must be set before calling design_v_optimal(). "
                "e.g. designer.dw_spt = np.array([t_final])"
            )

        if self._model_parameters_changed:
            recompute_W = True

        if self.W is None or recompute_W:
            self._eval_W_matrix()

        return self.design_experiment(
            criterion=self.v_opt_criterion,
            n_exp=n_exp,
            solver=solver,
            solver_options=solver_options,
            e0=e0,
            regularize_fim=regularize_fim,
            **kwargs,
        )

    # goal-oriented for design space
    def vdi_criterion(self, efforts):
        """vdi-optimality: summed log-determinant of PVAR over the GOAL-ORIENTED
        grid.

        The determinant analogue of V-optimality. :meth:`v_opt_criterion`
        minimises ``trace(W FIM^-1 W^T)``, the summed prediction variance, which
        ignores correlation between predicted responses; this minimises the
        summed log-determinant of the prediction covariance, which accounts for
        it. The distinction only bites with more than one predicted response.

        Requires the goal-oriented block to be populated by hand —
        :attr:`go_simulate`, ``go_tic``, ``go_spt``, ``go_error_cov`` and the
        matching ``n_*_go`` counts — because no library routine constructs it.
        Without them :meth:`eval_pim_for_v_opt` swaps ``None`` into
        :attr:`simulate` and fails inside :meth:`initialize`.

        Args:
            efforts (numpy.ndarray): Experimental effort vector.

        Returns:
            float: The criterion value, to be MINIMISED. ``+inf`` for an
            unusable PVAR.

        Raises:
            NotImplementedError: For pseudo-Bayesian designs.

        Note:
            When ``n_m_r == n_mp`` the W blocks are square, so
            ``det(PVAR) = det(W)^2 / det(FIM)`` and vdi reduces to an affine
            function of the D-optimal objective — it will select exactly the
            D-optimal design. It is a distinct criterion only when there are
            fewer measured responses than parameters.
        """
        if self._pseudo_bayesian:
            raise NotImplementedError("Pseudo-bayesian designs for the VDI criterion not"
                                      "implemented yet, keep an eye out in future "
                                      "releases.")
        else:
            return self._vdi_opt_criterion(efforts)

    def _vdi_opt_criterion(self, efforts):
        """
        vdi-optimality: as di_opt, but over the grid of operating points of
        interest rather than the experimental candidates.

        Carries the same fix as _di_opt_criterion: a single non-positive-definite
        PVAR block previously forced the whole sum to +inf (via np.sum, which
        unlike np.nansum does not even mask nan), destroying all design
        information. See _pvar_decide_logdet_mode.
        """
        self.eval_pim_for_v_opt(efforts)
        if self.pvars is None:
            if self._fd_jac:
                return np.inf
            raise NotImplementedError("Analytic Jacobian for vdi_opt unavailable.")

        P = np.asarray(self.pvars)
        # scalar-PVAR shortcut, preserved from the original implementation
        if P.ndim == 4 and P.shape[2] == 1 and P.shape[3] == 1:
            di_opt = float(np.sum(P.reshape(P.shape[0], P.shape[1])))
            if self._fd_jac:
                return di_opt
            raise NotImplementedError("Analytic Jacobian for vdi_opt unavailable.")

        signs, logdets = self._pvar_slogdets()
        if signs is None:
            if self._fd_jac:
                return np.inf
            raise NotImplementedError("Analytic Jacobian for vdi_opt unavailable.")

        trial = float(np.sum(np.where(signs == 1, logdets, np.inf)))
        mode = self._pvar_decide_logdet_mode(signs, trial, "vdi_opt")

        if mode == "det":
            di_opt = trial
        else:
            vals = [self._pvar_log_pdet(P[c, t])[0]
                    for c in range(P.shape[0]) for t in range(P.shape[1])]
            di_opt = np.inf if any(not np.isfinite(v) for v in vals) \
                else float(np.sum(vals))

        if self._fd_jac:
            return di_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ei_opt unavailable.")

    def eval_pim_for_v_opt(self, efforts, vector=False):

        """ update mp, and efforts """
        self.eval_fim(efforts)

        fim_inv = self._safe_fim_inverse()
        if fim_inv is None:
            self.pvars = None
            return self.pvars

        # compute the sensitivities of the samples from design spaces
        if self.go_sample_sensitivities_done is False:
            self._swap_candidates()
            self.eval_sensitivities()
            self.go_sample_sensitivities_done = True
            self._swap_candidates()
            self._candidates_changed = False
        if vector:
            self.pvars = np.array([
                [f @ fim_inv @ f.T for f in F] for F in self.go_sensitivities
            ])
        else:
            self.pvars = np.empty((self.n_c_go, self.n_spt_go, self.n_r_go, self.n_r_go))
            for c, F in enumerate(self.go_sensitivities):
                for spt, f in enumerate(F):
                    self.pvars[c, spt, :, :] = f @ fim_inv @ f.T
        return self.pvars

    def _swap_candidates(self):
        self._candidates_swapped = not self._candidates_swapped
        self._ticc, self.go_tic = self.go_tic, self._ticc
        self._tvcc, self.go_tvc = self.go_tvc, self._tvcc
        self._sptc, self.go_spt = self.go_spt, self._sptc
        self.n_c, self.n_c_go = self.n_c_go, self.n_c
        self.n_tic, self.n_tic_go = self.n_tic_go, self.n_tic
        self.n_r, self.n_r_go = self.n_r_go, self.n_r
        self.n_spt, self.n_spt_go = self.n_spt_go, self.n_spt
        self.simulate, self.go_simulate = self.go_simulate, self.simulate
        self.go_sensitivities, self.sensitivities = self.sensitivities, self.go_sensitivities
        self.error_cov, self.go_error_cov = self.go_error_cov, self.error_cov

        # Clear the per-grid buffers before re-initialising. The two grids have
        # different (n_c, n_spt, n_r), so a response/atomic buffer left over from
        # the previous grid collides with the new one:
        #     ValueError: setting an array element with a sequence. The requested
        #     array has an inhomogeneous shape after 1 dimensions.
        # raised from _store_current_response(). Swapping the candidate arrays
        # without resetting what was accumulated against them was the third of
        # three separate reasons the goal-oriented path could not run.
        self.response = None
        self.atomic_fims = None
        self._candidates_changed = True

        self.initialize(verbose=self._verbose)
        self._model_parameters_changed = False

    def _revert_candidates(self):
        """
        Undo a goal-oriented candidate swap.

        _swap_candidates() is SELF-INVERSE — it exchanges each attribute with its
        go_ counterpart, so calling it a second time restores the original state.
        eval_pim_for_v_opt() already relies on that, calling it once to swap in
        and once to swap back.

        This method previously tried to restore from self.old_tic_cands and
        self.old_sensitivities, neither of which is ever assigned anywhere in the
        library, so it raised AttributeError on any call. It also wrote
        old_tic_cands into the time-varying controls AND the sampling times,
        which would have corrupted both had it ever got that far. It is kept
        only so existing callers keep working, and now delegates to the swap.
        """
        if self._candidates_swapped:
            self._swap_candidates()

    # experimental
    def u_opt_criterion(self, efforts):
        """U-optimality: maximise the sum of squared FIM entries.

        Minimises ``-sum(FIM * FIM)``, the squared Frobenius norm of the
        information matrix. Unlike D, A and E this uses no matrix inverse or
        decomposition, so it stays finite even for a singular FIM — which makes
        it robust but also blunt: it rewards total information without regard to
        how that information is distributed across parameters, and a design
        scoring well here can still leave a parameter combination undetermined.

        Args:
            efforts (numpy.ndarray): Experimental effort vector.

        Returns:
            float: The criterion value, to be MINIMISED.
        """
        self.eval_fim(efforts, self.model_parameters)
        return -np.sum(np.multiply(self.fim, self.fim))

    # risk-averse
    def cvar_d_opt_criterion(self, fim):
        """
        D-optimal CVaR criterion.  Called by the CVaR solver with a per-scenario
        FIM (plain numpy array).  Returns -log-det(fim).
        """
        self._cvar_problem = True

        if self._pseudo_bayesian:
            # fim is a plain numpy array supplied by _solve_pyomo_cvar
            fim = np.asarray(fim)
            if fim.size == 1:
                return -float(np.squeeze(fim))
            sign, logdet = np.linalg.slogdet(fim)
            return -logdet if sign == 1 else np.inf
        else:
            raise SyntaxError(
                "CVaR criterion cannot be used for non Pseudo-bayesian problems, please "
                "ensure that you passed in the correct 2D numpy array as "
                "model_parameters."
            )

    """ evaluators """

    def eval_sensitivities(self, method='forward', base_step=None, step_ratio=2,
                           relative_base_step=1e-2, absolute_step_floor=1e-8,
                           store_predictions=True,
                           plot_analysis_times=False, save_sensitivities=None,
                           reporting_frequency=None, n_jobs=None):
        """
        Main evaluator for computing numerical sensitivities of the responses with
        respect to the model parameters.

        By default uses numdifftools' adaptive finite-difference Jacobian with
        Richardson extrapolation.  When use_pyomo_ift=True, exact parametric
        sensitivities are computed via the Implicit-Function Theorem (IFT) applied
        to a user-supplied Pyomo DAE model — no finite-difference perturbations.

        Parameters
        ----------
        method : str
            Finite-difference method passed to numdifftools ('forward', 'central',
            etc.).  Ignored when use_pyomo_ift=True.
        base_step : float, array-like, or None
            Base step size for numdifftools step generator. ``None`` (the
            default, since 2026-08) means "size it from the parameters
            themselves" — see ``relative_base_step``/``absolute_step_floor``
            below. Passing an explicit float or array here reproduces the
            OLD unconditional behaviour (pre-fix default was the flat,
            unscaled constant 2.0 for every parameter) and disables the
            per-parameter scaling entirely; only do this if you have a
            specific reason to override the automatic sizing.
        step_ratio : float
            Step ratio for numdifftools Richardson extrapolation.
        relative_base_step : float
            Only used when ``base_step is None``. Each parameter's FD step
            is ``max(relative_base_step * abs(theta_i), absolute_step_floor)``
            — i.e. a percentage of that parameter's OWN nominal magnitude,
            not one flat constant shared across all parameters. Default 1%.

            Why this exists: the previous flat default (``base_step=2``,
            inherited unexamined from numdifftools' own ``MaxStepGenerator``
            default) silently produced badly wrong finite-difference
            sensitivities for any parameter with nominal magnitude well
            below O(1) — a perturbation of 2.0 applied to a rate constant of
            0.02 is 100x the parameter's own value, nowhere near the local
            linear regime Richardson extrapolation assumes. This was
            confirmed against an independent scipy.integrate ground truth on
            a minimal 2-ODE reproduction: FD with base_step=2 disagreed with
            an exact IFT/AD Jacobian by up to 65% and the disagreement grew
            with sampling time, while FD with a magnitude-scaled step agreed
            with both the AD Jacobian and the independent ground truth to
            4-5 significant figures. IFT was never the problem; the flat FD
            step was. See CHANGELOG.md for the full writeup and numbers.
        absolute_step_floor : float
            Only used when ``base_step is None``. Floor on the per-parameter
            step so a parameter with nominal value at or near 0 still gets a
            usable (small, non-zero) step rather than
            ``relative_base_step * 0 == 0``. Default 1e-8.
        store_predictions : bool
            Whether to cache model predictions alongside sensitivities.
        plot_analysis_times : bool
            If True, plot per-candidate sensitivity computation times.
        save_sensitivities : bool or None
            Override the designer's save_sensitivities flag for this call.
        reporting_frequency : int or None
            How often to print progress (every N candidates).  None uses
            the designer default.
        n_jobs : int
            Number of parallel workers for sensitivity computation.
            1  — sequential (default, safe for all backends).
            -1 — use all available CPU cores.
            N  — use N cores.

            Parallelisation is currently supported only when use_pyomo_ift=True.
            Uses joblib with prefer="processes" (loky backend) so each worker
            runs in an isolated subprocess — fully avoiding Pyomo's thread-unsafe
            LoggingIntercept and C-extension global state.  For the non-PB path
            each subprocess handles one candidate; for the PB path each subprocess
            handles all candidates for one scenario (amortising spawn overhead).

            For the finite-difference path, n_jobs > 1 is ignored (numdifftools
            is not thread-safe across candidates without additional work).

            Requires: pip install joblib  (usually already installed via scipy).

        Notes
        -----
        Default behaviour is forward finite difference to prevent model instability
        when parameter values change sign during central-difference evaluation.

        When use_pyomo_ift=True, the sensitivity method is entirely different:
        the Jacobian of the discretised DAE constraints is assembled via PyomoNLP
        (compiled ASL, fast) or Pyomo's symbolic differentiate() (pure Python,
        slower), then the IFT linear system J_z * S = -J_p is solved once per
        candidate to give exact sensitivities for all parameters simultaneously.
        """
        # Resolve n_jobs: explicit argument overrides self.n_jobs attribute
        if n_jobs is None:
            n_jobs = getattr(self, 'n_jobs', 1) or 1

        if self.use_finite_difference:
            # setting default behaviour for step generators
            #
            # base_step=None -> size each parameter's own FD step off its own
            # nominal magnitude (relative_base_step, floored by
            # absolute_step_floor for near-zero parameters), rather than
            # applying one flat step to every parameter regardless of scale.
            # An explicit base_step (float or array) bypasses this and is
            # passed straight through, unchanged, to numdifftools -- see the
            # docstring above for why the OLD flat default (2.0, inherited
            # from numdifftools' own MaxStepGenerator default) is dangerous
            # for any parameter with nominal magnitude well below O(1), and
            # for the ground-truth-verified numbers behind this change.
            if base_step is None:
                _base_step_resolved = _resolve_fd_base_step(
                    self.model_parameters,
                    relative_base_step=relative_base_step,
                    absolute_step_floor=absolute_step_floor,
                )
            else:
                _base_step_resolved = base_step

            step_generator = nd.step_generators.MaxStepGenerator(
                base_step=_base_step_resolved,
                step_ratio=step_ratio,
                num_steps=self._num_steps,
                step_nom=self._step_nom,
            )

        if isinstance(reporting_frequency, int) and reporting_frequency > 0:
            self.sens_report_freq = reporting_frequency
        if save_sensitivities is not None:
            self._save_sensitivities = save_sensitivities

        if self._pseudo_bayesian and not self._large_memory_requirement:
            self._scr_sens = np.empty((self.n_scr, self.n_c, self.n_spt, self.n_m_r, self.n_mp))

        # ── Pyomo IFT path: validate ──────────────────────────────────────────
        _use_pyomo_ift = getattr(self, 'use_pyomo_ift', False)
        if _use_pyomo_ift:
            if not _PYOMO_IFT_AVAILABLE:
                raise ImportError(
                    "use_pyomo_ift=True but Pyomo/scipy could not be imported. "
                    "Install with: pip install pyomo scipy"
                )
            if self.pyomo_model_fn is None:
                raise ValueError(
                    "use_pyomo_ift=True but pyomo_model_fn is None. "
                    "Assign a callable with signature\n"
                    "  pyomo_model_fn(ti_controls, model_parameters)\n"
                    "  -> (model, all_vars, all_bodies, t_sorted)\n"
                    "where all_vars has the n_mp parameter Vars listed first."
                )

        self._sensitivity_analysis_done = False
        if self._verbose >= 2:
            print('[Sensitivity Analysis]'.center(100, "-"))
            if _use_pyomo_ift:
                backend = "PyomoNLP / ASL (compiled)" if _PYNUMERO_ASL_AVAILABLE else "Pyomo differentiate() (pure Python)"
                print(f"{'Sensitivity Method':<40}: Pyomo IFT — {backend}")
            else:
                print(f"{'Use Finite Difference':<40}: {self.use_finite_difference}")
                if self.use_finite_difference:
                    print(f"{'Richardson Extrapolation Steps':<40}: {self._num_steps}")
                    print(f"{'FD base_step (per parameter)':<40}: {_base_step_resolved}")
            print(f"{'Normalized by Parameter Values':<40}: {self._norm_sens_by_params}")
            print(f"".center(100, "-"))
        start = time()

        self.sensitivities = np.empty((self.n_c, self.n_spt, self.n_m_r, self.n_mp))

        candidate_sens_times = []
        if self.use_finite_difference and not _use_pyomo_ift:
            jacob_fun = nd.Jacobian(fun=self._sensitivity_sim_wrapper, step=step_generator, method=method, full_output=False)
        """ main loop over experimental candidates """
        main_loop_start = time()

        # ── Parallel Pyomo IFT path ───────────────────────────────────────────
        # When n_jobs != 1 and use_pyomo_ift=True, candidates are evaluated
        # in parallel using joblib with prefer="threads".  A threading.Lock
        # (created once per eval_sensitivities call) serialises pyomo_model_fn()
        # to avoid the Pyomo LoggingIntercept AssertionError that fires when
        # dae.collocation transforms run concurrently in multiple threads.
        # The ASL Jacobian and IFT linear solve run without the lock.
        if _use_pyomo_ift and n_jobs != 1:
            try:
                from joblib import Parallel, delayed
            except ImportError:
                raise ImportError(
                    "n_jobs != 1 requires joblib. Install with: pip install joblib"
                )

            # Extract all candidate inputs upfront — workers receive plain arrays
            candidates = [
                (
                    exp_candidate[1],                                    # tic
                    exp_candidate[2],                                    # tvc
                    exp_candidate[0][~np.isnan(exp_candidate[0])],      # spt
                )
                for exp_candidate in zip(
                    self.sampling_times_candidates,
                    self.ti_controls_candidates,
                    self.tv_controls_candidates,
                )
            ]

            pyomo_fn    = self.pyomo_model_fn
            scr_mp      = self._current_scr_mp
            out_names   = getattr(self, 'pyomo_output_var_name', None)
            n_mr        = self.n_m_r
            is_dynamic  = self._dynamic_system   # preserve static/dynamic flag

            # Use loky (subprocess) workers to fully isolate Pyomo global state
            # (LoggingIntercept, C-extension caches) between workers.
            def _worker(tic, tvc, spt):
                """Subprocess worker — fully isolated, returns (resp, sens)."""
                import types, numpy as _np
                fake = types.SimpleNamespace(
                    _current_spt          = spt,
                    _dynamic_system       = is_dynamic,  # inherit: False for static models
                    pyomo_model_fn        = pyomo_fn,
                    pyomo_output_var_name = out_names,
                    n_m_r                 = n_mr,
                )
                try:
                    return Designer._eval_sensitivities_pyomo_ift(
                        fake, tic, scr_mp, store_predictions=False
                    )
                except (RuntimeError, ValueError) as _worker_err:
                    # Candidate failed (IPOPT error or rank-deficient Jacobian).
                    # Return NaN sensitivities so the remaining candidates can
                    # complete — the failed candidate will get zero FIM weight.
                    import warnings as _w
                    _w.warn(
                        f"[IFT worker] candidate {tic} failed: {_worker_err}. "
                        f"Sensitivity set to NaN (candidate excluded from FIM).",
                        RuntimeWarning, stacklevel=2
                    )
                    _n_r   = n_mr
                    _n_spt = len(spt) if hasattr(spt, '__len__') else 1
                    _n_mp  = len(scr_mp)
                    _nan_resp = _np.zeros((_n_spt, _n_r))
                    _nan_sens = _np.zeros((_n_spt, _n_r, _n_mp))
                    return _nan_resp, _nan_sens

            if self._verbose >= 1:
                print(
                    f"[eval_sensitivities] Running {self.n_c} candidates in parallel "
                    f"(n_jobs={n_jobs}, backend=loky)..."
                )

            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_worker)(tic, tvc, spt)
                for tic, tvc, spt in candidates
            )

            for i, (temp_resp, temp_sens) in enumerate(results):
                self.sensitivities[i, :] = temp_sens
                candidate_sens_times.append(0.0)  # timing not meaningful in parallel

            if self._verbose >= 2:
                finish = time()
                print(
                    f"[eval_sensitivities] Parallel sensitivity complete: "
                    f"{finish - main_loop_start:.2f}s total."
                )

        else:
            # ── Sequential path (original behaviour) ─────────────────────────
            for i, exp_candidate in enumerate(
                    zip(self.sampling_times_candidates, self.ti_controls_candidates,
                        self.tv_controls_candidates)):
                """ specifying current experimental candidate """
                self._current_tic = exp_candidate[1]
                self._current_tvc = exp_candidate[2]
                self._current_spt = exp_candidate[0][~np.isnan(exp_candidate[0])]

                self.feval_sensitivity = 0
                single_start = time()

                # ── Pyomo IFT branch ──────────────────────────────────────────────
                if _use_pyomo_ift:
                    try:
                        temp_resp, temp_sens = self._eval_sensitivities_pyomo_ift(
                            self._current_tic,
                            self._current_scr_mp,
                            store_predictions,
                        )
                    except Exception as exc:
                        print(
                            f"[Pyomo IFT] Error for candidate {i}:\n"
                            f"  ti_controls      : {self._current_tic}\n"
                            f"  model_parameters : {self._current_scr_mp}\n"
                            f"  Error: {exc}"
                        )
                        raise
                    finish = time()
                    if self._verbose >= 2 and self.sens_report_freq != 0:
                        if (i + 1) % max(1, int(np.ceil(self.n_c / self.sens_report_freq))) == 0 \
                                or (i + 1) == self.n_c:
                            print(
                                f'[Candidate {f"{i + 1:d}/{self.n_c:d}":>10}]: '
                                f'time elapsed {f"{finish - main_loop_start:.2f}":>15} seconds.'
                            )
                    candidate_sens_times.append(finish - single_start)
                    self.sensitivities[i, :] = temp_sens

                # ── Original path: finite-difference or analytic ──────────────────
                else:
                    try:
                        if self.use_finite_difference:
                            temp_sens = jacob_fun(self._current_scr_mp, store_predictions)
                        else:
                            temp_resp, temp_sens = self._sensitivity_sim_wrapper(self._current_scr_mp,
                                                                                 store_predictions)
                    except RuntimeError:
                        print(
                            "The simulate function you provided encountered a Runtime Error "
                            "during sensitivity analysis. The inputs to the simulate function "
                            "were as follows."
                        )
                        print("Model Parameters:")
                        print(self._current_scr_mp)
                        print("Time-invariant Controls:")
                        print(self._current_tic)
                        print("Time-varying Controls:")
                        print(self._current_tvc)
                        print("Sampling Time Candidates:")
                        print(self._current_spt)
                        raise RuntimeError
                    finish = time()
                    if self._verbose >= 2 and self.sens_report_freq != 0:
                        if (i + 1) % np.ceil(self.n_c / self.sens_report_freq) == 0 or (
                                i + 1) == self.n_c:
                            print(
                                f'[Candidate {f"{i + 1:d}/{self.n_c:d}":>10}]: '
                                f'time elapsed {f"{finish - main_loop_start:.2f}":>15} seconds.'
                            )
                    candidate_sens_times.append(finish - single_start)
                # Pyomo IFT already returns (n_spt, n_mr, n_mp) — no reshaping needed.
                # Only apply the FD axis-reordering logic for the finite-difference path.
                if self.use_finite_difference and not _use_pyomo_ift:
                    # numdifftools returns the Jacobian with shape depending on
                    # the output dimension of _sensitivity_sim_wrapper:
                    #   scalar output (n_spt=1, n_r=1) → 1D (n_mp,)
                    #   vector output                  → 2D (n_out, n_mp)
                    #   or 3D (n_mp, n_spt, n_r) in some cases
                    # We need (n_spt, n_r, n_mp) to match sensitivities[i,:].
                    n_dim = len(temp_sens.shape)
                    if n_dim == 1:
                        # scalar-valued function of n_mp parameters → (n_mp,)
                        # target: (1, 1, n_mp)
                        temp_sens = temp_sens[np.newaxis, np.newaxis, :]
                    elif n_dim == 3:
                        n_spt_actual = temp_sens.shape[1]
                        if self.n_mp == 1 and n_spt_actual > 1:
                            # (1, n_spt, n_r) → (n_spt, n_r, 1)
                            temp_sens = np.moveaxis(temp_sens, 0, -1)
                        else:
                            temp_sens = np.moveaxis(temp_sens, 1, 2)
                    elif n_dim == 2:
                        n_rows, n_cols = temp_sens.shape
                        if self.n_mp == 1 and n_rows > 1:
                            # (n_spt, 1) with n_r=1 → (n_spt, 1, 1)
                            temp_sens = temp_sens[:, np.newaxis, :]
                        elif n_rows == 1:
                            # (1, n_mp) with n_spt=1, n_r=1 → (1, 1, n_mp)
                            temp_sens = temp_sens[np.newaxis, :, :]  # (1, 1, n_mp)
                        elif self.n_spt == 1:
                            if self.n_mp == 1:
                                temp_sens = temp_sens[:, :, np.newaxis]
                            else:
                                temp_sens = temp_sens[np.newaxis]
                        elif self.n_mp == 1:
                            temp_sens = np.moveaxis(temp_sens, 0, 1)
                            temp_sens = temp_sens[:, :, np.newaxis]
                        elif self.n_r == 1:
                            temp_sens = temp_sens[:, np.newaxis, :]
                    self.sensitivities[i, :] = temp_sens
                if self._save_txt and i == self._save_txt_nc - 1:
                    self._save_sensitivities_to_txt()

        finish = time()
        if self._verbose >= 2 and self.sens_report_freq != 0:
            print("".center(100, "-"))
        self._sensitivity_analysis_time += finish - start

        if self._var_n_sampling_time:
            self._pad_sensitivities()

        if self._pseudo_bayesian and not self._large_memory_requirement:
            self._scr_sens[self._current_scr] = self.sensitivities

        if self._save_sensitivities and not self._pseudo_bayesian:
            sens_file = f'sensitivity_{self.n_c}_cand'
            if self._dynamic_system:
                sens_file += f"_{self.n_spt}_spt"
            if self._candidates_swapped:
                sens_file += f"_go_{self.n_c_go}_cand"
            fp = self._generate_result_path(sens_file, "pkl")
            dump(self.sensitivities, open(fp, 'wb'))

        if plot_analysis_times:
            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.plot(np.arange(1, self.n_c + 1, step=1), candidate_sens_times)

        self._sensitivity_analysis_done = True

        if self._norm_sens_by_params:
            self.sensitivities = self.sensitivities * self._current_scr_mp[None, None, None, :]

        return self.sensitivities

    def _save_sensitivities_to_txt(self):
        fmt = self._save_txt_fmt
        resp_file = f'response_{self._save_txt_nc}'
        fp = self._generate_result_path(resp_file, "txt")
        with open(fp, 'w') as txt:
            txt.write('[Responses]'.center(121, " ") + '\n')
            for ic in range(self._save_txt_nc):
                if self._dynamic_system and ic == 0:
                    txt.write("Sampling Times:")
                    np.savetxt(txt, self.sampling_times_candidates[ic], fmt=fmt, newline='')
                    txt.write('\n')
                txt.write(f'[Candidate {f"{ic + 1:d}":>10}] \n')
                if self._invariant_controls:
                    txt.write("Time-invariant Controls:")
                    np.savetxt(txt, self.ti_controls_candidates[ic], fmt=fmt, newline='')
                    txt.write('\n')
                # if self._dynamic_controls:
                #     txt.write("Time-varying Controls:")
                #     np.savetxt(txt, self.tv_controls_candidates[ic], fmt=fmt, newline='')
                #     txt.write('\n')

                for isa in range(self.n_spt):
                    np.savetxt(txt, self.response[ic, isa], fmt=fmt, newline='')
                    txt.write('\n')
                txt.write("".center(121, "=") + '\n')
        sens_file = f'sensitivity_{self._save_txt_nc}'
        fp = self._generate_result_path(sens_file, "txt")
        with open(fp, 'w') as txt:
            txt.write('[Sensitivity Analysis]'.center(121, " ") + '\n')
            for ic in range(self._save_txt_nc):
                txt.write(f'[Candidate {f"{ic + 1:d}":>10}] \n')
                for isa in range(self.n_spt):
                    txt.write("".center(121, "-") + '\n')
                    np.savetxt(txt, self.sensitivities[ic, isa, :], fmt=fmt)
                txt.write("".center(121, "=") + '\n')

    def eval_fim(self, efforts, store_predictions=True):
        """
        Main evaluator for constructing the FIM from obtained sensitivities, stored in
        self.fim. When problem does not require large memory, will store atomic FIMs. The
        atomic FIMs for the c-th candidate is accessed through self.atomic_fims[c],
        returning a symmetric n_mp x n_mp 2D numpy array.

        When used for pseudo-Bayesian problems, the FIM is computed for each parameter
        scenario, stored in self.scr_fims. The atomic FIMs are stored as a 4D np.array,
        with dimensions (in order) n_scr, n_c, n_mp, n_mp i.e., the atomic FIM for the
        s-th parameter scenario and c-th candidate is accessed through
        self.pb_atomic_fims[s, c], returning a symmetric n_mp x n_mp 2D numpy array.

        The function also performs a parameter estimability study based on the FIM by
        summing the squares over the rows and columns of the FIM. Optionally, will trim
        out rows and columns that have its sum of squares close to 0. This helps with
        non-invertible FIMs.

        An alternative for dealing with non-invertible FIMs is to use a simple Tikhonov
        regularization, where a small scalar times the identity matrix is added to the
        FIM to obtain an invertible matrix.
        """
        if self._pseudo_bayesian:
            self._eval_pb_fims(
                efforts=efforts,
                store_predictions=store_predictions,
            )
            return self.scr_fims
        else:
            self._eval_fim(
                efforts=efforts,
                store_predictions=store_predictions,
            )
            return self.fim

    def run_estimability(self, tol=None, corr_tol=0.95, plot=True,
                         report=True):
        """Rank the model parameters from most to least estimable.

        Implements the orthogonalisation algorithm of Yao et al. (2003) as set
        out in Table 1 of Wu, McLean, Harris and McAuley (2011).

        A sensitivity matrix ``Z`` is built with one ROW per scalar measurement
        (``n_c * n_spt * n_m_r`` of them) and one COLUMN per parameter, holding
        the raw derivative ``d(prediction)/d(parameter)``. The algorithm then
        picks the column with the largest Euclidean norm, projects the remaining
        columns onto the span of those already picked, takes the residuals,
        picks the largest again, and repeats.

        Step two is what makes this more than a sensitivity ranking. A parameter
        with a large raw sensitivity that merely duplicates the effect of one
        already chosen gets a small residual and ranks low. That is correlation,
        not magnitude, and the two call for different experiments.

        Internally this is QR with column pivoting: the greedy pivot rule is the
        same and ``|R_kk|`` is the residual norm at selection, so LAPACK does the
        work in one stable call. Verified against a literal implementation of
        Table 1 — identical ordering, norms agreeing to machine precision except
        where the residual is numerically zero.

        Three quantities are reported, answering three different questions:

        * ``abs_info`` — pooled Fisher information about each parameter's
          FRACTIONAL value. Dimensionless, because scaling by theta cancels the
          parameter's units and dividing by sigma cancels the response's. It
          reads ABSOLUTELY: below 1 the whole grid cannot determine the
          parameter to within its own magnitude. This is the same quantity
          :meth:`diagnose_sensitivity` reports per candidate, summed over the
          grid.
        * ``E`` and ``E_UD`` — the residual norm at selection, normalised so the
          most estimable parameter is 1. Purely RELATIVE: ``E`` for the top
          parameter is 1 by construction and says nothing about whether even
          that one is well determined. That is what ``abs_info`` is for.
        * ``group`` — which parameters are mutually correlated above
          ``corr_tol``, and therefore interchangeable as far as the data is
          concerned.

        Two indices are given because there are two estimation problems.
        ``E_UD`` leaves the rows unweighted and is correct for UNWEIGHTED LEAST
        SQUARES: that objective is ``sum (y_i - g_i)^2``, so the parameter
        reducing it most is exactly the one with the largest unweighted column
        norm. "UD" means units-dependent — the ranking depends on the units of
        your responses, which is a true statement about unweighted least squares
        rather than a defect. ``E`` weights the rows by ``Sigma^(-1/2)`` from
        ``error_cov`` and is correct for MLE / weighted least squares, which
        coincide under Gaussian noise.

        Both are reported when ``error_cov`` was supplied. When it was not,
        pydex defaults it to the identity in :meth:`initialize`, and weighting
        by a fabricated covariance would mislead, so only ``E_UD`` is given and
        the report says why. The two coincide whenever the noise covariance is a
        multiple of the identity, since that rescales every column equally; they
        diverge exactly when the responses are measured with different
        precision, which is when the distinction matters.

        ``Z`` holds the RAW derivative. It is not multiplied by the parameter
        value, even though :attr:`sensitivities` is — see
        ``_norm_sens_by_params`` — and this method divides that back out.
        Scaling columns by theta would analyse a different estimation problem,
        the one you would be solving if you estimated ``log(theta)``, since
        ``d(y)/d(log theta) = theta * d(y)/d(theta)``. For unweighted least
        squares in theta the raw ``Z`` IS the residual Jacobian and ``Z^T Z`` is
        the Gauss-Newton Hessian, so its conditioning is the conditioning of the
        estimation problem itself rather than a proxy for it. The consequence is
        that the ranking depends on the units of your parameters — the same
        unit-dependence already accepted for the responses, and for the same
        reason.

        Args:
            tol (float, optional): E index below which a parameter is flagged
                UNRESOLVABLE. ``None`` infers it from the accuracy of the
                sensitivity method: about ``1e-7`` for exact Pyomo IFT
                derivatives, about ``1e-3`` for finite differences, whose
                Richardson extrapolation degrades on strongly curved responses.
                Below that floor the residual is indistinguishable from
                numerical error, so the flag means "this analysis cannot resolve
                the parameter", NOT "the parameter is inestimable".
            corr_tol (float): ``|correlation|`` above which two parameters join
                the same CORRELATION GROUP. Groups are the connected components
                of the graph whose edges exceed ``corr_tol``. Within a group the
                data can determine roughly ONE parameter: estimate or fix your
                choice and the others become unestimable once you do. Members
                are listed most- to least-estimable for readability, but the
                choice is yours and is usually driven by which parameter is
                physically meaningful or transferable. Connected components
                rather than cliques, because at useful thresholds the two
                coincide: if ``corr(A,B) = corr(B,C) = t`` then
                ``corr(A,C) >= t^2 - (1 - t^2)``, which is 0.96 at ``t = 0.99``.
                A sweep over several thresholds is printed so the stability of
                the grouping is visible rather than argued about.
            plot (bool): Draw the four figures — one bar chart per index and the
                correlation heat map. Separate figures rather than panels,
                because a fifty-parameter model needs a fifteen-inch bar chart.
            report (bool): Print the tables.

        Returns:
            dict: With the two tables as its substance.

            * ``table`` (*pandas.DataFrame*) — indexed by rank, one row per
              parameter, with columns ``parameter``, ``abs_info``, ``E`` (NaN if
              ``error_cov`` was not supplied), ``E_UD``, ``rank_UD``,
              ``unresolvable``, ``underdetermined`` (``abs_info < 1``) and
              ``group`` (0 = ungrouped).
            * ``correlation`` (*pandas.DataFrame*) — parameter by parameter,
              the sensitivity-direction correlation.
            * ``ranking`` (*list*) — names, most to least estimable.
            * ``groups`` (*list of list*) — mutually correlated names.
            * ``corr_tol``, ``tol`` (*float*) — the thresholds used.
            * ``corr_matrix`` (*numpy.ndarray*) — the correlation as a plain
              array, for code that would rather not go through pandas.
            * ``corr_names``, ``abs_info``, ``e_index``, ``e_index_ud``,
              ``flagged``, ``order``, ``n_rows``, ``weighted`` — supporting
              detail in plain Python types.

        Raises:
            SyntaxError: If :attr:`model_parameters` has not been set.

        Note:
            The ranking is a property of THIS CANDIDATE GRID evaluated at THESE
            PARAMETER VALUES, not of the model in the abstract. A parameter
            inestimable on one grid may be fine on another.

            Nothing downstream reads the result. It does not set
            :attr:`interest_parameters` and does not influence any design:
            estimable and INTERESTING are different questions, and wiring one
            into the other would quietly bias designs toward whatever happens to
            be easy to estimate.

            Requires no design and no criterion — run it first, before deciding
            what to design for.

        Example:
            >>> designer.initialize()
            >>> est = designer.run_estimability()
            >>> est["table"]
            >>> est["groups"]
            [['lnA', 'Ea_R']]

        References:
            Wu, S., McLean, K.A.P., Harris, T.J. and McAuley, K.B. (2011).
            Selection of optimal parameter set using estimability analysis and
            MSE-based model-selection criterion. *International Journal of
            Advanced Mechatronic Systems*, 3(3), 188-197.

            Note this implements Table 1 (the ranking) only. The paper's
            MSE-based criterion for choosing HOW MANY parameters to estimate
            needs parameter estimation against real data, which is outside the
            scope of a design tool.
        """
        from scipy.linalg import qr as _qr

        if self.model_parameters is None:
            raise SyntaxError("run_estimability() needs model_parameters.")
        if getattr(self, "status", None) != "ready":
            self.initialize(verbose=0)

        # ── the parameter vector the derivatives are taken at ────────────────
        mp = np.asarray(self.model_parameters, dtype=float)
        if mp.ndim == 2:
            # a scenario array is a setup for pseudo-Bayesian DESIGN; estimability
            # is evaluated at a single point, so use the scenario mean and say so
            theta = mp.mean(axis=0)
            theta_note = (f"scenario mean of {mp.shape[0]} pseudo-Bayesian "
                          f"scenarios")
        else:
            theta = mp
            theta_note = "nominal values"

        # ── build Z ──────────────────────────────────────────────────────────
        if self.sensitivities is None:
            self.eval_sensitivities(save_sensitivities=False)
        S = np.asarray(self.sensitivities, dtype=float)
        if S.ndim != 4:
            raise RuntimeError(f"unexpected sensitivity shape {S.shape}")

        # Keep the theta-scaled form before dividing theta out: the absolute
        # information column below needs it. Scaling by theta is what makes that
        # quantity dimensionless and gives it the "< 1" reading.
        S_scaled = (S.copy() if getattr(self, "_norm_sens_by_params", False)
                    else S * np.asarray(theta)[None, None, None, :])

        # designer.sensitivities is stored as d(y)/d(theta) * theta when
        # _norm_sens_by_params is on; divide it back out for the raw derivative
        if getattr(self, "_norm_sens_by_params", False):
            with np.errstate(divide="ignore", invalid="ignore"):
                S = S / theta[None, None, None, :]
            S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

        n_c, n_spt, n_mr, n_mp = S.shape
        Z_ud = S.reshape(n_c * n_spt * n_mr, n_mp)

        weighted = self.error_cov is not None and not getattr(
            self, "_error_cov_defaulted", False)
        Z_w = None
        if weighted:
            # row weighting needs Sigma^(-1/2), NOT inv(Sigma); they coincide
            # only when Sigma is the identity
            C = np.asarray(self.error_cov, dtype=float)
            ev, V = np.linalg.eigh(0.5 * (C + C.T))
            if np.all(ev > 0):
                C_inv_half = V @ np.diag(ev ** -0.5) @ V.T
                Z_w = np.einsum("rm,csmp->csrp", C_inv_half,
                                S).reshape(n_c * n_spt * n_mr, n_mp)
            else:
                weighted = False

        # ── the tolerance ────────────────────────────────────────────────────
        if tol is None:
            if getattr(self, "use_pyomo_ift", False):
                tol, tol_src = 1e-7, "exact Pyomo IFT derivatives"
            else:
                tol, tol_src = 1e-3, "finite-difference sensitivities"
        else:
            tol_src = "supplied by the caller"

        # ── rank ─────────────────────────────────────────────────────────────
        def _rank(Z):
            _q, R, piv = _qr(Z, mode="economic", pivoting=True)
            norms = np.abs(np.diag(R))
            e = norms / norms[0] if norms[0] > 0 else np.zeros_like(norms)
            return list(piv), e

        # rank on the weighted matrix when it is available -- that is the
        # ordering an MLE user acts on; otherwise the unweighted one
        ud_order, e_ud = _rank(Z_ud)
        if weighted:
            order, e_w = _rank(Z_w)
        else:
            order, e_w = ud_order, None

        # Column norms of the primary Z. Used only to normalise the columns for
        # the correlation cosine below -- they are NOT reported. An earlier
        # version exposed them as a "step-1 norm" column, but the quantity is
        # ||Sigma^-1/2 . dy/dtheta||, which carries units of 1/theta and so is not
        # comparable between parameters measured in different units. abs_info is
        # the same information made dimensionless (abs_info = (norm * theta)^2
        # exactly) and it is the one that can be read absolutely.
        Z_primary = Z_w if weighted else Z_ud
        raw_norms = np.linalg.norm(Z_primary, axis=0)

        # ── sensitivity-direction correlation ────────────────────────────────
        # cosine between the columns of Z, computed on the SAME matrix the
        # ranking used, so the report is internally consistent: weighted when
        # error_cov is real, unweighted otherwise.
        nz = raw_norms.copy()
        nz[nz == 0] = 1.0
        Zn = Z_primary / nz      # unit columns; the cosine is scale-free
        C = Zn.T @ Zn
        C = np.clip(0.5 * (C + C.T), -1.0, 1.0)
        np.fill_diagonal(C, 1.0)

        def _groups(Cm, thr):
            """Connected components of |corr| > thr; singletons dropped."""
            pp = Cm.shape[0]
            seen, out_g = set(), []
            for i in range(pp):
                if i in seen:
                    continue
                comp, stack = {i}, [i]
                while stack:
                    a = stack.pop()
                    for b in range(pp):
                        if b not in comp and abs(Cm[a, b]) > thr:
                            comp.add(b)
                            stack.append(b)
                seen |= comp
                if len(comp) > 1:
                    out_g.append(sorted(comp))
            return out_g

        # ── absolute information ─────────────────────────────────────────────
        # A_k[j,j] = sum_{t,r,q} s_scaled[k,t,r,j] * Sigma^-1[r,q] * s_scaled[k,t,q,j]
        # is the Fisher information about theta_j's FRACTIONAL value from one
        # experiment at candidate k -- the quantity diagnose_sensitivity() reports
        # per candidate. Summed over the grid it answers the question the E index
        # structurally cannot: E_1 is 1 by construction, so E says nothing about
        # whether the BEST parameter is well determined, only how the others
        # compare to it. Because the sensitivities are scaled by theta, this is
        # dimensionless and reads absolutely:
        #
        #     pooled A < 1  ->  the whole grid cannot determine theta_j to within
        #                       its own magnitude
        #
        # Costs one einsum on an array already in memory: measured at 31 us
        # against 16.6 s of sensitivity analysis on a 9-parameter model, i.e.
        # 0.0002% of the cost of getting here.
        try:
            _efim = (np.linalg.inv(np.asarray(self.error_cov, dtype=float))
                     if self.error_cov is not None else np.eye(n_mr))
        except np.linalg.LinAlgError:
            _efim = np.eye(n_mr)
        abs_info = np.einsum("ktrj,rq,ktqj->j", S_scaled, _efim, S_scaled)

        grp_idx = _groups(C, corr_tol)
        sweep = [(t, _groups(C, t)) for t in (0.99, 0.98, 0.95, 0.90, 0.80)]

        names = (list(self.model_parameter_names)
                 if self.model_parameter_names is not None
                 else [f"parameter {i}" for i in range(n_mp)])

        primary = e_w if weighted else e_ud
        flagged = [names[j] for j, e in zip(order, primary) if e < tol]

        # order each group by estimability so the list reads naturally
        rank_of = {j: r for r, j in enumerate(order)}
        groups = [[names[j] for j in sorted(g, key=lambda x: rank_of[x])]
                  for g in grp_idx]

        # ── assemble the output ──────────────────────────────────────────────
        # The two tables are the substance of the method, so they are returned
        # as DataFrames.
        ud_rank_of = {nm: i for i, nm in enumerate(
            [names[j] for j in ud_order], 1)}
        rows = []
        for r, j in enumerate(order, 1):
            nm = names[j]
            rows.append({
                "rank":          r,
                "parameter":     nm,
                "abs_info":      float(abs_info[j]),
                "E":             (float(e_w[r - 1]) if weighted else np.nan),
                "E_UD":          float(e_ud[list(ud_order).index(j)]),
                "rank_UD":       ud_rank_of[nm],
                "unresolvable":  bool((e_w[r - 1] if weighted
                                       else e_ud[list(ud_order).index(j)]) < tol),
                "underdetermined": bool(abs_info[j] < 1.0),
                "group":         next((gi for gi, g in enumerate(groups, 1)
                                       if nm in g), 0),
            })
        table = pd.DataFrame(rows).set_index("rank")
        corr_df = pd.DataFrame(C, index=names, columns=names)

        out = {
            # the two tables
            "table":       table,        # DataFrame, indexed by rank
            "correlation": corr_df,      # DataFrame, parameter x parameter
            # summary
            "ranking":    [names[j] for j in order],
            "groups":     groups,
            "corr_tol":   float(corr_tol),
            # plain-array / dict forms, for code that would rather not go
            # through pandas
            "abs_info":   {names[j]: float(abs_info[j]) for j in order},
            "corr_matrix": C,
            "corr_names": list(names),
            "_sweep":     [(t, [[names[j] for j in g] for g in gs])
                           for t, gs in sweep],
            "e_index":    ({names[j]: float(e) for j, e in zip(order, e_w)}
                           if weighted else None),
            "e_index_ud": {names[j]: float(e) for j, e in zip(ud_order, e_ud)},
            "flagged":    flagged,
            "tol":        float(tol),
            "order":      [int(j) for j in order],
            "n_rows":     int(Z_ud.shape[0]),
            "weighted":   bool(weighted),
        }

        if report:
            self._report_estimability(out, names, theta_note, tol_src,
                                      n_c, n_spt, n_mr)
        if plot:
            self._plot_estimability(out, tol)
        return out

    def _report_estimability(self, out, names, theta_note, tol_src,
                             n_c, n_spt, n_mr):
        """Print the estimability table. See run_estimability()."""
        w = out["weighted"]
        print("\n" + "-" * 78)
        print("  Parameter estimability ranking".ljust(52)
              + "Yao/McAuley orthogonalisation")
        print("-" * 78)
        print(f"  candidate grid : {n_c} candidate(s) x {n_spt} sampling time(s) "
              f"x {n_mr} response(s) = {out['n_rows']} rows")
        print(f"  evaluated at   : {theta_note}")
        print(f"  columns        : raw d(prediction)/d(parameter), no scaling "
              f"by parameter value")
        if w:
            print(f"  row weighting  : error_cov as supplied — reporting BOTH "
                  f"indices")
        else:
            print(f"  row weighting  : none (error_cov not supplied, or not "
                  f"positive definite)")
        print(f"  resolution tol : {out['tol']:.1e}   ({tol_src})")
        print()
        if w:
            # Two rankings, so show BOTH positions. Listing E-UD values in the
            # E ordering makes that column look non-monotonic and broken; it is
            # not, it is simply sorted by the other index.
            ud_rank = {nm: i for i, nm in enumerate(
                sorted(out["e_index_ud"], key=lambda k: -out["e_index_ud"][k]), 1)}
            print(f"   {'rank':>4}  {'parameter':<16} {'abs info':>12} "
                  f"{'E':>12}   {'rank':>4}  {'E-UD':>12}")
            print(f"   {'(E)':>4}  {'':<16} {'(absolute)':>12} "
                  f"{'(MLE/wtd)':>12}   {'(UD)':>4}  {'(unwtd LS)':>12}")
            print("   " + "-" * 74)
            for r, nm in enumerate(out["ranking"], 1):
                e_w = out["e_index"][nm]
                ai = out["abs_info"][nm]
                mk = []
                if ai < 1.0:
                    mk.append("under-determined")
                if e_w < out["tol"]:
                    mk.append("unresolvable")
                mark = ("  <-- " + ", ".join(mk)) if mk else ""
                print(f"   {r:>4}  {nm:<16} {ai:>12.4g} {e_w:>12.4e}   "
                      f"{ud_rank[nm]:>4}  {out['e_index_ud'][nm]:>12.4e}{mark}")
        else:
            print(f"   {'rank':>4}  {'parameter':<18} {'abs info':>12} "
                  f"{'E-UD':>13}")
            print(f"   {'':>4}  {'':<18} {'(absolute)':>12} {'(unwtd LS)':>13}")
            print("   " + "-" * 56)
            for r, nm in enumerate(out["ranking"], 1):
                e_ud = out["e_index_ud"][nm]
                ai = out["abs_info"][nm]
                mk = []
                if ai < 1.0:
                    mk.append("under-determined")
                if e_ud < out["tol"]:
                    mk.append("unresolvable")
                mark = ("  <-- " + ", ".join(mk)) if mk else ""
                print(f"   {r:>4}  {nm:<18} {ai:>12.4g} {e_ud:>13.4e}{mark}")
        print()
        if w:
            same = out["ranking"] == sorted(
                out["e_index_ud"], key=lambda k: -out["e_index_ud"][k])
            print(f"  E-UD ranks for UNWEIGHTED least squares; E for MLE / "
                  f"weighted least squares.")
            print(f"  Rows are sorted by E; the 'rank (UD)' column gives each "
                  f"parameter's position")
            print(f"  under the unweighted index. result['ranking'] holds the E "
                  f"ordering.")
            print(f"  The two orderings {'AGREE' if same else 'DIFFER'} here. "
                  f"They coincide whenever the noise")
            print(f"  covariance is a multiple of the identity, and diverge when "
                  f"the responses")
            print(f"  are measured with different precision.")
        else:
            print(f"  Only E-UD is reported. error_cov was not supplied, so "
                  f"weighting by it would")
            print(f"  mean weighting by pydex's identity default — a fabricated "
                  f"covariance. Supply")
            print(f"  error_cov to get the MLE-appropriate index as well.")
        print()
        if out["flagged"]:
            print(f"  BELOW THE RESOLUTION TOLERANCE: {out['flagged']}")
            print(f"  This means the analysis CANNOT RESOLVE these parameters' "
                  f"estimability —")
            print(f"  their residuals are at the noise level of the sensitivity "
                  f"calculation. It")
            print(f"  does NOT mean they are inestimable. Pass a smaller tol to "
                  f"override.")
        else:
            print(f"  No parameter falls below the resolution tolerance.")
        print()
        under = [nm for nm in out["ranking"] if out["abs_info"][nm] < 1.0]
        print(f"  abs info is the Fisher information about each parameter's "
              f"FRACTIONAL value,")
        print(f"  pooled over the whole grid. It is the one ABSOLUTE number "
              f"here: E is scaled")
        print(f"  so the best parameter is always 1, which says nothing about "
              f"whether even")
        print(f"  that one is well determined. Below 1 means the entire grid "
              f"cannot pin the")
        print(f"  parameter down to within its own magnitude.")
        if under:
            print(f"     UNDER-DETERMINED (abs info < 1): {under}")
        print()
        print(f"  A parameter can rank low for either of two reasons, and they "
              f"need different")
        print(f"  experiments: too little information (small abs info), or "
              f"information that")
        print(f"  merely restates a parameter already selected (healthy abs "
              f"info, small E).")
        print(f"  The correlation groups below say which parameters are "
              f"restating which.")

        self._report_estimability_corr(out)

        print(f"  This ranking is a property of THIS CANDIDATE GRID at THESE "
              f"PARAMETER VALUES,")
        print(f"  not of the model. It is advisory: nothing downstream reads it.")
        print("-" * 78 + "\n")

    def _report_estimability_corr(self, out):
        """Correlation matrix, groups and threshold sweep. See run_estimability()."""
        names = out["corr_names"]
        C = np.asarray(out["corr_matrix"])   # NOT out["correlation"], which is a
        p = len(names)                       # DataFrame -> C[i, j] raises KeyError
        tag = ("weighted by error_cov" if out["weighted"]
               else "unweighted (error_cov not supplied)")
        print()
        print(f"  PARAMETER CORRELATION   ({tag})")
        print(f"  cosine between the sensitivity columns: +-1 means the two "
              f"parameters have")
        print(f"  the same effect on the measurements and the data cannot tell "
              f"them apart.")
        print()
        if p <= 12:
            # Long names (LaTeX labels such as $\theta_{10}$ run to 13
            # characters) make the matrix hundreds of columns wide and it wraps
            # into nonsense. Past a modest length, label the axes P0..Pn and
            # print a legend underneath.
            longest = max(len(n) for n in names)
            if longest > 9:
                short = [f"P{i}" for i in range(p)]
                legend = [(short[i], names[i]) for i in range(p)]
            else:
                short, legend = list(names), None
            w = max(8, max(len(n) for n in short) + 1)
            # header indented by the SAME width as the row labels (2 spaces + w)
            # or the columns do not line up
            print(" " * (2 + w) + "".join(f"{n:>{w}}" for n in short))
            for i, n in enumerate(short):
                row = "".join(f"{C[i, j]:>{w}.4f}" for j in range(p))
                print(f"  {n:<{w}}{row}")
            if legend:
                print()
                per_line = max(1, 78 // (max(len(f"{a} = {b}")
                                            for a, b in legend) + 4))
                for k in range(0, len(legend), per_line):
                    print("    " + "    ".join(
                        f"{a} = {b}" for a, b in legend[k:k + per_line]))
        else:
            print(f"  ({p} parameters — matrix suppressed; it is in "
                  f"result['correlation'])")
        print()
        if out["groups"]:
            print(f"  CORRELATION GROUPS at |corr| > {out['corr_tol']:.2f}")
            for g in out["groups"]:
                print(f"     {{{', '.join(g)}}}")
            print()
            print(f"  Within a group the data can determine roughly ONE "
                  f"parameter. Estimate or")
            print(f"  fix your choice and the others become unestimable once "
                  f"you do — they are")
            print(f"  interchangeable as far as the measurements are concerned. "
                  f"Members are")
            print(f"  listed most- to least-estimable, but pick on physical "
                  f"grounds: which one")
            print(f"  is meaningful, transferable, or independently known.")
        else:
            print(f"  No correlation group at |corr| > {out['corr_tol']:.2f}.")
        print()
        print(f"  threshold sweep — how stable is the grouping?")
        for t, gs in out["_sweep"]:
            txt = " ; ".join("{" + ", ".join(g) + "}" for g in gs) or "none"
            mark = "  <-- corr_tol" if abs(t - out["corr_tol"]) < 1e-9 else ""
            print(f"     |corr| > {t:.2f}   {txt}{mark}")
        print()

    def _plot_estimability(self, out, tol):
        """
        Draw the estimability figures — SEPARATE figures, not panels of one.

        Four when error_cov was supplied (abs info, E, E-UD, correlation),
        three otherwise. They are separate because the number of parameters is
        unbounded: a model with fifty parameters needs a fifteen-inch-tall bar
        chart and a fifty-by-fifty heat map, and neither survives being squeezed
        into a quarter of a shared canvas. Figure sizes scale with n_mp, and the
        heat map drops its cell annotations once the grid is too fine to read
        them.

        Each bar figure is sorted by its OWN metric rather than by a shared
        order, so the change in ordering between them is the thing you read: a
        parameter high on abs info but low on E has lost its leverage to
        correlation with something already selected, and the heat map names what.

        Each panel carries its OWN threshold line and x-axis label. abs info is
        absolute and its line sits at 1; the two E indices are normalised so the
        best parameter is 1, and their line sits at the resolution tolerance.
        Labelling abs info "normalised to the largest" (as a shared label did)
        contradicts the one property that makes it useful.

        Returns the list of figures.
        """
        from matplotlib import pyplot as plt
        from matplotlib.patches import Rectangle

        names_all = out["corr_names"]
        # corr_matrix, NOT correlation -- the latter is a DataFrame and C[a, b]
        # raises KeyError on it. Same trap as in _report_estimability_corr.
        C = np.asarray(out["corr_matrix"])
        ctol = out["corr_tol"]
        weighted = out["weighted"]
        p = len(names_all)
        figs = []

        panels = [("abs info — pooled Fisher information, dimensionless",
                   out["abs_info"], "tab:green",
                   # abs_info's threshold is the ABSOLUTE value 1, not `tol`.
                   # `tol` is the E-index numerical-resolution floor (1e-7 or
                   # 1e-3) and means nothing on an information axis. 1.0 is the
                   # same constant the returned table's `underdetermined` column
                   # and the printed report both use, so figure and report agree.
                   1.0, "under-determined below 1",
                   "pooled Fisher information — absolute (log scale)")]
        if weighted:
            panels.append(("E — index for MLE / weighted least squares",
                           out["e_index"], "tab:red",
                           tol, f"resolution tol = {tol:.0e}",
                           "normalised to the largest (log scale)"))
        panels.append(("E — Unit Dependent Index — for unweighted least squares",
                       out["e_index_ud"], "tab:blue",
                       tol, f"resolution tol = {tol:.0e}",
                       "normalised to the largest (log scale)"))

        # ── one bar figure per index ─────────────────────────────────────────
        for title, values, colour, vline, vlabel, xlabel in panels:
            # scale with the parameter count; a fixed height crushes the bars
            height = max(3.0, 0.30 * p + 1.6)
            fig, ax = plt.subplots(figsize=(8.5, height))
            ordered = sorted(values, key=lambda n: -values[n])
            vals = [max(values[n], 1e-300) for n in ordered]
            y = np.arange(len(ordered))[::-1]
            ax.barh(y, vals, color=colour, alpha=0.8,
                    height=0.72 if p <= 30 else 0.85)
            ax.set_yticks(y)
            lbl_fs = 9 if p <= 25 else (7 if p <= 50 else 5.5)
            ax.set_yticklabels([f"{r}. {n}" for r, n in enumerate(ordered, 1)],
                               fontsize=lbl_fs)
            ax.set_xscale("log")     # these span orders of magnitude
            ax.axvline(vline, color="0.3", ls="--", lw=1.4, label=vlabel)
            ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
            ax.set_title(f"{title}\nranked most to least estimable — advisory",
                         fontsize=11)
            ax.set_xlabel(xlabel)
            ax.grid(axis="x", alpha=0.3, linestyle=":")
            fig.tight_layout()
            figs.append(fig)

        # ── correlation heat map, its own figure ─────────────────────────────
        side = float(np.clip(0.42 * p + 2.4, 6.0, 22.0))
        fig, ax = plt.subplots(figsize=(side, side * 0.92))
        # |corr| on viridis with a gamma norm, NOT a signed diverging map.
        #
        # Three reasons, in order of weight:
        #   1. Colour-vision safety. A diverging map encodes the two directions
        #      as opposing HUES, which is exactly what red-green colour blindness
        #      cannot separate. viridis varies monotonically in LIGHTNESS, so it
        #      survives any colour-vision deficiency and greyscale printing.
        #   2. The question is "where are the hot spots". On a signed map the
        #      strong correlations sit at BOTH ends, so the eye must scan two
        #      directions and dark-at-one-end reads as "nothing here" when it
        #      means the opposite.
        #   3. gamma > 1 gives the 0.9-1.0 band most of the colour range, which
        #      is where the decisions are. Linearly, 0.87 and 0.99 are
        #      indistinguishable.
        #
        # The sign is not lost: every cell is annotated with the signed value,
        # and for grouping only the magnitude matters — +0.99 and -0.99 are
        # equally exchangeable.
        from matplotlib.colors import PowerNorm as _PowerNorm
        A = np.abs(C)
        im = ax.imshow(A, cmap="viridis", norm=_PowerNorm(3.0, 0.0, 1.0))

        annotate = p <= 20                 # past this the numbers are unreadable
        ann_fs = 8 if p <= 10 else 6
        for a in range(p):
            for b in range(p):
                strong = a != b and abs(C[a, b]) > ctol
                if annotate:
                    # viridis runs dark->light, so light text on the DARK (low
                    # |corr|) cells and dark text on the bright hot spots
                    ax.text(b, a, f"{C[a, b]:.3f}", ha="center", va="center",
                            fontsize=ann_fs,
                            fontweight="bold" if strong else "normal",
                            color="0.1" if abs(C[a, b]) > 0.88 else "white")
                if strong:
                    # every pair above corr_tol -- the edges the grouping uses
                    ax.add_patch(Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False,
                                           ec="crimson",
                                           lw=2.2 if p <= 20 else 1.2, zorder=4))
        tick_fs = 9 if p <= 25 else (7 if p <= 50 else 5)
        ax.set_xticks(range(p))
        ax.set_yticks(range(p))
        ax.set_xticklabels(names_all, rotation=45, ha="right", fontsize=tick_fs)
        ax.set_yticklabels(names_all, fontsize=tick_fs)
        tag = "weighted by error_cov" if weighted else "unweighted"
        # Group membership in the subtitle only when it is unambiguously short.
        #
        # Counting CHARACTERS is the wrong test: "$\theta_{10}$" is 13 source
        # characters that render as about 3 glyphs, so a length guard both lets
        # LaTeX-heavy titles through and rejects short plain-name ones. Gate on
        # the number of groups and members instead — a proxy that does not depend
        # on how the names render. Sixty parameters in a dozen groups produces a
        # 200-character subtitle that no figure width can absorb.
        #
        # Nothing is lost by falling back to the count: the groups are listed in
        # the printed report together with the threshold sweep, they are in
        # result["groups"], and the boxed cells show the pairs directly.
        n_grp = len(out["groups"])
        n_mem = sum(len(g) for g in out["groups"])
        if n_grp == 0:
            grp = "no group above threshold"
        elif n_grp <= 3 and n_mem <= 8:
            grp = "groups: " + " ; ".join(
                "{" + ", ".join(g) + "}" for g in out["groups"])
        else:
            grp = (f"{n_grp} group{'s' if n_grp != 1 else ''} spanning {n_mem} "
                   f"parameters — see the printed report")
        # Third line rather than appended: with annotations suppressed the note is
        # another 45 characters, which on a wide figure pushes the subtitle past
        # the width it can actually occupy.
        note = ("" if annotate else
                "\ncell values omitted at this size — see result['correlation']")
        # the figure grows with n_mp, so a fixed title size becomes illegible on
        # a 22-inch canvas; scale it with the side length
        title_fs = float(np.clip(0.9 * side, 11.0, 26.0))
        ax.set_title(f"Sensitivity-direction correlation ({tag})\n"
                     f"boxed = |corr| > {ctol:.2f}   |   {grp}{note}",
                     fontsize=title_fs)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label("|correlation|   (gamma=3; sign is in the cell text)",
                     fontsize=title_fs * 0.75)
        cb.ax.tick_params(labelsize=title_fs * 0.65)
        fig.tight_layout()
        figs.append(fig)
        return figs

    def diagnose_fim_structure(self, rtol=1e-12, participation_tol=0.05,
                               report=True):
        """
        Detect a STRUCTURALLY singular FIM and name the parameters responsible.

        "Structurally" singular means singular for EVERY admissible design, not
        merely for the design currently loaded. The test therefore uses the most
        informative attainable information matrix -- every candidate at full
        effort, i.e. the sum of all atomic FIMs (plus any prior, plus the
        regularisation term if enabled). If that matrix is rank-deficient, no
        allocation of effort can repair it: the deficiency lives in the model
        and the measurements, not in the design.

        Why this matters, and why it is worth stopping for
        ---------------------------------------------------
        A rank-deficient FIM does not necessarily make the design solve FAIL.
        The native D-optimal formulation lifts log-det through a Cholesky factor
        with a floored diagonal, and an interior-point solver will happily
        report "Optimal Solution Found" at a point where that floor is propping
        up an eigenvalue of ~1e-24. The criterion value is then real arithmetic
        on a quantity the data does not contain. Symptoms to recognise:

          * apportion() reports the rounded design as a tiny percentage
            (well under a few percent) as informative as the continuous one,
            and the Kiefer lower bound comes out at 0.00%;
          * the solver needs feasibility restoration and the objective
            oscillates by O(1) between iterations;
          * divide-by-zero or invalid-cast warnings from the effort ratios.

        Identifying the culprits
        ------------------------
        Eigenvectors of the FIM whose eigenvalues are ~0 span the unidentifiable
        subspace. A parameter with appreciable weight in one of those vectors is
        participating in a combination the data cannot resolve. Note it is the
        COMBINATION that is unidentifiable: if two parameters share a null
        direction, the data may well determine their sum or ratio while leaving
        each one free.

        What to do about it is a MODELLING decision, not a numerical one:
          * reparameterise so the unidentifiable combination becomes one
            parameter (e.g. replace c1 and c2 by their sum);
          * add measurements that inform the direction;
          * fix the offending parameter at a known value;
          * or keep it in the model but stop asking for its precision --
            move it to the nuisance set and use ds_opt_criterion.

        Parameters
        ----------
        rtol : float
            Relative eigenvalue cutoff. An eigenvalue below
            rtol * max(eigenvalue) counts as null.
        participation_tol : float
            A parameter counts as participating in a null direction when the
            magnitude of its component in that eigenvector exceeds this.
        report : bool
            Print the diagnosis table.

        Returns
        -------
        dict with keys:
            singular          : bool
            rank, n_mp        : int
            eigenvalues       : (n_mp,) ascending
            eigenvectors      : (n_mp, n_mp) columns matching eigenvalues
            null_indices      : indices of the null eigenvalues
            culprits          : parameter NAMES participating in a null direction
            culprit_indices   : their positions
            directions        : list of (eigenvalue, [names]) per null direction
        """
        if self.atomic_fims is None:
            raise SyntaxError(
                "diagnose_fim_structure() needs the atomic FIMs. Call "
                "eval_fim() (or design_experiment(), or eval_sensitivities() "
                "followed by eval_fim()) first."
            )
        A = np.asarray(self.atomic_fims, dtype=float)
        if A.ndim == 4:                       # pseudo-Bayesian: average scenarios
            A = A.mean(axis=0)
        fim_max = A.sum(axis=0)
        if self._prior_fim is not None:
            fim_max = fim_max + np.asarray(self._prior_fim, dtype=float)
        if getattr(self, "_regularize_fim", False):
            fim_max = fim_max + self._eps * np.identity(fim_max.shape[0])
        fim_max = 0.5 * (fim_max + fim_max.T)

        w, V = np.linalg.eigh(fim_max)
        scale = max(1.0, float(abs(w).max()))
        null_idx = [j for j in range(len(w)) if w[j] <= rtol * scale]

        names = (list(self.model_parameter_names)
                 if self.model_parameter_names is not None
                 else [f"parameter {i}" for i in range(len(w))])

        directions, culprit_idx = [], set()
        for j in null_idx:
            idx = [i for i in range(len(w)) if abs(V[i, j]) > participation_tol]
            culprit_idx |= set(idx)
            directions.append((float(w[j]), [names[i] for i in idx]))

        out = {
            "singular": bool(null_idx),
            "rank": int(len(w) - len(null_idx)),
            "n_mp": int(len(w)),
            "eigenvalues": w,
            "eigenvectors": V,
            "null_indices": null_idx,
            "culprits": [names[i] for i in sorted(culprit_idx)],
            "culprit_indices": sorted(culprit_idx),
            "directions": directions,
        }

        if report:
            print("\n" + "-" * 100)
            print("  FIM Structural Diagnosis".center(100))
            print("-" * 100)
            print(f"  Evaluated at the fully-supported design (every candidate "
                  f"at full effort) -- the most")
            print(f"  informative matrix attainable, so any deficiency here is "
                  f"structural.")
            print(f"  Rank            : {out['rank']} of {out['n_mp']}")
            print(f"  cond(FIM_max)   : {np.linalg.cond(fim_max):.3e}")
            print(f"  Null cutoff     : eigenvalue <= {rtol:.0e} x "
                  f"{scale:.3e} = {rtol*scale:.3e}")
            print()
            hdr = f"    {'eigenvalue':>13}  " + "  ".join(f"{n[:11]:>11}" for n in names)
            print(hdr)
            for j in range(len(w)):
                row = "  ".join(f"{abs(V[i, j]):>11.4f}" for i in range(len(w)))
                tag = "   <-- NULL" if j in null_idx else ""
                print(f"    {w[j]:>13.3e}  {row}{tag}")
            print()
            if out["singular"]:
                for ev, who in out["directions"]:
                    print(f"  Unidentifiable direction (eigenvalue {ev:.3e}) "
                          f"involves: {who}")
                print(f"\n  Parameters implicated : {out['culprits']}")
                print("  It is the COMBINATION that is unresolved -- the data may "
                      "still determine a")
                print("  function of these parameters (a sum or ratio) while "
                      "leaving each one free.")
                print()
                print("  Options:")
                print("    * reparameterise so the unidentifiable combination "
                      "becomes a single parameter")
                print("    * add measurements that inform this direction")
                print("    * fix the offending parameter(s) at known values")
                print("    * keep them in the model but stop designing for their "
                      "precision: move them")
                print("      to the nuisance set and use ds_opt_criterion "
                      "(designer.interest_parameters)")
            else:
                print("  FIM is full rank at the fully-supported design -- no "
                      "structural deficiency.")
            print("-" * 100 + "\n")
        return out

    def diagnose_sensitivity(self, tol_diag=1.0, tol_cond=1e4, plot=False,
                             figsize=None, write=False, dpi=360,
                             report=True):
        """
        Diagnose rank-deficiency and near-zero sensitivity in the candidate grid
        using scale-free, physically motivated thresholds.

        Background
        ----------
        pydex normalises every sensitivity by the nominal parameter value::

            s_norm[c, t, r, j] = (∂y_r / ∂θ_j) · θ_j

        This makes sensitivities dimensionless — they represent the fractional
        change in response per fractional change in parameter (local elasticity).
        The atomic FIM diagonal is therefore also dimensionless::

            A_k[j, j] = Σ_{t,r}  s_norm[k,t,r,j]² / σ_r²

        Its inverse is the Cramér–Rao lower bound on the variance of θ_j / θ_j
        (relative variance) from a **single** experiment at candidate k.
        This gives a natural, grid-independent threshold:

        - ``A_k[j,j] < 1`` — one experiment cannot determine θⱼ to within its
          own magnitude; you need at least ``1/A_k[j,j]`` experiments at this
          candidate just to get a signal-to-noise ratio of 1 for θⱼ.
        - ``A_k[j,j] < tol_diag`` (default 1.0) — flags the above condition.

        Unlike a relative-norm threshold (which depends on which other candidates
        are in the grid), this criterion is entirely self-contained: it only
        depends on the model physics, the measurement noise, and the nominal
        parameters.

        Two quantities are computed per candidate
        ------------------------------------------
        1. **Atomic FIM diagonal** ``diag_A[c, j] = A_k[j, j]``
           — Fisher information for θⱼ from one experiment at candidate c.
           Flagged when below ``tol_diag``.

        2. **Condition number** of the full atomic FIM ``A_k``
           — ratio of largest to smallest eigenvalue.  A large condition number
           (even when no diagonal entry is near zero) means two or more parameters
           are nearly collinear at this candidate: allocating many experiments
           there still leaves a linear combination of parameters poorly determined.
           Flagged when above ``tol_cond``.

        The singular values of each ``A_k`` are also returned so users can inspect
        the full spectrum and identify which parameter directions are unobservable.

        Parameters
        ----------
        tol_diag : float
            Threshold for flagging a near-zero atomic FIM diagonal entry.
            ``A_k[j,j] < tol_diag`` → flag parameter θⱼ as unobservable at
            candidate k.  Default: 1.0 (one experiment cannot determine θⱼ to
            within its own magnitude).  Increase to be stricter (e.g. 10 means
            the single-experiment SNR must be at least √10 ≈ 3).

        tol_cond : float
            Condition number threshold above which a candidate is flagged as
            ill-conditioned.  Default: 1e4.

        plot : bool
            If True, produce two figures.  Default: False.
              - Heatmap of ``log10(A_k[j,j])`` (candidates × parameters),
                with ``tol_diag`` threshold line and flagged cells marked.
              - Bar chart of per-candidate condition numbers.

            Opt-in rather than automatic, because on a collinear model the
            second figure carries no information: every candidate exceeds
            ``tol_cond`` and the whole chart is red.  The heatmap can still be
            worth a look — a candidate red across ALL parameters is an
            experiment that informs nothing and could be dropped from the grid,
            which is the one thing this diagnostic sees that the pooled
            :meth:`run_estimability` cannot.

        report : bool
            Print the per-candidate table and summary.  Default: True.
            Set False to keep only the returned dict — the table is one row per
            candidate, which on a large grid is a great deal of low-density
            output.

        figsize : tuple or None
            Figure size.  None uses automatic sizing.

        write : bool
            Save figures to the result directory.

        dpi : int
            DPI for saved figures.

        Returns
        -------
        dict with keys:
            ``"diag_A"``        : np.ndarray (n_c, n_mp) — atomic FIM diagonal
            ``"cond"``          : np.ndarray (n_c,)      — condition numbers
            ``"singular_vals"`` : list of np.ndarray     — eigenvalues of A_k per candidate
            ``"flagged_diag"``  : list of (cand_idx, param_idx) — below tol_diag
            ``"flagged_cond"``  : list of int             — above tol_cond
            ``"param_names"``   : list of str
            ``"candidate_names"``: list of str
            ``"figs"``          : list of matplotlib Figure

        Raises
        ------
        RuntimeError
            If ``eval_sensitivities()`` has not been called yet.

        Examples
        --------
        >>> d.eval_sensitivities()
        >>> result = d.diagnose_sensitivity(tol_diag=1.0, tol_cond=1e4)
        >>> # result["flagged_diag"] — (candidate, parameter) pairs: one experiment
        >>> #   cannot determine that parameter to within its own magnitude here.
        >>> # result["flagged_cond"] — candidates where parameters are collinear.
        >>> # result["figs"] is empty unless plot=True is passed:
        >>> result = d.diagnose_sensitivity(tol_diag=1.0, plot=True)
        """
        if self.sensitivities is None:
            raise RuntimeError(
                "Sensitivities have not been computed yet. "
                "Call eval_sensitivities() first."
            )

        sens = self.sensitivities   # (n_c, n_spt, n_m_r, n_mp)

        # --- names ---
        param_names = (
            list(self.model_parameter_names)
            if self.model_parameter_names is not None
            else [f"θ_{j}" for j in range(self.n_mp)]
        )
        cand_names = (
            [str(cn) for cn in self.candidate_names]
            if self.candidate_names is not None
            else [f"C{c+1}" for c in range(self.n_c)]
        )

        # --- error FIM ---
        err_fim = self.error_fim if self.error_fim is not None else np.eye(self.n_m_r)

        # --- measurable responses only ---
        sens_m = sens[:, :, self.measurable_responses, :]  # (n_c, n_spt, n_m_r, n_mp)

        # --- per-candidate atomic FIM, diagonal, condition number, eigenvalues ---
        diag_A       = np.zeros((self.n_c, self.n_mp))
        cond_numbers = np.zeros(self.n_c)
        singular_vals = []

        for c in range(self.n_c):
            # sens_m[c] shape: (n_spt, n_m_r, n_mp)
            # Accumulate A_c = Σ_t  S_t.T @ err_fim @ S_t  (sum over time points)
            # This is equivalent to S_flat.T @ block_diag(err_fim,...) @ S_flat
            # but avoids building the large block-diagonal matrix explicitly.
            A_c = np.zeros((self.n_mp, self.n_mp))
            for t in range(sens_m.shape[1]):
                S_t = sens_m[c, t]            # (n_m_r, n_mp)
                A_c += S_t.T @ err_fim @ S_t  # (n_mp, n_mp)
            diag_A[c] = np.diag(A_c)

            # eigenvalues (symmetric matrix — use eigvalsh for stability)
            ev = np.linalg.eigvalsh(A_c)               # ascending order
            ev_pos = ev[ev > 0]
            cond_numbers[c] = (ev_pos[-1] / ev_pos[0]) if len(ev_pos) >= 2 else np.inf
            singular_vals.append(ev[::-1])             # store descending

        # --- flags ---
        flagged_diag = [
            (c, j)
            for c in range(self.n_c)
            for j in range(self.n_mp)
            if diag_A[c, j] < tol_diag
        ]
        flagged_cond = [c for c in range(self.n_c) if cond_numbers[c] > tol_cond]

        # --- print report ---
        # Gated on `report`. The per-candidate table is n_c rows wide by n_mp
        # columns; on a 60-candidate 9-parameter model that is 60 lines of
        # low-density output that buries whatever else the script prints. The
        # counts and the returned dict carry the same information.
        if report:
            # Build the header FIRST so the rules can be derived from its true
            # width. A hardcoded width is wrong in both directions: this table
            # is 2 + 20 + n_mp*(pcw+2) + 22 characters wide, which is 158 for a
            # 6-parameter model with long parameter names (rule too short) and
            # about 82 for a 2-parameter model (rule too long).
            pcw = max(10, max(len(p) for p in param_names))
            header = f"  {'Candidate':<20}"
            for p in param_names:
                header += f"  {p:>{pcw}}"
            header += f"  {'Cond#':>12}  Status"
            sub_header = (f"  {'':20}  "
                          + "  ".join(f"{'A_k[j,j]':>{pcw}}" for _ in param_names)
                          + f"  {'':>12}")
            tbl_w = len(header)

            sep = "─" * tbl_w
            print(f"\n{' Sensitivity Diagnosis ':─^{tbl_w}}")
            print(f"  Candidates         : {self.n_c}")
            print(f"  Parameters         : {self.n_mp}")
            print(f"  tol_diag           : {tol_diag:.1g}"
                  f"  (flag A_k[j,j] < {tol_diag:.1g}  ← {tol_diag:.1g} experiment(s) needed"
                  f" for SNR≥1 on θⱼ)")
            print(f"  tol_cond           : {tol_cond:.1g}")
            print(f"{sep}")

            print(header)
            print(sub_header)
            print(sep)

            for c in range(self.n_c):
                row    = f"  {cand_names[c]:<20}"
                issues = []
                for j in range(self.n_mp):
                    val = diag_A[c, j]
                    s   = f"{val:>{pcw}.3f}"
                    if val < tol_diag:
                        s = f"{'!'+f'{val:.1e}':>{pcw}}"
                        issues.append(f"{param_names[j]}")
                    row += f"  {s}"
                cn     = cond_numbers[c]
                cn_str = f"{cn:>12.2e}" if np.isfinite(cn) else f"{'∞':>12}"
                if cn > tol_cond:
                    cn_str = f"{'!'+f'{cn:.1e}':>12}"
                    issues.append("ill-cond")
                row += f"  {cn_str}"
                if issues:
                    row += f"  ⚠ {', '.join(issues)}"
                print(row)

            print(sep)
            print(f"\n  Summary:")
            print(f"    Near-zero diagonal flags  : {len(flagged_diag)} "
                  f"(candidate, parameter) pairs")
            if flagged_diag:
                for c, j in flagged_diag[:10]:
                    print(f"      {cand_names[c]:<22}  {param_names[j]:<20}"
                          f"  A_k[j,j] = {diag_A[c,j]:.2e}"
                          f"  → need ≥{1/max(diag_A[c,j],1e-30):.1f} experiments here for SNR≥1")
                if len(flagged_diag) > 10:
                    print(f"      ... and {len(flagged_diag)-10} more")
            print(f"    Ill-conditioned candidates : {len(flagged_cond)}")
            if flagged_cond:
                for c in flagged_cond[:10]:
                    print(f"      {cand_names[c]:<22}  cond = {cond_numbers[c]:.2e}")
                if len(flagged_cond) > 10:
                    print(f"      ... and {len(flagged_cond)-10} more")
            print(f"{sep}\n")

        # --- plots ---
        # figs must be initialised OUTSIDE the report block: it is returned
        # unconditionally, and gating its creation on `report` made
        # diagnose_sensitivity(report=False) raise UnboundLocalError.
        figs = []
        if plot:
            if figsize is None:
                figsize = (max(8, self.n_mp * 1.4), max(4, self.n_c * 0.32))

            log_diag = np.log10(np.clip(diag_A, 1e-30, None))
            log_tol  = np.log10(tol_diag)

            # heatmap of log10(A_k[j,j])
            fig1, ax1 = plt.subplots(figsize=figsize)
            vmin = min(log_tol - 2, log_diag.min())
            vmax = max(log_tol + 2, log_diag.max())
            im = ax1.imshow(
                log_diag, aspect='auto', cmap='RdYlGn',
                vmin=vmin, vmax=vmax, interpolation='nearest',
            )
            cb = plt.colorbar(im, ax=ax1)
            cb.set_label('log₁₀(A_k[j,j])  — Fisher info per experiment')
            cb.ax.axhline(log_tol, color='black', lw=1.5, ls='--')
            cb.ax.text(1.05, (log_tol - vmin) / (vmax - vmin),
                       f'tol={tol_diag:.0g}', transform=cb.ax.transAxes,
                       va='center', fontsize=7)
            ax1.set_xticks(range(self.n_mp))
            ax1.set_xticklabels(param_names, rotation=30, ha='right', fontsize=8)
            ax1.set_yticks(range(self.n_c))
            ax1.set_yticklabels(cand_names, fontsize=7)
            ax1.set_title(
                "Atomic FIM diagonal  —  A_k[j,j] = Fisher info for θⱼ per experiment\n"
                f"(green = informative, red = near-zero, threshold = {tol_diag:.0g})"
            )
            for c, j in flagged_diag:
                ax1.text(j, c, '!', ha='center', va='center',
                         color='black', fontsize=8, fontweight='bold')
            _safe_tight_layout(fig1)
            figs.append(fig1)

            # bar chart of condition numbers
            fig2, ax2 = plt.subplots(figsize=(max(8, self.n_c * 0.25), 4))
            colors = ['#d62728' if cond_numbers[c] > tol_cond else '#2ca02c'
                      for c in range(self.n_c)]
            cn_plot = np.where(np.isfinite(cond_numbers), cond_numbers, 1e15)
            ax2.bar(range(self.n_c), np.log10(cn_plot + 1), color=colors)
            ax2.axhline(np.log10(tol_cond),
                        color='orange', ls='--',
                        label=f'threshold = 10^{np.log10(tol_cond):.0f}')
            ax2.set_xticks(range(self.n_c))
            ax2.set_xticklabels(cand_names, rotation=90, fontsize=6)
            ax2.set_ylabel('log₁₀(condition number of A_k)')
            ax2.set_title(
                'Per-candidate condition number  —  A_k = Sₖᵀ Σ⁻¹ Sₖ\n'
                '(red = ill-conditioned, parameters are collinear at this candidate)'
            )
            ax2.legend(fontsize=8)
            _safe_tight_layout(fig2)
            figs.append(fig2)

            if write:
                fp1 = self._generate_result_path("sensitivity_diag_heatmap", "png")
                fp2 = self._generate_result_path("sensitivity_condition",    "png")
                fig1.savefig(fp1, dpi=dpi)
                fig2.savefig(fp2, dpi=dpi)

        return {
            "diag_A"         : diag_A,
            "cond"           : cond_numbers,
            "singular_vals"  : singular_vals,
            "flagged_diag"   : flagged_diag,
            "flagged_cond"   : flagged_cond,
            "param_names"    : param_names,
            "candidate_names": cand_names,
            "figs"           : figs,
        }


    def eval_fim(self, efforts, store_predictions=True):
        """
        Construct the FIM from sensitivities. See diagnose_sensitivity() for
        per-candidate rank and condition diagnostics.
        """
        if self._pseudo_bayesian:
            self._eval_pb_fims(
                efforts=efforts,
                store_predictions=store_predictions,
            )
            return self.scr_fims
        else:
            self._eval_fim(
                efforts=efforts,
                store_predictions=store_predictions,
            )
            return self.fim

    def _eval_fim(self, efforts, store_predictions=True, save_atomics=None,
                  skip_sens_eval=False):
        """
        skip_sens_eval : bool
            When True, skip the eval_sensitivities() call and use whatever is
            already stored in self.sensitivities.  Used by the parallel
            pseudo-Bayesian path in _eval_pb_fims() which pre-computes all
            sensitivities in one flat parallel job and injects them directly.
        """
        if save_atomics is not None:
            self._save_atomics = save_atomics

        def add_candidates(s_in, e_in, error_info_mat):
            if not np.any(np.isnan(s_in)):
                _atom_fim = s_in.T @ error_info_mat @ s_in
                self.fim += e_in * _atom_fim
            else:
                _atom_fim = np.zeros((self.n_mp, self.n_mp))
            if not self._large_memory_requirement:
                if self.atomic_fims is None:
                    self.atomic_fims = []
                if self._compute_atomics:
                    self.atomic_fims.append(_atom_fim)

        """ update efforts """
        self.efforts = efforts

        """ eval_sensitivities, only runs if model parameters changed """
        self._compute_sensitivities = self._model_parameters_changed
        self._compute_sensitivities = self._compute_sensitivities or self._candidates_changed
        self._compute_sensitivities = self._compute_sensitivities or self.sensitivities is None

        self._compute_atomics = self._model_parameters_changed
        self._compute_atomics = self._compute_atomics or self._candidates_changed
        self._compute_atomics = self._compute_atomics or self.atomic_fims is None

        # Invalidate cached atomic FIMs when the number of effort variables has
        # changed since they were last built.  This happens when design_experiment
        # is called with a different n_spt argument (e.g. n_spt=1 then n_spt=2):
        # n_spt_comb = C(n_spt_candidates, n_spt) changes, so the expected number
        # of atomics is n_c * n_spt_comb — different from the cached n_c * old_n_spt_comb.
        # Without this guard, _solve_pyomo indexes A[i,j,k] up to n_e-1 = n_c*n_spt_comb-1
        # while A only has n_c*old_n_spt_comb entries, causing an IndexError.
        if not self._compute_atomics and self.atomic_fims is not None:
            expected_n_atomics = (
                self.n_c * self.n_spt_comb
                if self._specified_n_spt
                else self.n_c * self.n_spt
            )
            if len(self.atomic_fims) != expected_n_atomics:
                self._compute_atomics = True

        if self._pseudo_bayesian:
            self._compute_sensitivities = self._compute_atomics or self.scr_fims is None

        if self._compute_sensitivities and self._compute_atomics and not skip_sens_eval:
            self.eval_sensitivities(
                save_sensitivities=self._save_sensitivities,
                store_predictions=store_predictions,
            )

        """ evaluate fim """
        start = time()

        # reshape efforts to (n_c, n_spt) for iteration
        if self._specified_n_spt:
            self.efforts = self.efforts.reshape((self.n_c, self.n_spt_comb))
        else:
            self.efforts = self.efforts.reshape((self.n_c, self.n_spt))
            if self.n_spt == 1:
                self.efforts = self.efforts[:, None]
        # if atomic is not given
        if self._compute_atomics:
            self.atomic_fims = []
            self.fim = 0
            if self._specified_n_spt:
                for c, (eff, sen, spt_combs) in enumerate(zip(self.efforts, self.sensitivities, self.spt_candidates_combs)):
                    for comb, (e, spt) in enumerate(zip(eff, spt_combs)):
                        s = np.mean(sen[spt], axis=0)
                        add_candidates(s, e, self.error_fim)
            else:
                for c, (eff, sen) in enumerate(zip(self.efforts, self.sensitivities)):
                    for spt, (e, s) in enumerate(zip(eff, sen)):
                        add_candidates(s, e, self.error_fim)
            if self._save_atomics and not self._pseudo_bayesian:
                sens_file = f"atomics_{self.n_c}_cand"
                if self._dynamic_system:
                    sens_file += f"_{self.n_spt}_spt"
                if self._pseudo_bayesian:
                    sens_file += f"_{self.n_scr}_scr"
                if self._candidates_swapped:
                    sens_file += f"_go_{self.n_c_go}_cand"
                fp = self._generate_result_path(sens_file, "pkl")
                dump(self.atomic_fims, open(fp, 'wb'))
        # if atomic is given
        else:
            self.fim = 0
            # Use a local 4-D view for the loop so that self.atomic_fims stays
            # in its flat (n_c*n_spt, n_mp, n_mp) shape.  Overwriting
            # self.atomic_fims here would cause _d_opt_criterion (and others)
            # to iterate over only n_c rows when computing the analytic
            # Jacobian, returning a gradient of length n_c instead of
            # n_c*n_spt and crashing IPOPT's gradient callback.
            atomic_fims_4d = self.atomic_fims.reshape(
                (self.n_c, self.n_spt, self.n_mp, self.n_mp)
            )
            if self._specified_n_spt:
                for c, (eff, atom, spt_combs) in enumerate(
                    zip(self.efforts, atomic_fims_4d, self.spt_candidates_combs)
                ):
                    for comb, (e, spt) in enumerate(zip(eff, spt_combs)):
                        a = np.mean(atom[spt], axis=0)
                        self.fim += e * a
            else:
                for c, (eff, atom) in enumerate(zip(self.efforts, atomic_fims_4d)):
                    for spt, (e, a) in enumerate(zip(eff, atom)):
                        self.fim += e * a

        finish = time()

        if np.all(np.asarray(self.fim) == 0):
            return np.array([0])

        # --- add prior experimental information (sequential MBDoE) ---
        if self._prior_fim is not None:
            prior = self._prior_fim.copy()
            # rescale to current model_parameters if they changed since prior was computed
            if not np.allclose(self._current_scr_mp, self._prior_fim_mp, rtol=1e-10):
                scale = self._current_scr_mp / self._prior_fim_mp   # (n_mp,)
                rescale = np.outer(scale, scale)
                prior = prior * rescale
            self.fim = self.fim + prior

        if self._regularize_fim:
            if self._verbose >= 3:
                print(
                    f"Applying Tikhonov regularization to FIM by adding "
                    f"{self._eps:.2f} * identity to the FIM. "
                    f"Warning: design is likely to be affected for large scalars!"
                )
            self.fim += self._eps * np.identity(self.n_mp)

        self._fim_eval_time = finish - start
        if self._verbose >= 3:
            print(
                f"Evaluation of fim took {self._fim_eval_time:.2f} seconds."
            )

        if not self._large_memory_requirement:
            self.atomic_fims = np.asarray(self.atomic_fims)

        """ set current mp as completed to prevent recomputation of atomics """
        self._model_parameters_changed = False
        self._candidates_changed = False

        return self.fim

    def _eval_pb_fims(self, efforts, store_predictions=True):
        """ only recompute pb_atomics if the full parameter scenarios are changed """
        self._compute_pb_atomics = self._model_parameters_changed
        self._compute_pb_atomics = self._compute_pb_atomics or self._candidates_changed
        self._compute_pb_atomics = self._compute_pb_atomics or self.pb_atomic_fims is None

        self.scr_fims = []
        if self._compute_pb_atomics:
            if self._verbose >= 2:
                print(f"{' Pseudo-bayesian ':#^100}")
            if self._verbose >= 1:
                print(f'Evaluating information for each scenario...')
            if store_predictions:
                self.scr_responses = []
            if not self._large_memory_requirement:
                self.pb_atomic_fims = np.empty((self.n_scr, self.n_c * self.n_spt, self.n_mp, self.n_mp))

            # ── Parallel pseudo-Bayesian path (Pyomo IFT only) ───────────────
            # Parallelise over scenarios using loky (subprocess) workers.
            # Each subprocess handles all n_c candidates for one scenario
            # sequentially — this isolates Pyomo global state (logging,
            # C-extension caches) between workers, eliminating the thread-
            # safety issues that affect prefer="threads".
            # Spawn overhead (~0.3 s per worker) is amortised over n_c
            # candidates per job, so net cost is small.
            _use_pyomo_ift = getattr(self, "use_pyomo_ift", False)
            _n_jobs = getattr(self, "n_jobs", 1)
            if _use_pyomo_ift and _n_jobs != 1:
                try:
                    from joblib import Parallel, delayed
                except ImportError:
                    raise ImportError(
                        "n_jobs != 1 requires joblib. Install with: pip install joblib"
                    )

                pyomo_fn  = self.pyomo_model_fn
                out_names = getattr(self, "pyomo_output_var_name", None)
                n_mr      = self.n_m_r
                tic_list  = self.ti_controls_candidates
                spt_list  = self.sampling_times_candidates
                mp_list   = self.model_parameters   # shape (n_scr, n_mp)
                n_c_      = self.n_c
                n_spt_    = self.n_spt
                n_mp_     = self.n_mp

                def _pb_scenario_worker(scr, mp, tic_list_, spt_list_, out_names_,
                                            n_mr_, n_spt__, n_mp__, norm_sens,
                                            dyn_sys):
                    """Process all candidates for one scenario; runs in a subprocess.

                    mp          : parameter vector for THIS scenario.
                    norm_sens   : whether to apply parameter-value normalization
                                  (mirrors the _norm_sens_by_params step that
                                  eval_sensitivities applies in the sequential path).
                    """
                    import types, numpy as _np
                    mp = _np.asarray(mp, dtype=float)
                    sens_scr = _np.empty((len(tic_list_), n_spt__, n_mr_, n_mp__))
                    for c, tic in enumerate(tic_list_):
                        # _eval_sensitivities_pyomo_ift gates the
                        # sampling_times kwarg on _dynamic_system. That flag was
                        # previously ABSENT from this namespace, so getattr
                        # defaulted it to False and sampling_times was forced to
                        # None regardless of the designer's real setting -- fine
                        # for static models (where the sequential path also
                        # passes None, which is why both agreed) but wrong for
                        # genuinely dynamic ones, where it crashes builders that
                        # accept the kwarg. Propagate the REAL flag: static
                        # models keep their previous behaviour exactly, dynamic
                        # models now receive their sampling times.
                        if dyn_sys and spt_list_ is not None:
                            # Dynamic model: use THIS candidate's sampling
                            # times, mirroring the sequential path
                            # (self._current_spt = spt_arr[k][~nan]).
                            _spt_c = _np.atleast_1d(spt_list_[c]).astype(float)
                            _spt_c = _spt_c[~_np.isnan(_spt_c)]
                        else:
                            # Static model: keep the original expression
                            # verbatim. It reads oddly -- these are the
                            # time-invariant CONTROLS, not times -- but
                            # _current_spt is only used for array shapes and
                            # the time loop here, its length matches n_spt == 1,
                            # and sampling_times_candidates for a static model
                            # holds uninitialised placeholder values that must
                            # NOT be fed to the builder. Changing this branch
                            # makes the parallel path disagree with the
                            # sequential one (pydex test 22 / 27).
                            _spt_c = _np.atleast_1d(tic)
                        fake = types.SimpleNamespace(
                            _current_spt          = _spt_c,
                            _dynamic_system       = dyn_sys,
                            pyomo_model_fn        = pyomo_fn,
                            pyomo_output_var_name = out_names_,
                            n_m_r                 = n_mr_,
                        )
                        _, sens = Designer._eval_sensitivities_pyomo_ift(
                            fake, tic, mp, store_predictions=False
                        )
                        sens_scr[c] = sens
                    # Apply parameter-value normalization (S_ij *= theta_j)
                    # This mirrors the _norm_sens_by_params step in eval_sensitivities
                    # which is bypassed when skip_sens_eval=True.
                    if norm_sens:
                        sens_scr *= mp[_np.newaxis, _np.newaxis, _np.newaxis, :]
                    return scr, sens_scr

                if self._verbose >= 1:
                    print(
                        f"[_eval_pb_fims] Running {self.n_scr} scenario jobs "
                        f"({self.n_scr} scenarios × {self.n_c} candidates) "
                        f"in parallel (n_jobs={_n_jobs}, backend=loky)..."
                    )

                scr_sens = np.empty((self.n_scr, self.n_c, self.n_spt, self.n_m_r, self.n_mp))
                _pb_par_start = time()
                _norm_sens = getattr(self, "_norm_sens_by_params", True)
                _dyn_sys   = bool(getattr(self, "_dynamic_system", False))
                raw = Parallel(n_jobs=_n_jobs, prefer="processes")(
                    delayed(_pb_scenario_worker)(
                        scr, mp_list[scr].copy(), list(tic_list),
                        (list(spt_list) if spt_list is not None else None),
                        out_names, n_mr, n_spt_, n_mp_, _norm_sens, _dyn_sys
                    )
                    for scr in range(self.n_scr)
                )
                self._sensitivity_analysis_time = time() - _pb_par_start
                for scr, sens_scr in raw:
                    scr_sens[scr] = sens_scr

                # Build per-scenario FIMs from the collected sensitivities
                for scr, mp in enumerate(self.model_parameters):
                    self._current_scr     = scr
                    self._current_scr_mp  = mp
                    self.sensitivities    = scr_sens[scr]
                    self.atomic_fims      = None
                    self._eval_fim(efforts, store_predictions,
                                   skip_sens_eval=True)
                    self.scr_fims.append(self.fim)
                    if not self._large_memory_requirement:
                        self.pb_atomic_fims[scr] = self.atomic_fims

            else:
                # ── Sequential scenario loop (original behaviour) ─────────────
                for scr, mp in enumerate(self.model_parameters):
                    self.atomic_fims = None
                    self._current_scr = scr
                    self._current_scr_mp = mp
                    if self._verbose >= 2:
                        print(f"{f'[Scenario {scr+1}/{self.n_scr}]':=^100}")
                        print("Model Parameters:")
                        print(mp)
                    self._eval_fim(efforts, store_predictions)
                    self.scr_fims.append(self.fim)
                    if self._verbose >= 2:
                        print(f"Time elapsed: {self._sensitivity_analysis_time:.2f} seconds.")
                    if store_predictions:
                        self.scr_responses.append(self.response)
                        self.response = None
                    if not self._large_memory_requirement:
                        self.pb_atomic_fims[scr] = self.atomic_fims
            if store_predictions:
                self.scr_responses = np.array(self.scr_responses)

            """ set current mp as completed to prevent recomputation of atomics """
            self._model_parameters_changed = False
        else:
            for scr, atomic_fims in enumerate(self.pb_atomic_fims):
                self.atomic_fims = atomic_fims
                self._eval_fim(efforts, store_predictions)
                self.scr_fims.append(self.fim)

        if self._save_atomics:
            fn = f"atomics_{self.n_c}_can_{self.n_scr}_scr"
            fp = self._generate_result_path(fn, "pkl")
            dump(self.pb_atomic_fims, open(fp, "wb"))

        return self.scr_fims

    def eval_pim(self, efforts, vector=False):

        """ update mp, and efforts """
        self.eval_fim(efforts)

        # Guarded inverse. This was previously an unguarded np.linalg.inv, which
        # raises LinAlgError on an exactly singular FIM and silently returns
        # garbage on a nearly singular one. Setting pvars to None lets the
        # consuming criteria report an infeasible design (+inf) instead, which
        # is what an optimiser can actually act on.
        fim_inv = self._safe_fim_inverse()
        if fim_inv is None:
            self.pvars = None
            return self.pvars
        if vector:
            self.pvars = np.array([
                [f @ fim_inv @ f.T for f in F] for F in self.sensitivities
            ])
        else:
            self.pvars = np.empty((self.n_c, self.n_spt, self.n_r, self.n_r))
            for c, F in enumerate(self.sensitivities):
                for spt, f in enumerate(F):
                    self.pvars[c, spt, :, :] = f @ fim_inv @ f.T

        return self.pvars

    def eval_atom_fims(self, mp, store_predictions=True):
        """Evaluate the atomic FIM of every candidate at given parameter values.

        The atomic FIM is one candidate's information contribution at unit
        effort; the design FIM is their effort-weighted sum. Computing them once
        is what makes the optimisation cheap relative to the sensitivity
        analysis.

        Args:
            mp (numpy.ndarray): Parameter values to evaluate at.
            store_predictions (bool): Keep the predicted responses.
        """
        self._current_scr_mp = mp

        """ eval_sensitivities, only runs if model parameters changed """
        self.eval_sensitivities(save_sensitivities=self._save_sensitivities,
                                store_predictions=store_predictions)

        """ deal with unconstrained form, i.e. transform efforts """
        self._transform_efforts()  # only transform if required, logic incorporated there

        """ deal with opt_sampling_times """
        sens = self.sensitivities.reshape(self.n_c * self.n_spt, self.n_m_r, self.n_mp)

        """ main """
        start = time()
        if self._large_memory_requirement:
            confirmation = input(
                f"Memory requirement is large. Slow solution expected, continue?"
                f"Y/N."
            )
            if confirmation != "Y":
                return
        self.atomic_fims = []
        for e, f in zip(self.efforts.flatten(), sens):
            if not np.any(np.isnan(f)):
                _atom_fim = f.T @ f
            else:
                _atom_fim = np.zeros(shape=(self.n_mp, self.n_mp))
            self.atomic_fims.append(_atom_fim)
        finish = time()
        self._fim_eval_time = finish - start

        return self.atomic_fims

    """ getters (filters) """

    def get_optimal_candidates(self, tol=1e-4):
        """Collect the candidates carrying non-negligible effort.

        Args:
            tol (float): Effort below which a candidate is treated as unused.

        Returns:
            list: One entry per supported candidate, holding its index, control
            values, sampling times, sampling schedules and efforts. Also stored
            on :attr:`optimal_candidates`.
        """
        if self.efforts is None:
            raise SyntaxError(
                'Please solve an experiment design before attempting to get optimal '
                'candidates.'
            )

        self._remove_zero_effort_candidates(tol=tol)
        self.optimal_candidates = []

        for i, eff_sp in enumerate(self.efforts):
            if self._dynamic_system and self._opt_sampling_times:
                optimal = np.any(eff_sp > tol)
            else:
                optimal = np.sum(eff_sp) > tol
            if optimal:
                opt_candidate = [
                    i,  # index of optimal candidate
                    self.ti_controls_candidates[i],
                    self.tv_controls_candidates[i],
                    [],
                    [],
                    [],
                    []
                ]
                if self._opt_sampling_times:
                    for j, eff in enumerate(eff_sp):
                        if eff > tol:
                            if self._specified_n_spt:
                                opt_spt = self.sampling_times_candidates[i, self.spt_candidates_combs[i, j]]
                                opt_candidate[3].append(opt_spt)
                                opt_candidate[4].append(np.ones_like(opt_spt) * eff / len(opt_spt))
                                opt_candidate[5].append(self.spt_candidates_combs[i, j])
                            else:
                                opt_candidate[3].append(self.sampling_times_candidates[i][j])
                                opt_candidate[4].append(eff)
                                opt_candidate[5].append(j)
                else:
                    opt_candidate[3] = self.sampling_times_candidates[i]
                    opt_candidate[4] = eff_sp
                    opt_candidate[5].append([t for t in range(self.n_spt)])
                self.optimal_candidates.append(opt_candidate)

        self.n_opt_c = len(self.optimal_candidates)
        if self.n_opt_c == 0:
            print(
                f"[Warning]: empty optimal candidates. Likely failed optimization; if "
                f"prediction-orriented design is used, try avoiding dg, ag, or eg "
                f"criteria as they are notoriously hard to optimize with gradient-based "
                f"optimizers."
            )

        self.n_factor_sups = 0
        self.n_spt_sups = 0
        self.max_n_opt_spt = 0
        for i, opt_cand in enumerate(self.optimal_candidates):
            if self._dynamic_system and self._opt_sampling_times:
                self.n_factor_sups += len(opt_cand[4])
            else:
                self.n_factor_sups += 1
            self.max_n_opt_spt = max(self.max_n_opt_spt, len(opt_cand[4]))

        return self.optimal_candidates

    """ optional operations """

    def _d_opt_criterion(self, efforts):
        """ D-optimality: maximise log-det(FIM). """
        self.eval_fim(efforts)

        if self.fim.size == 1:
            d_opt = -self.fim
            if self._fd_jac:
                return np.squeeze(d_opt)
            else:
                jac = -np.array([1 / self.fim * m for m in self.atomic_fims])
                return d_opt, jac

        sign, d_opt = np.linalg.slogdet(self.fim)
        if self._fd_jac:
            return -d_opt if sign == 1 else np.inf
        else:
            fim_inv = np.linalg.inv(self.fim)
            jac = -np.array([np.sum(fim_inv.T * m) for m in self.atomic_fims])
            return (-d_opt, jac) if sign == 1 else (np.inf, jac)

    def _b_opt_criterion(self, efforts):
        """
        Numpy evaluator for a GIVEN discrete design -- post-hoc reporting
        only (e.g. compute_criterion_value() on an already-solved design).
        The actual OPTIMIZATION happens in _solve_pyomo_b_opt via the
        Pyomo/solver path; this never calls self.eval_fim() and never
        touches self.atomic_fims / self.fim, by design -- b_opt is not a
        parameter-sensitivity criterion.

        Returns the value to MINIMISE, matching every other criterion's
        sign convention (b_opt maximises the combined log-det, so this
        returns its negative).
        """
        efforts = np.asarray(efforts, dtype=float).flatten()
        selected = np.where(efforts > 1e-6)[0]
        if selected.size < 2:
            return np.inf

        U_raw = np.asarray(self.ti_controls_candidates, dtype=float)
        lb, ub = U_raw.min(axis=0), U_raw.max(axis=0)
        span = np.where(ub > lb, ub - lb, 1.0)
        U = 2.0 * (U_raw - lb) / span - 1.0
        M_in = U[selected].T @ U[selected]
        sign_in, logdet_in = np.linalg.slogdet(M_in)
        if sign_in <= 0:
            return np.inf

        if self.response is None:
            raise RuntimeError(
                "b_opt_criterion needs predicted responses. Call "
                "designer.simulate_candidates() first."
            )
        Y_raw = np.asarray(self.response, dtype=float).reshape(U_raw.shape[0], -1)
        Y_mean, Y_std = Y_raw.mean(axis=0), Y_raw.std(axis=0)
        Y_std = np.where(Y_std > 0, Y_std, 1.0)
        Y_sel = (Y_raw[selected] - Y_mean) / Y_std
        yc = Y_sel.mean(axis=0)
        M_out = (Y_sel - yc).T @ (Y_sel - yc) / max(selected.size - 1, 1)
        sign_out, logdet_out = np.linalg.slogdet(M_out)
        if sign_out <= 0:
            return np.inf

        wout = float(getattr(self, "_b_opt_output_weight", 0.5))
        win = 1.0 - wout
        return -(win * logdet_in + wout * logdet_out)

    def _resolve_ds_idx(self):
        """
        Resolve the user-declared interest-parameter NAMES
        (self.ds_interest_names) into positional indices into the FIM, by
        matching each name against self.model_parameter_names.

        Matching is by exact name, never by position/order: the position of
        a parameter in the FIM follows the order of self.model_parameters as
        supplied to the designer, which is independent of (and not
        guaranteed to track) the order in which a Pyomo model happens to
        declare its equations or variables. Resolving lazily here — rather
        than at `interest_parameters` assignment time — also allows
        interest_parameters to be set before initialize() has populated
        (or defaulted) model_parameter_names.

        Results are cached in self.ds_interest_idx / self.ds_nuisance_idx
        and reused on subsequent calls.
        """
        if self.ds_interest_idx is not None and self.ds_nuisance_idx is not None:
            return self.ds_interest_idx, self.ds_nuisance_idx

        if self.ds_interest_names is None:
            raise SyntaxError(
                "Ds-optimal design requires designer.interest_parameters to "
                "be set to the NAMES of the parameters of interest, e.g. "
                "designer.interest_parameters = ['Ka', 'A0']."
            )
        if self.model_parameter_names is None:
            raise SyntaxError(
                "designer.model_parameter_names is not set, but "
                "interest_parameters matches parameters BY NAME against it. "
                "Assign designer.model_parameter_names (or call "
                "initialize() first, which defaults them) before evaluating "
                "ds_opt_criterion."
            )

        name_list = list(self.model_parameter_names)
        idx_s = []
        for nm in self.ds_interest_names:
            if nm not in name_list:
                raise ValueError(
                    f"interest_parameters: '{nm}' not found in "
                    f"model_parameter_names {name_list}. Names must match "
                    f"exactly."
                )
            idx_s.append(name_list.index(nm))
        idx_s = np.array(sorted(set(idx_s)), dtype=int)
        idx_n = np.array(
            [j for j in range(self.n_mp) if j not in idx_s], dtype=int
        )
        self.ds_interest_idx = idx_s
        self.ds_nuisance_idx = idx_n
        return idx_s, idx_n

    def _ds_opt_criterion(self, efforts):
        """
        Ds-optimality: maximise log-det of the Schur complement of the
        nuisance-parameter block of the FIM, i.e. D-optimal design for a
        SUBSET of model_parameters ("interest" parameters, indices idx_s)
        while marginalising out the remaining ("nuisance") parameters
        (indices idx_n).

        Partitioning the FIM (after conceptually reordering rows/columns so
        interest parameters come first) as

            FIM = [[M_ss, M_sn],
                   [M_ns, M_nn]]

        the Schur complement of the nuisance block M_nn is

            S = M_ss - M_sn @ M_nn^{-1} @ M_ns

        and, by the Schur determinant identity,

            det(S) = det(FIM) / det(M_nn)
            log-det(S) = log-det(FIM) - log-det(M_nn)

        The determinant identity above is NOT used to evaluate the criterion,
        because it is 0/0-indeterminate precisely when an unidentifiable
        nuisance parameter makes both determinants vanish — the very case
        Ds-optimality exists to handle. S is instead formed explicitly (it is
        only n_s x n_s) via a least-squares nuisance solve; see
        _ds_eval_schur() for the full numerical rationale. When there are no
        nuisance parameters (idx_n is empty), Ds-optimality collapses exactly
        to D-optimality and is delegated to _d_opt_criterion.

        The analytic (non finite-difference) Jacobian uses S = Pᵀ·FIM·P with
        P = [[I_s], [-M_nn^{-1} M_ns]], giving

            d/de_i log-det(S) = trace(S^{-1} Pᵀ A_i P)

        for the i-th atomic FIM A_i (the FIM is linear in the efforts). This
        form needs only S^{-1}, so it remains valid when M_nn is singular,
        unlike the identity-based gradient which needs both FIM^{-1} and
        M_nn^{-1}.
        """
        idx_s, idx_n = self._resolve_ds_idx()
        self.eval_fim(efforts)

        if len(idx_n) == 0:
            # no nuisance parameters: Ds-optimality reduces to D-optimality
            return self._d_opt_criterion(efforts)

        n_grad = self._n_grad(efforts)
        logdet_S, P, S_inv, info = self._ds_eval_schur(
            self.fim, idx_s, idx_n, want_grad=not self._fd_jac
        )

        if logdet_S is None:
            return self._ds_infeasible(n_grad, info)

        if self._fd_jac:
            return -logdet_S

        atoms = self.atomic_fims
        if atoms is None:
            raise RuntimeError(
                "Analytic Jacobian for ds_opt_criterion requires atomic FIMs, "
                "but self.atomic_fims is None (large-memory mode). Set "
                "designer._fd_jac = True, or reduce the problem size."
            )
        # d/de_i log-det(S) = trace(S^{-1} Pᵀ A_i P), exact because the FIM is
        # linear in the efforts and the P-dependence contributes nothing (the
        # [S 0] structure of Pᵀ·FIM makes the envelope/Danskin term vanish).
        jac = -np.array([
            np.sum(S_inv.T * (P.T @ np.asarray(a) @ P)) for a in atoms
        ])
        return -logdet_S, jac

    """ Ds-optimality numerical kernel """

    def _n_grad(self, efforts):
        """
        Length the criterion gradient must have. Prefer the number of atomic
        FIMs (the authoritative count of effort variables); fall back to the
        size of the effort vector when atomics are unavailable, so that an
        infeasible return still has the right shape instead of crashing.

        Used by the Ds and A criteria. Note self.n_e is NOT usable for this:
        it is initialised to None and never assigned anywhere in the class.
        """
        if self.atomic_fims is not None:
            try:
                return len(self.atomic_fims)
            except TypeError:
                pass
        return int(np.asarray(efforts).size)

    def _ds_infeasible(self, n_grad, info=None):
        """ Uniform infeasible return, shaped for the active gradient mode. """
        if info is not None and self._verbose >= 2:
            reason = info.get("reason", "unspecified")
            print(f"[ds_opt_criterion] infeasible FIM ({reason}).")
        if self._fd_jac:
            return np.inf
        return np.inf, np.zeros(n_grad)

    def _prepare_fim(self, fim):
        """
        Coerce a candidate FIM into a well-formed (n_mp, n_mp) finite array,
        or return None if it is degenerate. Shared by the Ds and A criteria.

        This guards several shapes that _eval_fim can legitimately produce and
        that would otherwise crash the sub-block extraction below:
          * python int 0 — self.fim is initialised to the scalar 0 and is only
            incremented for candidates with non-NaN sensitivities, so a grid
            whose sensitivities are all NaN leaves it a scalar;
          * np.array([0]) — the 1-D sentinel returned on an all-zero FIM;
          * an all-zero (n_mp, n_mp) array — every effort driven to zero;
          * non-finite entries from a diverged simulation.
        """
        if fim is None:
            return None
        fim = np.asarray(fim, dtype=float)
        if fim.ndim != 2 or fim.shape != (self.n_mp, self.n_mp):
            return None
        if not np.all(np.isfinite(fim)):
            return None
        if np.all(fim == 0.0):
            return None
        return fim

    def _ds_eval_schur(self, fim, idx_s, idx_n, want_grad=False):
        """
        Evaluate log-det of the Schur complement of the nuisance block.

        Returns (logdet_S, P, S_inv, info). logdet_S is None when Ds is
        infeasible at this FIM; info carries diagnostics.

        Why this does NOT use the determinant identity
        ----------------------------------------------
        log-det(S) = log-det(FIM) - log-det(M_nn) is algebraically correct but
        numerically fragile in exactly the case Ds-optimality exists to serve.
        If a nuisance parameter is wholly unidentified by the data, BOTH
        determinants vanish and the identity becomes 0/0 — reported as
        infeasible — even though S itself remains finite and positive definite.
        That is the single most valuable Ds use case (design for the parameters
        you care about while an uninteresting parameter stays unidentifiable),
        so it must not be rejected.

        Conditioning note
        -----------------
        The nuisance block is never the conditioning bottleneck. For a PSD FIM,
        Cauchy eigenvalue interlacing gives
            lambda_min(FIM) <= lambda_min(M_nn) <= lambda_max(M_nn) <= lambda_max(FIM)
        hence cond(M_nn) <= cond(FIM): a principal submatrix of a PSD matrix is
        PSD and no worse conditioned than its parent. So M_nn singular IMPLIES
        FIM singular, but not conversely. The quantity whose conditioning
        actually governs Ds is S, which is why S is formed explicitly (it is
        only n_s x n_s, so this is cheap) and tested directly.

        Solving for W = M_nn^+ M_ns by least squares yields the minimum-norm
        solution when M_nn is rank-deficient, which is precisely the limiting
        (generalised) Schur complement — valid whenever range(M_ns) is
        contained in range(M_nn). That containment is automatic for any PSD
        FIM: if M_nn v = 0 then v'M_nn v = 0, and PSD-ness of the FIM forces
        FIM[0; v] = 0, hence M_sn v = 0. A non-negligible least-squares
        residual therefore indicates a NON-PSD FIM, for which S genuinely
        diverges — correctly reported as infeasible.
        """
        info = {"reason": None, "nuisance_rank": None,
                "nuisance_rank_deficient": False, "cond_S": np.inf,
                "residual": 0.0}

        fim = self._prepare_fim(fim)
        if fim is None:
            info["reason"] = "FIM absent, wrong shape, all-zero, or non-finite"
            return None, None, None, info

        # Cheap sanity check on the FULL FIM. It is PSD by construction
        # (Σᵢ eᵢ·SᵢᵀΣ⁻¹Sᵢ with eᵢ >= 0), so an indefinite FIM means something is
        # wrong upstream — almost always a user-supplied _prior_fim that is not
        # itself PSD. Worth surfacing loudly, because the Schur complement can
        # still come out positive definite in that situation and would
        # otherwise hand back a plausible-looking number built on an invalid
        # information matrix. Not treated as fatal: S being PD is what the
        # criterion actually requires.
        _eig_fim = np.linalg.eigvalsh(0.5 * (fim + fim.T))
        _psd_tol = -max(1.0, float(np.abs(_eig_fim).max())) * 1e-10
        info["fim_not_psd"] = bool(_eig_fim.min() < _psd_tol)
        if info["fim_not_psd"] and self._verbose >= 1:
            if "fim_psd" not in self._ds_warned:
                self._ds_warned.add("fim_psd")
                print(
                    f"[WARNING][ds_opt_criterion] the FIM is not positive "
                    f"semi-definite (lambda_min = {_eig_fim.min():.3e}). A FIM "
                    f"assembled from sensitivities is PSD by construction, so "
                    f"this usually means designer._prior_fim is not PSD. The "
                    f"Ds value below is computed from the Schur complement and "
                    f"may be meaningless."
                )

        m_ss = fim[np.ix_(idx_s, idx_s)]
        m_sn = fim[np.ix_(idx_s, idx_n)]
        m_ns = fim[np.ix_(idx_n, idx_s)]
        m_nn = fim[np.ix_(idx_n, idx_n)]

        # W = M_nn^+ M_ns via least squares (handles singular M_nn).
        try:
            W, _res, rank, svals = np.linalg.lstsq(
                m_nn, m_ns, rcond=self._ds_rcond
            )
        except np.linalg.LinAlgError:
            info["reason"] = "nuisance-block least-squares solve failed (SVD)"
            return None, None, None, info

        info["nuisance_rank"] = int(rank)
        info["nuisance_rank_deficient"] = bool(rank < len(idx_n))

        # Inconsistent solve => non-PSD FIM => S diverges.
        resid = float(np.linalg.norm(m_nn @ W - m_ns))
        scale = max(1.0, float(np.linalg.norm(m_ns)))
        info["residual"] = resid / scale
        if info["residual"] > self._ds_resid_tol:
            info["reason"] = (
                f"nuisance solve inconsistent (rel. residual "
                f"{info['residual']:.2e}) — FIM is not positive semi-definite, "
                f"so the Schur complement diverges"
            )
            return None, None, None, info

        S = m_ss - m_sn @ W
        S = 0.5 * (S + S.T)     # re-symmetrise against round-off asymmetry

        # Genuine positive-definiteness test. A determinant-sign check is NOT
        # sufficient: det > 0 only requires an EVEN number of negative
        # eigenvalues, so an indefinite matrix (e.g. diag(1, 1, -1, -1), which
        # a rescaled user-supplied prior FIM can produce) passes a sign test
        # while being meaningless as an information matrix. Cholesky is the
        # correct test and simultaneously gives a stable log-det.
        try:
            chol_S = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            info["reason"] = "Schur complement not positive definite"
            return None, None, None, info

        diag_S = np.diag(chol_S)
        if not np.all(diag_S > 0) or not np.all(np.isfinite(diag_S)):
            info["reason"] = "Schur complement Cholesky degenerate"
            return None, None, None, info

        logdet_S = float(2.0 * np.sum(np.log(diag_S)))
        if not np.isfinite(logdet_S):
            info["reason"] = "log-det(S) non-finite"
            return None, None, None, info

        # Conditioning diagnostics (S is small, so this is cheap).
        eig_S = np.linalg.eigvalsh(S)
        info["cond_S"] = (
            float(eig_S.max() / eig_S.min()) if eig_S.min() > 0 else np.inf
        )
        if self._verbose >= 1 and info["cond_S"] > self._ds_cond_warn:
            key = "cond_S"
            if key not in self._ds_warned:
                self._ds_warned.add(key)
                print(
                    f"[WARNING][ds_opt_criterion] Schur complement is "
                    f"ill-conditioned (cond = {info['cond_S']:.2e}). The "
                    f"interest parameters are close to collinear after "
                    f"marginalising the nuisance parameters; the Ds design may "
                    f"be numerically unreliable. Consider reducing "
                    f"interest_parameters, or enabling regularize_fim."
                )
        if (self._verbose >= 2 and info["nuisance_rank_deficient"]
                and "nuis_rank" not in self._ds_warned):
            self._ds_warned.add("nuis_rank")
            print(
                f"[INFO][ds_opt_criterion] nuisance block is rank-deficient "
                f"(rank {info['nuisance_rank']}/{len(idx_n)}). Using the "
                f"generalised (minimum-norm) Schur complement — this is well "
                f"defined and is a normal situation for Ds-optimal design."
            )

        P, S_inv = None, None
        if want_grad:
            P = np.zeros((self.n_mp, len(idx_s)))
            P[idx_s, :] = np.eye(len(idx_s))
            P[idx_n, :] = -W
            try:
                from scipy.linalg import cho_solve
                S_inv = cho_solve((chol_S, True), np.eye(len(idx_s)))
            except Exception:
                S_inv = np.linalg.inv(S)
            S_inv = 0.5 * (S_inv + S_inv.T)

        return logdet_S, P, S_inv, info

    def _a_opt_criterion(self, efforts):
        """
        A-optimality: minimise trace(FIM^{-1}), i.e. the total (summed)
        variance of the parameter estimates.

        Infeasibility convention
        ------------------------
        A-optimality is MINIMISED, and trace(FIM^{-1}) diverges to +inf as the
        FIM approaches singularity. An unusable FIM must therefore return +inf.
        Earlier revisions returned 0, which is the BEST attainable value and
        made a singular FIM look like a perfect design -- actively attracting
        the optimiser toward rank-deficient supports. The sign of that error
        matters more than its magnitude, so it is called out here explicitly.

        Positive-definiteness is tested by eigenvalue in BOTH the
        finite-difference and analytic branches. Testing only for
        LinAlgError (as the analytic branch previously did) is insufficient:
        an indefinite matrix such as diag(1, -1) inverts cleanly and yields a
        NEGATIVE trace, which to a minimiser looks better than any feasible
        design. It also made the two branches disagree about feasibility, so
        flipping _fd_jac silently changed the answer.
        """
        self.eval_fim(efforts)

        n_grad = self._n_grad(efforts)
        fim_raw = self.fim

        # Single-parameter case, checked BEFORE _prepare_fim because the
        # original implementation short-circuited on fim.size == 1 without
        # consulting n_mp; preserving that ordering keeps genuine
        # single-parameter studies bit-identical. Kept as the historical
        # monotone-equivalent surrogate -maximise(FIM) rather than
        # minimise(1/FIM): both order designs identically for FIM > 0.
        # The finite/positive guard is new, so that the np.array([0]) sentinel
        # _eval_fim can return routes to the infeasible branch instead of
        # yielding a meaningless -0.
        if isinstance(fim_raw, np.ndarray) and fim_raw.size == 1:
            _v = float(np.asarray(fim_raw, dtype=float).reshape(-1)[0])
            if np.isfinite(_v) and _v > 0.0:
                if self._fd_jac:
                    return -fim_raw
                jac = np.array([m for m in self.atomic_fims])
                return -fim_raw, jac
            return np.inf if self._fd_jac else (np.inf, np.zeros(n_grad))

        fim = self._prepare_fim(fim_raw)
        if fim is None:
            # degenerate FIM (absent, wrong shape, all-zero, non-finite)
            return np.inf if self._fd_jac else (np.inf, np.zeros(n_grad))

        eigvals = np.linalg.eigvalsh(0.5 * (fim + fim.T))
        if not np.all(eigvals > 0):
            return np.inf if self._fd_jac else (np.inf, np.zeros(n_grad))

        if self._fd_jac:
            return float(np.sum(1.0 / eigvals))

        try:
            fim_inv = np.linalg.inv(fim)
        except np.linalg.LinAlgError:
            return np.inf, np.zeros(n_grad)
        a_opt = float(fim_inv.trace())
        if not np.isfinite(a_opt):
            return np.inf, np.zeros(n_grad)
        jac = -np.array([
            np.sum((fim_inv @ fim_inv) * np.asarray(m)) for m in self.atomic_fims
        ])
        return a_opt, jac

    def _e_opt_criterion(self, efforts):
        """ E-optimality: maximise minimum eigenvalue of FIM. """
        self.eval_fim(efforts)

        if self.fim.size == 1:
            return -self.fim

        if self._fd_jac:
            return -np.linalg.eigvalsh(self.fim).min()
        else:
            raise NotImplementedError  # TODO: implement analytic jac for e-opt

    # prediction-oriented
    """ Prediction-variance criterion helpers (dg / di / vdi) """

    def _safe_fim_inverse(self):
        """
        Inverse of the current FIM, or None when the FIM is not usable.

        Positive-definiteness is tested by eigenvalue rather than relying on
        np.linalg.inv raising: an indefinite matrix inverts cleanly and yields
        a plausible-looking but meaningless PVAR.
        """
        fim = getattr(self, "fim", None)
        if fim is None:
            return None
        fim = np.asarray(fim, dtype=float)
        if fim.ndim != 2 or fim.shape[0] != fim.shape[1]:
            return None
        if not np.all(np.isfinite(fim)):
            return None
        eig = np.linalg.eigvalsh(0.5 * (fim + fim.T))
        if not np.all(eig > 0):
            return None
        try:
            return np.linalg.inv(fim)
        except np.linalg.LinAlgError:
            return None

    def reset_pvar_logdet_mode(self):
        """
        Clear the latched det/pseudo-det decision for dg / di / vdi.

        Called automatically at the start of design_experiment(). Call it
        manually if you change model_parameters, candidates or measurable
        responses and then re-evaluate a determinant-based prediction-variance
        criterion outside of design_experiment().
        """
        self._pvar_logdet_mode = None
        self._pvar_warned = set()

    def _pvar_log_pdet(self, pvar):
        """
        log pseudo-determinant: the sum of log-eigenvalues above a relative
        cutoff. Well defined for a singular or near-singular PSD matrix, where
        the ordinary determinant is either exactly zero or numerical noise.
        Reduces to log-det when the matrix is well conditioned.

        Returns (value, n_kept).
        """
        P = np.asarray(pvar, dtype=float)
        ev = np.linalg.eigvalsh(0.5 * (P + P.T))
        if ev.size == 0 or not np.all(np.isfinite(ev)):
            return -np.inf, 0
        ev_max = ev.max()
        if not np.isfinite(ev_max) or ev_max <= 0.0:
            return -np.inf, 0
        keep = ev[ev > self._pvar_rcond * ev_max]
        if keep.size == 0:
            return -np.inf, 0
        return float(np.sum(np.log(keep))), int(keep.size)

    def _pvar_slogdets(self):
        """
        Per-block (sign, logdet) of self.pvars, plus the near-singularity
        diagnostic on the underlying sensitivity blocks.

        Returns (signs, logdets) as (n_blk0, n_blk1) arrays, or (None, None)
        when pvars is unavailable.
        """
        pvars = getattr(self, "pvars", None)
        if pvars is None:
            return None, None
        P = np.asarray(pvars, dtype=float)
        if P.ndim != 4:
            return None, None
        n0, n1 = P.shape[0], P.shape[1]
        signs = np.empty((n0, n1))
        logdets = np.empty((n0, n1))
        for c in range(n0):
            for t in range(n1):
                signs[c, t], logdets[c, t] = np.linalg.slogdet(P[c, t])
        return signs, logdets

    def _pvar_warn_near_singular(self):
        """
        Warn once when the sensitivity blocks feeding PVAR are near-singular.

        This is a MODELLING signal, not a numerical one: it means the measurable
        responses are close to linearly dependent in sensitivity space at some
        conditions, so the response set carries fewer independent directions
        than it appears to. Only the determinant-based criteria are sensitive to
        it, which is why it is surfaced here rather than treated as an error.
        """
        if self._verbose < 1 or "near_sing" in self._pvar_warned:
            return
        sens = getattr(self, "sensitivities", None)
        if sens is None:
            return
        S = np.asarray(sens, dtype=float)
        if S.ndim != 4:
            return
        worst = np.inf
        for c in range(S.shape[0]):
            for t in range(S.shape[1]):
                sv = np.linalg.svd(S[c, t], compute_uv=False)
                if sv.size and sv[0] > 0:
                    worst = min(worst, float(sv[-1] / sv[0]))
        if np.isfinite(worst) and worst < self._pvar_cond_warn:
            self._pvar_warned.add("near_sing")
            print(
                f"[WARNING][pvar] the response sensitivity blocks are "
                f"near-singular (smallest sv_min/sv_max over all "
                f"candidate/sampling-time pairs = {worst:.2e}). The measurable "
                f"responses are close to linearly dependent in sensitivity "
                f"space, so PVAR has a near-null direction. Determinant-based "
                f"criteria (dg, di, vdi) multiply that direction in and lose "
                f"meaning; trace-based (ag, ai) and eigenvalue-based (eg, ei) "
                f"criteria are unaffected."
            )

    def _pvar_decide_logdet_mode(self, signs, trial, label, scale_floor=None):
        """
        Decide ONCE per design run whether the ordinary determinant form of a
        prediction-variance criterion is usable, and latch the answer.

        Latching matters: the branch must not flip while the optimiser is
        running, or the objective becomes discontinuous and SLSQP breaks. The
        test is deliberately BEHAVIOURAL ("did the original form produce a
        usable number?") rather than rank-based, because the numerical rank of
        PVAR is tolerance-dependent and was observed to flip between values
        across effort vectors for blocks sitting on the cutoff.

        Returns 'det' (use the original definition, bit-identically) or 'pdet'.
        """
        if self._pvar_logdet_mode is not None:
            return self._pvar_logdet_mode

        reasons = []
        if signs is not None and not np.all(signs == 1):
            n_bad = int((signs != 1).sum())
            reasons.append(
                f"{n_bad} of {signs.size} PVAR blocks are not positive definite"
            )
        if trial is not None and not np.isfinite(trial):
            reasons.append(f"the aggregate evaluates to {trial}")
        if (scale_floor is not None and trial is not None
                and np.isfinite(trial) and abs(trial) < scale_floor):
            reasons.append(
                f"|aggregate| = {abs(trial):.3e} is below the usable floor "
                f"{scale_floor:.0e}, i.e. numerical noise rather than signal"
            )

        if reasons:
            self._pvar_logdet_mode = "pdet"
            if self._verbose >= 1 and "mode" not in self._pvar_warned:
                self._pvar_warned.add("mode")
                print(
                    f"[WARNING][{label}] the determinant form of this criterion "
                    f"is not usable here ({'; '.join(reasons)}). Falling back to "
                    f"the log-PSEUDO-determinant (sum of log-eigenvalues above "
                    f"a relative cutoff of {self._pvar_rcond:.0e}), which stays "
                    f"finite and O(1) for a near-singular PVAR. NOTE this "
                    f"CHANGES the criterion definition, and the reported value "
                    f"is on a LOG scale -- it is not comparable with a "
                    f"determinant from a well-conditioned problem. The decision "
                    f"is latched for this design run; see "
                    f"reset_pvar_logdet_mode()."
                )
        else:
            self._pvar_logdet_mode = "det"
        return self._pvar_logdet_mode

    def _dg_opt_criterion(self, efforts):
        """
        dg-optimality: the WORST (maximum) determinant of the prediction
        variance matrix over all candidate / sampling-time pairs, minimised.

        Two defects in the previous implementation are fixed here.

        1. `dg_opts[c, spt] = sign * np.exp(temp_dg)` with `temp_dg` set to inf
           when `sign != 1` evaluates to `0 * inf = nan` for a SINGULAR block
           (sign == 0), and np.nanmax then silently DISCARDS it. Degenerate
           blocks -- exactly the ones that should dominate a worst-case
           criterion -- were being dropped from the maximum.
        2. Even with valid blocks, det() of an n_r x n_r matrix of small
           variances underflows toward zero. An objective below the solver's
           absolute ftol makes SLSQP declare convergence at iteration 1 and
           return the starting design untouched.

        When every block is positive definite and the aggregate is on a usable
        scale, the original determinant definition is kept EXACTLY (the same
        slogdet call, so values are bit-identical to previous releases).
        Otherwise a log-pseudo-determinant is substituted; see
        _pvar_decide_logdet_mode.
        """
        self.eval_pim(efforts)
        if self.pvars is None:
            if self._fd_jac:
                return np.inf
            raise NotImplementedError("Analytic Jacobian for dg_opt unavailable.")

        self._pvar_warn_near_singular()
        signs, logdets = self._pvar_slogdets()
        if signs is None:
            if self._fd_jac:
                return np.inf
            raise NotImplementedError("Analytic Jacobian for dg_opt unavailable.")

        # trial value under the ORIGINAL definition, used only to decide the mode
        with np.errstate(over="ignore"):
            trial = float(np.nanmax(np.where(signs == 1, np.exp(logdets), -np.inf)))
        mode = self._pvar_decide_logdet_mode(
            signs, trial, "dg_opt", scale_floor=self._pvar_scale_floor
        )

        if mode == "det":
            dg_opt = trial
        else:
            vals = [
                self._pvar_log_pdet(self.pvars[c, t])[0]
                for c in range(np.asarray(self.pvars).shape[0])
                for t in range(np.asarray(self.pvars).shape[1])
            ]
            finite = [v for v in vals if np.isfinite(v)]
            dg_opt = float(max(finite)) if finite else np.inf

        if self._fd_jac:
            return dg_opt
        else:
            raise NotImplementedError("Analytic Jacobian for dg_opt unavailable.")

    def _di_opt_criterion(self, efforts):
        """
        di-optimality: the SUM of log-determinants of the prediction variance
        matrix over all candidate / sampling-time pairs, minimised.

        Note the original comment said "average det"; the computation is a sum
        of log-determinants. The computation is preserved -- only the comment
        was wrong.

        The previous implementation set a block's contribution to +inf whenever
        slogdet reported a non-positive-definite PVAR, so a SINGLE degenerate
        block drove the whole sum to +inf and destroyed all design information.
        When every block is positive definite the original definition is kept
        exactly; otherwise a log-pseudo-determinant is substituted. See
        _pvar_decide_logdet_mode.
        """
        self.eval_pim(efforts)
        if self.pvars is None:
            if self._fd_jac:
                return np.inf
            raise NotImplementedError("Analytic Jacobian for di_opt unavailable.")

        self._pvar_warn_near_singular()
        signs, logdets = self._pvar_slogdets()
        if signs is None:
            if self._fd_jac:
                return np.inf
            raise NotImplementedError("Analytic Jacobian for di_opt unavailable.")

        trial = float(np.nansum(np.where(signs == 1, logdets, np.inf)))
        mode = self._pvar_decide_logdet_mode(signs, trial, "di_opt")

        if mode == "det":
            di_opt = trial
        else:
            vals = [
                self._pvar_log_pdet(self.pvars[c, t])[0]
                for c in range(np.asarray(self.pvars).shape[0])
                for t in range(np.asarray(self.pvars).shape[1])
            ]
            di_opt = np.inf if any(not np.isfinite(v) for v in vals) \
                else float(np.sum(vals))

        if self._fd_jac:
            return di_opt
        else:
            raise NotImplementedError("Analytic Jacobian for di_opt unavailable.")

    def _ag_opt_criterion(self, efforts):

        self.eval_pim(efforts)
        # ag_opt: max trace of the pvar matrix over candidates and sampling times
        ag_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.trace(pvar)
                ag_opts[c, spt] = temp_dg
        ag_opt = np.nanmax(ag_opts)

        if self._fd_jac:
            return ag_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ag_opt unavailable.")

    def _ai_opt_criterion(self, efforts):

        self.eval_pim(efforts)
        # ai_opt: average trace of the pvar matrix over candidates and sampling times
        ai_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.trace(pvar)
                ai_opts[c, spt] = temp_dg
        ag_opt = np.nansum(ai_opts)

        if self._fd_jac:
            return ag_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ai_opt unavailable.")

    def _eg_opt_criterion(self, efforts):

        self.eval_pim(efforts)
        # eg_opt: max of the max_eigenval of the pvar matrix over candidates and sampling times
        eg_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                # eigvalsh: PVAR is symmetric by construction, so use the
                # symmetric solver -- it is faster and guarantees a real
                # spectrum, whereas the general eigvals can return a complex
                # array whose imaginary part is then silently discarded on
                # assignment into a float buffer.
                _P = np.asarray(pvar, dtype=float)
                temp_dg = np.linalg.eigvalsh(0.5 * (_P + _P.T)).max()
                eg_opts[c, spt] = temp_dg
        eg_opt = np.nanmax(eg_opts)

        if self._fd_jac:
            return eg_opt
        else:
            raise NotImplementedError("Analytic Jacobian for eg_opt unavailable.")

    def _ei_opt_criterion(self, efforts):

        self.eval_pim(efforts)
        # ei_opts: average of the max_eigenval of the pvar matrix over candidates and sampling times
        ei_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                # eigvalsh: PVAR is symmetric by construction, so use the
                # symmetric solver -- it is faster and guarantees a real
                # spectrum, whereas the general eigvals can return a complex
                # array whose imaginary part is then silently discarded on
                # assignment into a float buffer.
                _P = np.asarray(pvar, dtype=float)
                temp_dg = np.linalg.eigvalsh(0.5 * (_P + _P.T)).max()
                ei_opts[c, spt] = temp_dg
        ei_opt = np.nansum(ei_opts)

        if self._fd_jac:
            return ei_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ei_opt unavailable.")

    """ pseudo_bayesian criterion """

    # calibration-oriented
    def _pb_d_opt_criterion(self, efforts):
        """ Pseudo-Bayesian D-optimality. """
        self.eval_fim(efforts)

        if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
            avg_fim = np.mean([fim for fim in self.scr_fims], axis=0)
            sign, d_opt = np.linalg.slogdet(avg_fim)
            return np.inf if sign != 1 else -d_opt
        elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
            d_opt = 0
            for fim in self.scr_fims:
                sign, scr_d_opt = np.linalg.slogdet(fim)
                d_opt += scr_d_opt if sign == 1 else np.inf
            return -d_opt / self.n_scr
        else:
            # Fail loudly rather than falling through and returning None, which
            # surfaces later as an opaque TypeError inside the optimizer.
            raise ValueError(
                f"_pseudo_bayesian_type is {self._pseudo_bayesian_type!r}; "
                f"expected 0/'avg_inf'/'average_information' or "
                f"1/'avg_crit'/'average_criterion'. It is normally defaulted by "
                f"design_experiment(); set it explicitly when calling a "
                f"pseudo-Bayesian criterion directly."
            )


    def _pb_ds_opt_criterion(self, efforts):
        """
        Pseudo-Bayesian Ds-optimality: as _ds_opt_criterion, but averaged
        over parameter scenarios (either by averaging the FIM itself, or by
        averaging the per-scenario Ds criterion value, per
        self._pseudo_bayesian_type).
        """
        idx_s, idx_n = self._resolve_ds_idx()
        self.eval_fim(efforts)

        def scr_ds(fim):
            """ Per-scenario Ds value; +inf when infeasible. """
            fim = self._prepare_fim(fim)
            if fim is None:
                return np.inf
            if len(idx_n) == 0:
                # degenerates to D-optimality; use a Cholesky PD test rather
                # than a determinant-sign test (see _ds_eval_schur).
                try:
                    chol = np.linalg.cholesky(0.5 * (fim + fim.T))
                except np.linalg.LinAlgError:
                    return np.inf
                return -float(2.0 * np.sum(np.log(np.diag(chol))))
            logdet_S, _P, _S_inv, _info = self._ds_eval_schur(
                fim, idx_s, idx_n, want_grad=False
            )
            return np.inf if logdet_S is None else -logdet_S

        scr_fims = self.scr_fims
        if scr_fims is None:
            return np.inf

        if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
            prepared = [self._prepare_fim(f) for f in scr_fims]
            if any(f is None for f in prepared):
                # a degenerate scenario FIM would poison the average
                return np.inf
            return scr_ds(np.mean(prepared, axis=0))
        elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
            vals = [scr_ds(fim) for fim in scr_fims]
            return np.inf if any(np.isinf(v) for v in vals) else float(np.mean(vals))
        else:
            # Fail loudly rather than falling through and returning None. The
            # sibling _pb_*_opt_criterion methods return None silently when
            # _pseudo_bayesian_type is unset (it is only defaulted inside
            # design_experiment), which surfaces much later as an opaque
            # TypeError inside the optimizer. Surface it here instead.
            raise ValueError(
                f"_pseudo_bayesian_type is {self._pseudo_bayesian_type!r}; "
                f"expected 0/'avg_inf'/'average_information' or "
                f"1/'avg_crit'/'average_criterion'. It is normally defaulted by "
                f"design_experiment(); set it explicitly when calling "
                f"ds_opt_criterion directly."
            )

    def _pb_a_opt_criterion(self, efforts):
        """
        Pseudo-Bayesian A-optimality.

        Carries the same infeasibility convention as _a_opt_criterion: an
        unusable FIM returns +inf, never 0, because A-optimality is minimised
        and 0 is its best attainable value. Positive-definiteness is checked by
        eigenvalue rather than relying on np.linalg.inv raising, since an
        indefinite scenario FIM inverts cleanly and returns a negative trace.
        """
        self.eval_fim(efforts)

        def scr_a(fim):
            """ trace(FIM^-1) for one scenario; +inf when unusable. """
            fim = self._prepare_fim(fim)
            if fim is None:
                return np.inf
            eigvals = np.linalg.eigvalsh(0.5 * (fim + fim.T))
            if not np.all(eigvals > 0):
                return np.inf
            # Deliberately keep the original inv().trace() arithmetic rather
            # than the algebraically equivalent sum(1/eigvals), so that values
            # for well-conditioned FIMs stay bit-identical to previous releases.
            try:
                val = float(np.linalg.inv(fim).trace())
            except np.linalg.LinAlgError:
                return np.inf
            return val if np.isfinite(val) else np.inf

        scr_fims = self.scr_fims
        if scr_fims is None:
            return np.inf

        if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
            prepared = [self._prepare_fim(f) for f in scr_fims]
            if any(f is None for f in prepared):
                # a degenerate scenario FIM would poison the average
                return np.inf
            return scr_a(np.mean(prepared, axis=0))
        elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
            vals = [scr_a(fim) for fim in scr_fims]
            return np.inf if any(np.isinf(v) for v in vals) else float(np.mean(vals))
        else:
            # Fail loudly rather than falling through and returning None, which
            # surfaces later as an opaque TypeError inside the optimizer.
            raise ValueError(
                f"_pseudo_bayesian_type is {self._pseudo_bayesian_type!r}; "
                f"expected 0/'avg_inf'/'average_information' or "
                f"1/'avg_crit'/'average_criterion'. It is normally defaulted by "
                f"design_experiment(); set it explicitly when calling "
                f"a_opt_criterion directly."
            )

    def _pb_e_opt_criterion(self, efforts):
        """ Pseudo-Bayesian E-optimality. """
        self.eval_fim(efforts)

        if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
            avg_fim = np.mean([fim for fim in self.scr_fims], axis=0)
            return -np.linalg.eigvalsh(avg_fim).min()
        elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
            return np.mean([
                -np.linalg.eigvalsh(fim).min() for fim in self.scr_fims
            ])

        else:
            # Fail loudly rather than falling through and returning None, which
            # surfaces later as an opaque TypeError inside the optimizer.
            raise ValueError(
                f"_pseudo_bayesian_type is {self._pseudo_bayesian_type!r}; "
                f"expected 0/'avg_inf'/'average_information' or "
                f"1/'avg_crit'/'average_criterion'. It is normally defaulted by "
                f"design_experiment(); set it explicitly when calling a "
                f"pseudo-Bayesian criterion directly."
            )
    # prediction-oriented

    def _pb_dg_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    def _pb_di_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    def _pb_ag_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    def _pb_ai_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    def _pb_eg_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    def _pb_ei_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    """ private methods """

    def _generate_result_path(self, name, extension, iteration=None):
        self.create_result_dir()

        while True:
            now = datetime.now()
            if not self.result_dir:
                self.result_dir = self.result_dir_daily + f"time_{now.hour:d}-{now.minute:d}-{now.second}/"
                if not path.exists(self.result_dir):
                    makedirs(self.result_dir)
            fn = f"{name}.{extension}"
            if iteration is not None:
                fn = f"iter_{iteration:d}_" + fn
            fp = self.result_dir + fn
            return fp

    def _plot_optimal_sensitivities(self, absolute=False, legend=None,
                                   markersize=10, colour_map="jet",
                                   write=False, dpi=720, figsize=None):
        if not self._dynamic_system:
            raise SyntaxError("Sensitivity plots are only for dynamic systems.")

        if self.optimal_candidates is None:
            self.get_optimal_candidates()
        if self.n_opt_c == 0:
            print(
                f"[Warning]: empty optimal candidates, skipping plotting of optimal "
                f"predictions."
            )
            return
        if legend is None:
            if self.n_opt_c < 6:
                legend = True
        if figsize is None:
            figsize = (self.n_mp * 4.0, 1.0 + 2.5 * self.n_m_r)

        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=self.n_m_r,
            ncols=self.n_mp,
            sharex=True,
        )
        if self.n_m_r == 1 and self.n_mp == 1:
            axes = np.array([[axes]])
        elif self.n_m_r == 1:
            axes = np.array([axes])
        elif self.n_mp == 1:
            axes = np.array([axes]).T

        if self._pseudo_bayesian:
            mean_sens = np.nanmean(self._scr_sens, axis=0)
            std_sens = np.nanstd(self._scr_sens, axis=0)

        for row in range(self.n_m_r):
            for col in range(self.n_mp):
                cmap = plt.get_cmap(colour_map, len(self.optimal_candidates))
                colors = itertools.cycle(
                    cmap(_) for _ in np.linspace(0, 1, len(self.optimal_candidates))
                )
                for c, cand in enumerate(self.optimal_candidates):
                    opt_spt = self.sampling_times_candidates[cand[0]]
                    if self._pseudo_bayesian:
                        sens = mean_sens[
                                   cand[0],
                                   :,
                                   self.measurable_responses[row],
                                   col
                               ]
                        std = std_sens[
                                  cand[0],
                                  :,
                                  self.measurable_responses[row],
                                  col
                              ]
                    else:
                        sens = self.sensitivities[
                                   cand[0],
                                   :,
                                   self.measurable_responses[row],
                                   col
                               ]
                    color = next(colors)
                    if absolute:
                        sens = np.abs(sens)
                    ax = axes[row, col]
                    ax.plot(
                        opt_spt,
                        sens,
                        linestyle="--",
                        label=f"Candidate {cand[0] + 1:d}",
                        color=color
                    )
                    if not self._specified_n_spt:
                        if self._opt_sampling_times:
                            plot_sens = sens[cand[5]]
                        else:
                            plot_sens = sens[tuple(cand[5])]
                        ax.scatter(
                            cand[3],
                            plot_sens,
                            marker="o",
                            s=markersize * 50 * np.array(cand[4]),
                            color=color,
                            facecolors="none",
                        )
                    else:
                        markers = itertools.cycle(["o", "s", "h", "P"])
                        for i, (eff, spt, spt_idx) in enumerate(zip(cand[4], cand[3], cand[5])):
                            marker = next(markers)
                            ax.scatter(
                                spt,
                                sens[spt_idx],
                                marker=marker,
                                s=markersize * 50 * np.array(eff),
                                color=color,
                                label=f"Sampling schedule {i + 1}",
                                facecolors="none",
                            )
                    if self._pseudo_bayesian:
                        ax.fill_between(
                            opt_spt,
                            sens + std,
                            sens - std,
                            facecolor=color,
                            alpha=0.1,
                        )
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

                    if row == self.n_m_r - 1:
                        if self.time_unit_name is not None:
                            ax.set_xlabel(f"Time ({self.time_unit_name})")
                        else:
                            ax.set_xlabel('Time')
                    if self.response_names is None or self.model_parameter_names is None:
                        pass
                    else:
                        ylabel = r"$\partial$"
                        ylabel += self.response_names[self.measurable_responses[row]]
                        ylabel += r"/$\partial$"
                        ylabel += self.model_parameter_names[col]
                        if self.response_unit_names is None or self.model_parameter_unit_names is None:
                            pass
                        else:
                            ylabel += f" [({self.response_unit_names[row]})/({self.model_parameter_unit_names[col]})]"
                        ax.set_ylabel(ylabel)
                        # ax.set_ylabel(
                        #     f"$\\partial {self.response_names[self.measurable_responses[row]]}"
                        #     f"/"
                        #     f"\\partial {self.model_parameter_names[col]}$"
                        # )
        if legend and len(self.optimal_candidates) > 1:
            axes[-1, -1].legend()

        _safe_tight_layout(fig)

        if write:
            fn = f"sensitivity_plot_{self.oed_result['optimality_criterion']}"
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)
            self.run_no = 1

        return fig

    def _plot_optimal_sensitivities_interactive(self, figsize=None, markersize=10,
                                                colour_map="jet"):
        if not self._dynamic_system:
            raise SyntaxError("Sensitivity plots are only for dynamic systems.")

        if self.sensitivities is None:
            self.eval_sensitivities()
        if figsize is None:
            figsize = (18, 7)
        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=2,
            ncols=3,
            gridspec_kw={
                "width_ratios": [2, 1, 1],
                "height_ratios": [2, 1],
            }
        )

        for axis_list in axes[:, 1:]:
            for ax in axis_list:
                ax.remove()

        gs = axes[0, 0].get_gridspec()
        res_rad_ax = fig.add_subplot(gs[:, 1])
        mp_rad_ax = fig.add_subplot(gs[:, 2])

        if self.time_unit_name is not None:
            axes[0, 0].set_xlabel(f"Time ({self.time_unit_name})")
        else:
            axes[0, 0].set_xlabel('Time')

        lines = []
        fill_lines = []
        cmap = plt.get_cmap(colour_map)
        colors = itertools.cycle(
            cmap(_)
            for _ in np.linspace(0, 1, len(self.optimal_candidates))
        )

        if self._pseudo_bayesian:
            mean_sens = np.nanmean(
                self._scr_sens,
                axis=0,
            )
            std_sens = np.nanstd(
                self._scr_sens,
                axis=0,
            )

        for opt_c in self.optimal_candidates:
            color = next(colors)
            label = f"Candidate {opt_c[0]+1}"
            if self._pseudo_bayesian:
                line, = axes[0, 0].plot(
                    self.sampling_times_candidates[opt_c[0]],
                    mean_sens[opt_c[0], :, 0, 0],
                    visible=True,
                    label=label,
                    marker="o",
                    markersize=markersize,
                    color=color,
                )
                fill_line = axes[0, 0].fill_between(
                    self.sampling_times_candidates[opt_c[0]],
                    mean_sens[opt_c[0], :, 0, 0] + std_sens[opt_c[0], :, 0, 0],
                    mean_sens[opt_c[0], :, 0, 0] - std_sens[opt_c[0], :, 0, 0],
                    facecolor=color,
                    alpha=0.1,
                    visible=True,
                )
            else:
                line, = axes[0, 0].plot(
                    self.sampling_times_candidates[opt_c[0]],
                    self.sensitivities[opt_c[0], :, 0, 0],
                    visible=True,
                    label=label,
                    marker="o",
                    markersize=markersize,
                    color=color,
                )
            lines.append(line)
            if self._pseudo_bayesian:
                fill_lines.append(fill_line)
            axes[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        labels = [str(line.get_label()) for line in lines]
        visibilities = [line.get_visible() for line in lines]
        cand_check = CheckButtons(
            axes[1, 0],
            labels=labels,
            actives=visibilities,
        )

        def _cand_check(label):
            index = labels.index(label)
            lines[index].set_visible(not lines[index].get_visible())
            if self._pseudo_bayesian:
                fill_lines[index].set_visible(not fill_lines[index].get_visible())
            plt.draw()

        cand_check.on_clicked(_cand_check)

        res_dict = {
            f"{res_name}": i
            for i, res_name in enumerate(self.response_names)
        }
        mp_dict = {
            f"{mp_name}": j
            for j, mp_name in enumerate(self.model_parameter_names)
        }

        res_rad = RadioButtons(
            res_rad_ax,
            labels=[
                f"{res_name}"
                for res_name in self.response_names
            ],
        )

        def _res_rad(label):
            res_idx = res_dict[label]
            mp_idx = mp_dict[mp_rad.value_selected]
            for i, (opt_c, line) in enumerate(zip(self.optimal_candidates, lines)):
                color = next(colors)
                if self._pseudo_bayesian:
                    sens_data = mean_sens[opt_c[0], :, res_idx, mp_idx]
                    fill_lines[i].remove()
                    fill_lines[i] = axes[0, 0].fill_between(
                        self.sampling_times_candidates[opt_c[0]],
                        sens_data + std_sens[opt_c[0], :, res_idx, mp_idx],
                        sens_data - std_sens[opt_c[0], :, res_idx, mp_idx],
                        facecolor=color,
                        alpha=0.1,
                    )
                else:
                    sens_data = self.sensitivities[opt_c[0], :, res_idx, mp_idx]
                line.set_ydata(sens_data)
            axes[0, 0].relim()
            axes[0, 0].autoscale()
            plt.draw()
        res_rad.on_clicked(_res_rad)

        mp_rad = RadioButtons(
            mp_rad_ax,
            labels=[
                f"{mp_name}"
                for mp_name in self.model_parameter_names
            ],
        )

        def _mp_rad(label):
            res_idx = res_dict[res_rad.value_selected]
            mp_idx = mp_dict[label]
            for i, (opt_c, line) in enumerate(zip(self.optimal_candidates, lines)):
                color = next(colors)
                if self._pseudo_bayesian:
                    sens_data = mean_sens[opt_c[0], :, res_idx, mp_idx]
                    fill_lines[i].remove()
                    fill_lines[i] = axes[0, 0].fill_between(
                        self.sampling_times_candidates[opt_c[0]],
                        sens_data + std_sens[opt_c[0], :, res_idx, mp_idx],
                        sens_data - std_sens[opt_c[0], :, res_idx, mp_idx],
                        facecolor=color,
                        alpha=0.1,
                    )
                else:
                    sens_data = self.sensitivities[opt_c[0], :, res_idx, mp_idx]
                line.set_ydata(sens_data)
            axes[0, 0].relim()
            axes[0, 0].autoscale()
            plt.draw()
        mp_rad.on_clicked(_mp_rad)

        _safe_tight_layout(fig)
        plt.show()
        return fig


    def _eval_sensitivities_pyomo_ift(self, ti_controls, model_parameters,
                                      store_predictions=True):
        """
        Compute response and exact parametric sensitivities via the
        Implicit-Function Theorem (IFT) applied to a user-supplied Pyomo DAE model.

        Two Jacobian backends, selected automatically:
        1. PyomoNLP / ASL (fast, compiled C) — when pynumero_ASL is available.
           Parameters must be temporarily unfixed so the NL writer includes them.
        2. Pyomo differentiate() (pure Python fallback) — always available.

        In both cases the IFT linear solve is identical:
            J = [J_p | J_z]  where J_p = dc/dp, J_z = dc/dz
            S = lstsq(J_z, -J_p)   shape (n_state, n_mp)

        Returns
        -------
        responses : ndarray shape (n_spt, n_m_r)
        sens      : ndarray shape (n_spt, n_m_r, n_mp)
        """
        import pyomo.environ as _pyo
        import scipy.sparse as _sp

        theta = np.asarray(model_parameters, dtype=float)
        n_mp  = len(theta)
        n_mr  = self.n_m_r

        # 1. Build and initialise the Pyomo model.
        # Pass _current_spt as sampling_times so the model embeds the requested
        # measurement times into the collocation grid.  The IFT Jacobian is then
        # evaluated at collocation points that coincide with (or are very close to)
        # the actual sampling times, enabling correct sensitivity discrimination
        # across different sampling times when optimize_sampling_times=True.
        #
        # For static/signature-1 systems, _current_spt holds the ti_controls
        # value (not a sampling time), so we guard with _is_dynamic to avoid
        # passing nonsensical time arguments to the model builder.
        #
        # pyomo_model_fn must accept sampling_times as a keyword argument;
        # if it doesn't (older model builders), the TypeError is caught and
        # we fall back to the original two-argument call unchanged.
        _is_dynamic = getattr(self, '_dynamic_system', False)
        _spt = getattr(self, '_current_spt', None) if _is_dynamic else None
        try:
            m, all_vars, all_bodies, t_sorted = self.pyomo_model_fn(
                ti_controls, theta, sampling_times=_spt
            )
        except TypeError:
            m, all_vars, all_bodies, t_sorted = self.pyomo_model_fn(
                ti_controls, theta
            )

        # 2. Resolve output variable name(s)
        out_names = getattr(self, 'pyomo_output_var_name', None)
        if out_names is None:
            # Derive the response variable BASE names from the state block of
            # all_vars, taking the first n_mr DISTINCT bases in order of first
            # appearance.
            #
            # The previous rule was  [str(all_vars[n_mp + r]) for r in range(n_mr)],
            # which assumed one all_vars entry per response. That holds only for
            # SCALAR responses. For a time-indexed dynamic model the layout is
            #
            #     [params...] + [ca[t] for ALL t] + [cb[t] for ALL t] + ...
            #
            # so all_vars[n_mp + r] walks along ONE response's time index instead
            # of across responses: with 61 collocation points, all_vars[4] and
            # all_vars[5] are 'ca[0.0]' and 'ca[0.007753]' — both CA. Every
            # response then resolved to the same variable, so the extractor
            # returned the CA row n_mr times and never read CB at all. Both the
            # predicted responses and their sensitivity rows came back
            # duplicated, and any parameter appearing only in the unread
            # response (here nu, which enters solely through the CB material
            # balance) presented as unidentifiable — a structurally singular FIM
            # from a perfectly well-posed problem.
            #
            # A single-response model cannot exhibit this, which is why the
            # FD-vs-IFT agreement test passes on the first-order example.
            _bases = []
            for _v in all_vars[n_mp:]:
                _b = str(_v).split('[', 1)[0]
                if _b not in _bases:
                    _bases.append(_b)
            if len(_bases) < n_mr:
                raise RuntimeError(
                    f"[Pyomo IFT] Cannot identify {n_mr} response variables: the "
                    f"state block of all_vars contains only {len(_bases)} "
                    f"distinct variable name(s) {_bases}. Order all_vars as "
                    f"[parameters..., response_1[...], response_2[...], ..., "
                    f"auxiliaries...], or set "
                    f"designer.pyomo_output_var_name explicitly."
                )
            out_names = _bases[:n_mr]
        elif isinstance(out_names, str):
            out_names = [out_names]

        state_var_strs = [str(v) for v in all_vars[n_mp:]]

        def _find_state_idx(base_name, t_val):
            # Snap to the nearest time in t_sorted
            t_key = min(t_sorted, key=lambda tt: abs(tt - t_val))
            # Build candidate string targets.  Pyomo renders indexed variables
            # as "VarName[t]" where t is formatted by Python's str() — which
            # may differ from the float repr used in t_key (e.g. "0.37" vs
            # "0.37000000000000004").  We therefore try several formats and
            # also fall back to a substring search as a last resort.
            for target in (
                f"{base_name}[{t_key}]",
                f"{base_name}[{t_key:.10g}]",
                f"{base_name}[{t_key:.6g}]",
                base_name,
            ):
                for idx, vname in enumerate(state_var_strs):
                    # Compare the BASE name exactly (everything before '[')
                    # rather than by prefix. A startswith test would let 'ca'
                    # match 'cah', 'ca_total', 'catalyst' and so on, silently
                    # returning a different variable's row.
                    _vbase = vname.split('[', 1)[0]
                    if vname == target or (
                        _vbase == base_name and
                        vname.endswith(f"[{t_key}]")
                    ):
                        return idx
            # Last resort: find the state variable whose time index is
            # numerically closest to t_key, using string parsing
            best_idx, best_dist = None, float('inf')
            prefix = f"{base_name}["
            for idx, vname in enumerate(state_var_strs):
                # exact base-name match, then nearest time
                if (vname.split('[', 1)[0] == base_name
                        and vname.startswith(prefix) and vname.endswith("]")):
                    try:
                        t_var = float(vname[len(prefix):-1])
                        dist = abs(t_var - t_key)
                        if dist < best_dist:
                            best_dist, best_idx = dist, idx
                    except ValueError:
                        pass
            if best_idx is not None:
                return best_idx
            raise RuntimeError(
                f"[Pyomo IFT] Cannot find state variable '{base_name}[{t_key}]' "
                f"or scalar '{base_name}'.\n"
                f"Available: {state_var_strs[:10]}..."
            )

        # 3. Build Jacobian — choose backend
        _has_free_vars = any(
            not v.is_fixed()
            for v in m.component_data_objects(_pyo.Var, active=True)
        )
        global _PYNUMERO_ASL_AVAILABLE
        _asl_ok = False
        if _PYNUMERO_ASL_AVAILABLE and _has_free_vars:
            # Fast path: unfix param vars, get ASL Jacobian, re-fix.
            # _PYNUMERO_ASL_AVAILABLE only proves the Python class imported;
            # _PyomoNLP(m) can still fail here if the compiled ASL extension
            # underneath it is missing or broken. That failure must not
            # propagate uncaught — fall back to the pure-Python Jacobian
            # instead, and downgrade the flag so later calls in this process
            # skip straight to the fallback rather than re-attempting a
            # backend already shown not to work.
            param_vars = all_vars[:n_mp]
            for pv in param_vars:
                pv.unfix()
            try:
                nlp      = _PyomoNLP(m)
                J_sparse = nlp.evaluate_jacobian_eq()
                J_dense  = J_sparse.toarray()
                nlp_var_names = nlp.primals_names()
                _asl_ok = True
            except Exception as _asl_exc:
                warnings.warn(
                    "[Pyomo IFT / ASL] PyomoNLP failed at runtime despite "
                    "importing successfully (likely a missing or broken "
                    "compiled ASL extension). Falling back to the "
                    "pure-Python differentiate() Jacobian for the "
                    f"remainder of this process. Original error: {_asl_exc}",
                    RuntimeWarning,
                )
                _PYNUMERO_ASL_AVAILABLE = False
            finally:
                for pv in param_vars:
                    pv.fix()

            if _asl_ok:
                all_var_strs = [str(v) for v in all_vars]
                col_order = []
                for vname in all_var_strs:
                    matched = _match_nlp_var(vname, nlp_var_names)
                    if matched is None:
                        raise RuntimeError(
                            f"[Pyomo IFT / ASL] Cannot match variable '{vname}' "
                            f"in NLP variable list.\nNLP vars: {nlp_var_names}"
                        )
                    col_order.append(matched)
                J = J_dense[:, col_order]

        if not _asl_ok:
            # Fallback: pure-Python differentiate() loop. Reached either
            # because _PYNUMERO_ASL_AVAILABLE was False to begin with, or
            # because the ASL backend just failed above at runtime.
            J = _pyomo_ift_fd_jacobian(all_vars, all_bodies)

        # 4. Split J into parameter and state columns
        J_p = J[:, :n_mp]
        J_z = J[:, n_mp:]

        # 5. Solve J_z * S = -J_p
        S, *_ = _scipy_linalg.lstsq(J_z, -J_p)

        # 6. Extract responses and sensitivities.
        # For causal (sequential) sensitivities each sampling time needs its
        # own model integrated from 0 to t_i.  We therefore loop over the
        # requested sampling times and, when there are multiple distinct times,
        # rebuild + re-solve the model for each one so that dCB(t_i)/dθ
        # reflects only the history up to t_i — exactly matching FD behaviour.
        # When all sampling times are the same (e.g. single endpoint), the
        # model is built only once (the S from above is reused).
        responses = np.zeros((len(self._current_spt), n_mr))
        sens      = np.zeros((len(self._current_spt), n_mr, n_mp))

        unique_spt = sorted(set(float(t) for t in self._current_spt))

        # Scale needed to map an ABSOLUTE sampling time into the grid returned
        # by pyomo_model_fn. Model builders are free to return their grid in
        # absolute time OR normalised to [0, 1] (case_2_model normalises by
        # tau = max(sampling_times)), and the designer cannot know which. Both
        # are handled by proportional scaling: a build that was asked for
        # sampling times with maximum `build_tau` and returned a grid whose
        # maximum is `grid_max` maps absolute t to  t / build_tau * grid_max.
        #   normalised grid: grid_max = 1        -> t / build_tau
        #   absolute grid  : grid_max = build_tau -> t
        #
        # Getting this wrong was a real bug: the snap used
        #     min(t_sorted, key=lambda tt: abs(tt - t_val))
        # comparing an absolute time against a normalised grid. For t_val > 1
        # the min clamps to the largest grid point, which is accidentally
        # correct; for t_val < 1 it lands in the grid interior and reads the
        # response at the WRONG time. Sensitivities were exact at late sampling
        # times and silently wrong at early ones, with an error that does not
        # shrink as the collocation grid is refined.
        def _grid_target(t_abs, t_grid, build_tau):
            g_max = max(t_grid) if len(t_grid) else 1.0
            if build_tau is None or not np.isfinite(build_tau) or build_tau <= 0:
                return float(t_abs)
            return float(t_abs) / float(build_tau) * float(g_max)

        _spt_arr0 = np.asarray(self._current_spt, dtype=float).ravel()
        _spt_arr0 = _spt_arr0[np.isfinite(_spt_arr0) & (_spt_arr0 >= 0)]
        _build_tau0 = float(_spt_arr0.max()) if _spt_arr0.size else None
        # Cache: maps t_val → (S_t, t_sorted_t, m_t) to avoid rebuilding
        # the same model twice if the same time appears more than once.
        _spt_cache = {}

        for spt_i, t_val in enumerate(self._current_spt):
            t_val_f = float(t_val)
            if t_val_f not in _spt_cache:
                if len(unique_spt) > 1:
                    # Rebuild model integrated only to t_val for causal sens.
                    m_t, all_vars_t, all_bodies_t, t_sorted_t = \
                        self.pyomo_model_fn(
                            ti_controls, theta,
                            sampling_times=[t_val_f]
                        )
                    n_mp_t = n_mp   # same number of parameters
                    state_var_strs_t = [str(v) for v in all_vars_t[n_mp_t:]]
                    # Build Jacobian for this sub-model. Same runtime-failure
                    # guard as the main build above: a True
                    # _PYNUMERO_ASL_AVAILABLE only means the Python class
                    # imported, not that _PyomoNLP(m_t) will actually work.
                    _asl_ok_t = False
                    if _PYNUMERO_ASL_AVAILABLE:
                        param_vars_t = all_vars_t[:n_mp_t]
                        for pv in param_vars_t:
                            pv.unfix()
                        try:
                            nlp_t      = _PyomoNLP(m_t)
                            J_sparse_t = nlp_t.evaluate_jacobian_eq()
                            J_dense_t  = J_sparse_t.toarray()
                            nlp_vars_t = nlp_t.primals_names()
                            _asl_ok_t  = True
                        except Exception as _asl_exc_t:
                            warnings.warn(
                                "[Pyomo IFT causal / ASL] PyomoNLP failed at "
                                "runtime despite importing successfully. "
                                "Falling back to the pure-Python "
                                "differentiate() Jacobian for the remainder "
                                f"of this process. Original error: {_asl_exc_t}",
                                RuntimeWarning,
                            )
                            _PYNUMERO_ASL_AVAILABLE = False
                        finally:
                            for pv in param_vars_t:
                                pv.fix()

                        if _asl_ok_t:
                            all_var_strs_t = [str(v) for v in all_vars_t]
                            col_order_t = []
                            for vname in all_var_strs_t:
                                matched = _match_nlp_var(vname, nlp_vars_t)
                                if matched is None:
                                    raise RuntimeError(
                                        f"[Pyomo IFT causal] Cannot match '{vname}'"
                                    )
                                col_order_t.append(matched)
                            J_t = J_dense_t[:, col_order_t]

                    if not _asl_ok_t:
                        J_t = _pyomo_ift_fd_jacobian(all_vars_t, all_bodies_t)
                    J_p_t = J_t[:, :n_mp_t]
                    J_z_t = J_t[:, n_mp_t:]
                    try:
                        S_t, *_ = _scipy_linalg.lstsq(J_z_t, -J_p_t)
                    except np.linalg.LinAlgError as _lae:
                        raise RuntimeError(
                            f"IFT lstsq failed (SVD did not converge) — "
                            f"J_z_t is rank-deficient for this candidate."
                        ) from _lae
                    # this sub-model was built for the single time t_val_f,
                    # so that is its requested-time scale
                    _spt_cache[t_val_f] = (S_t, t_sorted_t,
                                           state_var_strs_t, m_t,
                                           all_vars_t, t_val_f)
                else:
                    # Single unique spt — reuse the already-solved model, which
                    # was built from the full _current_spt vector
                    _spt_cache[t_val_f] = (S, t_sorted, state_var_strs,
                                           m, all_vars, _build_tau0)

            S_use, t_sorted_use, sv_strs_use, m_use, av_use, tau_use = \
                _spt_cache[t_val_f]
            # map the absolute sampling time into this model's grid coordinates
            t_tgt = _grid_target(t_val_f, t_sorted_use, tau_use)
            t_key = min(t_sorted_use, key=lambda tt: abs(tt - t_tgt))

            def _find_state_idx_t(base_name, t_v, _sv=sv_strs_use,
                                  _t_key=t_key, _tau=tau_use):
                # t_v arrives as an ABSOLUTE sampling time, but t_sorted_use may
                # be normalised — map it through the same proportional scaling
                # used for t_key rather than snapping on the raw value. Snapping
                # an absolute time against a normalised grid reads the response
                # at the wrong time whenever t_v < 1 (see _grid_target).
                if float(t_v) == float(t_val_f):
                    t_k = _t_key            # already mapped by the caller
                else:
                    t_k = min(t_sorted_use,
                              key=lambda tt: abs(tt - _grid_target(
                                  t_v, t_sorted_use, _tau)))
                for target in (
                    f"{base_name}[{t_k}]",
                    f"{base_name}[{t_k:.10g}]",
                    f"{base_name}[{t_k:.6g}]",
                    base_name,
                ):
                    for idx, vname in enumerate(_sv):
                        # exact BASE-name comparison; a startswith test would let
                        # 'ca' match 'cah', 'ca_total', 'catalyst', ...
                        if vname == target or (
                            vname.split('[', 1)[0] == base_name and
                            vname.endswith(f"[{t_k}]")
                        ):
                            return idx
                prefix = f"{base_name}["
                best_idx2, best_dist2 = None, float('inf')
                for idx, vname in enumerate(_sv):
                    if (vname.split('[', 1)[0] == base_name
                            and vname.startswith(prefix)
                            and vname.endswith("]")):
                        try:
                            t_v2  = float(vname[len(prefix):-1])
                            dist2 = abs(t_v2 - t_k)
                            if dist2 < best_dist2:
                                best_dist2, best_idx2 = dist2, idx
                        except ValueError:
                            pass
                if best_idx2 is not None:
                    return best_idx2
                raise RuntimeError(
                    f"[Pyomo IFT causal] Cannot find '{base_name}[{t_k}]'"
                )

            for r_i, out_name in enumerate(out_names):
                base_name = out_name.split("[")[0]
                var_comp  = m_use.find_component(base_name)
                if var_comp is None:
                    var_comp = m_use.find_component(out_name)
                if var_comp is None:
                    raise RuntimeError(
                        f"[Pyomo IFT] Output variable '{out_name}' not found."
                    )
                if hasattr(var_comp, 'is_indexed') and var_comp.is_indexed():
                    val = _pyo.value(var_comp[t_key])
                else:
                    val = _pyo.value(var_comp)
                responses[spt_i, r_i] = val
                sens[spt_i, r_i, :]   = S_use[
                    _find_state_idx_t(base_name, t_val_f), :]

        # 7. Store responses
        if store_predictions:
            self._current_res = responses
            self._store_current_response()

        return responses, sens

    def _sensitivity_sim_wrapper(self, theta_try, store_responses=True):
        if self.use_finite_difference:
            response = self._simulate_internal(self._current_tic, self._current_tvc,
                                               theta_try, self._current_spt)
        else:
            self.do_sensitivity_analysis = True
            response, sens = self._simulate_internal(self._current_tic, self._current_tvc,
                                                     theta_try, self._current_spt)
            self.do_sensitivity_analysis = False
        self.feval_sensitivity += 1
        """ store responses whenever required, and model parameters are the same as 
        current model's """
        if store_responses and np.allclose(theta_try, self._current_scr_mp,
                                           rtol=self._store_responses_rtol,
                                           atol=self._store_responses_atol):
            self._current_res = response
            self._store_current_response()
        if self.use_finite_difference:
            if self.n_m_r == 1 and len(response.flatten()) == 1:
                return response[0]
            else:
                return response
        else:
            return response, sens

    def _plot_current_efforts_2d(self, tol=1e-4, width=None, write=False, dpi=720,
                                 figsize=None):
        self.get_optimal_candidates(tol=tol)

        if self._verbose >= 2:
            print("Plotting current continuous design.")

        if width is None:
            width = 0.7

        if self.efforts.ndim == 2:
            p_plot = np.array([np.sum(opt_cand[4]) for opt_cand in self.optimal_candidates])
        else:
            p_plot = np.array([opt_cand[4][0] for opt_cand in self.optimal_candidates])

        x = np.array([opt_cand[0]+1 for opt_cand in self.optimal_candidates]).astype(str)
        if figsize is None:
            fig = plt.figure(figsize=(15, 7))
        else:
            fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(111)

        axes.bar(x, p_plot, width=width)

        axes.set_xticks(x)
        axes.set_xlabel("Candidate Number")

        axes.set_ylabel("Optimal Experimental Effort")
        if not self._discrete_design:
            axes.set_ylim([0, 1])
            axes.set_yticks(np.linspace(0, 1, 11))
        else:
            axes.set_ylim([0, self.efforts.max()])
            axes.set_yticks(
                np.linspace(0, self.efforts.max(), self.efforts.max().astype(int))
            )

        if write:
            fn = f"efforts_{self._current_criterion}"
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)

        _safe_tight_layout(fig)
        return fig

    def _plot_current_efforts_3d(self, width=None, write=False, dpi=720, tol=1e-4,
                                 figsize=None):
        self.get_optimal_candidates(tol=tol)

        if self._specified_n_spt:
            # For n_spt designs each candidate has sampling-time variants
            # (combinations of n_spt_spec times).  Plot one bar per variant,
            # positioned at the mean sampling time of that combination, with
            # height = total effort allocated to that variant.
            if self._verbose >= 2:
                print("Plotting current continuous design.")

            if width is None:
                width = 0.7

            # Collect all sampling times to compute a sensible bar depth
            all_times = []
            for opt_cand in self.optimal_candidates:
                for spt_comb in opt_cand[3]:
                    all_times.extend(spt_comb)
            all_times  = np.array(all_times)
            time_range = np.nanmax(all_times) - np.nanmin(all_times)
            dy         = max(time_range * 0.02, 1.0)

            if figsize is None:
                fig = plt.figure(figsize=(12, 8))
            else:
                fig = plt.figure(figsize=figsize)
            axes = fig.add_subplot(111, projection='3d')

            for c_plot, opt_cand in enumerate(self.optimal_candidates):
                for spt_comb, eff_arr in zip(opt_cand[3], opt_cand[4]):
                    eff_total = float(np.nansum(eff_arr))
                    if eff_total < tol:
                        continue
                    # Position bar at mean sampling time of this variant
                    y_pos = float(np.mean(spt_comb))
                    axes.bar3d(
                        x  = c_plot - width / 2,
                        y  = y_pos  - dy / 2,
                        z  = 0,
                        dx = width,
                        dy = dy,
                        dz = eff_total,
                    )

            axes.grid(False)
            axes.set_xlabel('Candidate')
            axes.set_xticks(range(len(self.optimal_candidates)))
            axes.set_xticklabels(
                [str(opt_c[0] + 1) for opt_c in self.optimal_candidates]
            )
            if self.time_unit_name is not None:
                axes.set_ylabel(f"Sampling Times ({self.time_unit_name})")
            else:
                axes.set_ylabel('Sampling Times')
            axes.set_zlabel('Experimental Effort')
            axes.set_zlim([0, 1])
            axes.set_zticks(np.linspace(0, 1, 6))
            _safe_tight_layout(fig)

            if write:
                fn = f'efforts_{self.oed_result["optimality_criterion"]}'
                fp = self._generate_result_path(fn, "png")
                fig.savefig(fname=fp, dpi=dpi)
            return fig

        if self._verbose >= 2:
            print("Plotting current continuous design.")

        if width is None:
            width = 0.7

        p = self.efforts.reshape([self.n_c, self.n_spt])

        sampling_time_scale = np.nanmin(np.diff(self.sampling_times_candidates, axis=1))

        if figsize is None:
            fig = plt.figure(figsize=(12, 8))
        else:
            fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(111, projection='3d')
        opt_cand = np.unique(np.where(p > tol)[0], axis=0)
        for c, spt in enumerate(self.sampling_times_candidates[opt_cand]):
            x = np.array([c] * self.n_spt) - width / 2
            z = np.zeros(self.n_spt)

            dx = width
            dy = width * sampling_time_scale * width
            dz = p[opt_cand[c], :]

            x = x[~np.isnan(spt)]
            y = spt[~np.isnan(spt)]
            z = z[~np.isnan(spt)]
            dz = dz[~np.isnan(spt)]

            axes.bar3d(
                x=x,
                y=y,
                z=z,
                dx=dx,
                dy=dy,
                dz=dz
            )

        axes.grid(False)
        axes.set_xlabel('Candidate')
        xticks = opt_cand + 1
        axes.set_xticks(
            [c for c, _ in enumerate(self.sampling_times_candidates[opt_cand])])
        axes.set_xticklabels(labels=xticks)

        if self.time_unit_name is not None:
            axes.set_ylabel(f"Sampling Times ({self.time_unit_name})")
        else:
            axes.set_ylabel('Sampling Times')

        axes.set_zlabel('Experimental Effort')
        axes.set_zlim([0, 1])
        axes.set_zticks(np.linspace(0, 1, 6))

        _safe_tight_layout(fig)

        if write:
            fn = f'efforts_{self.oed_result["optimality_criterion"]}'
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)
        return fig

    def _pad_sampling_times(self):
        """ check the required number of sampling times """
        max_num_sampling_times = 1
        for sampling_times in self.sampling_times_candidates:
            num_sampling_times = len(sampling_times)
            if num_sampling_times > max_num_sampling_times:
                max_num_sampling_times = num_sampling_times

        for i, sampling_times in enumerate(self.sampling_times_candidates):
            num_sampling_times = len(sampling_times)
            if num_sampling_times < max_num_sampling_times:
                diff = max_num_sampling_times - num_sampling_times
                self.sampling_times_candidates[i] = np.pad(sampling_times,
                                                           pad_width=(0, diff),
                                                           mode='constant',
                                                           constant_values=np.nan)
        self.sampling_times_candidates = np.array(
            self.sampling_times_candidates.tolist())
        return self.sampling_times_candidates

    def _pad_sensitivities(self):
        """ padding sensitivities to accommodate for missing sampling times """
        for i, row in enumerate(self.sensitivities):
            if row.ndim < 3:  # check if row has less than 3 dim
                if self.n_mp == 1:  # potential cause 1: we only have 1 mp
                    row = np.expand_dims(row, -1)  # add last dimension
                if self.n_r == 1:  # potential cause 2: we only have 1 response
                    row = np.expand_dims(row, -2)  # add second to last
            if row.ndim != 3:  # check again if already 3 dims
                # only reason: we only have 1 spt, add dim to first position
                row = np.expand_dims(row, 0)
            # pad sampling times
            diff = self.n_spt - row.shape[0]
            self.sensitivities[i] = np.pad(row,
                                           pad_width=[(0, diff), (0, 0), (0, 0)],
                                           mode='constant', constant_values=np.nan)
        self.sensitivities = self.sensitivities.tolist()
        self.sensitivities = np.asarray(self.sensitivities)
        return self.sensitivities

    def _store_current_response(self):
        """ padding responses to accommodate for missing sampling times """
        start = time()
        if self.response is None:  # if it is the first response to be stored,
            # initialize response list
            self.response = []

        if self._dynamic_system and self.n_spt == 1:
            self._current_res = self._current_res[np.newaxis]
        if self.n_r == 1:
            self._current_res = self._current_res[:, np.newaxis]

        if self._var_n_sampling_time:
            self._current_res = np.pad(
                self._current_res,
                pad_width=((0, self.n_spt - self._current_res.shape[0]), (0, 0)),
                mode='constant',
                constant_values=np.nan
            )

        """ convert to list if np array """
        if isinstance(self.response, np.ndarray):
            self.response = self.response.tolist()
        self.response.append(self._current_res)

        """ convert to numpy array """
        self.response = np.array(self.response)
        end = time()
        if self._verbose >= 3:
            print('Storing response took %.6f CPU ms.' % (1000 * (end - start)))
        return self.response

    def _simulate_internal(self, ti_controls, tv_controls, theta, sampling_times):
        raise SyntaxError(
            "Make sure you have initialized the designer, and specified the simulate "
            "function correctly."
        )

    def _initialize_internal_simulate_function(self):
        if self._simulate_signature == 1:
            self._simulate_internal = lambda tic, tvc, mp, spt: \
                self.simulate(tic, mp)
        elif self._simulate_signature == 2:
            self._simulate_internal = lambda tic, tvc, mp, spt: \
                self.simulate(tic, spt, mp)
        elif self._simulate_signature == 3:
            self._simulate_internal = lambda tic, tvc, mp, spt: \
                self.simulate(tvc, spt, mp)
        elif self._simulate_signature == 4:
            self._simulate_internal = lambda tic, tvc, mp, spt: \
                self.simulate(tic, tvc, spt, mp)
        elif self._simulate_signature == 5:
            self._simulate_internal = lambda tic, tvc, mp, spt: \
                self.simulate(spt, mp)
        else:
            raise SyntaxError(
                'Cannot initialize simulate function properly, check your syntax.'
            )

    def _transform_efforts(self):
        if self._unconstrained_form:
            if not self._efforts_transformed:
                self.efforts = np.square(self.efforts)
                self.efforts /= np.sum(self.efforts)
                self._efforts_transformed = True
                if self._verbose >= 3:
                    print("Efforts transformed.")

        return self.efforts

    def _check_missing_components(self):
        # basic components
        if self.model_parameters is None:
            raise SyntaxError("Please specify nominal model parameters.")

        # invariant controls
        if self._invariant_controls and self.ti_controls_candidates is None:
            raise SyntaxError(
                "Simulate function suggests time-invariant controls are needed, but "
                "ti_controls_candidates is empty."
            )

        # dynamic system
        if self._dynamic_system:
            if self.sampling_times_candidates is None:
                raise SyntaxError(
                    "Simulate function suggests dynamic system, but "
                    "sampling_times_candidates is empty."
                )
            if self._dynamic_controls:
                if self.tv_controls_candidates is None:
                    raise SyntaxError(
                        "Simulate function suggests time-varying controls are needed, "
                        "but tv_controls_candidates is empty."
                    )

    def _handle_simulate_sig(self):
        """
        Determines type of model from simulate signature. Five supported types:
        =================================================================================
        1. simulate(ti_controls, model_parameters).
        2. simulate(ti_controls, sampling_times, model_parameters).
        3. simulate(tv_controls, sampling_times, model_parameters).
        4. simulate(ti_controls, tv_controls, sampling_times, model_parameters).
        5. simulate(sampling_times, model_parameters).
        =================================================================================
        If a pyomo.dae model is specified a special signature is recommended that adds
        two input arguments to the beginning of the simulate signatures e.g., for type 3:
        simulate(model, simulator, tv_controls, sampling_times, model_parameters).
        """
        sim_sig = list(signature(self.simulate).parameters.keys())
        unspecified_sig = ["unspecified"]
        if np.all([entry in sim_sig for entry in unspecified_sig]):
            raise SyntaxError("Don't forget to specify the simulate function.")

        t1_sig = ["ti_controls"]
        t2_sig = ["ti_controls", "sampling_times"]
        t3_sig = ["tv_controls", "sampling_times"]
        t4_sig = ["ti_controls", "tv_controls", "sampling_times"]
        t5_sig = ["sampling_times"]
        # initialize simulate id
        self._simulate_signature = 0
        # check if model_parameters is present
        if "model_parameters" not in sim_sig:
            raise SyntaxError(
                f"The input argument \"model_parameters\" is not found in the simulate "
                f"function, please fix simulate signature."
            )
        if np.all([entry in sim_sig for entry in t4_sig]):
            self._simulate_signature = 4
            self._dynamic_system = True
            self._dynamic_controls = True
            self._invariant_controls = True
        elif np.all([entry in sim_sig for entry in t3_sig]):
            self._simulate_signature = 3
            self._dynamic_system = True
            self._dynamic_controls = True
            self._invariant_controls = False
        elif np.all([entry in sim_sig for entry in t2_sig]):
            self._simulate_signature = 2
            self._dynamic_system = True
            self._dynamic_controls = False
            self._invariant_controls = True
        elif np.all([entry in sim_sig for entry in t1_sig]):
            self._simulate_signature = 1
            self._dynamic_system = False
            self._dynamic_controls = False
            self._invariant_controls = True
        elif np.all([entry in sim_sig for entry in t5_sig]):
            self._simulate_signature = 5
            self._dynamic_system = True
            self._dynamic_controls = False
            self._invariant_controls = False
        if self._simulate_signature == 0:
            raise SyntaxError(
                "Unrecognized simulate function signature, please check if you have "
                "specified it correctly. The base signature requires "
                "'model_parameters'. Adding 'sampling_times' makes it dynamic,"
                "adding 'tv_controls' and 'sampling_times' makes a dynamic system with"
                " time-varying controls. Adding 'tv_controls' without 'sampling_times' "
                "does not work. Adding 'model' and 'simulator' makes it a pyomo "
                "simulate signature. 'ti_controls' are optional in all cases."
            )
        self._initialize_internal_simulate_function()

    def _check_stats_framework(self):
        """ check if local or Pseudo-bayesian designs """
        if self.model_parameters.ndim == 1:
            self._pseudo_bayesian = False
        elif self.model_parameters.ndim == 2:
            self._pseudo_bayesian = True
        else:
            raise SyntaxError(
                "model_parameters must be fed in as a 1D numpy array for local "
                "designs, and a 2D numpy array for Pseudo-bayesian designs."
            )

    def _check_candidate_lengths(self):
        if self._invariant_controls:
            self.n_c = self.n_c_tic
        if self._dynamic_controls:
            if not self.n_c:
                self.n_c = self.n_c_tvc
            else:
                assert self.n_c == self.n_c_tvc, f"Inconsistent candidate lengths. " \
                                                 f"tvc_candidates has {self.n_c_tvc}, " \
                                                 f"but {self.n_c} is expected."
        if self._dynamic_system:
            if not self.n_c:
                self.n_c = self.n_c_spt
            else:
                assert self.n_c == self.n_c_spt, f"Inconsistent candidate lengths. " \
                                                 f"spt_candidates has {self.n_c_spt}, " \
                                                 f"but {self.n_c} is expected."

    def _check_var_spt(self):
        if np.all([len(spt) == len(self.sampling_times_candidates[0]) for spt in
                   self.sampling_times_candidates]) \
                and np.all(~np.isnan(self.sampling_times_candidates)):
            self._var_n_sampling_time = False
        else:
            self._var_n_sampling_time = True
            self._pad_sampling_times()

    def _get_component_sizes(self):

        if self._simulate_signature == 1:
            self.n_c_tic, self.n_tic = self.ti_controls_candidates.shape
            self.tv_controls_candidates = np.empty((self.n_c_tic, 1))
            self.n_c_tvc, self.n_tvc = self.n_c_tic, 1
            self.sampling_times_candidates = np.empty_like(self.ti_controls_candidates)
            self.n_c_spt, self.n_spt = self.n_c_tic, 1
        elif self._simulate_signature == 2:
            self.n_c_tic, self.n_tic = self.ti_controls_candidates.shape
            self.tv_controls_candidates = np.empty((self.n_c_tic, 1))
            self.n_c_tvc, self.n_tvc = self.n_c_tic, 1
            self.n_c_spt, self.n_spt = self.sampling_times_candidates.shape
        elif self._simulate_signature == 3:
            self.n_c_tvc, self.n_tvc = self.tv_controls_candidates.shape
            self.ti_controls_candidates = np.empty((self.n_c_tvc, 1))
            self.n_c_tic, self.n_tic = self.n_c_tvc, 1
            self.n_c_spt, self.n_spt = self.sampling_times_candidates.shape
        elif self._simulate_signature == 4:
            self.n_c_tic, self.n_tic = self.ti_controls_candidates.shape
            self.n_c_tvc, self.n_tvc = self.tv_controls_candidates.shape
            self.n_c_spt, self.n_spt = self.sampling_times_candidates.shape
        elif self._simulate_signature == 5:
            self.n_c_spt, self.n_spt = self.sampling_times_candidates.shape
            self.ti_controls_candidates = np.empty((self.n_c_spt, 1))
            self.n_c_tic, self.n_tic = self.n_c_spt, 1
            self.tv_controls_candidates = np.empty((self.n_c_spt, 1))
            self.n_c_tvc, self.n_tvc = self.n_c_spt, 1
        else:
            raise SyntaxError("Unrecognized simulate signature, unable to proceed.")

        # number of model parameters, and scenarios (if pseudo_bayesian)
        if self._pseudo_bayesian:
            self.n_scr, self.n_mp = self.model_parameters.shape
            self._current_scr_mp = self.model_parameters[0]
        else:
            self.n_mp = self.model_parameters.shape[0]
            self._current_scr_mp = self.model_parameters

        # number of responses
        if self.n_r is None:
            if self._verbose >= 3:
                print(
                    "Running one simulation for initialization "
                    "(required to determine number of responses)."
                )
            y = self._simulate_internal(
                self.ti_controls_candidates[0],
                self.tv_controls_candidates[0],
                self._current_scr_mp,
                self.sampling_times_candidates[0][~np.isnan(self.sampling_times_candidates[0])]
            )
            try:
                self.n_spt_r, self.n_r = y.shape
            except ValueError:  # output not two dimensional
                # case 1: n_r is 1
                if self._dynamic_system and self.n_spt > 1:
                    self.n_r = 1
                # case 2: n_spt is 1
                else:
                    self.n_r = y.shape[0]

        # number of measurable responses (if not all)
        if self.measurable_responses is None:
            self.n_m_r = self.n_r
            self.measurable_responses = np.array([_ for _ in range(self.n_r)])
        elif self.n_m_r != len(self.measurable_responses):
            self.n_m_r = len(self.measurable_responses)
            if self.n_m_r > self.n_r:
                raise SyntaxError(
                    "Given number of measurable responses is greater than number of "
                    "responses given."
                )

    def _check_memory_req(self, threshold):
        # check problem size (affects if designer will be memory-efficient or quick)
        self._memory_threshold = threshold
        memory_req = self.n_c * self.n_spt * self.n_m_r * self.n_mp * 8
        if self._pseudo_bayesian:
            memory_req *= self.n_scr
        if memory_req > self._memory_threshold:
            print(
                f'Sensitivity matrix will take {memory_req / 1e9:.2f} GB of memory space '
                f'(more than {self._memory_threshold / 1e9:.2f} GB threshold).'
            )
            self._large_memory_requirement = True

    def _initialize_names(self):
        if self.response_names is None:
            self.response_names = np.array([
                f"Response {_}"
                for _ in range(self.n_m_r)
            ])
        if self.model_parameter_names is None:
            self.model_parameter_names = np.array([
                f"Model Parameter {_}"
                for _ in range(self.n_mp)
            ])
        if self.candidate_names is None:
            self.candidate_names = np.array([
                f"Candidate {_}"
                for _ in range(self.n_c)
            ])
        if self.ti_controls_names is None and self._invariant_controls:
            self.ti_controls_names = np.array([
                f"Time-invariant Control {_}"
                for _ in range(self.n_tic)
            ])
        if self.tv_controls_names is None and self._dynamic_controls:
            self.tv_controls_names = np.array([
                f"Time-varying Control {_}"
                for _ in range(self.n_tvc)
            ])

    def _remove_zero_effort_candidates(self, tol):
        self.efforts[self.efforts < tol] = 0
        self.efforts = self.efforts / self.efforts.sum()
        return self.efforts
