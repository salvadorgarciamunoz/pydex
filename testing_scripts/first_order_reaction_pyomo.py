"""
first_order_reaction_pyomo.py
==============================
D-optimal design of experiments for a first-order reaction
    A -> B,   dA/dt = -k*A   ->   A(t) = A0*exp(-k*t)

This example shows how to use the Pyomo IFT sensitivity path in pydex.
The user only needs to provide build_pyomo_model() — all the generic
IFT machinery (Jacobian assembly, lstsq solve, output extraction) lives
in designer._eval_sensitivities_pyomo_ift().

Usage
-----
Set two attributes on the designer before calling design_experiment():

    d.use_pyomo_ift  = True
    d.pyomo_model_fn = build_pyomo_model

pyomo_model_fn(ti_controls, model_parameters) must return:
    (model, all_vars, all_bodies, t_sorted)

where all_vars has the n_mp parameter Vars FIRST (declared as fixed Var,
not Param), followed by the state variables.  The designer uses n_mp
(already known) to split the Jacobian into J_p and J_z automatically.

Runs
----
  1. Local D-optimal        — nominal k=0.5, A0=1.0
  2. Pseudo-Bayesian        — 200 samples of k ~ U[0.1, 1.0], A0=1 fixed
  3. Pseudo-Bayesian        — 200 samples of both k and A0 uncertain
"""

import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae

from pydex.core.designer import Designer



# ─────────────────────────────────────────────────────────────────────────────
# Model builder  —  the ONLY model-specific function needed
# ─────────────────────────────────────────────────────────────────────────────

def build_pyomo_model(ti_controls, model_parameters, sampling_times=None, nfe=20, ncp=3):
    """
    Build and solve a Pyomo.DAE model for:
        dA/dt = -k * A,   A(0) = A0

    Uses Lagrange-Radau orthogonal collocation (nfe=3, ncp=3 by default),
    which gives near-exact accuracy for this smooth ODE with very few
    elements.  With ncp=3 the local error is O(h^5), giving essentially
    machine-precision results for a simple exponential decay.

    This is far superior to backward-Euler, which needs nfe~500 to achieve
    comparable accuracy for large t values (e.g. at t=9.2 with k=0.5,
    backward-Euler with nfe=20 gives A=0.81 vs the true value 0.01).

    The model is solved with IPOPT so that variable values are at the
    true collocation solution — the Jacobian from PyomoNLP is then
    evaluated at the correct point and the IFT sensitivities are exact
    for this discretisation.

    k and A0 are declared as fixed Var (not Param) so that PyomoNLP
    includes them in the primal vector once temporarily unfixed,
    providing dc/dk and dc/dA0 columns in the Jacobian.

    Parameters
    ----------
    ti_controls      : array-like — [t_sample], the observation time
    model_parameters : array-like — [k, A0]
    nfe              : int        — number of finite elements
    ncp              : int        — collocation points per element

    Returns
    -------
    m           : solved ConcreteModel
    all_vars    : [k, A0, A[t0], ..., A[tn], dAdt[t0], ..., dAdt[tn]]
    all_bodies  : equality constraint body expressions
    t_sorted    : sorted list of time-point floats
    """
    # ------------------------------------------------------------------
    # IMPORTANT — always unpack inputs with np.asarray().flatten()[i].
    #
    # Do NOT use float(model_parameters[0]) or float(ti_controls[0])
    # directly.  pydex and the ASL diagnostic can pass these arguments
    # as plain Python scalars, 1-D arrays, OR 0-dimensional numpy arrays
    # depending on the call site.  float() raises
    #   "TypeError: only 0-dimensional arrays can be converted to Python
    #    scalars"
    # when given a multi-element array, and silently does the wrong thing
    # on a 0-d array produced by indexing a numpy array.
    #
    # np.asarray(...).flatten()[i] normalises all three cases to a plain
    # 1-D array before indexing, so float() always receives a true scalar.
    # ------------------------------------------------------------------
    k_val  = float(np.asarray(model_parameters).flatten()[0])
    A0_val = float(np.asarray(model_parameters).flatten()[1])
    t_f    = float(np.asarray(ti_controls).flatten()[0])

    # ------------------------------------------------------------------
    # IMPORTANT — guard against t_f=0 when called by the ASL diagnostic.
    #
    # Designer.initialize() calls build_pyomo_model() via
    # diagnose_asl_elimination() to verify that all parameter Vars are
    # reachable in the ASL primal vector.  The diagnostic uses
    # ti_controls_candidates[0] as its probe point — for this script that
    # is t=0, which triggers the trivial guard below and returns a
    # degenerate model that PyomoNLP cannot compile.
    #
    # When sampling_times is provided (as the diagnostic always does),
    # promote t_f to the largest finite sampling time so that a full
    # collocation model is built and the ASL check can complete.
    # ------------------------------------------------------------------
    if t_f <= 0.0 and sampling_times is not None:
        spt = np.asarray(sampling_times, dtype=float).flatten()
        spt = spt[np.isfinite(spt) & (spt > 0)]
        if len(spt) > 0:
            t_f = float(np.max(spt))

    # Guard: t=0 — trivial case, A=A0, no ODE needed
    if t_f <= 0.0:
        m = pyo.ConcreteModel()
        m.A  = pyo.Var(initialize=A0_val);  m.A.fix(A0_val)
        m.k  = pyo.Var(initialize=k_val);   m.k.fix(k_val)
        m.A0 = pyo.Var(initialize=A0_val);  m.A0.fix(A0_val)
        m.trivial = pyo.Constraint(expr=m.A == m.A0)
        # Dummy objective required by PyomoNLP — this model is a square
        # feasibility problem (simulation), not an optimisation. The zero
        # objective has no effect on the solution.
        m.obj = pyo.Objective(expr=0.0)
        all_vars   = [m.k, m.A0, m.A]
        all_bodies = [m.trivial.body]
        return m, all_vars, all_bodies, [0.0]

    # Use an explicit uniform grid — for this single-endpoint model (t_f is
    # the only measurement time and is the domain boundary, not an interior
    # point), the explicit linspace grid and the ContinuousSet(bounds, initialize)
    # pattern are mathematically equivalent but the explicit grid is faster
    # because Pyomo can skip internal grid-construction work.
    # The standard Pyomo.DAE parameter-estimation pattern applies when
    # intermediate measurement times need to be preserved as FE boundaries.
    t_grid = np.linspace(0.0, t_f, nfe + 1).tolist()

    m = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(initialize=t_grid)

    # Parameter vars declared as fixed — PyomoNLP will include them
    # once unfixed temporarily, giving dc/dk and dc/dA0 in the Jacobian.
    m.k  = pyo.Var(initialize=k_val);   m.k.fix(k_val)
    m.A0 = pyo.Var(initialize=A0_val);  m.A0.fix(A0_val)

    m.A    = pyo.Var(m.t, initialize=A0_val, bounds=(0, None))
    m.dAdt = dae.DerivativeVar(m.A, withrespectto=m.t)

    @m.Constraint(m.t)
    def ode(m, t):
        return m.dAdt[t] == -m.k * m.A[t]

    @m.Constraint()
    def ic(m):
        return m.A[0.0] == m.A0

    # Dummy objective required by PyomoNLP — this model is a square
    # feasibility problem (simulation), not an optimisation. The zero
    # objective has no effect on the solution.
    m.obj = pyo.Objective(expr=0.0)

    # Lagrange-Radau orthogonal collocation
    disc = pyo.TransformationFactory('dae.collocation')
    disc.apply_to(m, nfe=nfe, ncp=ncp, scheme='LAGRANGE-RADAU')

    # Solve with IPOPT — variable values must be at the true collocation
    # solution for the PyomoNLP Jacobian to give correct IFT sensitivities.
    solver = pyo.SolverFactory('ipopt')
    solver.options['print_level'] = 0
    solver.options['tol'] = 1e-12
    result = solver.solve(m, tee=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"IPOPT did not converge for t={t_f}: "
            f"{result.solver.termination_condition}"
        )

    t_sorted_full = sorted(m.t)

    # Return only t_f as the "sampling time" — for signature-1 models pydex
    # passes self._current_spt=[0.0] as a placeholder, and the designer snaps
    # to the nearest time in t_sorted. By returning [t_f] as t_sorted, the
    # snap always lands on A[t_f], which is the correct output variable.
    t_sorted = [t_sorted_full[-1]]   # only the endpoint matters for output extraction

    # all_vars: parameter vars FIRST, then ALL state vars (needed for full Jacobian)
    all_vars = (
        [m.k, m.A0]
        + [m.A[t] for t in t_sorted_full]
        + [m.dAdt[t] for t in t_sorted_full]
    )

    # Collect all active equality constraint bodies
    all_bodies = []
    for con in m.component_objects(pyo.Constraint, active=True):
        for idx in con:
            c = con[idx]
            if c.equality:
                all_bodies.append(c.body - c.upper)

    return m, all_vars, all_bodies, t_sorted  # t_sorted = [t_f] for output snapping


# ─────────────────────────────────────────────────────────────────────────────
# Standard pydex simulate function  —  signature 1
# ─────────────────────────────────────────────────────────────────────────────

def simulate(ti_controls, model_parameters):
    """
    Pydex simulate signature 1: simulate(ti_controls, model_parameters)

    Calls build_pyomo_model() directly to guarantee that simulate() and
    the Pyomo IFT sensitivity model use the exact same discretisation.
    This is the only way to ensure self-consistency: if both use the same
    collocation scheme and the same IPOPT solve, the sensitivities from
    the IFT are exact derivatives of the values returned by simulate().
    """
    t_sample = float(ti_controls[0])
    A0_val   = float(model_parameters[1])
    if t_sample == 0.0:
        return np.array([A0_val])
    m, all_vars, all_bodies, t_sorted = build_pyomo_model(ti_controls, model_parameters)
    t_end = t_sorted[-1]
    return np.array([pyo.value(m.A[t_end])])


# ─────────────────────────────────────────────────────────────────────────────
# Candidate grid  (same as first_order_reaction_ipopt.py)
# ─────────────────────────────────────────────────────────────────────────────

t_max        = 10.0
n_candidates = 51
t_candidates = np.linspace(0.0, t_max, n_candidates).reshape(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def make_designer(model_parameters, verbose=0):
    d = Designer()
    d.simulate               = simulate
    d.model_parameters       = model_parameters
    d.ti_controls_candidates = t_candidates
    d.pyomo_model_fn         = build_pyomo_model
    # use_pyomo_ift=True and n_jobs=-1 are set automatically by initialize()
    # when pyomo_model_fn is provided.
    d.initialize(verbose=verbose)
    return d


def print_result(d, label):
    p_min    = 1e-4
    efforts  = d.efforts.flatten()
    t_vals   = t_candidates.flatten()
    supports = [(t_vals[i], efforts[i])
                for i in range(len(efforts)) if efforts[i] > p_min]
    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'─'*60}")
    print(f"  Criterion value    : {d._criterion_value:.6f}")
    print(f"  Sensitivity method : Pyomo IFT (exact symbolic Jacobian)")
    print(f"  Sensitivity time   : {d._sensitivity_analysis_time:.3f} s")
    print(f"  Optimal sampling times ({len(supports)} support points):")
    for t, p in sorted(supports):
        print(f"    t = {t:6.3f}   effort = {p:.4f}")


if __name__ == "__main__":
    np.random.seed(42)
    # ═════════════════════════════════════════════════════════════════════════════
    # RUN 1 — Local D-optimal
    # ═════════════════════════════════════════════════════════════════════════════
    print("\n" + "█"*64)
    print("  RUN 1: Local D-optimal  (k=0.5, A0=1.0)")
    print("█"*64)

    d1 = make_designer(np.array([0.5, 1.0]))
    d1.design_experiment(
        d1.d_opt_criterion,
        solver="ipopt",
        solver_options={"linear_solver": "ma57",
            "tol": 1e-10, "max_iter": 3000},
    )
    print_result(d1, "IPOPT / MA57 — local D-optimal")


    # ═════════════════════════════════════════════════════════════════════════════
    # RUN 2 — Pseudo-Bayesian, k uncertain
    # ═════════════════════════════════════════════════════════════════════════════
    print("\n" + "█"*64)
    print("  RUN 2: Pseudo-Bayesian D-optimal  (k uncertain, A0=1 fixed)")
    print("█"*64)

    N_scr     = 200
    scenarios2 = np.column_stack([np.random.uniform(0.1, 1.0, N_scr), np.ones(N_scr)])


    d2 = make_designer(scenarios2, verbose=1)
    d2.design_experiment(
        d2.d_opt_criterion,
        solver="ipopt",
        solver_options={"linear_solver": "ma57", "tol": 1e-8, "max_iter": 5000},
        pseudo_bayesian_type=0,   # 0 = average-information (avg FIM then log-det)
    )
    print_result(d2, "IPOPT / MA57 — pseudo-Bayesian D-optimal (k uncertain)")


    # ═════════════════════════════════════════════════════════════════════════════
    # RUN 3 — Pseudo-Bayesian, both k and A0 uncertain
    # ═════════════════════════════════════════════════════════════════════════════
    print("\n" + "█"*64)
    print("  RUN 3: Pseudo-Bayesian D-optimal  (both k and A0 uncertain)")
    print("█"*64)

    scenarios3 = np.column_stack([
        np.random.uniform(0.1, 1.0, N_scr),
        np.random.uniform(0.5, 2.0, N_scr),
    ])

    d3 = make_designer(scenarios3)
    d3.design_experiment(
        d3.d_opt_criterion,
        solver="ipopt",
        solver_options={"linear_solver": "ma57", "tol": 1e-8, "max_iter": 5000},
        pseudo_bayesian_type=0,
    )
    print_result(d3, "IPOPT / MA57 — pseudo-Bayesian D-optimal (k and A0 uncertain)")


    print(f"\n{'='*64}")
    print("  All runs complete.")
    print(f"{'='*64}\n")
