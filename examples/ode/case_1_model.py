"""
case_1_model.py
===============
First-order reaction:  A -> B,   dA/dt = -k * CA,   CA(0) = CA0

Two functions are provided:

  build_pyomo_model()
      Pyomo.DAE model with Lagrange-Radau orthogonal collocation, solved by
      IPOPT.  Assigned to designer.pyomo_model_fn — provides exact IFT
      sensitivities via PyomoNLP.

      NOTE: The Pyomo.DAE Simulator (scipy/vode) is intentionally NOT used
      here.  The Simulator integrates numerically but does not discretise the
      DAE into algebraic form, so PyomoNLP cannot produce a correct Jacobian.
      IFT requires a fully discretised algebraic system — collocation provides
      this; the Simulator does not.

  simulate()
      scipy solve_ivp (RK45).  Assigned to designer.simulate — used only for
      fast response evaluation.  Sensitivities always come from
      build_pyomo_model / PyomoNLP IFT, never from this function.
"""

import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

# Collocation settings
NFE = 20   # number of finite elements
NCP = 3    # collocation points per element (Lagrange-Radau)


# =============================================================================
# build_pyomo_model — collocation + IPOPT
# Assign to designer.pyomo_model_fn
# =============================================================================

def build_pyomo_model(ti_controls, model_parameters, sampling_times=None,
                      nfe=NFE, ncp=NCP):
    """
    Build and solve a Pyomo.DAE model for dCA/dt = -k*CA using
    Lagrange-Radau orthogonal collocation, solved by IPOPT.

    Sampling times are embedded as finite-element boundaries so the
    collocation grid hits them exactly, enabling the IFT to extract
    sensitivities at the correct time points.

    Parameters are declared as fixed Var (not Param) so PyomoNLP includes
    them in the primal vector once temporarily unfixed, providing the
    dc/dk Jacobian column needed for the IFT.

    CA0 is a ti_control, not a model parameter — encoded via the ic
    equality constraint (ca[0] == CA0_val) so that ca[0.0] remains free
    in the NLP and is included by PyomoNLP.

    Parameters
    ----------
    ti_controls      : array-like  [CA0]
    model_parameters : array-like  [k]
    sampling_times   : array-like or None  — measurement times (absolute)
    nfe              : int  — finite elements
    ncp              : int  — collocation points per element

    Returns  (pydex IFT contract)
    -------
    m           : solved ConcreteModel
    all_vars    : [k,  ca[t]...,  dca_dt[t]...]   parameter var FIRST
    all_bodies  : equality constraint bodies
    t_sorted    : full collocation grid — sampling times are exact members
                  (embedded as FE boundaries in t_grid before discretisation)
    """
    k_val   = float(model_parameters[0])
    CA0_val = float(ti_controls[0])

    if sampling_times is None or len(sampling_times) == 0:
        t_final = 10.0
        spt_abs = np.array([t_final])
    else:
        spt_abs = np.asarray(sampling_times, dtype=float)
        spt_abs = spt_abs[np.isfinite(spt_abs) & (spt_abs > 0)]
        t_final = float(np.max(spt_abs)) if len(spt_abs) > 0 else 10.0
        spt_abs = np.array([t_final]) if len(spt_abs) == 0 else spt_abs

    # Embed sampling times as finite-element boundaries so they appear
    # exactly in the collocation grid after disc.apply_to()
    t_grid = sorted(set(
        np.linspace(0.0, t_final, nfe + 1).tolist() + spt_abs.tolist()
    ))

    m = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(initialize=t_grid)

    # k is the only model parameter — fixed Var so PyomoNLP sees it as a
    # primal once temporarily unfixed (gives dc/dk column in Jacobian)
    m.k = pyo.Var(initialize=k_val);  m.k.fix(k_val)

    m.ca     = pyo.Var(m.t, initialize=CA0_val, bounds=(0, None))
    m.dca_dt = dae.DerivativeVar(m.ca, wrt=m.t)

    @m.Constraint(m.t)
    def material_balance(m, t):
        return m.dca_dt[t] == -m.k * m.ca[t]

    # IC as equality constraint with numeric RHS — keeps ca[0.0] free in
    # the NLP so PyomoNLP includes it (fixing ca[0] would drop it)
    m.ic = pyo.Constraint(expr=m.ca[0] == CA0_val)

    # Dummy objective (square feasibility problem — zero has no effect)
    m.obj = pyo.Objective(expr=0.0)

    # ── Discretise with Lagrange-Radau collocation ────────────────────────
    disc = pyo.TransformationFactory('dae.collocation')
    disc.apply_to(m, nfe=nfe, ncp=ncp, scheme='LAGRANGE-RADAU')

    # ── Solve with IPOPT ──────────────────────────────────────────────────
    solver = pyo.SolverFactory('ipopt')
    solver.options['print_level'] = 0
    solver.options['tol']         = 1e-12
    result = solver.solve(m, tee=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"IPOPT did not converge: {result.solver.termination_condition}"
        )

    # ── Assemble IFT contract ─────────────────────────────────────────────
    t_sorted_full = sorted(m.t)

    # Sampling times were embedded as FE boundaries in t_grid, so they
    # appear as exact members of the collocation grid — no approximation.
    all_vars = (
        [m.k]
        + [m.ca[t]     for t in t_sorted_full]
        + [m.dca_dt[t] for t in t_sorted_full]
    )

    all_bodies = []
    for con in m.component_objects(pyo.Constraint, active=True):
        for idx in con:
            c = con[idx]
            if c.equality:
                all_bodies.append(c.body - c.upper)

    return m, all_vars, all_bodies, t_sorted_full


# =============================================================================
# simulate — scipy solve_ivp
# Assign to designer.simulate  (pydex signature 2)
# =============================================================================

def simulate(ti_controls, sampling_times, model_parameters):
    """
    Fast response evaluation using scipy solve_ivp (RK45).
    Used by pydex for response storage and plotting only.
    Sensitivities come from build_pyomo_model / PyomoNLP IFT.

    Parameters
    ----------
    ti_controls      : array-like  [CA0]
    sampling_times   : array-like  measurement times
    model_parameters : array-like  [k]

    Returns
    -------
    ca : np.ndarray, shape (n_spt,)
    """
    k   = float(model_parameters[0])
    CA0 = float(ti_controls[0])
    spt = np.asarray(sampling_times, dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: [-k * y[0]],
        t_span=(0.0, float(np.max(spt))),
        y0=[CA0],
        t_eval=spt,
        method='RK45',
        rtol=1e-9,
        atol=1e-11,
    )
    return sol.y[0]


# =============================================================================
# Main: quick sanity check
# =============================================================================

if __name__ == '__main__':
    spt = np.linspace(0, 10, 11)

    y = simulate(
        ti_controls=[1],
        sampling_times=spt,
        model_parameters=[0.25],
    )

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(spt, y)
    axes.set_xlabel('Time')
    axes.set_ylabel('$C_A$')
    axes.set_title('First-order reaction  (k=0.25, CA0=1)')
    plt.show()
