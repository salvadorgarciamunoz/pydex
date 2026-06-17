"""
case_1_no_ift_model.py
======================
First-order reaction:  A -> B,   dCA/dt = -k * CA,   CA(0) = CA0

Variant of case_1_model.py that does NOT use the IFT path.

One function is provided:

  build_pyomo_model()
      Pyomo.DAE model with Lagrange-Radau orthogonal collocation, solved by
      IPOPT.  Returns the full IFT contract tuple — but is NOT assigned to
      designer.pyomo_model_fn.

  simulate()
      Thin wrapper around build_pyomo_model() that extracts the response
      array from the solved Pyomo model and returns it in the format
      designer.simulate expects.  Assigned to designer.simulate only.

      Sensitivities are computed by pydex via finite differences on top of
      simulate() — no IFT, no PyomoNLP Jacobian.

This example demonstrates the wrapper pattern: a single Pyomo collocation
model serves as the sole source of truth for both response evaluation and
(indirectly) sensitivities, with no scipy integrator involved.

NOTE: Finite-difference sensitivities are less accurate and slower than IFT
for stiff or high-dimensional models.  For production use, prefer
case_1_model.py (IFT path).  This file exists to illustrate the architecture.
"""

import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

# Collocation settings
NFE = 20   # number of finite elements
NCP = 3    # collocation points per element (Lagrange-Radau)


# =============================================================================
# build_pyomo_model — collocation + IPOPT
# NOT assigned to designer.pyomo_model_fn in this variant
# =============================================================================

def build_pyomo_model(ti_controls, model_parameters, sampling_times=None,
                      nfe=NFE, ncp=NCP):
    """
    Build and solve a Pyomo.DAE model for dCA/dt = -k*CA using
    Lagrange-Radau orthogonal collocation, solved by IPOPT.

    Sampling times are embedded as finite-element boundaries so they appear
    as exact members of the collocation grid after disc.apply_to().

    Parameters are declared as fixed Var (not Param) so PyomoNLP includes
    them in the primal vector once temporarily unfixed — this is only
    relevant when IFT is active (case_1_model.py).  Here the fixed Var
    declaration is kept for consistency and future reuse.

    CA0 is a ti_control, not a model parameter — encoded via the ic
    equality constraint (ca[0] == CA0_val) so that ca[0.0] remains free
    in the NLP.

    Parameters
    ----------
    ti_controls      : array-like  [CA0]
    model_parameters : array-like  [k]
    sampling_times   : array-like or None  — measurement times (absolute)
    nfe              : int  — finite elements
    ncp              : int  — collocation points per element

    Returns  (pydex IFT contract — unused in this variant)
    -------
    m           : solved ConcreteModel
    all_vars    : [k,  ca[t]...,  dca_dt[t]...]   parameter var FIRST
    all_bodies  : equality constraint bodies
    t_sorted    : full collocation grid — sampling times are exact members
                  (embedded as FE boundaries in t_grid before discretisation)
    """
    k_val   = float(model_parameters[0])
    CA0_val = float(ti_controls[0])

    spt_abs = np.asarray(sampling_times, dtype=float)
    t_final = float(np.max(spt_abs))

    # Embed sampling times as finite-element boundaries so they appear
    # exactly in the collocation grid after disc.apply_to()
    t_grid = sorted(set(
        np.linspace(0.0, t_final, nfe + 1).tolist() + spt_abs.tolist()
    ))

    m = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(initialize=t_grid)

    m.k = pyo.Var(initialize=k_val);  m.k.fix(k_val)

    m.ca     = pyo.Var(m.t, initialize=CA0_val, bounds=(0, None))
    m.dca_dt = dae.DerivativeVar(m.ca, wrt=m.t)

    def material_balance_rule(m, t):
        return m.dca_dt[t] == -m.k * m.ca[t]

    m.material_balance = pyo.Constraint(m.t, rule=material_balance_rule)

    m.ic = pyo.Constraint(expr=m.ca[0] == CA0_val)
    m.obj = pyo.Objective(expr=0.0)

    disc = pyo.TransformationFactory('dae.collocation')
    disc.apply_to(m, nfe=nfe, ncp=ncp, scheme='LAGRANGE-RADAU')

    solver = pyo.SolverFactory('ipopt')
    solver.options['print_level'] = 0
    solver.options['tol']         = 1e-12
    result = solver.solve(m, tee=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"IPOPT did not converge: {result.solver.termination_condition}"
        )

    t_sorted_full = sorted(m.t)

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
# simulate — wrapper around build_pyomo_model
# Assigned to designer.simulate  (pydex signature 2)
# =============================================================================

def simulate(ti_controls, sampling_times, model_parameters):
    """
    Thin wrapper around build_pyomo_model().

    Calls the collocation model and extracts CA at each requested sampling
    time from the solved Pyomo model.  This makes build_pyomo_model() the
    single source of truth for both response evaluation and (via finite
    differences) sensitivities — no separate scipy integrator needed.

    pydex calls this function to:
      - evaluate responses for all candidates during initialize()
      - compute finite-difference sensitivities during design_experiment()
      - extract predictions for plotting

    Parameters
    ----------
    ti_controls      : array-like  [CA0]
    sampling_times   : array-like  measurement times
    model_parameters : array-like  [k]

    Returns
    -------
    ca : np.ndarray, shape (n_spt,)
    """
    # Flatten to 1D and strip non-finite values (pydex may pass NaN padding)
    spt = np.asarray(sampling_times, dtype=float).flatten()
    spt = spt[np.isfinite(spt)]

    m, _, _, t_sorted_full = build_pyomo_model(ti_controls, model_parameters, spt)

    # Extract CA at every collocation point, then interpolate to the
    # requested sampling times.  Sampling times were embedded as FE
    # boundaries so interpolation error is at most machine epsilon —
    # but interpolation is cleaner than snapping to the nearest key.
    t_grid  = np.array(t_sorted_full)
    ca_grid = np.array([pyo.value(m.ca[t]) for t in t_sorted_full])
    ca_interp = interp1d(t_grid, ca_grid, kind='cubic', assume_sorted=True)
    return ca_interp(spt)


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
    axes.plot(spt, y, label='Pyomo collocation')
    axes.set_xlabel('Time')
    axes.set_ylabel('$C_A$')
    axes.set_title('First-order reaction  (k=0.25, CA0=1)  —  no IFT')
    axes.legend()
    plt.show()
