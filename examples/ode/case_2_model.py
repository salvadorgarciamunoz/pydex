"""
case_2_model.py
===============
A → B reaction with Arrhenius kinetics:

    dCA/dt = -k * CA^α
    dCB/dt =  ν * k * CA^α

    k = exp(θ₀ + θ₁ * (T - 273.15) / T)

Time is normalised to [0, 1] using τ = max(sampling_times), so the ODE
system solved by collocation is always on a unit interval regardless of the
experiment duration.  This is a common pattern for variable-duration
experiments and keeps the collocation grid well-conditioned.

Two functions are provided:

  build_pyomo_model()
      Pyomo.DAE model with Lagrange-Radau orthogonal collocation, solved by
      IPOPT.  Returns the pydex IFT contract tuple.  Assigned to
      designer.pyomo_model_fn — provides exact IFT sensitivities via
      PyomoNLP.

      alpha_b (reaction order on CB) is hardcoded to 0, consistent with the
      original example.  The four model parameters are [θ₀, θ₁, α, ν].

  simulate()
      Thin wrapper around build_pyomo_model().  Solves the collocation model
      and extracts [CA, CB] at each requested sampling time by direct lookup
      (sampling times are embedded as FE boundaries so they are exact grid
      members after discretisation).  Returns shape (n_spt, 2).
      Assigned to designer.simulate — used for response evaluation and
      plotting.
"""

import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
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
    Build and solve a Pyomo.DAE model for the A→B reaction using
    Lagrange-Radau orthogonal collocation on a normalised time domain,
    solved by IPOPT.

    Time is normalised: t_norm = t_abs / tau,  tau = max(sampling_times).
    The ODE system is integrated on t_norm ∈ [0, 1].  Sampling times are
    converted to normalised form and embedded as finite-element boundaries
    so they appear as exact members of the collocation grid.

    Parameters are declared as fixed Var so PyomoNLP includes them in the
    primal vector once temporarily unfixed, providing the Jacobian columns
    needed for IFT.

    alpha_b is hardcoded to 0 (B does not autocatalyse).

    Parameters
    ----------
    ti_controls      : array-like  [CA0, T]   — initial concentration, temperature (K)
    model_parameters : array-like  [θ₀, θ₁, α, ν]
    sampling_times   : array-like or None  — absolute measurement times
    nfe              : int  — finite elements
    ncp              : int  — collocation points per element

    Returns  (pydex IFT contract)
    -------
    m           : solved ConcreteModel
    all_vars    : [θ₀, θ₁, α, ν,  ca[t]...,  cb[t]...,  dca_dt[t]...,  dcb_dt[t]...]
                  parameter vars FIRST
    all_bodies  : equality constraint bodies
    t_sorted    : full collocation grid (normalised time) — sampling times
                  are exact members (embedded as FE boundaries before discretisation)
    """
    CA0_val = float(ti_controls[0])
    T_val   = float(ti_controls[1])

    theta_0_val = float(model_parameters[0])
    theta_1_val = float(model_parameters[1])
    alpha_val   = float(model_parameters[2])
    nu_val      = float(model_parameters[3])

    # Flatten and strip non-finite values (pydex may pass NaN padding)
    spt_abs = np.asarray(sampling_times, dtype=float).flatten()
    spt_abs = spt_abs[np.isfinite(spt_abs) & (spt_abs >= 0)]
    tau     = float(np.max(spt_abs))

    # Normalise sampling times to [0, 1]
    spt_norm = spt_abs / tau

    # Embed normalised sampling times as FE boundaries so they appear
    # exactly in the collocation grid after disc.apply_to()
    t_grid = sorted(set(
        np.linspace(0.0, 1.0, nfe + 1).tolist() + spt_norm.tolist()
    ))

    m = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(initialize=t_grid)

    # Model parameters — fixed Var so PyomoNLP includes them once unfixed
    m.theta_0 = pyo.Var(initialize=theta_0_val);  m.theta_0.fix(theta_0_val)
    m.theta_1 = pyo.Var(initialize=theta_1_val);  m.theta_1.fix(theta_1_val)
    m.alpha_a = pyo.Var(initialize=alpha_val);    m.alpha_a.fix(alpha_val)
    m.nu      = pyo.Var(initialize=nu_val);       m.nu.fix(nu_val)

    # Time scale — fixed Var
    m.tau = pyo.Var(initialize=tau);  m.tau.fix(tau)

    # Temperature — declared as a FREE Var, pinned via equality constraint.
    #
    # WHY NOT m.temp.fix(T_val):
    #
    # If temp is fixed, ASL sees theta_1 * (temp-273.15)/temp as
    # theta_1 * constant.  Since theta_1 is also fixed, the entire product
    # collapses to a single number.  ASL substitutes it away and theta_1
    # disappears from the NLP primal vector — PyomoNLP then cannot find it
    # when building the IFT Jacobian.
    #
    # By leaving temp FREE and anchoring it through an equality constraint
    # (temp_fix: temp == T_val), temp remains a live NLP variable.  The
    # expression theta_1 * (temp-273.15)/temp now involves a free variable,
    # so ASL cannot eliminate theta_1 from the Jacobian.
    m.temp     = pyo.Var(initialize=T_val)
    m.temp_fix = pyo.Constraint(expr=m.temp == T_val)

    # State variables on normalised time
    m.ca     = pyo.Var(m.t, initialize=CA0_val, bounds=(0, 50))
    m.cb     = pyo.Var(m.t, initialize=0.0,     bounds=(0, 50))
    m.dca_dt = dae.DerivativeVar(m.ca, wrt=m.t)
    m.dcb_dt = dae.DerivativeVar(m.cb, wrt=m.t)

    # ── Auxiliary variables for the Arrhenius rate constant ──────────────
    # k = exp(θ₀ + θ₁ * (T - 273.15) / T)
    #
    # WHY THE MODEL IS STRUCTURED THIS WAY — ASL variable elimination:
    #
    # For IFT to work, every model parameter must appear in the NLP primal
    # vector.  ASL (the compiled NLP backend used by PyomoNLP) aggressively
    # eliminates variables that are fixed or only appear in expressions that
    # reduce to pure constants.  Three attempts were needed:
    #
    # Attempt 1 — inline expression (FAILS):
    #   `pyo.exp(m.theta_0 + m.theta_1 * ...)` in the material balances.
    #   All of theta_0, theta_1, temp are fixed → the whole expression is
    #   a constant → both parameters eliminated from the NLP.
    #
    # Attempt 2 — single auxiliary k[t] (STILL FAILS for theta_1):
    #   k[t] == exp(theta_0 + theta_1 * (temp-273.15)/temp)
    #   theta_0 survives (chained through k[t]).  But (temp-273.15)/temp
    #   is a fixed constant, so theta_1 * constant is also constant → ASL
    #   eliminates theta_1.
    #
    # Attempt 3 — split ln_k[t] + k[t], WITH temp as free Var (WORKS):
    #   temp is left free and pinned via temp_fix constraint (see above).
    #   ln_k[t] == theta_0 + theta_1 * (temp-273.15)/temp
    #   k[t]    == exp(ln_k[t])
    #   Now (temp-273.15)/temp involves the free variable temp, so ASL
    #   cannot reduce it to a constant.  theta_1 survives in the NLP.
    #   Both theta_0 and theta_1 appear alongside free variables and
    #   PyomoNLP can extract the full 4-column IFT Jacobian.
    #
    # alpha_b hardcoded to 0 → CB term drops out: CA^α * CB^0 = CA^α

    ln_k_init = theta_0_val + theta_1_val * (T_val - 273.15) / T_val
    k_init    = float(np.exp(ln_k_init))

    m.ln_k = pyo.Var(m.t, initialize=ln_k_init)
    m.k    = pyo.Var(m.t, initialize=k_init)

    def ln_k_def_rule(m, t):
        # theta_0 and theta_1 appear here alongside the free var ln_k[t]
        # — ASL cannot eliminate either parameter
        return m.ln_k[t] == m.theta_0 + m.theta_1 * (m.temp - 273.15) / m.temp

    m.ln_k_def = pyo.Constraint(m.t, rule=ln_k_def_rule)

    def k_def_rule(m, t):
        return m.k[t] == pyo.exp(m.ln_k[t])

    m.k_def = pyo.Constraint(m.t, rule=k_def_rule)

    def material_balance_a_rule(m, t):
        return m.dca_dt[t] / m.tau == -m.k[t] * (m.ca[t] ** m.alpha_a)

    m.material_balance_a = pyo.Constraint(m.t, rule=material_balance_a_rule)

    def material_balance_b_rule(m, t):
        return m.dcb_dt[t] / m.tau == m.nu * m.k[t] * (m.ca[t] ** m.alpha_a)

    m.material_balance_b = pyo.Constraint(m.t, rule=material_balance_b_rule)

    # Initial conditions as equality constraints — keeps ca[0], cb[0] free
    # in the NLP so PyomoNLP includes them in the primal vector
    m.ic_a = pyo.Constraint(expr=m.ca[0] == CA0_val)
    m.ic_b = pyo.Constraint(expr=m.cb[0] == 0.0)

    # Dummy objective
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

    # Parameter vars first, then the RESPONSE state vars (ca, cb) immediately
    # after — pydex's IFT extractor identifies response variables by their
    # position: it expects responses to follow directly after the n_mp
    # parameter vars.  Auxiliary vars (ln_k, k) and derivatives must come
    # after the response vars, otherwise pydex picks up ln_k/k as responses
    # and the sensitivity solve (lstsq on J_z_t) sees a singular Jacobian.
    all_vars = (
        [m.theta_0, m.theta_1, m.alpha_a, m.nu]
        + [m.ca[t]     for t in t_sorted_full]
        + [m.cb[t]     for t in t_sorted_full]
        + [m.ln_k[t]   for t in t_sorted_full]
        + [m.k[t]      for t in t_sorted_full]
        + [m.dca_dt[t] for t in t_sorted_full]
        + [m.dcb_dt[t] for t in t_sorted_full]
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
# Assign to designer.simulate  (pydex signature 2)
# =============================================================================

def simulate(ti_controls, sampling_times, model_parameters):
    """
    Thin wrapper around build_pyomo_model().

    Solves the collocation model and extracts [CA, CB] at each requested
    sampling time by direct lookup on the normalised grid.  Sampling times
    are embedded as FE boundaries in build_pyomo_model(), so each normalised
    sampling time is an exact member of the collocation grid — no
    interpolation needed.

    Parameters
    ----------
    ti_controls      : array-like  [CA0, T]
    sampling_times   : array-like  absolute measurement times
    model_parameters : array-like  [θ₀, θ₁, α, ν]

    Returns
    -------
    y : np.ndarray, shape (n_spt, 2)   columns: [CA, CB]
    """
    # Flatten and strip non-finite values
    spt_abs = np.asarray(sampling_times, dtype=float).flatten()
    spt_abs = spt_abs[np.isfinite(spt_abs) & (spt_abs >= 0)]
    tau     = float(np.max(spt_abs))

    m, _, _, _ = build_pyomo_model(ti_controls, model_parameters, spt_abs)

    # Normalised sampling times — exact members of the collocation grid
    spt_norm = spt_abs / tau

    ca = np.array([pyo.value(m.ca[t]) for t in spt_norm])
    cb = np.array([pyo.value(m.cb[t]) for t in spt_norm])

    return np.column_stack([ca, cb])


# =============================================================================
# Main: quick sanity check
# =============================================================================

if __name__ == '__main__':
    pre_exp_constant = 0.1
    activ_energy     = 5000
    theta_0 = np.log(pre_exp_constant) - activ_energy / (8.314159 * 273.15)
    theta_1 = activ_energy / (8.314159 * 273.15)
    theta_nom = np.array([theta_0, theta_1, 1.0, 0.5])

    tic = [1.0, 323.15]
    spt = np.linspace(0, 200, 11)
    spt[0] = 0.001  # avoid t=0 division issues with normalisation

    y = simulate(
        ti_controls=tic,
        sampling_times=spt,
        model_parameters=theta_nom,
    )

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(spt, y[:, 0], label='$c_A$', marker='o')
    axes.plot(spt, y[:, 1], label='$c_B$', marker='o')
    axes.set_xlabel('Time (min)')
    axes.set_ylabel('Concentration (mol/L)')
    axes.set_title('A→B reaction  (collocation + IPOPT)')
    axes.legend()
    fig.tight_layout()
    plt.show()
