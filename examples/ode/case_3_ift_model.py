"""
case_3_ift_model.py
===================
Pyomo.DAE collocation model for the Michaelis-Menten-style reaction network,
providing exact IFT sensitivities via PyomoNLP.

Reaction system
---------------
    A → B    (irreversible, inhibited power-law rate)

    dCA/dt = -τ * r
    dCB/dt =  τ * ν * r

    r(CA, T) = k1(T) * CA^α / (k2(T) + k3(T) * CA^β)

    ki(T) = exp(θ_i0 + θ_i1 * (T - 273.15) / T)    i = 1, 2, 3

Time is normalised to [0, 1]: t_norm = t_abs / τ.
The ODEs are solved on t_norm ∈ [0, 1] with Lagrange-Radau collocation.

Nine model parameters : [θ_10, θ_11, θ_20, θ_21, θ_30, θ_31, ν, α, β]
Three time-invariant controls : [CA0 (mol/L), T (K), τ]
Two measurable responses : [CA(t), CB(t)]

Two functions are provided:

  build_pyomo_model()
      Pyomo.DAE model with Lagrange-Radau orthogonal collocation, solved by
      IPOPT.  Returns the pydex IFT contract tuple:
          (model, all_vars, all_bodies, t_sorted)
      Assigned to designer.pyomo_model_fn.

  simulate()
      Thin wrapper around build_pyomo_model().  Extracts [CA, CB] at each
      requested sampling time by direct lookup on the collocation grid
      (sampling times are embedded as FE boundaries so they are exact grid
      members after discretisation).  Returns shape (n_spt, 2).
      Assigned to designer.simulate.

ASL variable elimination — why the model is structured this way
---------------------------------------------------------------
For IFT to work, every model parameter must survive in the NLP primal vector
that PyomoNLP assembles.  AMPL Solver Library (ASL), the compiled NLP backend
used by PyomoNLP, aggressively eliminates variables whose values are determined
purely by fixed quantities.

This model has 9 parameters and 3 Arrhenius expressions.  The pattern from
case_2_model.py is extended here:

  Problem: if T is fixed AND θ_i0, θ_i1 are fixed, the expression
           θ_i0 + θ_i1*(T-273.15)/T is a compile-time constant.
           ASL substitutes it away → the parameters disappear from the NLP.

  Fix applied (same as case_2):
    1. T is declared as a FREE Var, pinned via an equality constraint
       (temp_fix: temp == T_val).  This keeps (T-273.15)/T as a live
       NLP sub-expression involving a free variable, preventing collapse.
    2. Each Arrhenius expression is split into two auxiliary Var arrays:
         ln_ki[t] == θ_i0 + θ_i1 * (temp - 273.15) / temp
         ki[t]    == exp(ln_ki[t])
       The chained free-variable dependency (temp → ln_ki → ki) ensures
       all six Arrhenius parameters survive in the Jacobian.
    3. α and β appear inline in the material balance constraints as
       (ca[t] + ε)^α and (ca[t] + ε)^β.  Since ca[t] is a free NLP
       variable, ASL cannot collapse these power expressions to constants,
       so α and β survive in the Jacobian without needing extra auxiliaries.
       This mirrors the approach in case_2_model.py (ca[t]**alpha_a inline).
       The earlier ln_ca/ca_alpha/ca_beta/denom auxiliary approach was
       abandoned because it made J_z rank-deficient (SVD failure in lstsq).

Variable ordering in all_vars
------------------------------
pydex's IFT extractor identifies the response variables by position:
parameters FIRST (n_mp entries), then response state vars (CA, CB),
then all auxiliary vars.  Any deviation causes the IFT lstsq solve to
see wrong columns and produce garbage sensitivities.

Order used here:
  [θ_10, θ_11, θ_20, θ_21, θ_30, θ_31, ν, α, β]   ← 9 parameter vars
  [ca[t] for t in grid]                               ← response 1
  [cb[t] for t in grid]                               ← response 2
  [ln_k1[t], k1[t], ln_k2[t], k2[t],
   ln_k3[t], k3[t],
   dca_dt[t], dcb_dt[t]]                              ← auxiliaries last
"""

import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
from matplotlib import pyplot as plt
import logging

# Collocation settings — increase NFE for stiffer conditions (high T, large τ)
NFE = 20   # number of finite elements
NCP = 3    # collocation points per element (Lagrange-Radau, order 2*NCP-1 = 5)

# Small epsilon to guard log(CA) against CA → 0 in the auxiliary constraints.
# This is a modelling regularisation, not a numerical hack: at CA = ε the
# reaction rate is effectively zero and the profile is flat, so ε only affects
# candidates that are already degenerate (and would be removed by the
# feasibility filter in case_3_ift.py).
_LOG_EPS = 1e-8


# =============================================================================
# build_pyomo_model — collocation + IPOPT
# Assign to designer.pyomo_model_fn
# =============================================================================

def build_pyomo_model(ti_controls, model_parameters, sampling_times=None,
                      nfe=NFE, ncp=NCP):
    """
    Build and solve a Pyomo.DAE model for the Michaelis-Menten reaction
    using Lagrange-Radau orthogonal collocation on normalised time [0, 1].

    Parameters
    ----------
    ti_controls      : array-like, length 3
        [CA0 (mol/L), T (K), τ]
    model_parameters : array-like, length 9
        [θ_10, θ_11, θ_20, θ_21, θ_30, θ_31, ν, α, β]
    sampling_times   : array-like or None
        Absolute measurement times.  Embedded as FE boundaries so they
        appear as exact members of the collocation grid.
    nfe : int
        Number of finite elements (default 20).
    ncp : int
        Collocation points per element (default 3, Lagrange-Radau order 5).

    Returns  (pydex IFT contract)
    -------
    m           : solved ConcreteModel
    all_vars    : list of Pyomo Var references — parameters first, then
                  response state vars, then auxiliaries (see module docstring)
    all_bodies  : list of equality constraint body expressions
    t_sorted    : sorted list of all collocation time points (normalised)
    """
    # ── Unpack controls and parameters ───────────────────────────────────────
    CA0_val = float(ti_controls[0])
    T_val   = float(ti_controls[1])
    tau_val = float(ti_controls[2])

    th10, th11 = float(model_parameters[0]), float(model_parameters[1])
    th20, th21 = float(model_parameters[2]), float(model_parameters[3])
    th30, th31 = float(model_parameters[4]), float(model_parameters[5])
    nu_val     = float(model_parameters[6])
    alpha_val  = float(model_parameters[7])
    beta_val   = float(model_parameters[8])

    # ── Sampling times ────────────────────────────────────────────────────────
    spt_abs = np.asarray(sampling_times, dtype=float).flatten()
    spt_abs = spt_abs[np.isfinite(spt_abs) & (spt_abs >= 0)]

    # Normalise to [0, 1] using τ from ti_controls.
    # Unlike case_2 (where τ = max(spt)), here τ is an explicit control — the
    # model time scale is fixed by the experiment design, not the sampling grid.
    spt_norm = spt_abs / tau_val

    # Embed normalised sampling times as FE boundaries so they are exact
    # collocation grid members after disc.apply_to()
    t_grid = sorted(set(
        np.linspace(0.0, 1.0, nfe + 1).tolist() + spt_norm.tolist()
    ))

    # ── Initialisation values ─────────────────────────────────────────────────
    T_shift    = (T_val - 273.15) / T_val
    ln_k1_init = th10 + th11 * T_shift
    ln_k2_init = th20 + th21 * T_shift
    ln_k3_init = th30 + th31 * T_shift
    k1_init    = float(np.exp(ln_k1_init))
    k2_init    = float(np.exp(ln_k2_init))
    k3_init    = float(np.exp(ln_k3_init))

    # Initialisation for rate (used to warm-start ca, cb)
    ca_reg_init  = max(CA0_val, _LOG_EPS)
    r_init       = (k1_init * ca_reg_init**alpha_val /
                    (k2_init + k3_init * ca_reg_init**beta_val))

    # ── Build ConcreteModel ───────────────────────────────────────────────────
    m   = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(initialize=t_grid)

    # ── Model parameters — fixed Var ─────────────────────────────────────────
    # Declared as Var (not Param) so PyomoNLP includes them in the primal
    # vector once temporarily unfixed during the IFT Jacobian extraction.
    m.theta_10 = pyo.Var(initialize=th10);   m.theta_10.fix(th10)
    m.theta_11 = pyo.Var(initialize=th11);   m.theta_11.fix(th11)
    m.theta_20 = pyo.Var(initialize=th20);   m.theta_20.fix(th20)
    m.theta_21 = pyo.Var(initialize=th21);   m.theta_21.fix(th21)
    m.theta_30 = pyo.Var(initialize=th30);   m.theta_30.fix(th30)
    m.theta_31 = pyo.Var(initialize=th31);   m.theta_31.fix(th31)
    m.nu       = pyo.Var(initialize=nu_val);  m.nu.fix(nu_val)
    m.alpha    = pyo.Var(initialize=alpha_val); m.alpha.fix(alpha_val)
    m.beta     = pyo.Var(initialize=beta_val);  m.beta.fix(beta_val)

    # ── Temperature — free Var pinned via equality constraint ─────────────────
    # See module docstring (ASL elimination, item 1).  If temp is fixed,
    # (temp-273.15)/temp collapses to a compile-time constant and all six
    # Arrhenius parameters are eliminated from the NLP Jacobian.
    m.temp     = pyo.Var(initialize=T_val)
    m.temp_fix = pyo.Constraint(expr=m.temp == T_val)

    # ── State variables on normalised time ───────────────────────────────────
    m.ca     = pyo.Var(m.t, initialize=CA0_val, bounds=(0, None))
    m.cb     = pyo.Var(m.t, initialize=0.0,     bounds=(0, None))
    m.dca_dt = dae.DerivativeVar(m.ca, wrt=m.t)
    m.dcb_dt = dae.DerivativeVar(m.cb, wrt=m.t)

    # ── Auxiliary variables — Arrhenius rate constants ────────────────────────
    # Split into ln_ki[t] and ki[t] to prevent ASL eliminating θ_i0, θ_i1.
    # See module docstring (ASL elimination, item 2).
    m.ln_k1 = pyo.Var(m.t, initialize=ln_k1_init)
    m.k1    = pyo.Var(m.t, initialize=k1_init, bounds=(0, None))
    m.ln_k2 = pyo.Var(m.t, initialize=ln_k2_init)
    m.k2    = pyo.Var(m.t, initialize=k2_init, bounds=(0, None))
    m.ln_k3 = pyo.Var(m.t, initialize=ln_k3_init)
    m.k3    = pyo.Var(m.t, initialize=k3_init, bounds=(0, None))

    def ln_k1_def(m, t):
        # θ_10 and θ_11 appear alongside free var temp → ASL cannot eliminate
        return m.ln_k1[t] == m.theta_10 + m.theta_11 * (m.temp - 273.15) / m.temp
    m.ln_k1_def = pyo.Constraint(m.t, rule=ln_k1_def)

    def k1_def(m, t):
        return m.k1[t] == pyo.exp(m.ln_k1[t])
    m.k1_def = pyo.Constraint(m.t, rule=k1_def)

    def ln_k2_def(m, t):
        return m.ln_k2[t] == m.theta_20 + m.theta_21 * (m.temp - 273.15) / m.temp
    m.ln_k2_def = pyo.Constraint(m.t, rule=ln_k2_def)

    def k2_def(m, t):
        return m.k2[t] == pyo.exp(m.ln_k2[t])
    m.k2_def = pyo.Constraint(m.t, rule=k2_def)

    def ln_k3_def(m, t):
        return m.ln_k3[t] == m.theta_30 + m.theta_31 * (m.temp - 273.15) / m.temp
    m.ln_k3_def = pyo.Constraint(m.t, rule=ln_k3_def)

    def k3_def(m, t):
        return m.k3[t] == pyo.exp(m.ln_k3[t])
    m.k3_def = pyo.Constraint(m.t, rule=k3_def)

    # ── Material balances ─────────────────────────────────────────────────────
    # CA^α and CA^β are written inline as (ca[t] + ε)^α and (ca[t] + ε)^β.
    # This mirrors the pattern in case_2_model.py which uses ca[t]**alpha_a
    # directly in the constraint body.  The power expression involves a free
    # variable (ca[t]) so ASL cannot collapse it to a constant — α and β
    # survive in the NLP primal vector without needing extra auxiliary Vars.
    # The small ε guard avoids pyo.log(0) / 0^β at CA=0.
    #
    # The division k1*CA^α / (k2 + k3*CA^β) is kept inline — the denominator
    # is nonzero by construction (k2 > 0) so there is no division-by-zero risk
    # at nominal parameters. This eliminates the ln_ca, ca_alpha, ca_beta and
    # denom auxiliary arrays, greatly reducing the size of J_z and curing the
    # SVD rank deficiency.
    m.tau_param = pyo.Param(initialize=tau_val, mutable=False)

    def material_balance_a(m, t):
        ca_reg = m.ca[t] + _LOG_EPS
        r = m.k1[t] * (ca_reg ** m.alpha) / (m.k2[t] + m.k3[t] * (ca_reg ** m.beta))
        return m.dca_dt[t] == -m.tau_param * r
    m.material_balance_a = pyo.Constraint(m.t, rule=material_balance_a)

    def material_balance_b(m, t):
        ca_reg = m.ca[t] + _LOG_EPS
        r = m.k1[t] * (ca_reg ** m.alpha) / (m.k2[t] + m.k3[t] * (ca_reg ** m.beta))
        return m.dcb_dt[t] == m.tau_param * m.nu * r
    m.material_balance_b = pyo.Constraint(m.t, rule=material_balance_b)

    # ── Initial conditions as equality constraints ────────────────────────────
    # Kept as constraints (not .fix()) so ca[0] and cb[0] remain free NLP
    # variables — required for PyomoNLP to include them in the primal vector.
    m.ic_a = pyo.Constraint(expr=m.ca[0] == CA0_val)
    m.ic_b = pyo.Constraint(expr=m.cb[0] == 0.0)

    # ── Dummy objective ───────────────────────────────────────────────────────
    m.obj = pyo.Objective(expr=0.0)

    # ── Discretise with Lagrange-Radau collocation ────────────────────────────
    # Suppress the "More finite elements were found" warning that Pyomo emits
    # when the ContinuousSet has more breakpoints than nfe (which is expected
    # here because we embed sampling times as additional FE boundaries).
    # This warning fires in every worker process during parallel IFT evaluation
    # and cannot be suppressed from the main process — it must be silenced here.
    _pyomo_log = logging.getLogger('pyomo')
    _prev_level = _pyomo_log.level
    _pyomo_log.setLevel(logging.ERROR)
    disc = pyo.TransformationFactory('dae.collocation')
    disc.apply_to(m, nfe=nfe, ncp=ncp, scheme='LAGRANGE-RADAU')
    _pyomo_log.setLevel(_prev_level)

    # ── Solve with IPOPT ──────────────────────────────────────────────────────
    solver = pyo.SolverFactory('ipopt')
    solver.options['print_level'] = 0
    solver.options['tol']         = 1e-8    # 1e-12 caused internalSolverError under parallel forking
    # IMPORTANT: use load_solutions=False so Pyomo does NOT automatically call
    # load_from() before we have checked the solver status.  When IPOPT returns
    # a hard error (status='error', not just 'infeasible') the automatic load
    # raises "Cannot load a SolverResults object with bad status: error" inside
    # the joblib worker, crashing the entire parallel sensitivity run.  By
    # deferring the load we can inspect the status first and raise a clean
    # RuntimeError, which pydex catches and converts to a NaN sensitivity so
    # the remaining candidates can complete normally.
    result = solver.solve(m, tee=False, load_solutions=False)

    _ok_conditions = (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
    )
    if (result.solver.status != pyo.SolverStatus.ok or
            result.solver.termination_condition not in _ok_conditions):
        raise RuntimeError(
            f"IPOPT failed for candidate "
            f"[CA0={CA0_val:.3f}, T={T_val:.2f}, tau={tau_val:.3f}]: "
            f"status={result.solver.status}, "
            f"condition={result.solver.termination_condition}"
        )

    m.solutions.load_from(result)

    # ── Assemble IFT contract ─────────────────────────────────────────────────
    # t_sorted_full: the complete collocation grid — used for all_vars and
    #                all_bodies so the full Jacobian is available to pydex.
    # t_sorted_spt:  only the normalised sampling times snapped to the nearest
    #                collocation point — returned as the 4th element of the
    #                contract tuple so pydex knows which rows of the Jacobian
    #                correspond to the requested measurement times.
    #
    # IMPORTANT: returning the full collocation grid as the 4th element causes
    # pydex to build an over-determined J_z (one row per collocation point
    # rather than per sampling time) which makes the IFT lstsq singular.
    t_sorted_full = sorted(m.t)

    # Snap each normalised sampling time to the nearest collocation grid point
    # (difference is ~machine epsilon since spt_norm was embedded as FE
    # boundaries before disc.apply_to()).
    t_sorted_spt = sorted(set(
        min(t_sorted_full, key=lambda tt: abs(tt - float(t)))
        for t in spt_norm
    ))

    # CRITICAL ordering — see module docstring:
    #   parameters (9) → response states (ca, cb) → all auxiliaries
    # all_vars and all_bodies span the FULL collocation grid so the complete
    # constraint Jacobian is available; pydex uses t_sorted_spt to select
    # the relevant rows after the IFT solve.
    all_vars = (
        # 9 model parameters — must be first
        [m.theta_10, m.theta_11,
         m.theta_20, m.theta_21,
         m.theta_30, m.theta_31,
         m.nu, m.alpha, m.beta]
        # Response state variables — immediately after parameters
        + [m.ca[t] for t in t_sorted_full]
        + [m.cb[t] for t in t_sorted_full]
        # Auxiliary variables — after responses
        # (ln_ki / ki pairs only — alpha and beta survive ASL via inline power)
        + [m.ln_k1[t] for t in t_sorted_full]
        + [m.k1[t]    for t in t_sorted_full]
        + [m.ln_k2[t] for t in t_sorted_full]
        + [m.k2[t]    for t in t_sorted_full]
        + [m.ln_k3[t] for t in t_sorted_full]
        + [m.k3[t]    for t in t_sorted_full]
        + [m.dca_dt[t] for t in t_sorted_full]
        + [m.dcb_dt[t] for t in t_sorted_full]
    )

    all_bodies = []
    for con in m.component_objects(pyo.Constraint, active=True):
        for idx in con:
            c = con[idx]
            if c.equality:
                all_bodies.append(c.body - c.upper)

    return m, all_vars, all_bodies, t_sorted_spt


# =============================================================================
# simulate — wrapper around build_pyomo_model
# Assign to designer.simulate  (pydex signature 2)
# =============================================================================

def simulate(ti_controls, sampling_times, model_parameters):
    """
    Solve the collocation model and return [CA, CB] at each requested
    sampling time.

    Sampling times are embedded as FE boundaries in build_pyomo_model(),
    so each normalised sampling time is an exact collocation grid member —
    no interpolation needed.

    Parameters
    ----------
    ti_controls      : array-like, length 3  [CA0, T, τ]
    sampling_times   : array-like            absolute measurement times
    model_parameters : array-like, length 9

    Returns
    -------
    y : np.ndarray, shape (n_spt, 2)   columns: [CA, CB]
    """
    spt_abs = np.asarray(sampling_times, dtype=float).flatten()
    spt_abs = spt_abs[np.isfinite(spt_abs) & (spt_abs >= 0)]
    tau_val = float(ti_controls[2])

    m, _, _, _ = build_pyomo_model(ti_controls, model_parameters, spt_abs)

    # Normalised sampling times were embedded as FE boundaries so they are
    # exact members of the collocation grid — direct lookup, no interpolation
    spt_norm = spt_abs / tau_val
    ca = np.array([pyo.value(m.ca[t]) for t in spt_norm])
    cb = np.array([pyo.value(m.cb[t]) for t in spt_norm])

    return np.column_stack([ca, cb])


# =============================================================================
# Main: quick sanity check
# =============================================================================

if __name__ == '__main__':
    from time import time

    tic = [10.0, 303.15, 10.0]
    mp  = [5.4, 5.0, 6.2, 0.5, 1.4, 2.5, 7/3, 3, 5]
    spt = np.linspace(0.001, 1, 11)

    start = time()
    c = simulate(tic, spt, mp)
    print(f"One simulation took {time() - start:.3f} CPU seconds.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(spt, c[:, 0], label="$c_A$", marker='o')
    ax.plot(spt, c[:, 1], label="$c_B$", marker='o')
    ax.set_xlabel("Normalised time")
    ax.set_ylabel("Concentration (mol/L)")
    ax.set_title("Michaelis-Menten reaction  (Pyomo collocation + IPOPT, IFT path)")
    ax.legend()
    fig.tight_layout()
    plt.show()
