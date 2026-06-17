"""
case_2_no_ift_model.py
======================
Pyomo.DAE model for the A→B reaction with Arrhenius kinetics, used by the
finite-difference (FD) sensitivity path in case_2_no_ift.py.

Reaction system
---------------
    A → B    (irreversible, power-law rate)

    dCA/dt = -k * CA^α
    dCB/dt =  ν * k * CA^α

    k(T) = exp(θ₀ + θ₁ * (T - 273.15) / T)      [Arrhenius, reparametrised]

State variables : CA(t), CB(t)   [mol/L]
Parameters      : θ = [θ₀, θ₁, α, ν]
Controls        : ti_controls = [CA0 (mol/L), T (K)]

Relationship to case_2_model.py (IFT variant)
----------------------------------------------
build_pyomo_model() is structurally identical to case_2_model.py so that
the FD and IFT variants produce the same discretised system and their
sensitivities can be directly compared.  The only difference is that this
file does NOT assign build_pyomo_model to designer.pyomo_model_fn — pydex
therefore falls back to its finite-difference (Richardson extrapolation)
sensitivity path using numdifftools.

Why the FD path can fail — and how this file defends against it
---------------------------------------------------------------
During Richardson extrapolation, numdifftools perturbs each parameter θⱼ
by a finite step h and evaluates simulate(θ + h·eⱼ).  For some step sizes
the perturbed parameters may leave the physically meaningful region:

  • Very large θ₁  → k becomes huge → CA collapses to 0, hitting the lower
    bound before IPOPT converges → IPOPT returns "infeasible".
  • Negative α     → CA^α becomes complex (undefined) for CA < 1.
  • Negative ν     → CB accumulation becomes negative, unphysical.

The original code raised RuntimeError directly from the IPOPT convergence
check, which propagated all the way up through numdifftools and crashed the
entire sensitivity analysis.

Three layers of defence are applied in this fixed version:

  1. Parameter clamping in build_pyomo_model():
     α and ν are clamped to a small positive epsilon before the solve.
     This prevents the ODE RHS from becoming undefined or reversing sign
     for aggressive FD perturbations.  θ₀ and θ₁ are NOT clamped because
     they appear in an exponential and clamping would silently bias the FD
     Jacobian — instead, the infeasible-solve fallback handles them.

  2. Relaxed CA lower bound (0 → -_CA_FLOOR_ABS):
     A tiny negative slack on the CA lower bound prevents IPOPT from
     declaring infeasibility when a large-k perturbation drives CA to
     near-zero.  The slack is chosen so that the physical solution is never
     affected (nominal CA values are O(1)), but IPOPT can still converge
     past the apparent lower-bound constraint.

  3. NaN-returning fallback in simulate():
     If IPOPT still fails to converge (e.g. a truly extreme perturbation),
     simulate() catches the RuntimeError and returns an array of NaN values
     with the correct shape.  numdifftools / Richardson extrapolation treats
     NaN as a failed evaluation and silently skips it, using only the
     remaining valid evaluations to estimate the derivative.  This is the
     standard approach for robust FD Jacobians over partially-infeasible
     domains — the same mechanism scipy.optimize uses internally.

Normalised time convention
--------------------------
The collocation model is solved on the normalised domain τ ∈ [0, 1], where
τ = t_abs / t_max and t_max = max(sampling_times).  Normalisation keeps the
ODE right-hand side and the collocation matrices well-conditioned regardless
of the absolute time scale, which improves IPOPT convergence.

Sampling times are embedded as finite-element (FE) boundaries in normalised
form before discretisation, so they become exact grid members after
TransformationFactory('dae.collocation') is applied — no interpolation or
snapping is needed when extracting responses.

ASL variable elimination — why the model is structured as it is
---------------------------------------------------------------
For the IFT path (case_2_model.py), every model parameter must remain in
the NLP primal vector so that PyomoNLP / ASL can compute ∂x/∂θ.  ASL
aggressively pre-solves and eliminates fixed variables or constants.  Three
structural choices prevent spurious elimination:

  a) Temperature T is a *free* Var pinned by an equality constraint
     (m.temp_fix: m.temp == T_val) rather than a fixed Param.  If T were
     fixed, ASL would substitute it before building the NLP and θ₁ would
     disappear from the Jacobian.

  b) The Arrhenius rate is split into two auxiliary Vars:
       m.ln_k[t] = θ₀ + θ₁*(temp - 273.15)/temp     [defined by ln_k_def]
       m.k[t]    = exp(ln_k[t])                       [defined by k_def]
     This forces θ₁ to appear explicitly alongside the free Var m.temp in
     the ln_k constraint, which ASL cannot eliminate.

  c) α (alpha_a) and ν (nu) are fixed Vars rather than Params for the same
     reason — Params are substituted at model construction time in Pyomo.

This structure is retained in the FD variant for consistency and so that
the file can be promoted to the IFT path by simply assigning pyomo_model_fn.

Dependencies
------------
    numpy, pyomo, pyomo.dae, matplotlib (for __main__ sanity check only)

Typical usage (from case_2_no_ift.py)
--------------------------------------
    from case_2_no_ift_model import simulate
    designer.simulate = simulate
    # designer.pyomo_model_fn is NOT set → FD sensitivity path is used
"""

import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
from matplotlib import pyplot as plt

# ---------------------------------------------------------------------------
# Collocation defaults
# ---------------------------------------------------------------------------
NFE = 20   # number of finite elements for the collocation discretisation
NCP = 3    # number of collocation points per element (Lagrange-Radau)

# ---------------------------------------------------------------------------
# Physical bounds for robustness during FD perturbation
# ---------------------------------------------------------------------------
# Minimum allowed values for α (reaction order) and ν (stoichiometric ratio).
# numdifftools may perturb these parameters negative when the nominal value
# is small.  Values below _PARAM_MIN cause the ODE RHS to become undefined
# (complex power, negative accumulation) and make IPOPT infeasible.
# The clamp is only applied inside build_pyomo_model — the designer's own
# copy of model_parameters is never mutated.
_PARAM_MIN_ALPHA = 1e-6   # CA^α undefined for α ≤ 0 when CA < 1
_PARAM_MIN_NU    = 1e-6   # negative ν would reverse CB accumulation

# Tiny negative slack on the CA lower bound (mol/L).
# Allows IPOPT to converge when a large-k perturbation drives CA to ~0.
# Chosen to be << nominal CA values (O(1)) so nominal solutions are
# unaffected, but large enough to absorb floating-point boundary violations.
_CA_FLOOR_ABS = 1e-8


# =============================================================================
# build_pyomo_model
# =============================================================================

def build_pyomo_model(ti_controls, model_parameters, sampling_times=None,
                      nfe=NFE, ncp=NCP):
    """
    Build and solve a Pyomo.DAE collocation model for the A→B reaction.

    The model is solved on the normalised time domain τ ∈ [0, 1] using
    Lagrange-Radau orthogonal collocation, with IPOPT as the NLP solver.
    Sampling times are embedded as finite-element boundaries so they are
    exact collocation-grid members after discretisation.

    This function is NOT assigned to designer.pyomo_model_fn in this variant.
    It is called only from simulate() to obtain the state trajectory.
    If you want to switch to the IFT sensitivity path, assign this function
    to designer.pyomo_model_fn and set designer.use_pyomo_ift = True.

    Parameters
    ----------
    ti_controls : array-like, length 2
        [CA0 (mol/L), T (K)]
        CA0 : initial concentration of A.
        T   : isothermal reaction temperature (constant throughout batch).

    model_parameters : array-like, length 4
        [θ₀, θ₁, α, ν]
        θ₀  : Arrhenius pre-exponential offset  (ln(k_ref) - θ₁)
        θ₁  : Arrhenius activation-energy group  Ea / (R * T_ref)
        α   : reaction order in CA  (clamped to _PARAM_MIN_ALPHA if ≤ 0)
        ν   : stoichiometric coefficient CB/CA  (clamped to _PARAM_MIN_NU if ≤ 0)

    sampling_times : array-like or None
        Absolute measurement times (same units as the ODE time variable).
        These are embedded as FE boundaries so the collocation grid includes
        them exactly.  If None, a uniform grid over [0, nfe] is used.

    nfe : int, optional
        Number of finite elements.  Default: NFE (module-level constant).

    ncp : int, optional
        Collocation points per element.  Default: NCP (module-level constant).

    Returns
    -------
    m : pyo.ConcreteModel
        Solved Pyomo model.  State variables ca[t] and cb[t] are accessible
        via pyo.value(m.ca[t]) for any t in sorted(m.t).

    all_vars : list of Pyomo Var objects
        Parameter vars first (θ₀, θ₁, α, ν), then all state and auxiliary
        vars in time order.  This ordering satisfies the pydex IFT contract
        (parameter vars must be first) even though IFT is not used here.

    all_bodies : list of Pyomo expressions
        Equality constraint bodies (lhs - rhs) for every active equality
        constraint in the model.  Used by PyomoNLP to build the KKT system
        for IFT (not used in this FD variant, but returned for API consistency).

    t_sorted_full : list of float
        Sorted list of all collocation time points on [0, 1] (normalised).
        These are the valid keys for indexing m.ca[t] and m.cb[t].

    Raises
    ------
    RuntimeError
        If IPOPT does not return an optimal solution.  The caller (simulate)
        catches this and returns NaN so the FD Richardson extrapolation
        can degrade gracefully rather than aborting the sensitivity analysis.
    """
    # ------------------------------------------------------------------
    # Unpack controls and parameters
    # ------------------------------------------------------------------
    CA0_val = float(ti_controls[0])
    T_val   = float(ti_controls[1])

    theta_0_val = float(model_parameters[0])
    theta_1_val = float(model_parameters[1])

    # Clamp α and ν to their physical lower bounds.
    # numdifftools may perturb these parameters into negative territory when
    # their nominal values are small.  A negative α makes CA^α undefined
    # for CA < 1; a negative ν reverses CB accumulation.  Neither is
    # physically meaningful, and both make IPOPT declare infeasibility.
    # Clamping is safe here because:
    #   (a) the designer's model_parameters array is never mutated,
    #   (b) perturbations to θ₀ and θ₁ are handled by the NaN fallback,
    #   (c) the sensitivity error from clamping α/ν is only triggered when
    #       numdifftools is exploring a step size that has already left the
    #       feasible region — Richardson extrapolation will prefer the
    #       smaller, non-clamped steps anyway.
    alpha_val = max(float(model_parameters[2]), _PARAM_MIN_ALPHA)
    nu_val    = max(float(model_parameters[3]), _PARAM_MIN_NU)

    # ------------------------------------------------------------------
    # Sampling time processing and normalisation
    # ------------------------------------------------------------------
    spt_abs = np.asarray(sampling_times, dtype=float).flatten()

    # Strip non-finite and negative entries — pydex may pad with NaN when
    # the candidate has fewer sampling times than the maximum in the grid.
    spt_abs = spt_abs[np.isfinite(spt_abs) & (spt_abs >= 0)]

    # τ = max(sampling_times) is the normalisation constant.
    # The ODE is re-expressed on [0, 1]: d(·)/dτ = τ * d(·)/dt_abs.
    # This keeps the ODE RHS O(1) regardless of the absolute time horizon,
    # improving IPOPT convergence for both short and long experiments.
    tau = float(np.max(spt_abs))

    # Normalised sampling times — these become FE boundaries so they are
    # exact grid members after collocation discretisation.
    spt_norm = spt_abs / tau

    # Build the collocation grid: uniform base grid ∪ normalised spt.
    # Using a set eliminates duplicates before sorting.
    t_grid = sorted(set(
        np.linspace(0.0, 1.0, nfe + 1).tolist() + spt_norm.tolist()
    ))

    # ------------------------------------------------------------------
    # Pyomo model construction
    # ------------------------------------------------------------------
    m   = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(initialize=t_grid)

    # ── Parameter Vars ────────────────────────────────────────────────
    # Fixed Vars rather than Params so ASL retains them in the NLP primal
    # vector (required for IFT compatibility — see module docstring).
    m.theta_0 = pyo.Var(initialize=theta_0_val);  m.theta_0.fix(theta_0_val)
    m.theta_1 = pyo.Var(initialize=theta_1_val);  m.theta_1.fix(theta_1_val)
    m.alpha_a = pyo.Var(initialize=alpha_val);    m.alpha_a.fix(alpha_val)
    m.nu      = pyo.Var(initialize=nu_val);       m.nu.fix(nu_val)

    # ── Time-scale Var ────────────────────────────────────────────────
    # Fixed so IPOPT knows the normalisation factor when evaluating d(·)/dτ.
    m.tau = pyo.Var(initialize=tau);  m.tau.fix(tau)

    # ── Temperature: free Var pinned by equality constraint ───────────
    # IMPORTANT: T must be a free Var (not fixed or a Param) so that θ₁
    # is retained in the ASL NLP for the IFT Jacobian.  See module docstring
    # section "ASL variable elimination" for the full explanation.
    # In this FD variant, temp_fix still ensures T is correctly set.
    m.temp     = pyo.Var(initialize=T_val)
    m.temp_fix = pyo.Constraint(expr=m.temp == T_val)

    # ── State variables ───────────────────────────────────────────────
    # Lower bound is relaxed to -_CA_FLOOR_ABS (a tiny negative slack)
    # rather than the strict physical bound of 0.  This prevents IPOPT from
    # declaring infeasibility when a large-k perturbation (from a big FD
    # step in θ₀ or θ₁) drives CA to near-zero and the interior-point
    # barrier cannot step past the hard bound.  The slack is negligibly
    # small relative to nominal CA values (O(1) mol/L) so the solution at
    # the nominal parameters is unaffected.
    m.ca     = pyo.Var(m.t, initialize=CA0_val, bounds=(-_CA_FLOOR_ABS, 50))
    m.cb     = pyo.Var(m.t, initialize=0.0,     bounds=(-_CA_FLOOR_ABS, 50))
    m.dca_dt = dae.DerivativeVar(m.ca, wrt=m.t)
    m.dcb_dt = dae.DerivativeVar(m.cb, wrt=m.t)

    # ── Arrhenius auxiliary variables ─────────────────────────────────
    # k(T) = exp(θ₀ + θ₁*(T - 273.15)/T) is split into:
    #   ln_k[t] = θ₀ + θ₁*(temp - 273.15)/temp    [ln_k_def constraint]
    #   k[t]    = exp(ln_k[t])                      [k_def constraint]
    #
    # This two-variable split serves two purposes:
    #   1. IFT compatibility: θ₁ appears alongside the free Var m.temp in
    #      ln_k_def, preventing ASL from eliminating it (see module docstring).
    #   2. Numerical conditioning: separating the log and exp operations
    #      gives IPOPT cleaner Hessian structure than a single nested exp.
    #
    # alpha_b (reaction order in CB) is hardcoded to 0 so CB^0 = 1 drops out:
    #   rate = k[t] * CA[t]^α * CB[t]^0 = k[t] * CA[t]^α
    ln_k_init = theta_0_val + theta_1_val * (T_val - 273.15) / T_val
    k_init    = float(np.exp(ln_k_init))

    m.ln_k = pyo.Var(m.t, initialize=ln_k_init)
    m.k    = pyo.Var(m.t, initialize=k_init,    bounds=(0, None))

    def ln_k_def_rule(m, t):
        # ln(k) = θ₀ + θ₁*(T - 273.15)/T
        # Written with m.temp (free Var) rather than T_val (constant) so
        # that θ₁ appears in the NLP alongside a non-constant expression.
        return m.ln_k[t] == m.theta_0 + m.theta_1 * (m.temp - 273.15) / m.temp

    m.ln_k_def = pyo.Constraint(m.t, rule=ln_k_def_rule)

    def k_def_rule(m, t):
        # k = exp(ln_k) — separate constraint keeps the Hessian cleaner.
        return m.k[t] == pyo.exp(m.ln_k[t])

    m.k_def = pyo.Constraint(m.t, rule=k_def_rule)

    # ── ODE constraints (on normalised time) ──────────────────────────
    # The chain rule gives d(·)/dτ = τ * d(·)/dt_abs, so:
    #   dCA/dτ / τ = -k * CA^α      →   dCA/dτ = -τ * k * CA^α
    # pydex stores dca_dt as d/dτ (the derivative w.r.t. the normalised
    # ContinuousSet), so we divide by τ on the LHS to recover d/dt_abs.
    def material_balance_a_rule(m, t):
        return m.dca_dt[t] / m.tau == -m.k[t] * (m.ca[t] ** m.alpha_a)

    m.material_balance_a = pyo.Constraint(m.t, rule=material_balance_a_rule)

    def material_balance_b_rule(m, t):
        # CB accumulates at rate ν * (rate of A consumption)
        return m.dcb_dt[t] / m.tau == m.nu * m.k[t] * (m.ca[t] ** m.alpha_a)

    m.material_balance_b = pyo.Constraint(m.t, rule=material_balance_b_rule)

    # ── Initial conditions ────────────────────────────────────────────
    m.ic_a = pyo.Constraint(expr=m.ca[0] == CA0_val)
    m.ic_b = pyo.Constraint(expr=m.cb[0] == 0.0)

    # ── Dummy objective ───────────────────────────────────────────────
    # IPOPT requires an objective; we use 0 since this is a feasibility solve.
    m.obj = pyo.Objective(expr=0.0)

    # ── Collocation discretisation ────────────────────────────────────
    # Lagrange-Radau scheme: the last collocation point in each element
    # coincides with the right FE boundary, which is the sampling time.
    # This guarantees that the response at each sampling time is read
    # directly from a collocation variable — no interpolation needed.
    disc = pyo.TransformationFactory('dae.collocation')
    disc.apply_to(m, nfe=nfe, ncp=ncp, scheme='LAGRANGE-RADAU')

    # ── Solve ─────────────────────────────────────────────────────────
    solver = pyo.SolverFactory('ipopt')
    solver.options['print_level'] = 0
    solver.options['tol']         = 1e-12   # tight tolerance for accurate FD
    result = solver.solve(m, tee=False)

    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        # Raise so that simulate()'s NaN-fallback catches this.
        # Do NOT silently continue — a non-optimal solve would give wrong
        # state values that numdifftools would treat as valid, poisoning the
        # FD Jacobian more insidiously than a NaN would.
        raise RuntimeError(
            f"IPOPT did not converge: {result.solver.termination_condition}"
        )

    # ------------------------------------------------------------------
    # Collect variables and constraint bodies for IFT contract
    # (not used in FD path, but returned for API compatibility)
    # ------------------------------------------------------------------
    t_sorted_full = sorted(m.t)

    # Parameter vars MUST come first in all_vars for the IFT Jacobian
    # to correctly map columns to θⱼ.  State and auxiliary vars follow
    # in time order.
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
# simulate — pydex signature 2 wrapper
# =============================================================================

def simulate(ti_controls, sampling_times, model_parameters):
    """
    Evaluate the A→B model at the given conditions and return concentrations
    at each requested sampling time.

    This function is assigned to designer.simulate and is called by pydex
    during both prediction (plotting, print_optimal_candidates) and
    sensitivity analysis (finite-difference Jacobian via numdifftools).

    Pydex signature type 2: simulate(ti_controls, sampling_times, model_parameters)

    Parameters
    ----------
    ti_controls : array-like, length 2
        [CA0 (mol/L), T (K)]

    sampling_times : array-like
        Absolute measurement times.  pydex may pad shorter candidate rows
        with NaN — these are stripped before passing to build_pyomo_model.

    model_parameters : array-like, length 4
        [θ₀, θ₁, α, ν]
        During FD sensitivity analysis, pydex / numdifftools perturbs each
        element of this array by a finite step.  The perturbed vector is
        passed directly here, so model_parameters[j] may be outside its
        nominal physical range.  The parameter clamping in build_pyomo_model
        and the NaN fallback below handle these cases robustly.

    Returns
    -------
    y : np.ndarray, shape (n_spt, 2)
        Columns: [CA (mol/L), CB (mol/L)] at each sampling time.
        Returns an array of NaN (same shape) if IPOPT does not converge,
        allowing numdifftools Richardson extrapolation to skip the failed
        evaluation and continue with smaller FD steps.

    Notes on the NaN fallback
    --------------------------
    numdifftools uses multi-point Richardson extrapolation: it evaluates
    the function at several step sizes and combines the results to cancel
    truncation error.  When one evaluation returns NaN, numdifftools
    silently ignores it and uses the remaining evaluations.  As long as at
    least one step size produces a valid solve — which is always true near
    the nominal parameters — the Jacobian estimate is valid.

    This is strictly better than raising an exception, which would abort
    the entire sensitivity analysis.  It is also better than returning a
    zero array, which would silently inject a wrong (zero) FD increment and
    bias the Jacobian toward zero for that column.
    """
    # Flatten and strip non-finite / negative padding
    spt_abs = np.asarray(sampling_times, dtype=float).flatten()
    spt_abs = spt_abs[np.isfinite(spt_abs) & (spt_abs >= 0)]
    n_spt   = len(spt_abs)

    try:
        m, _, _, _ = build_pyomo_model(ti_controls, model_parameters, spt_abs)
    except RuntimeError:
        # IPOPT failed to converge for this perturbed parameter vector.
        # Return NaN so Richardson extrapolation can skip this evaluation.
        # The shape (n_spt, 2) matches the expected output so pydex and
        # numdifftools can proceed without type errors.
        return np.full((n_spt, 2), np.nan)

    # Normalised sampling times are exact collocation grid members —
    # direct lookup, no interpolation or snapping needed.
    tau      = float(np.max(spt_abs))
    spt_norm = spt_abs / tau

    ca = np.array([pyo.value(m.ca[t]) for t in spt_norm])
    cb = np.array([pyo.value(m.cb[t]) for t in spt_norm])

    return np.column_stack([ca, cb])


# =============================================================================
# Sanity check — run directly to verify the model produces a sensible profile
# =============================================================================

if __name__ == '__main__':
    # Nominal parameters: k_ref = 0.1 L/(mol·min), Ea = 5000 J/mol
    pre_exp_constant = 0.1
    activ_energy     = 5000.0
    R                = 8.314159
    T_ref            = 273.15   # K  (reference temperature for Arrhenius)

    # Reparametrised form: k(T) = exp(θ₀ + θ₁*(T - T_ref)/T)
    # At T = T_ref: k(T_ref) = exp(θ₀) = k_ref  →  θ₀ = ln(k_ref)
    # θ₁ = Ea / (R * T_ref)  (dimensionless activation energy group)
    theta_0 = np.log(pre_exp_constant) - activ_energy / (R * T_ref)
    theta_1 = activ_energy / (R * T_ref)
    theta_nom = np.array([theta_0, theta_1, 1.0, 0.5])

    print(f"Nominal parameters: θ = {theta_nom}")
    print(f"  k at T=323.15 K : "
          f"{np.exp(theta_0 + theta_1*(323.15-273.15)/323.15):.4f} 1/min")

    tic = [1.0, 323.15]                 # CA0 = 1 mol/L, T = 50°C
    spt = np.linspace(0.001, 200, 11)   # 11 sampling times over 200 min

    y = simulate(
        ti_controls=tic,
        sampling_times=spt,
        model_parameters=theta_nom,
    )

    print(f"\nConcentrations at sampling times:")
    print(f"  {'t (min)':>10}  {'CA (mol/L)':>12}  {'CB (mol/L)':>12}")
    print(f"  {'-'*38}")
    for t, row in zip(spt, y):
        print(f"  {t:>10.2f}  {row[0]:>12.6f}  {row[1]:>12.6f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(spt, y[:, 0], label='$c_A$', marker='o')
    ax.plot(spt, y[:, 1], label='$c_B$', marker='o')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mol/L)')
    ax.set_title('A→B reaction  (collocation + IPOPT, FD sensitivity path)')
    ax.legend()
    fig.tight_layout()
    plt.show()
