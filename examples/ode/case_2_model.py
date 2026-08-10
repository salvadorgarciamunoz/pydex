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

===============================================================================
 PITFALL: sampling times that nearly coincide with a collocation node
===============================================================================
This is the non-obvious failure this example exists to warn about. It shipped
undetected for a long time, produced a silently wrong answer, and every
diagnostic that should have caught it reported success.

WHAT HAPPENS
------------
Sampling times are embedded as finite-element boundaries so each measurement
lands exactly on a collocation node. The obvious way to build that grid is

    t_grid = sorted(set(np.linspace(0, 1, nfe+1).tolist() + spt_norm.tolist()))

Time here is normalised by tau = max(sampling_times), so an ABSOLUTE sampling
time t maps to t/tau. With the sampling grid this example uses,

    spt = np.linspace(0.001, 200, 11)     ->  tau = 200
    spt_norm[0] = 0.001/200 = 5e-6

the first measurement lands 5e-6 away from the node at 0.0 -- close, but not
equal. Both points survive into the grid, and after disc.apply_to() the Radau
collocation points collide with the pair and produce a finite element of width
1.11e-16: machine epsilon, against a maximum element width of 5e-2. Ratio
2.2e+14.

Note the same thing happens WITHOUT a tiny absolute time: set() de-duplicates
by exact float equality, so a sampling time that is mathematically equal to a
node -- 1.2/3.0 against linspace's 0.4 -- also survives as a separate point
differing in the last bit.

WHY IT IS HARD TO SPOT
----------------------
  * IPOPT reports "EXIT: Optimal Solution Found." The NLP is feasible; it is
    just not the NLP you meant to solve.
  * The returned trajectory is wildly non-physical -- CA rising to 31 mol/L
    from CA0 = 5 -- but only at LATER sampling times. Early times look right.
  * Refining the discretisation does NOT help. nfe 20 -> 200 leaves the
    mass-balance residual at ~6e+01, and non-monotonically. That is the
    signature of a formulation problem, not truncation error.
  * The IFT sensitivity path is UNAFFECTED, because pydex's causal rebuild
    calls the model with ONE sampling time, so tau = t and the measurement
    lands on the existing node at 1.0. Only simulate() -- and therefore
    finite-difference sensitivities, predictions and plots -- passes the full
    sampling-time vector and hits the sliver.

    That asymmetry is what made the bug so confusing: the design produced by
    case_2.py was correct while the plots beside it were not, and
    case_2_no_ift.py returned a D-optimal criterion of 45.31 against the
    correct 10.72 -- an apparent 34-unit improvement in log-det that was
    entirely an artefact.

HOW IT IS FIXED HERE
--------------------
build_collocation_grid() drops any normalised sampling time closer than
MIN_NODE_GAP/nfe to a node already in the grid; read_at_times() then reads the
measurement from the nearest node rather than requiring an exact one. Time
placement error is at most MIN_NODE_GAP/nfe of the horizon -- negligible
against the collocation error -- and every element stays well proportioned.

HOW TO DETECT IT IN YOUR OWN MODEL
----------------------------------
  1. Assert a conservation law. This system has an exact invariant,
         CA(t) + CB(t)/nu = CA0
     and simulate() checks it on every call. It costs nothing and it is the
     check that would have caught this immediately: the residual was ~70 mol/L
     on a system bounded by 5 mol/L.

  2. Watch for "More finite elements were found in ContinuousSet 't' than the
     number of finite elements specified in apply." Pyomo emits this whenever
     embedded sampling times push the element count past nfe. It is benign on
     its own -- an extra legitimate node does it too -- but it tells you the
     grid is not what you specified, which is where to start looking.

  3. Print min/max finite-element width after building the grid. A ratio above
     ~1e6 is a red flag; 1e14 means the solve is meaningless.

  4. Cross-check against an independent integrator. See
     case_2_no_ift_no_collocation_model.py, which uses scipy.integrate and no
     collocation at all. After the fix the two agree to 7 significant figures
     (10.724136 vs 10.724134); before it, one of them was 45.31.
===============================================================================
"""

import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
from matplotlib import pyplot as plt

# Collocation settings
NFE = 20   # number of finite elements
NCP = 3    # collocation points per element (Lagrange-Radau)


# ---------------------------------------------------------------------------
# Collocation grid construction — degenerate-element guard
# ---------------------------------------------------------------------------
# Sampling times are embedded as finite-element boundaries so that each
# measurement lands exactly on a collocation node. Done naively as
#
#     t_grid = sorted(set(np.linspace(0, 1, nfe+1).tolist() + spt_norm.tolist()))
#
# this admits a sampling time arbitrarily close to an existing node, and the
# resulting sliver element destroys the solve.
#
# CONCRETE FAILURE (this is not hypothetical — it shipped):
#   spt = linspace(0.001, 200, 11), tau = 200  ->  spt_norm[0] = 5e-6, which
#   sits 5e-6 away from the node at 0.0. After disc.apply_to() the Radau
#   collocation points collide with it and produce a finite element of width
#   1.11e-16 -- machine epsilon -- against a maximum width of 5e-2, a ratio of
#   2.2e+14. IPOPT still reports "Optimal Solution Found", but the returned
#   trajectory is non-physical: CA rises to 31 mol/L from CA0 = 5, and the
#   exact conservation law CA + CB/nu = CA0 is violated by ~70 mol/L.
#
#   Refining nfe does NOT help (20 -> 200 leaves the residual at ~6e+01, and
#   non-monotonically), which is the signature of a formulation problem rather
#   than discretisation error.
#
#   The defect was invisible for a long time because it does not affect the IFT
#   sensitivity path: pydex's causal rebuild calls the model with ONE sampling
#   time, so tau = t and the measurement lands on the existing node at 1.0. Only
#   simulate() -- and therefore finite-difference sensitivities, predictions and
#   plots -- passes the full sampling-time vector and hits the sliver.
#
# The guard: drop any normalised sampling time closer than MIN_NODE_GAP (as a
# fraction of the nominal element width 1/nfe) to a node already present. The
# measurement is then read from the nearest node instead of an exact one, which
# costs at most MIN_NODE_GAP/nfe in time placement -- utterly negligible next to
# the collocation error -- and keeps every element well proportioned.
MIN_NODE_GAP = 1e-3      # fraction of the nominal element width 1/nfe

# The snap warning below is emitted once per session, not once per call.
# build_pyomo_model() is invoked for every candidate, every sampling time and
# every FD perturbation -- hundreds of times in a single design run -- so a
# per-call warning buries the actual output. The condition is a property of the
# sampling grid, not of any individual call, so saying it once is sufficient.
_SNAP_WARNED = False


def build_collocation_grid(spt_norm, nfe):
    """
    Finite-element boundaries for the normalised time domain [0, 1].

    Returns (t_grid, snapped) where `snapped` is True if any requested sampling
    time was too close to an existing node and will be read from a neighbour.
    """
    import numpy as _np
    base = _np.linspace(0.0, 1.0, nfe + 1)
    tol = MIN_NODE_GAP / float(nfe)
    extra, snapped = [], False
    for s in _np.atleast_1d(_np.asarray(spt_norm, dtype=float)).ravel():
        if _np.min(_np.abs(s - base)) <= tol or any(abs(s - e) <= tol for e in extra):
            snapped = True          # too close to an existing node — reuse it
        else:
            extra.append(float(s))
    return sorted(set(base.tolist() + extra)), snapped


def read_at_times(m, var, spt_norm, t_grid):
    """
    Read a time-indexed Pyomo Var at the requested normalised times, snapping
    to the nearest grid node. Direct lookup  var[t]  raises KeyError whenever a
    sampling time was dropped by build_collocation_grid, so always go through
    this rather than indexing with the requested value.
    """
    import numpy as _np
    import pyomo.environ as _pyo
    grid = _np.asarray(sorted(t_grid), dtype=float)
    out = []
    for s in _np.atleast_1d(_np.asarray(spt_norm, dtype=float)).ravel():
        out.append(_pyo.value(var[grid[int(_np.argmin(_np.abs(grid - s)))]]))
    return _np.asarray(out, dtype=float)


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
    t_grid, _snapped = build_collocation_grid(spt_norm, nfe)
    global _SNAP_WARNED
    # Only the main process reports. pydex sets n_jobs = -1 automatically when
    # pyomo_model_fn is present, and joblib's loky backend runs each candidate in
    # a SEPARATE PROCESS -- every worker imports this module afresh with
    # _SNAP_WARNED = False, so a module-level flag alone still produced one
    # warning per worker (17 on a 16-core machine). The snapping condition is a
    # property of the sampling grid and is identical in every worker, so a single
    # report from the parent is complete. The parent does call this function (via
    # simulate() when plotting predictions), so the warning is not lost.
    import multiprocessing as _mp
    _is_main = _mp.current_process().name == "MainProcess"
    if _snapped and not _SNAP_WARNED and _is_main:
        _SNAP_WARNED = True
        import warnings as _w
        _w.warn(
            "[case_2 model] One or more sampling times fell within "
            f"{MIN_NODE_GAP:g}/nfe of an existing collocation node and were "
            "read from the nearest node instead of being embedded exactly. "
            "Embedding them would have created a near-zero-width finite "
            "element, which silently corrupts the solve. Time placement error "
            f"is at most {MIN_NODE_GAP:g}/nfe of the horizon.",
            RuntimeWarning, stacklevel=2,
        )

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

    # Snap to the nearest collocation node. Direct lookup m.ca[t] raises
    # KeyError for any sampling time that build_collocation_grid dropped as
    # too close to an existing node.
    t_grid_solved = sorted(m.t)
    ca = read_at_times(m, m.ca, spt_norm, t_grid_solved)
    cb = read_at_times(m, m.cb, spt_norm, t_grid_solved)

    # Conservation check. A → B with stoichiometry nu gives the exact invariant
    #     CA(t) + CB(t)/nu = CA0     for all t
    # This is free to evaluate and is the check that would have caught the
    # degenerate-element failure immediately: it was violated by ~70 mol/L on a
    # system bounded by 5 mol/L while IPOPT reported "Optimal Solution Found".
    _nu = float(model_parameters[3])
    if abs(_nu) > 1e-12:
        _resid = np.max(np.abs(ca + cb / _nu - float(ti_controls[0])))
        _scale = max(abs(float(ti_controls[0])), 1.0)
        if _resid > 1e-4 * _scale:
            import warnings as _w
            _w.warn(
                f"[case_2 model] mass-balance residual "
                f"max|CA + CB/nu - CA0| = {_resid:.3e} "
                f"({_resid/_scale:.2%} of CA0). The collocation solve converged "
                f"to a non-physical branch; the returned trajectory should not "
                f"be trusted. Check the finite-element widths of the time grid.",
                RuntimeWarning, stacklevel=2,
            )

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
