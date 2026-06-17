"""
v_optimal_test_case_pyomo.py
============================
Example of V-optimal Model-Based Design of Experiments (MBDoE) using pydex.

Overview
--------
This example demonstrates a two-stage workflow for designing experiments that
are specifically optimised for prediction accuracy at a target operating
condition, rather than globally minimising parameter uncertainty.

    Stage 1 — Process optimisation:
        Find the operating point dw = (T0, Tjacket, catalyst_load) that
        maximises product yield subject to quality and safety constraints.
        This is solved as a nonlinear programme via IPOPT.

    Stage 2 — V-optimal MBDoE:
        Design experiments that minimise model prediction variance at dw,
        using the V-optimality criterion:

            J_V = trace( W * FIM^{-1} * W^T )

        where W is the scaled sensitivity matrix evaluated at dw.
        The resulting design is compared against A-optimal and D-optimal
        designs to quantify the benefit of prediction-oriented design.

Why V-optimal?
--------------
Classical D-optimal and A-optimal designs minimise parameter uncertainty
globally across the entire operating space. When the ultimate goal is to
deploy the model at a specific operating condition (e.g. the economically
optimal process point), these global designs may allocate experimental
effort to regions that are informative about parameters in general but not
about predictions at the point of interest.

V-optimal design targets the FIM inversion specifically towards the
prediction directions encoded in W, concentrating experimental effort
near dw. The trade-off is that V-optimal designs are less informative
away from dw — which is acceptable when the operating point is known.

Reference:
    Shahmohammadi, A. & McAuley, K.B. (2019). Sequential model-based A- and
    V-optimal design of experiments for building fundamental models of
    pharmaceutical production processes. Computers & Chemical Engineering,
    129, 106504. https://doi.org/10.1016/j.compchemeng.2019.06.029

Reaction system
---------------
Three parallel first-order reactions consuming reactant A:

    A -> B    desired product    endothermic   Ea_main = 55,000 J/mol
    A -> I    impurity           exothermic    Ea_imp  = 75,000 J/mol
    A -> D    decomposition      exothermic    Ea_dec  = 80,000 J/mol
                                              (reference temp = 85 C)

Because Ea_imp and Ea_dec are both larger than Ea_main, high temperatures
accelerate the side reactions disproportionately. This creates a genuine
three-way trade-off between productivity and quality.

The energy balance includes jacket heat transfer (U = 5000 J/hr/K), making
Tjacket a meaningful degree of freedom — the reactor temperature tracks
the jacket temperature rather than staying at T0.

Process optimisation constraints
---------------------------------
    CI_final <= 0.05 mol/L    impurity quality specification
    CD_final <= 0.05 mol/L    decomposition byproduct specification
    Tjacket  >= T0            physical (jacket must heat, not cool)

The optimal operating point sits in the interior of the feasible region,
where both quality constraints are simultaneously active.

Parameters estimated
--------------------
    theta = [k_ref, Ea, k_ref_imp, Ea_imp, k_ref_dec, Ea_dec]

    k_ref     : main reaction rate constant at T_ref = 60 C  (1/hr)
    Ea        : main reaction activation energy               (J/mol)
    k_ref_imp : impurity rate constant at T_ref = 60 C       (1/hr)
    Ea_imp    : impurity activation energy                    (J/mol)
    k_ref_dec : decomposition rate constant at T_ref = 85 C  (1/hr)
    Ea_dec    : decomposition activation energy               (J/mol)

Pydex interface
---------------
Signature type 2: simulate(ti_controls, sampling_times, model_parameters)

    ti_controls      : [T0 (C), Tjacket (C), catalyst_load]
    sampling_times   : 1-D array of measurement times (hr)
    model_parameters : [k_ref, Ea, k_ref_imp, Ea_imp, k_ref_dec, Ea_dec]
    returns          : np.ndarray shape (n_spt, 4) — [CA, CB, CI, CD]

Pyomo IFT sensitivity
---------------------
This version uses exact parametric sensitivities via the Implicit-Function
Theorem (IFT) computed from a Pyomo DAE model, replacing pydex's default
finite-difference approach.

The Pyomo model encodes the same 5-state ODE system using Lagrange-Radau
orthogonal collocation (nfe=10, ncp=3), following the standard Pyomo.DAE
parameter-estimation idiom: measurement/sampling times are seeded into the
ContinuousSet and become exact finite-element boundaries; the discretiser
adds collocation points between them.

Key advantages:
  - No finite-difference perturbations — exact sensitivities for the
    discretised model, consistent with simulate()
  - Parallel sensitivity computation via joblib threads (set designer.n_jobs)
  - Same model used for simulation and sensitivity — no consistency issues

Two-solver architecture
-----------------------
This script intentionally uses two solvers for different purposes:

  build_pyomo_model / simulate()  — used exclusively for all DoE-critical
      computations: pydex responses, IFT Jacobians, FIM, and criterion.
      Exact symbolic derivatives, fully consistent with the collocation model.

  _solve (scipy Radau)  — used for fast non-DoE evaluations: Stage 1 process
      optimisation (SLSQP calls _solve ~500 times per start), constraint
      landscape visualisation (~3600 grid evaluations), and trajectory plots.
      Pyomo would be ~100x slower for these many cheap evaluations.

The DoE results (support candidates, J_V, effort allocation) depend only on
the Pyomo path. The scipy path only affects Stage 1 operating point finding
and plotting — never the sensitivity analysis or criterion optimisation.

Parallel computation
--------------------
Set designer.n_jobs = -1 to use all available CPU cores for sensitivity
computation. Each candidate is solved independently (160 parallel IPOPT
calls), giving ~2-3× speedup on an 8-core machine.

    designer.n_jobs = -1   # all cores
    designer.n_jobs = 4    # explicit core count
    designer.n_jobs = 1    # sequential (default)

Dependencies
------------
    numpy, scipy, matplotlib
    pyomo, pyomo.dae
    pydex >= 0.0.9 with V-optimal and Pyomo IFT extensions:
        Designer.find_optimal_operating_point()
        Designer.design_v_optimal()
        Designer.v_opt_criterion()
        Designer._eval_W_matrix()
        Designer._eval_sensitivities_pyomo_ift()
        Designer.n_jobs

Solver
------
    IPOPT is used for Stage 1, sensitivity computation, and Stage 2.
    The default linear solver is "ma57" (HSL). If HSL is not available,
    change LINEAR_SOLVER = "mumps" near the top of this file.

Typical runtimes (Mac M-series, n_jobs=-1)
------------------------------------------
    Stage 1 (process optimisation):    ~30s
    Sensitivity computation (parallel): ~28s
    Stage 2 V-optimal IPOPT:           ~270s
    Stage 2 A-optimal IPOPT:           ~250s
    Stage 2 D-optimal IPOPT:             ~1s
    Total:                             ~10 min

Usage
-----
    Run directly:   python v_optimal_test_case_pyomo.py
    In Spyder:      F5  (or Run > Run File)

    The script produces four sets of figures:
      1. Representative model trajectories at three operating conditions
      2. Constraint landscape (CB, CI, CD vs T0 and catalyst load)
      3. Trajectory at the optimal operating point
      4. Design comparison: V-optimal vs A-optimal vs D-optimal
         - Bubble chart of effort allocation in (T0, Tjacket) space
         - Concentration trajectories at selected candidates
         - Summary table of J_V for each design

Expected results
----------------
    V-optimal   J_V ≈ 0.000962  (best by construction)
    A-optimal   J_V ≈ 0.001354  → ~1.41× worse prediction variance
    D-optimal   J_V ≈ 0.001547  → ~1.61× worse prediction variance

    V-optimal concentrates ~78% of effort on candidate 65
    (T0=50°C, Tjacket=75°C, catalyst=1.5), sampling at t=1hr (end of batch).
"""

# =============================================================================
# Imports
# =============================================================================


import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pyomo.environ as pyo
import pyomo.dae as dae
from pydex.core.designer import Designer


# =============================================================================
# CONFIG — edit here to explore different scenarios
# =============================================================================

# IPOPT linear solver.  Use "mumps" if HSL (ma57) is not available.
LINEAR_SOLVER = "ma57"

# Measurement noise standard deviation for each response (mol/L).
# Used to build the diagonal error covariance matrix passed to pydex.
SIGMA_RESPONSES = 0.01   # std dev ~1% of CA0 for all four species

# Quality and safety constraints (mol/L at t = T_FINAL)
CI_MAX = 0.05   # maximum allowable impurity concentration
CD_MAX = 0.05   # maximum allowable decomposition product concentration

# Sampling time mode for Stage 2 V-optimal design.
# The Pyomo IFT path may distribute effort uniformly across all sampling
# time candidates when the system is genuinely indifferent to when samples
# are taken (i.e. IFT sensitivities are nearly equal at all times).
#
# Minimum effort threshold for sampling time sparsity enforcement.
# When optimize_sampling_times=True, pydex may distribute effort equally
# across all sampling time candidates if the IFT sensitivities are nearly
# identical at all times (the system is genuinely indifferent to when
# samples are taken — this is mathematically correct, not a bug).
#
# Setting SPT_MIN_EFFORT > 0 adds binary variables to the sampling time
# dimension (same mechanism as for candidates) and forces the optimizer
# to concentrate effort at a single sampling time per experiment,
# revealing the single most informative time point.
#
# IMPORTANT: sparsity enforcement requires a MINLP solver — BARON via GAMS
# is used automatically when SPT_MIN_EFFORT > 0. GAMS/BARON must be available
# in your environment. With IPOPT the binary variables are relaxed to
# continuous and min_effort has no effect.
#
#   SPT_MIN_EFFORT = 0.0   → IPOPT, uniform spread (default, mathematically correct)
#   SPT_MIN_EFFORT = 0.1   → BARON/GAMS, concentrated at the most informative time
SPT_MIN_EFFORT = 0.1


# =============================================================================
# Fixed physical parameters
# (not estimated — treated as known from independent experiments)
# =============================================================================
R           = 8.314       # universal gas constant              (J/mol/K)
T_ref_C     = 60.0        # Arrhenius reference temp, A->B, A->I (C)
T_ref_K     = T_ref_C + 273.15
T_ref_dec_C = 85.0        # Arrhenius reference temp, A->D        (C)
T_ref_dec_K = T_ref_dec_C + 273.15
Hr_main     =  50000.0    # heat of reaction A->B  (J/mol, +ve = endothermic)
Hr_imp      = -30000.0    # heat of reaction A->I  (J/mol, -ve = exothermic)
Hr_dec      = -60000.0    # heat of reaction A->D  (J/mol, stronger exotherm)
Cp          = 4184.0      # heat capacity of reaction mixture    (J/kg/K)
mass        = 1.0         # reactor contents mass                (kg)
U           = 5000.0      # overall heat transfer coefficient    (J/hr/K)
A_area      = 1.0         # heat transfer area                   (m^2)
CA0_fixed   = 1.0         # initial concentration of A           (mol/L, fixed)
T_FINAL     = 1.0         # batch end time                       (hr)


# =============================================================================
# Parameter values
# =============================================================================

# True values — used to generate in-silico data and for plotting
THETA_TRUE = np.array([
    1.0,       # k_ref      (1/hr)   main reaction rate at T_ref
    55000.0,   # Ea         (J/mol)  main reaction activation energy
    0.08,      # k_ref_imp  (1/hr)   impurity rate at T_ref
    75000.0,   # Ea_imp     (J/mol)  impurity activation energy (> Ea)
    0.3,       # k_ref_dec  (1/hr)   decomposition rate at T_ref_dec
    80000.0,   # Ea_dec     (J/mol)  decomposition activation energy (> Ea_imp)
])

# Initial guess — represents knowledge available before experiments
# Purposely offset from THETA_TRUE to simulate realistic prior uncertainty
THETA_GUESS = np.array([
    0.8,       # k_ref      — 20% low
    50000.0,   # Ea         — ~9% low
    0.06,      # k_ref_imp  — 25% low
    70000.0,   # Ea_imp     — ~7% low
    0.2,       # k_ref_dec  — 33% low
    75000.0,   # Ea_dec     — ~6% low
])

PARAM_NAMES = ["k_ref", "Ea", "k_ref_imp", "Ea_imp", "k_ref_dec", "Ea_dec"]


# =============================================================================
# ODE system
# =============================================================================

def _odes(t, y, Tjacket_K, cat, k_ref, Ea, k_ref_imp, Ea_imp, k_ref_dec, Ea_dec):
    """
    ODE right-hand side for the three-reaction batch system.

    State vector y = [CA, CB, CI, CD, T]
        CA : concentration of reactant A         (mol/L)
        CB : concentration of desired product B  (mol/L)
        CI : concentration of impurity I         (mol/L)
        CD : concentration of decomposition D    (mol/L)
        T  : reactor temperature                 (K)

    All three reactions are first-order in CA and follow a modified
    Arrhenius expression with a reference-temperature formulation:

        k(T) = k_ref * catalyst_load * exp( -Ea/R * (1/T - 1/T_ref) )

    This form avoids the numerical sensitivity of the standard Arrhenius
    pre-exponential factor A = k_ref * exp(Ea / (R * T_ref)) when Ea is large.

    Reactions:
        A -> B   r_main = k_main(T) * CA   endothermic, T_ref = 60 C
        A -> I   r_imp  = k_imp(T)  * CA   exothermic,  T_ref = 60 C
        A -> D   r_dec  = k_dec(T)  * CA   exothermic,  T_ref = 85 C

    The A->D reaction uses a higher reference temperature (85 C) reflecting
    the fact that it only becomes significant at elevated temperatures.

    Energy balance:
        mass * Cp * dT/dt = Q - Hr_main*r_main - Hr_imp*r_imp - Hr_dec*r_dec
        Q = U * A_area * (Tjacket_K - T)

    Note on sign convention: Hr > 0 means endothermic (absorbs heat, cools
    reactor). Hr < 0 means exothermic (releases heat, warms reactor).
    The energy balance therefore subtracts Hr*r, so an endothermic reaction
    (positive Hr) reduces dT/dt and an exothermic one (negative Hr) increases it.

    Parameters
    ----------
    t          : float        current time (hr)
    y          : list[float]  state vector [CA, CB, CI, CD, T(K)]
    Tjacket_K  : float        jacket temperature (K)
    cat        : float        catalyst load (dimensionless multiplier on k_ref)
    k_ref      : float        main reaction rate constant at T_ref (1/hr)
    Ea         : float        main reaction activation energy (J/mol)
    k_ref_imp  : float        impurity rate constant at T_ref (1/hr)
    Ea_imp     : float        impurity activation energy (J/mol)
    k_ref_dec  : float        decomposition rate constant at T_ref_dec (1/hr)
    Ea_dec     : float        decomposition activation energy (J/mol)

    Returns
    -------
    list[float] : [dCA/dt, dCB/dt, dCI/dt, dCD/dt, dT/dt]
    """
    CA, CB, CI, CD, T = y
    CA = max(CA, 0.0)   # guard against numerical undershoot below zero

    k_main = max(k_ref     * cat * np.exp(-Ea     / R * (1/T - 1/T_ref_K)),     0.0)
    k_imp  = max(k_ref_imp * cat * np.exp(-Ea_imp / R * (1/T - 1/T_ref_K)),     0.0)
    k_dec  = max(k_ref_dec * cat * np.exp(-Ea_dec / R * (1/T - 1/T_ref_dec_K)), 0.0)

    r_main = k_main * CA   # rate of A->B  (mol/L/hr)
    r_imp  = k_imp  * CA   # rate of A->I  (mol/L/hr)
    r_dec  = k_dec  * CA   # rate of A->D  (mol/L/hr)

    # Mass balances: A is consumed by all three pathways
    # B, I, D each produced by their respective pathway only
    dCA = -(r_main + r_imp + r_dec)
    dCB =   r_main
    dCI =   r_imp
    dCD =   r_dec

    # Energy balance: jacket supplies heat Q, reactions absorb/release heat
    Q  = U * A_area * (Tjacket_K - T)
    dT = (Q - Hr_main*r_main - Hr_imp*r_imp - Hr_dec*r_dec) / (mass * Cp)

    return [dCA, dCB, dCI, dCD, dT]


def _solve(T0, Tjacket, catalyst_load, mp, t_eval):
    """
    Fast scipy ODE solver for non-DoE uses: Stage 1 process optimisation,
    constraint landscape visualisation, and trajectory plotting.

    This function is intentionally kept separate from build_pyomo_model.
    The DoE-critical path (sensitivities, FIM, criterion) uses build_pyomo_model
    exclusively via designer.simulate() and the Pyomo IFT machinery — ensuring
    exact symbolic Jacobians and full consistency between simulated responses
    and computed sensitivities.

    _solve is used wherever fast approximate evaluations are sufficient:
      - process_objective / process_constraints (Stage 1 SLSQP — called
        hundreds of times per start; Pyomo would be ~100x slower)
      - plot_constraint_landscape (60×60 grid, ~3600 evaluations)
      - plot_case / plot_design_comparison (trajectory visualisation)

    Uses Radau (implicit Runge-Kutta, order 5), suited for the moderately
    stiff dynamics that arise at high temperatures.

    Parameters
    ----------
    T0             : float        initial reactor temperature (C)
    Tjacket        : float        jacket temperature (C)
    catalyst_load  : float        catalyst load (dimensionless)
    mp             : array-like   model parameters [k_ref, Ea, k_ref_imp,
                                  Ea_imp, k_ref_dec, Ea_dec]
    t_eval         : np.ndarray   time points at which to store solution (hr)

    Returns
    -------
    scipy OdeResult
        .y  : shape (5, len(t_eval)) — rows are [CA, CB, CI, CD, T(K)]
        .t  : shape (len(t_eval),)   — time points (hr)
    """
    sol = solve_ivp(
        _odes,
        (0.0, t_eval[-1]),
        [CA0_fixed, 0.0, 0.0, 0.0, T0 + 273.15],   # initial conditions
        args=(Tjacket + 273.15, catalyst_load, *mp),
        t_eval=t_eval,
        method='Radau',
        rtol=1e-8,
        atol=1e-10,
    )
    return sol


# =============================================================================
# Pyomo DAE model builder  — for IFT sensitivity computation
# =============================================================================

def build_pyomo_model(ti_controls, model_parameters, sampling_times=None, nfe=10, ncp=3):
    """
    Build and solve a Pyomo.DAE model for the three-reaction batch system
    using Lagrange-Radau orthogonal collocation (nfe=10, ncp=3 by default).

    This replaces the scipy Radau solver for sensitivity computation only.
    The model is solved with IPOPT; PyomoNLP then provides the exact sparse
    Jacobian dc/d[params, states] via the ASL interface, enabling IFT
    sensitivities without finite-difference perturbations.

    State variables (5):  CA, CB, CI, CD, T_K (reactor temperature in K)
    Parameters (6, fixed Var): k_ref, Ea, k_ref_imp, Ea_imp, k_ref_dec, Ea_dec

    Parameters are declared as fixed Var (not Param) so they appear in the
    NLP primal vector — PyomoNLP includes them once temporarily unfixed,
    giving the parameter Jacobian columns needed for the IFT.

    The standard Pyomo.DAE pattern for parameter estimation is used:
    measurement/sampling times are passed as the initialisation set of the
    ContinuousSet.  The discretiser then adds ncp Radau collocation points
    inside each of the nfe intervals, guaranteeing that every measurement
    time is an exact finite-element boundary — no manual grid merging needed.

    Parameters
    ----------
    ti_controls      : [T0_C, Tjacket_C, catalyst_load]
    model_parameters : [k_ref, Ea, k_ref_imp, Ea_imp, k_ref_dec, Ea_dec]
    sampling_times   : list/array of measurement times (hr); when None
                       defaults to [T_FINAL] (endpoint only)
    nfe              : number of finite elements (intervals between breakpoints)
    ncp              : collocation points per element (Radau order)

    Returns
    -------
    m           : solved ConcreteModel
    all_vars    : [k_ref, Ea, k_ref_imp, Ea_imp, k_ref_dec, Ea_dec,
                   CA[t0..tn], CB[t0..tn], CI[t0..tn], CD[t0..tn], T_K[t0..tn],
                   dCAdt[..], dCBdt[..], dCIdt[..], dCDdt[..], dTdt[..]]
    all_bodies  : equality constraint body expressions
    t_sorted    : measurement times snapped to nearest collocation points
    """
    T0_C, Tjacket_C, cat = ti_controls
    Tjacket_K_val = Tjacket_C + 273.15
    T0_K_val      = T0_C    + 273.15

    # Unpack parameters
    k_ref_v, Ea_v, k_ref_imp_v, Ea_imp_v, k_ref_dec_v, Ea_dec_v = model_parameters

    if sampling_times is None:
        sampling_times = [T_FINAL]

    # ── Determine integration horizon ─────────────────────────────────────────
    # For causal (sequential) sensitivities, the model must integrate only up
    # to the requested measurement time — not always to T_FINAL.
    # When designer._eval_sensitivities_pyomo_ift calls build_pyomo_model with
    # a single sampling time t_s, we build the DAE from 0 to t_s.  This gives
    # dCB(t_s)/dθ that reflects only the history up to t_s, exactly matching
    # the FD sensitivity.  Building to T_FINAL and extracting the t_s row gives
    # the wrong simultaneous (non-causal) sensitivity from the full collocation.
    t_horizon = max(float(t) for t in sampling_times)
    if t_horizon <= 0.0:
        t_horizon = T_FINAL

    # ── Standard Pyomo.DAE parameter-estimation pattern ───────────────────────
    # Seed the ContinuousSet with the measurement times (plus 0 and t_horizon
    # as domain boundaries).  The discretiser adds ncp Radau collocation points
    # inside each of the nfe intervals between these breakpoints, so every
    # measurement time is guaranteed to be an exact finite-element boundary.
    meas_pts = sorted(set(
        [0.0, t_horizon] + [float(t) for t in sampling_times
                             if 0.0 < float(t) < t_horizon]
    ))

    m = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(bounds=(0.0, t_horizon), initialize=meas_pts)

    # ── Parameters as fixed Vars ───────────────────────────────────────────────
    m.k_ref     = pyo.Var(initialize=k_ref_v);     m.k_ref.fix(k_ref_v)
    m.Ea        = pyo.Var(initialize=Ea_v);         m.Ea.fix(Ea_v)
    m.k_ref_imp = pyo.Var(initialize=k_ref_imp_v); m.k_ref_imp.fix(k_ref_imp_v)
    m.Ea_imp    = pyo.Var(initialize=Ea_imp_v);     m.Ea_imp.fix(Ea_imp_v)
    m.k_ref_dec = pyo.Var(initialize=k_ref_dec_v); m.k_ref_dec.fix(k_ref_dec_v)
    m.Ea_dec    = pyo.Var(initialize=Ea_dec_v);     m.Ea_dec.fix(Ea_dec_v)

    # ── State variables ────────────────────────────────────────────────────────
    m.CA  = pyo.Var(m.t, initialize=CA0_fixed, bounds=(0, None))
    m.CB  = pyo.Var(m.t, initialize=0.0,       bounds=(0, None))
    m.CI  = pyo.Var(m.t, initialize=0.0,       bounds=(0, None))
    m.CD  = pyo.Var(m.t, initialize=0.0,       bounds=(0, None))
    m.T_K = pyo.Var(m.t, initialize=T0_K_val,  bounds=(200, 600))

    m.dCAdt = dae.DerivativeVar(m.CA,  withrespectto=m.t)
    m.dCBdt = dae.DerivativeVar(m.CB,  withrespectto=m.t)
    m.dCIdt = dae.DerivativeVar(m.CI,  withrespectto=m.t)
    m.dCDdt = dae.DerivativeVar(m.CD,  withrespectto=m.t)
    m.dTdt  = dae.DerivativeVar(m.T_K, withrespectto=m.t)

    # ── ODE constraints ────────────────────────────────────────────────────────
    # Arrhenius: k(T) = k_ref * cat * exp(-Ea/R * (1/T - 1/T_ref))
    # Using pyo.exp() for symbolic differentiation support in PyomoNLP
    @m.Constraint(m.t)
    def ode_CA(m, t):
        k_main = m.k_ref     * cat * pyo.exp(-m.Ea     / R * (1/m.T_K[t] - 1/T_ref_K))
        k_imp  = m.k_ref_imp * cat * pyo.exp(-m.Ea_imp / R * (1/m.T_K[t] - 1/T_ref_K))
        k_dec  = m.k_ref_dec * cat * pyo.exp(-m.Ea_dec / R * (1/m.T_K[t] - 1/T_ref_dec_K))
        return m.dCAdt[t] == -(k_main + k_imp + k_dec) * m.CA[t]

    @m.Constraint(m.t)
    def ode_CB(m, t):
        k_main = m.k_ref * cat * pyo.exp(-m.Ea / R * (1/m.T_K[t] - 1/T_ref_K))
        return m.dCBdt[t] == k_main * m.CA[t]

    @m.Constraint(m.t)
    def ode_CI(m, t):
        k_imp = m.k_ref_imp * cat * pyo.exp(-m.Ea_imp / R * (1/m.T_K[t] - 1/T_ref_K))
        return m.dCIdt[t] == k_imp * m.CA[t]

    @m.Constraint(m.t)
    def ode_CD(m, t):
        k_dec = m.k_ref_dec * cat * pyo.exp(-m.Ea_dec / R * (1/m.T_K[t] - 1/T_ref_dec_K))
        return m.dCDdt[t] == k_dec * m.CA[t]

    @m.Constraint(m.t)
    def ode_T(m, t):
        k_main = m.k_ref     * cat * pyo.exp(-m.Ea     / R * (1/m.T_K[t] - 1/T_ref_K))
        k_imp  = m.k_ref_imp * cat * pyo.exp(-m.Ea_imp / R * (1/m.T_K[t] - 1/T_ref_K))
        k_dec  = m.k_ref_dec * cat * pyo.exp(-m.Ea_dec / R * (1/m.T_K[t] - 1/T_ref_dec_K))
        r_main = k_main * m.CA[t]
        r_imp  = k_imp  * m.CA[t]
        r_dec  = k_dec  * m.CA[t]
        Q = U * A_area * (Tjacket_K_val - m.T_K[t])
        return m.dTdt[t] == (Q - Hr_main*r_main - Hr_imp*r_imp - Hr_dec*r_dec) / (mass * Cp)

    # ── Initial conditions ─────────────────────────────────────────────────────
    m.ic_CA  = pyo.Constraint(expr=m.CA[0.0]  == CA0_fixed)
    m.ic_CB  = pyo.Constraint(expr=m.CB[0.0]  == 0.0)
    m.ic_CI  = pyo.Constraint(expr=m.CI[0.0]  == 0.0)
    m.ic_CD  = pyo.Constraint(expr=m.CD[0.0]  == 0.0)
    m.ic_T   = pyo.Constraint(expr=m.T_K[0.0] == T0_K_val)

    # Dummy objective required by PyomoNLP — square feasibility problem,
    # not an optimisation. The zero objective has no effect on the solution.
    m.obj = pyo.Objective(expr=0.0)

    # ── Discretise with Lagrange-Radau collocation ─────────────────────────────
    # The ContinuousSet already contains the measurement times as breakpoints.
    # apply_to() adds ncp Radau collocation points inside each of the nfe
    # intervals, preserving the measurement times exactly.
    disc = pyo.TransformationFactory('dae.collocation')
    disc.apply_to(m, nfe=nfe, ncp=ncp, scheme='LAGRANGE-RADAU')

    # ── Solve with IPOPT ───────────────────────────────────────────────────────
    # Variable values must be at the true collocation solution for the
    # PyomoNLP Jacobian to give correct IFT sensitivities.
    solver = pyo.SolverFactory('ipopt')
    solver.options['print_level'] = 0
    solver.options['tol'] = 1e-10
    result = solver.solve(m, tee=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"IPOPT did not converge: {result.solver.termination_condition}"
        )

    t_sorted_full = sorted(m.t)

    # Measurement times are exact finite-element boundaries — snap each to the
    # nearest point in the discretised grid (difference should be ~machine eps).
    t_sorted = sorted(set(
        min(t_sorted_full, key=lambda tt: abs(tt - float(t)))
        for t in sampling_times
    ))

    # ── Assemble all_vars: parameters FIRST, then states ──────────────────────
    # Designer uses the first n_mp=6 entries as parameter columns when
    # splitting the Jacobian into J_p and J_z.
    all_vars = (
        [m.k_ref, m.Ea, m.k_ref_imp, m.Ea_imp, m.k_ref_dec, m.Ea_dec]
        + [m.CA[t]  for t in t_sorted_full]
        + [m.CB[t]  for t in t_sorted_full]
        + [m.CI[t]  for t in t_sorted_full]
        + [m.CD[t]  for t in t_sorted_full]
        + [m.T_K[t] for t in t_sorted_full]
        + [m.dCAdt[t] for t in t_sorted_full]
        + [m.dCBdt[t] for t in t_sorted_full]
        + [m.dCIdt[t] for t in t_sorted_full]
        + [m.dCDdt[t] for t in t_sorted_full]
        + [m.dTdt[t]  for t in t_sorted_full]
    )

    # ── Collect all active equality constraint bodies ──────────────────────────
    all_bodies = []
    for con in m.component_objects(pyo.Constraint, active=True):
        for idx in con:
            c = con[idx]
            if c.equality:
                all_bodies.append(c.body - c.upper)

    return m, all_vars, all_bodies, t_sorted


# =============================================================================
# Pydex simulate function  (signature type 2)
# =============================================================================

def simulate(ti_controls, sampling_times, model_parameters):
    """
    Pydex-compatible simulate function.

    This function is assigned to designer.simulate and is called internally
    by pydex during sensitivity analysis and optimal design.  It must follow
    pydex signature type 2: simulate(ti_controls, sampling_times, model_parameters).

    Parameters
    ----------
    ti_controls      : array-like, length 3
                       [T0 (C), Tjacket (C), catalyst_load]
    sampling_times   : 1-D array of measurement times (hr)
                       Pydex passes the candidate sampling times here.
                       When optimize_sampling_times=True, pydex optimises
                       over these times as additional decision variables.
    model_parameters : 1-D array, length 6
                       [k_ref, Ea, k_ref_imp, Ea_imp, k_ref_dec, Ea_dec]
                       During sensitivity analysis, pydex perturbs these
                       values via finite differences to build the FIM.

    Returns
    -------
    np.ndarray, shape (n_spt, 4)
        Columns: [CA, CB, CI, CD] — concentrations at each sampling time (mol/L).
        CA0 is fixed at CA0_fixed = 1.0 mol/L for all experiments.
        Temperature (state index 4) is not returned as it is not directly
        measured, but it drives the ODE dynamics through the energy balance.
    """
    # Call build_pyomo_model to guarantee exact consistency between simulate()
    # and the IFT sensitivity model — both use the same collocation solve.
    T0, Tjacket, cat = ti_controls
    m, all_vars, all_bodies, t_sorted = build_pyomo_model(
        ti_controls, model_parameters, sampling_times=sampling_times
    )
    # Sampling times are exact grid points — read directly, no snapping error
    responses = np.zeros((len(sampling_times), 4))
    for i, t_req in enumerate(sampling_times):
        t_key = min(t_sorted, key=lambda tt: abs(tt - float(t_req)))
        responses[i, 0] = pyo.value(m.CA[t_key])
        responses[i, 1] = pyo.value(m.CB[t_key])
        responses[i, 2] = pyo.value(m.CI[t_key])
        responses[i, 3] = pyo.value(m.CD[t_key])
    return responses


# =============================================================================
# Stage 1: process objective and constraints
# =============================================================================

def process_objective(tic, tvc, mp):
    """
    Process optimisation objective for Stage 1.

    Maximises CB at the end of the batch (t = T_FINAL).  Since CA0 is
    fixed, maximising CB is equivalent to maximising the yield of B.

    The quality constraints (CI_MAX, CD_MAX) prevent the optimizer from
    achieving high CB via aggressive high-temperature routes that produce
    excessive impurity or decomposition product.  The result is a solution
    that sits on the constraint boundary — the best achievable yield given
    the product quality requirements.

    Parameters
    ----------
    tic : array-like  [T0 (C), Tjacket (C), catalyst_load]
    tvc : array-like  time-varying controls (unused — empty list)
    mp  : array-like  current model parameters

    Returns
    -------
    float : CB at t = T_FINAL (mol/L)  — to be maximised
    """
    sol = _solve(tic[0], tic[1], tic[2], mp, np.array([T_FINAL]))
    return float(sol.y[1, 0])


def process_constraints(tic, tvc, mp):
    """
    Process optimisation constraints for Stage 1.

    Returns a list of constraint dictionaries in pydex / scipy format.
    Each dict has keys "type" ("ineq" or "eq") and "fun" (callable).
    For "ineq" constraints, IPOPT requires fun(tic, tvc, mp) >= 0.

    Constraints
    -----------
    1. CI_final <= CI_MAX  :  impurity quality specification
       fun = CI_MAX - CI_final >= 0

    2. CD_final <= CD_MAX  :  decomposition byproduct specification
       fun = CD_MAX - CD_final >= 0

    3. Tjacket >= T0       :  physical constraint — jacket must heat the
       fun = Tjacket - T0 >= 0   reactor, not cool it (no refrigeration)

    The CI and CD constraints create two separate feasibility boundaries
    in the (T0, Tjacket, catalyst_load) space, and the optimizer must
    navigate to the point of maximum CB that satisfies both simultaneously.

    Parameters
    ----------
    tic : array-like  [T0 (C), Tjacket (C), catalyst_load]
    tvc : array-like  time-varying controls (unused)
    mp  : array-like  current model parameters

    Returns
    -------
    list of dicts : [{"type": "ineq", "fun": callable}, ...]
    """
    def ci_con(tic, tvc, mp):
        """CI_MAX - CI_final >= 0"""
        sol = _solve(tic[0], tic[1], tic[2], mp, np.array([T_FINAL]))
        return CI_MAX - float(sol.y[2, 0])

    def cd_con(tic, tvc, mp):
        """CD_MAX - CD_final >= 0"""
        sol = _solve(tic[0], tic[1], tic[2], mp, np.array([T_FINAL]))
        return CD_MAX - float(sol.y[3, 0])

    def jacket_con(tic, tvc, mp):
        """Tjacket - T0 >= 0"""
        return tic[1] - tic[0]

    return [
        {"type": "ineq", "fun": ci_con},
        {"type": "ineq", "fun": cd_con},
        {"type": "ineq", "fun": jacket_con},
    ]


# =============================================================================
# Experimental candidate grid
# =============================================================================

# Time-invariant control candidates: [T0 (C), Tjacket (C), catalyst_load]
# Only physically valid combinations are included (Tjacket >= T0).
T0_cands      = np.array([45, 50, 55, 60, 65, 70])       # initial temperature (C)
Tjacket_cands = np.array([50, 55, 60, 65, 70, 75, 80])   # jacket temperature  (C)
cat_cands     = np.array([0.5, 0.75, 1.0, 1.25, 1.5])    # catalyst load       (-)

tic_candidates = np.array([
    [T0, Tj, cat]
    for T0  in T0_cands
    for Tj  in Tjacket_cands  if Tj >= T0   # physical constraint
    for cat in cat_cands
])   # shape (n_c, 3)

# Sampling time candidates: 20 uniformly spaced points over [0.05, 1.0] hr.
# When optimize_sampling_times=True, pydex allocates effort over these
# times as additional decision variables alongside the candidate selection.
spt_grid       = np.linspace(0.05, 1.0, 10)                          # (hr)
spt_candidates = np.tile(spt_grid, (len(tic_candidates), 1))          # (n_c, 10)


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_case(T0, Tjacket, catalyst_load, title, mp=THETA_TRUE):
    """
    Plot concentration trajectories, temperature profile, selectivity,
    and yield breakdown for a single operating condition.

    Four panels:
      1. Concentrations of CA, CB, CI, CD vs time with constraint lines
      2. Reactor temperature vs time
      3. Instantaneous selectivity to B (CB / (CB + CI + CD)) vs time
      4. Yield of each species as % of CA0 vs time

    Parameters
    ----------
    T0, Tjacket, catalyst_load : float  operating conditions
    title : str                         figure title
    mp    : array-like                  model parameters (default: THETA_TRUE)
    """
    t_eval = np.linspace(0.0, T_FINAL, 300)
    sol    = _solve(T0, Tjacket, catalyst_load, mp, t_eval)
    CA = sol.y[0]; CB = sol.y[1]; CI = sol.y[2]
    CD = sol.y[3]; T  = sol.y[4] - 273.15

    converted = CB + CI + CD
    with np.errstate(invalid='ignore', divide='ignore'):
        SB = np.where(converted > 1e-6, CB / converted * 100, 100.0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(title, fontsize=11)

    axes[0].plot(t_eval, CA, 'b', label='CA')
    axes[0].plot(t_eval, CB, 'g', label='CB')
    axes[0].plot(t_eval, CI, 'r', label='CI')
    axes[0].plot(t_eval, CD, 'm', label='CD')
    axes[0].axhline(CI_MAX, color='r', ls='--', alpha=0.5,
                    label=f'CI_max={CI_MAX}')
    axes[0].axhline(CD_MAX, color='m', ls='--', alpha=0.5,
                    label=f'CD_max={CD_MAX}')
    axes[0].set(xlabel='time (hr)', ylabel='Concentration (mol/L)')
    axes[0].legend(fontsize=7)

    axes[1].plot(t_eval, T, 'k')
    axes[1].set(xlabel='time (hr)', ylabel='T (C)')

    axes[2].plot(t_eval, SB, 'g')
    axes[2].set(xlabel='time (hr)', ylabel='Selectivity to B (%)',
                ylim=[0, 105])

    axes[3].plot(t_eval, CB / CA0_fixed * 100, 'g', label='B yield')
    axes[3].plot(t_eval, CI / CA0_fixed * 100, 'r', label='I yield')
    axes[3].plot(t_eval, CD / CA0_fixed * 100, 'm', label='D yield')
    axes[3].set(xlabel='time (hr)', ylabel='Yield (% of CA0)')
    axes[3].legend(fontsize=7)

    plt.tight_layout()
    return fig


def plot_constraint_landscape(dw_tic, mp=THETA_GUESS):
    """
    Visualise the process optimisation landscape.

    Produces a 3-row contour map showing CB, CI, and CD at t=T_FINAL
    as functions of (T0, catalyst_load) for several fixed Tjacket values.

    Columns shown are always 60°C, 70°C, 80°C, plus the optimal Tjacket
    from dw_tic if it is more than 3°C from all fixed values.

    Constraint boundaries are overlaid as black dashed/dash-dot lines:
      - Black dashed    : CI = CI_MAX  (impurity limit)
      - Black dash-dot  : CD = CD_MAX  (decomposition limit)
    Both constraint values are embedded in the fill-level arrays so each
    dashed line falls exactly on a color-band edge.

    The constrained optimum (T0, catalyst_load) is found independently
    for each Tjacket column via a quick SLSQP solve and marked with a
    gold star (★).  Every column gets its own star because the optimum
    legitimately depends on Tjacket.

    All columns share the same colorbar scale for direct comparison.

    Parameters
    ----------
    dw_tic : np.ndarray, shape (r_w, 3)
             Optimal operating points from find_optimal_operating_point().
             Duplicates are removed automatically.
    mp     : array-like   model parameters (default: THETA_GUESS)
    """
    T0_grid  = np.linspace(45, 72, 60)   # finer grid → smoother contours
    cat_grid = np.linspace(0.5, 2.0, 60)
    T0_mg, cat_mg = np.meshgrid(T0_grid, cat_grid)
    t_eval = np.array([T_FINAL])

    # ── deduplicate operating points ──────────────────────────────────────────
    _, uid    = np.unique(np.round(dw_tic, 2), axis=0, return_index=True)
    dw_unique = dw_tic[uid]
    Tj_opt    = float(dw_unique[0, 1])

    # ── Tjacket columns: fixed set + optimal Tjacket if far enough away ───────
    Tjacket_fixed = [60.0, 70.0, 80.0]
    if all(abs(Tj_opt - Tj) > 3.0 for Tj in Tjacket_fixed):
        Tjacket_vals = sorted(Tjacket_fixed + [round(Tj_opt)])
    else:
        Tjacket_vals = Tjacket_fixed
    n_cols = len(Tjacket_vals)

    # ── global optimum coordinates (T0, catalyst_load) from Stage 1 ──────────
    # The star is placed only in the column whose Tjacket is closest to dw_Tjacket.
    # No per-column optimisation is performed — there is one unique optimum.
    dw_T0      = float(dw_unique[0, 0])
    dw_cat     = float(dw_unique[0, 2])
    dw_col_idx = int(np.argmin([abs(Tj - Tj_opt) for Tj in Tjacket_vals]))

    # ── pre-compute concentration maps ────────────────────────────────────────
    all_maps = []
    for Tj in Tjacket_vals:
        CB_map = np.full_like(T0_mg, np.nan)
        CI_map = np.full_like(T0_mg, np.nan)
        CD_map = np.full_like(T0_mg, np.nan)
        for i in range(T0_mg.shape[0]):
            for j in range(T0_mg.shape[1]):
                T0  = T0_mg[i, j]
                cat = cat_mg[i, j]
                if Tj < T0:
                    continue   # physically invalid: jacket can't be cooler than T0
                sol = _solve(T0, Tj, cat, mp, t_eval)
                CB_map[i, j] = sol.y[1, 0]
                CI_map[i, j] = sol.y[2, 0]
                CD_map[i, j] = sol.y[3, 0]
        all_maps.append((CB_map, CI_map, CD_map))

    # ── shared colorbar ranges ────────────────────────────────────────────────
    CB_vmin = np.nanmin([m[0] for m in all_maps])
    CB_vmax = np.nanmax([m[0] for m in all_maps])
    CI_vmin = np.nanmin([m[1] for m in all_maps])
    CI_vmax = np.nanmax([m[1] for m in all_maps])
    CD_vmin = np.nanmin([m[2] for m in all_maps])
    CD_vmax = np.nanmax([m[2] for m in all_maps])

    # embed constraint values into fill levels so dashed line = band edge
    CB_levels = np.linspace(CB_vmin, CB_vmax, 15)
    CI_levels = np.unique(np.sort(
        np.append(np.linspace(CI_vmin, CI_vmax, 14), CI_MAX)))
    CD_levels = np.unique(np.sort(
        np.append(np.linspace(CD_vmin, CD_vmax, 14), CD_MAX)))

    # ── check which constraints are actually visible in the plotted domain ───
    # A constraint line is "in range" only if its threshold lies strictly
    # between the global min and max of that map — otherwise contour() would
    # draw nothing (or clip at the boundary) and the legend entry is misleading.
    ci_in_range = CI_vmin < CI_MAX < CI_vmax
    cd_in_range = CD_vmin < CD_MAX < CD_vmax

    # ── figure layout: reserve a narrow strip on the right for 3 colorbars ───
    # GridSpec lets us place the colorbars in their own column, completely
    # outside all data axes, so they never overlap any panel.
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    # Layout constants (all in figure-fraction units)
    fig_left    = 0.07
    fig_right   = 0.91   # data grid right edge; colorbar fits in [0.927, 0.945]
    fig_top     = 0.93
    fig_bottom  = 0.06
    cb_left     = 0.927  # left edge of colorbar strip
    cb_width    = 0.018  # width of each colorbar
    hspace_frac = 0.08   # fraction of row height reserved for row spacing

    # Total height available and per-row height
    total_h  = fig_top - fig_bottom
    row_h    = total_h / 3
    inner_h  = row_h * (1 - hspace_frac)

    # Bottom coordinate of each row's colorbar (row 0 = top)
    row_bottoms = [fig_bottom + (2 - r) * row_h + row_h * hspace_frac / 2
                   for r in range(3)]

    fig = plt.figure(figsize=(5 * n_cols + 1.2, 12))
    gs  = GridSpec(3, n_cols, figure=fig,
                   left=fig_left, right=fig_right,
                   top=fig_top,   bottom=fig_bottom,
                   hspace=0.30,   wspace=0.10)

    axes = np.empty((3, n_cols), dtype=object)
    for r in range(3):
        for c in range(n_cols):
            share_x = axes[0, 0] if r > 0 else None
            share_y = axes[r, 0] if c > 0 else None
            axes[r, c] = fig.add_subplot(gs[r, c],
                                         sharex=share_x, sharey=share_y)

    # One colorbar per row, vertically aligned with its row
    cbar_axes = [fig.add_axes([cb_left, row_bottoms[r], cb_width, inner_h])
                 for r in range(3)]

    fig.suptitle(
        f'Constraint landscape at t={T_FINAL} hr  |  '
        + (f'Black dashed = CI_MAX={CI_MAX}  ' if ci_in_range else '')
        + (f'Black dash-dot = CD_MAX={CD_MAX}  ' if cd_in_range else '')
        + '|  ★ = process optimum',
        fontsize=10
    )

    # legend handles — only include lines/patches that actually appear in the plots
    legend_handles = []
    if ci_in_range:
        legend_handles.append(
            Line2D([0], [0], color='k', lw=1.5, ls='--',
                   label=f'CI = {CI_MAX} (impurity limit)'))
    if cd_in_range:
        legend_handles.append(
            Line2D([0], [0], color='k', lw=1.5, ls='-.',
                   label=f'CD = {CD_MAX} (decomp. limit)'))
    if ci_in_range or cd_in_range:
        from matplotlib.patches import Patch
        legend_handles.append(
            Patch(facecolor='white', edgecolor='k', hatch='////',
                  label='Infeasible region'))
    legend_handles.append(
        Line2D([0], [0], marker='*', color='gold', lw=0,
               markeredgecolor='k', markersize=10,
               label='Process optimum'))

    cf_row = [None, None, None]   # store last mappable per row for colorbars

    for col, (Tj, (CB_map, CI_map, CD_map)) in enumerate(
            zip(Tjacket_vals, all_maps)):

        col_label   = f'Tjacket={Tj:.1f}°C'
        is_dw_col   = (col == dw_col_idx)

        def add_star(ax, _show=is_dw_col):
            """Mark the global optimum (T0, catalyst) — only in the dw Tjacket column."""
            if _show:
                ax.plot(dw_T0, dw_cat, '*', color='gold',
                        markersize=16, markeredgecolor='k',
                        markeredgewidth=0.8, zorder=5)

        def add_legend(ax):
            ax.legend(handles=legend_handles, fontsize=6, loc='upper right')

        def add_hatch(ax, data_map, threshold, vmax):
            """Overlay diagonal hatching on the infeasible region (data > threshold)."""
            ax.contourf(T0_mg, cat_mg, data_map,
                        levels=[threshold, vmax + 1],
                        colors='none', hatches=['////'],
                        zorder=2)

        # ── Row 0: CB (product yield) ─────────────────────────────────────────
        ax = axes[0, col]
        cf = ax.contourf(T0_mg, cat_mg, CB_map, levels=CB_levels,
                         cmap='Greens', vmin=CB_vmin, vmax=CB_vmax)
        if ci_in_range:
            add_hatch(ax, CI_map, CI_MAX, CI_vmax)
            ax.contour(T0_mg, cat_mg, CI_map, levels=[CI_MAX],
                       colors='k', linewidths=1.5, linestyles='--')
        if cd_in_range:
            add_hatch(ax, CD_map, CD_MAX, CD_vmax)
            ax.contour(T0_mg, cat_mg, CD_map, levels=[CD_MAX],
                       colors='k', linewidths=1.5, linestyles='-.')
        ax.set_title(f'CB  |  {col_label}')
        if col == 0:
            ax.set_ylabel('Catalyst load')
        cf_row[0] = cf
        add_star(ax)
        add_legend(ax)

        # ── Row 1: CI (impurity) ──────────────────────────────────────────────
        ax2 = axes[1, col]
        cf2 = ax2.contourf(T0_mg, cat_mg, CI_map, levels=CI_levels,
                           cmap='Reds', vmin=CI_vmin, vmax=CI_vmax)
        if ci_in_range:
            add_hatch(ax2, CI_map, CI_MAX, CI_vmax)
            ax2.contour(T0_mg, cat_mg, CI_map, levels=[CI_MAX],
                        colors='k', linewidths=1.5, linestyles='--')
        ax2.set_title(f'CI  |  {col_label}')
        if col == 0:
            ax2.set_ylabel('Catalyst load')
        cf_row[1] = cf2
        add_star(ax2)
        add_legend(ax2)

        # ── Row 2: CD (decomposition product) ────────────────────────────────
        ax3 = axes[2, col]
        cf3 = ax3.contourf(T0_mg, cat_mg, CD_map, levels=CD_levels,
                           cmap='Purples', vmin=CD_vmin, vmax=CD_vmax)
        if ci_in_range:
            add_hatch(ax3, CI_map, CI_MAX, CI_vmax)
            ax3.contour(T0_mg, cat_mg, CI_map, levels=[CI_MAX],
                        colors='k', linewidths=1.5, linestyles='--')
        if cd_in_range:
            add_hatch(ax3, CD_map, CD_MAX, CD_vmax)
            ax3.contour(T0_mg, cat_mg, CD_map, levels=[CD_MAX],
                        colors='k', linewidths=1.5, linestyles='-.')
        ax3.set_title(f'CD  |  {col_label}')
        ax3.set_xlabel('T0 (°C)')
        if col == 0:
            ax3.set_ylabel('Catalyst load')
        cf_row[2] = cf3
        add_star(ax3)
        add_legend(ax3)

    # ── colorbars in their own axes, fully outside the data panels ────────────
    fig.colorbar(cf_row[0], cax=cbar_axes[0], label='CB (mol/L)')
    fig.colorbar(cf_row[1], cax=cbar_axes[1], label='CI (mol/L)')
    fig.colorbar(cf_row[2], cax=cbar_axes[2], label='CD (mol/L)')

    return fig


def plot_design_comparison(efforts_dict, tic_candidates, dw_tic,
                           jv_dict, mp=THETA_GUESS):
    """
    Compare V-optimal, A-optimal, and D-optimal experimental designs.

    For each design, two panels are shown side by side:

    Left — Bubble chart in (T0, Tjacket) space:
        Each bubble represents a selected candidate experiment.
        Bubble size is proportional to the effort weight allocated to that
        candidate. Bubble colour encodes catalyst load.
        All candidate experiments (selected and unselected) are shown as
        faint background markers to give context.
        The gold star marks the optimal operating point dw from Stage 1.

    Right — Concentration trajectories:
        Model trajectories (CB solid, CI dashed, CD dotted) are plotted for
        each selected candidate, with opacity proportional to effort weight.
        The bold black/red/magenta lines are the trajectories at dw itself.
        Circular markers show the selected sampling times for each candidate.

    All rows share identical axis limits, enabling direct visual comparison
    of how the three designs differ in their coverage of the operating space.

    Parameters
    ----------
    efforts_dict  : dict  {name: effort_array}  — one entry per design
    tic_candidates: np.ndarray  shape (n_c, 3)  — full candidate grid
    dw_tic        : np.ndarray  shape (r_w, 3)  — optimal operating point(s)
    jv_dict       : dict  {name: J_V value}     — prediction variance per design
    mp            : array-like  model parameters for trajectory plotting
    """
    designs = list(efforts_dict.keys())
    colours = {
        'V-optimal': '#1f77b4',
        'A-optimal': '#ff7f0e',
        'D-optimal': '#2ca02c',
    }
    tol    = 1e-3
    t_plot = np.linspace(0.0, T_FINAL, 200)

    # remove duplicate operating points
    _, uid    = np.unique(np.round(dw_tic, 2), axis=0, return_index=True)
    dw_unique = dw_tic[uid]

    # axis limits that include the dw point
    T0_all   = np.append(tic_candidates[:, 0], dw_unique[:, 0])
    Tj_all   = np.append(tic_candidates[:, 1], dw_unique[:, 1])
    bub_xlim = (T0_all.min() - 2, T0_all.max() + 2)
    bub_ylim = (Tj_all.min() - 2, Tj_all.max() + 4)

    # global trajectory y-limits (shared across all rows)
    all_CB, all_CI, all_CD = [], [], []
    for name, eff in efforts_dict.items():
        n_c    = len(tic_candidates)
        eff_2d = np.array(eff).reshape(n_c, -1)
        eff_pc = eff_2d.sum(axis=1)
        for idx in np.where(eff_pc > tol)[0]:
            sol = _solve(*tic_candidates[idx], mp, t_plot)
            all_CB.append(sol.y[1])
            all_CI.append(sol.y[2])
            all_CD.append(sol.y[3])
    for dw_pt in dw_unique:
        sol_dw = _solve(*dw_pt, mp, t_plot)
        all_CB.append(sol_dw.y[1])
        all_CI.append(sol_dw.y[2])
        all_CD.append(sol_dw.y[3])
    traj_ymax = max(np.concatenate(all_CB).max(),
                    np.concatenate(all_CI).max(),
                    np.concatenate(all_CD).max()) * 1.15
    traj_ylim = (-0.02, traj_ymax)

    fig, axes = plt.subplots(len(designs), 2,
                             figsize=(14, 4.5 * len(designs)))
    fig.suptitle(
        'Design comparison  |  Bubble size = effort weight  |  '
        'Design comparison  |  Bubble size = effort weight  |  '
        'Star = process optimum',
        fontsize=11
    )

    for row, name in enumerate(designs):
        ax_bub  = axes[row, 0]
        ax_traj = axes[row, 1]

        n_c     = len(tic_candidates)
        eff_2d  = np.array(efforts_dict[name]).reshape(n_c, -1)
        eff_pc  = eff_2d.sum(axis=1)   # total effort per candidate
        sup_idx = np.where(eff_pc > tol)[0]
        T0s     = tic_candidates[sup_idx, 0]
        Tjs     = tic_candidates[sup_idx, 1]
        cats    = tic_candidates[sup_idx, 2]
        effs    = eff_pc[sup_idx]

        # background: all candidates as faint markers
        ax_bub.scatter(
            tic_candidates[:, 0], tic_candidates[:, 1],
            s=60, c=tic_candidates[:, 2], cmap='plasma',
            alpha=0.12, edgecolors='grey', linewidths=0.3,
            vmin=cat_cands.min(), vmax=cat_cands.max(), zorder=1,
        )
        # foreground: selected candidates sized by effort
        sc = ax_bub.scatter(
            T0s, Tjs,
            s=effs / effs.max() * 800,
            c=cats, cmap='plasma', alpha=0.85,
            edgecolors='k', linewidths=0.8,
            vmin=cat_cands.min(), vmax=cat_cands.max(), zorder=3,
        )
        plt.colorbar(sc, ax=ax_bub, label='Catalyst load')

        for w, dw_pt in enumerate(dw_unique):
            ax_bub.plot(dw_pt[0], dw_pt[1], '*', color='gold',
                        markersize=16, markeredgecolor='k', zorder=5,
                        label='Process optimum' if w == 0 else '_nolegend_')

        ax_bub.set_xlabel('T0 (C)')
        ax_bub.set_ylabel('Tjacket (C)')
        ax_bub.set_title(
            f'{name}   Pred. Var. = {jv_dict[name]:.4f}  '
            f'({len(sup_idx)} support point(s))'
        )
        ax_bub.set_xlim(bub_xlim)
        ax_bub.set_ylim(bub_ylim)
        ax_bub.legend(fontsize=7, loc='lower right')

        # trajectories at selected candidates
        n_spt = eff_2d.shape[1]
        for idx in sup_idx:
            T0c, Tjc, catc = tic_candidates[idx]
            sol   = _solve(T0c, Tjc, catc, mp, t_plot)
            alpha = float(eff_pc[idx]) / float(eff_pc[sup_idx].max())
            ax_traj.plot(t_plot, sol.y[1], color=colours[name],
                         alpha=max(0.2, alpha), linewidth=1.2)
            ax_traj.plot(t_plot, sol.y[2], 'r--',
                         alpha=max(0.15, alpha * 0.6), linewidth=1.0)
            ax_traj.plot(t_plot, sol.y[3], 'm:',
                         alpha=max(0.15, alpha * 0.6), linewidth=1.0)
            # selected sampling times as scatter markers
            spt_eff  = eff_2d[idx, :n_spt]
            spt_here = spt_grid[:n_spt]
            sel_spt  = spt_here[spt_eff > tol]
            if len(sel_spt) > 0:
                sol_s = _solve(T0c, Tjc, catc, mp, sel_spt)
                ax_traj.scatter(
                    sel_spt, sol_s.y[1],
                    color=colours[name], s=40, zorder=6,
                    alpha=max(0.5, alpha),
                    edgecolors='k', linewidths=0.4,
                )

        # dw trajectory highlighted with white halo for visibility
        for w, dw_pt in enumerate(dw_unique):
            sol_dw = _solve(*dw_pt, mp, t_plot)
            for y_idx, col, lbl in [
                (1, 'k',           'CB @ process optimum'),
                (2, 'darkred',     'CI @ process optimum'),
                (3, 'darkmagenta', 'CD @ process optimum'),
            ]:
                # white halo drawn first, then coloured line on top
                ax_traj.plot(t_plot, sol_dw.y[y_idx], 'w-',
                             linewidth=5.0, zorder=4)
                ax_traj.plot(t_plot, sol_dw.y[y_idx], color=col,
                             linewidth=2.5, zorder=5,
                             label=lbl if w == 0 else '_nolegend_')

        ax_traj.axhline(CI_MAX, color='red',     ls=':', alpha=0.6)
        ax_traj.axhline(CD_MAX, color='magenta', ls=':', alpha=0.6)
        ax_traj.set_xlabel('time (hr)')
        ax_traj.set_ylabel('Concentration (mol/L)')
        ax_traj.set_ylim(traj_ylim)
        ax_traj.set_title(
            f'{name}: solid = CB  dashed = CI  dotted = CD\n'
            f'dots = selected sampling times  |  opacity proportional to effort'
        )
        ax_traj.legend(fontsize=7, ncol=2, loc='lower right')

    plt.tight_layout()
    return fig


def plot_jv_bar(jv_dict):
    """
    Bar chart comparing prediction variance J_V across designs.

    J_V = trace(W * FIM^{-1} * W^T) evaluated at dw for each design.
    Lower J_V means lower prediction uncertainty at the operating point
    of interest.  V-optimal minimises J_V by construction and therefore
    always achieves the lowest bar.  The ratios quantify the cost of
    using a globally-oriented criterion (A or D) when predictions at a
    specific operating point are what matters.

    Parameters
    ----------
    jv_dict : dict  {design_name: J_V_value}
    """
    names  = list(jv_dict.keys())
    values = [jv_dict[n] for n in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        names, values,
        color=['#1f77b4', '#ff7f0e', '#2ca02c'],
        edgecolor='k', linewidth=0.7,
    )
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
    ax.set_ylabel('Prediction Variance at Process Optimum')
    ax.set_title(
        'Prediction variance at the optimal operating point\n'
        'Lower is better — V-optimal minimises this by construction'
    )
    ax.set_ylim(0, max(values) * 1.2)
    plt.tight_layout()
    return fig


# =============================================================================
# Main script
# =============================================================================

if __name__ == "__main__":

    # ── 0. Explore model behaviour before running MBDoE ───────────────────────
    # Three representative cases illustrating the trade-off:
    #   Low T:  feasible but slow — both constraints satisfied with slack
    #   Mid T:  approaching the feasibility boundary
    #   High T: fast conversion but both CI and CD constraints violated
    print("Plotting representative model cases...")
    plot_case(50, 60, 1.0, 'Low T  — slow, both constraints satisfied',
              mp=THETA_TRUE)
    plot_case(58, 68, 1.0, 'Mid T  — approaching constraint boundaries',
              mp=THETA_TRUE)
    plot_case(65, 80, 1.5, 'High T — fast, both constraints violated',
              mp=THETA_TRUE)
    plt.show()

    # ── 1. Initialise the pydex designer ──────────────────────────────────────
    print("\nInitialising pydex designer...")
    designer = Designer()
    designer.simulate                  = simulate
    designer.model_parameters          = THETA_GUESS
    designer.ti_controls_candidates    = tic_candidates
    designer.sampling_times_candidates = spt_candidates
    # diagonal error covariance: sigma^2 on each of the 4 responses
    designer.error_cov = np.diag([SIGMA_RESPONSES**2] * 4)
    designer.model_parameters_names    = PARAM_NAMES
    designer.ti_controls_names         = ["T0_C", "Tjacket_C", "catalyst_load"]
    designer.response_names            = ["CA", "CB", "CI", "CD"]
    # ── Pyomo IFT exact sensitivity ───────────────────────────────────────────
    # Replace finite-difference sensitivity computation with exact IFT
    # Jacobian from PyomoNLP (ASL path).
    #
    # build_pyomo_model is wrapped in a closure that captures the current
    # sampling times from the designer so the collocation grid always
    # includes the exact requested sampling time points — no snapping,
    # no interpolation error, perfect consistency with simulate().
    def pyomo_model_fn_with_spt(tic, theta, sampling_times=None):
        # sampling_times is injected by the parallel worker for each candidate.
        # In the sequential path, designer._current_spt is used instead.
        spt = sampling_times if sampling_times is not None else designer._current_spt
        if spt is None or len(spt) == 0:
            spt = [T_FINAL]
        return build_pyomo_model(tic, theta, sampling_times=spt)

    designer.use_pyomo_ift         = True
    designer.pyomo_model_fn        = pyomo_model_fn_with_spt
    designer.pyomo_output_var_name = ["CA", "CB", "CI", "CD"]
    # ─────────────────────────────────────────────────────────────────────────
    designer.initialize(verbose=1)
    print(f"  Candidate experiments: {len(tic_candidates)}")
    print(f"  Sampling time candidates per experiment: {len(spt_grid)}")
    print(f"  Model parameters: {len(THETA_GUESS)}")

    # ── 2. Stage 1: process optimisation — find dw ────────────────────────────
    # Solve the constrained process optimisation to find the operating
    # condition dw = (T0, Tjacket, catalyst_load) that maximises CB
    # subject to the quality and physical constraints.
    #
    # Multiple starting points are used because IPOPT is a local solver.
    # The problem is well-conditioned here and all starts converge to the
    # same solution, but this is good practice for general use.
    print("\n" + "=" * 60)
    print("STAGE 1: Process optimisation — finding optimal dw")
    print("=" * 60)

    designer.process_objective   = process_objective
    designer.process_constraints = process_constraints
    designer.dw_sense            = "maximize"
    designer.dw_bounds_tic       = [
        (45.0, 70.0),   # T0        (C)
        (50.0, 85.0),   # Tjacket   (C)  — capped at 85C (decomp dominates above)
        (0.5,  2.0),    # catalyst_load
    ]
    designer.dw_bounds_tvc = []   # no time-varying controls in this model

    init_guesses = np.array([
        [55.0, 65.0, 1.0],    # start 1: moderate conditions
        [50.0, 70.0, 1.5],    # start 2: lower T, higher cat
        [60.0, 75.0, 0.75],   # start 3: higher T, lower cat
    ])

    dw_tic, dw_tvc = designer.find_optimal_operating_point(
        init_guess     = init_guesses,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER, "tol": 1e-8, "max_iter": 500},
        n_starts       = 2,
    )

    # Report results — tolerance of 1e-6 accounts for floating-point equality
    print("\nOptimal operating point(s):")
    for w in range(dw_tic.shape[0]):
        T0_opt, Tj_opt, cat_opt = dw_tic[w]
        sol  = _solve(T0_opt, Tj_opt, cat_opt, THETA_GUESS, np.array([T_FINAL]))
        CB_f = float(sol.y[1, 0])
        CI_f = float(sol.y[2, 0])
        CD_f = float(sol.y[3, 0])
        ci_status = 'OK'   if CI_f <= CI_MAX + 1e-6 else 'VIOLATED'
        cd_status = 'OK'   if CD_f <= CD_MAX + 1e-6 else 'VIOLATED'
        print(f"  [{w+1}]  T0={T0_opt:.2f} C  Tjacket={Tj_opt:.2f} C  "
              f"catalyst={cat_opt:.3f}")
        print(f"         CB={CB_f:.4f}  CI={CI_f:.4f} ({ci_status})"
              f"  CD={CD_f:.4f} ({cd_status})")

    # Select the single best operating point (highest objective value).
    # Multi-start may find multiple local optima — we pass only the best
    # to Stage 2 so the W matrix encodes a single prediction target,
    # which gives a cleaner, more interpretable V-optimal design.
    best_idx = np.argmax(designer._dw_obj_vals)
    dw_tic   = dw_tic[[best_idx]]
    designer.dw_tic = dw_tic
    T0_opt, Tj_opt, cat_opt = dw_tic[0]
    print(f"\n  Best operating point selected (index {best_idx+1}):")
    print(f"  T0={T0_opt:.2f} C  Tjacket={Tj_opt:.2f} C  catalyst={cat_opt:.3f}")

    # Visualise the constraint landscape and the optimal point
    print("\nPlotting constraint landscape...")
    plot_constraint_landscape(dw_tic, mp=THETA_GUESS)
    best = dw_tic[0]
    plot_case(
        best[0], best[1], best[2],
        f"Optimal dw: T0={best[0]:.1f} C  Tjacket={best[1]:.1f} C  "
        f"catalyst={best[2]:.2f}  (on constraint boundary)",
        mp=THETA_GUESS,
    )
    plt.show()

    # ── 3. Stage 2: V-optimal design ──────────────────────────────────────────
    # Design experiments that minimise prediction variance at dw.
    # dw_spt specifies the time point(s) at which prediction accuracy matters.
    # Here we care about predictions at the end of the batch (t = T_FINAL).
    #
    # The W matrix is built internally by evaluating model sensitivities at
    # dw_tic / dw_tvc / dw_spt. The FIM is built from the candidate grid as
    # usual. The criterion J_V = trace(W * FIM^{-1} * W^T) is minimised.
    print("\n" + "=" * 60)
    print("STAGE 2: V-optimal experimental design")
    print("=" * 60)

    designer.dw_spt = np.array([T_FINAL])   # care about predictions at t=1 hr

    # Note: with the Pyomo IFT path, pydex may distribute effort equally across
    # all sampling time candidates when the system is genuinely indifferent to
    # sampling time (i.e. IFT sensitivities are nearly equal at all times).
    # This is mathematically correct — set SPT_MIN_EFFORT > 0 in CONFIG to
    # enforce sparsity and reveal the single most informative sampling time.
    #
    # Sparsity requires a MINLP solver — Bonmin is used automatically when
    # SPT_MIN_EFFORT > 0. IPOPT relaxes the binary variables to continuous,
    # so min_effort has no effect with IPOPT.
    if SPT_MIN_EFFORT > 0.0:
        v_solver         = "gams"
        v_solver_options = {"solver": "baron"}
        v_min_effort     = SPT_MIN_EFFORT
    else:
        v_solver         = "ipopt"
        v_solver_options = {"linear_solver": LINEAR_SOLVER, "tol": 1e-8, "max_iter": 1000}
        v_min_effort     = None

    designer.design_v_optimal(
        solver                  = v_solver,
        solver_options          = v_solver_options,
        regularize_fim          = False,
        optimize_sampling_times = True,   # let pydex choose when to sample
        min_effort              = v_min_effort,
    )
    print(f"\nV-optimal  J_V = {designer._criterion_value:.6f}")
    designer.print_optimal_candidates(tol=1e-3)
    v_opt_efforts = designer.efforts.copy()

    # ── Diagnostic: check if IFT sensitivities truly vary across sampling times ──
    print("\n" + "="*60)
    print("DIAGNOSTIC: IFT sensitivity variation across sampling times")
    print("="*60)
    sens = designer.sensitivities  # shape (n_c, n_spt, n_mr, n_mp)
    spts = designer.sampling_times_candidates
    # Use the dominant support candidate (highest effort)
    eff_2d = designer.efforts.reshape(designer.n_c, -1)
    c_diag = int(np.argmax(eff_2d.sum(axis=1)))
    print(f"\n  Candidate {c_diag+1}: "
          f"T0={designer.ti_controls_candidates[c_diag, 0]:.0f}°C  "
          f"Tj={designer.ti_controls_candidates[c_diag, 1]:.0f}°C  "
          f"cat={designer.ti_controls_candidates[c_diag, 2]:.1f}")
    print(f"\n  {'t (hr)':>8}  {'tr(FIM_t)':>12}  {'||S_t||_F':>12}  {'effort':>8}")
    print(f"  {'-'*46}")
    n_spt_diag = sens.shape[1]
    for t in range(n_spt_diag):
        S_t    = sens[c_diag, t]                    # (n_mr, n_mp)
        fim_t  = S_t.T @ S_t
        effort = float(eff_2d[c_diag, t])
        print(f"  {spts[c_diag, t]:>8.2f}  {np.trace(fim_t):>12.4f}  "
              f"{np.linalg.norm(S_t, 'fro'):>12.4f}  {effort:>8.4f}")
    print("="*60)

    # ── 4. A-optimal design (comparison) ──────────────────────────────────────
    # A-optimal minimises total parameter variance: J_A = trace(FIM^{-1})
    # This spreads experimental effort to reduce uncertainty in all parameters
    # equally, without regard for where predictions will be used.
    print("\n" + "=" * 60)
    print("COMPARISON: A-optimal design")
    print("=" * 60)

    designer.design_experiment(
        criterion               = designer.a_opt_criterion,
        solver                  = v_solver,
        solver_options          = v_solver_options,
        optimize_sampling_times = True,
        regularize_fim          = True,   # stabilises FIM^{-1} when FIM is near-singular
        e0                      = v_opt_efforts,  # warm-start from V-optimal solution
        min_effort              = v_min_effort,
    )
    print(f"\nA-optimal  J_A = {designer._criterion_value:.6f}")
    designer.print_optimal_candidates(tol=1e-3)
    a_opt_efforts = designer.efforts.copy()

    # ── 5. D-optimal design (comparison) ──────────────────────────────────────
    # D-optimal maximises det(FIM), minimising the volume of the joint
    # parameter confidence ellipsoid. It selects experiments that maximise
    # the overall information content without targeting any specific
    # prediction direction. Typically pushes experiments to the extremes
    # of the operating space to maximise sensitivity contrasts.
    print("\n" + "=" * 60)
    print("COMPARISON: D-optimal design")
    print("=" * 60)

    designer.design_experiment(
        criterion               = designer.d_opt_criterion,
        solver                  = v_solver,
        solver_options          = v_solver_options,
        optimize_sampling_times = True,
        regularize_fim          = True,
        e0                      = v_opt_efforts,  # warm-start from V-optimal solution
        min_effort              = v_min_effort,
    )
    print(f"\nD-optimal  J_D = {designer._criterion_value:.6f}")
    designer.print_optimal_candidates(tol=1e-3)
    d_opt_efforts = designer.efforts.copy()

    # ── 6. Summary: compare J_V across designs ────────────────────────────────
    # Evaluate J_V = trace(W * FIM^{-1} * W^T) for each design using the
    # same W matrix (fixed at dw).  This is the apples-to-apples comparison:
    # how well does each design reduce prediction uncertainty at the point
    # that actually matters for process operation?
    print("\n" + "=" * 60)
    print("SUMMARY: prediction variance at optimal operating point dw")
    print("=" * 60)

    def eval_JV(efforts):
        """Evaluate J_V for a given effort allocation."""
        designer._eval_fim(efforts)
        try:
            fim_inv = np.linalg.inv(designer.fim)
        except np.linalg.LinAlgError:
            fim_inv = np.linalg.pinv(designer.fim)
        return float(np.trace(designer.W @ fim_inv @ designer.W.T))

    jv_vopt = eval_JV(v_opt_efforts)
    jv_aopt = eval_JV(a_opt_efforts)
    jv_dopt = eval_JV(d_opt_efforts)

    jv_dict = {
        'V-optimal': jv_vopt,
        'A-optimal': jv_aopt,
        'D-optimal': jv_dopt,
    }
    efforts_dict = {
        'V-optimal': v_opt_efforts,
        'A-optimal': a_opt_efforts,
        'D-optimal': d_opt_efforts,
    }

    print(f"\n  {'Design':<12}  {'J_V':>12}  {'vs V-optimal':>12}")
    print(f"  {'-' * 40}")
    print(f"  {'V-optimal':<12}  {jv_vopt:>12.6f}  {'(best by construction)':>12}")
    print(f"  {'A-optimal':<12}  {jv_aopt:>12.6f}  "
          f"{f'{jv_aopt/jv_vopt:.2f}x worse':>12}")
    print(f"  {'D-optimal':<12}  {jv_dopt:>12.6f}  "
          f"{f'{jv_dopt/jv_vopt:.2f}x worse':>12}")

    # ── 7. Visualise results ───────────────────────────────────────────────────
    print("\nPlotting design comparison...")
    plot_design_comparison(
        efforts_dict   = efforts_dict,
        tic_candidates = tic_candidates,
        dw_tic         = dw_tic,
        jv_dict        = jv_dict,
        mp             = THETA_GUESS,
    )
    plot_jv_bar(jv_dict)
    plt.show()
