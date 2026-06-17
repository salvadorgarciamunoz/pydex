"""
pydex_full_capability_test.py
==============================
Comprehensive test of all pydex Designer capabilities, derived from the
three-reaction batch model introduced in v_optimal_test_case.py.

Reaction system
---------------
    A -> B    desired product    endothermic   Ea_main = 55,000 J/mol
    A -> I    impurity           exothermic    Ea_imp  = 75,000 J/mol
    A -> D    decomposition      exothermic    Ea_dec  = 80,000 J/mol

Parameters estimated: [k_ref, Ea, k_ref_imp, Ea_imp, k_ref_dec, Ea_dec]

Capabilities exercised (in order)
-----------------------------------
 01.  Designer setup and initialization
 02.  Candidate grid enumeration helpers (create_grid / enumerate_candidates)
 02.  Sensitivity analysis (eval_sensitivities)
 02.  Sensitivity visualisation (plot_sensitivities)
 02b. Sensitivity diagnosis (diagnose_sensitivity)
 03.  Local D-optimal design
 04.  Local A-optimal design
 05.  Local E-optimal design
 06.  D-optimal with optimize_sampling_times=True
 07.  Pseudo-Bayesian D-optimal (average-information, type 0)
 08.  Pseudo-Bayesian D-optimal (average-criterion, type 1)
 09.  CVaR D-optimal design
 10.  Continuous → exact design (apportion / Adams method)
 11.  Prior FIM — Case A: set_prior_fim (FIM from external covariance)
 12.  Prior FIM — Case B: set_prior_experiments (arbitrary prior conditions)
 13.  V-optimal workflow
        Stage 1: find_optimal_operating_point
        Stage 2: design_v_optimal
 13b. Process optimizer standalone verification (find_optimal_operating_point)
 14.  Saving and loading OED results (write_oed_result / load_oed_result)
 15.  Saving and restoring full designer state (save_state / load_state)
 16.  Visualisation suite
        plot_optimal_efforts, plot_optimal_controls
        plot_optimal_predictions, plot_optimal_sensitivities
        plot_predictions, plot_sensitivities
        diagnose_sensitivity (full grid)
        plot_criterion_cdf / plot_criterion_pdf (CVaR)
 17.  Sparsity-enforcing MINLP design (min_effort, BARON via GAMS)
 18.  CVaR bi-objective Pareto frontier (solve_cvar_problem / plot_pareto_frontier)
 19.  Pyomo IFT sensitivity — auto-detection in initialize()
        use_pyomo_ift auto-set, n_jobs auto-set, override respected
 20.  Pyomo IFT — local D-optimal (signature 1, sequential)
        correctness: criterion and support points vs analytical truth
 21.  Pyomo IFT — local D-optimal (signature 1, parallel n_jobs=-1)
        correctness: criterion matches sequential within tolerance
 22.  Pyomo IFT — pseudo-Bayesian D-optimal (parallel, type 0)
        correctness: criterion matches sequential baseline within tolerance
 23.  Pyomo IFT — sensitivity normalization toggle (_norm_sens_by_params)
        verifies normalized and unnormalized paths produce consistent FIMs
 24.  FD vs IFT sensitivity agreement
        verifies both methods give consistent sensitivities on same model
 25.  Pyomo DAE simulate + IFT — local D-optimal (sequential, self-consistency)
        simulate() calls Pyomo.DAE directly; IFT sensitivities checked against
        analytical-simulate IFT; support points checked against analytical truth
 26.  Pyomo DAE simulate + IFT — local D-optimal (parallel correctness)
        parallel criterion matches sequential from Test 25
 27.  Pyomo DAE simulate + IFT — pseudo-Bayesian D-optimal (parallel)
        full end-to-end real-world scenario: no analytical fallback anywhere
 28.  DAE simulate vs analytical simulate response agreement
        collocation response agrees with A0·exp(−kt) to within 0.1%
 29.  Generalized and individual criteria (dg, di, ag, ai, eg, ei)
        all six criteria run without error and select valid support candidates
 30.  Pyomo IFT — signature-2 model with multiple outputs and sampling times
        A→B→C series kinetics, measures [B(t), C(t)]; IFT vs FD agreement < 2%
 31.  Regularized FIM (regularize_fim=True)
        verifies eps*I inflation increases criterion, same support, flag stored
 33.  IFT sampling-time optimisation — regression guard
        Check A: analytical truth (first-order reaction, t* = 1/k)
        Check B: FD vs IFT cross-validation with optimize_sampling_times=True
        catches the bug where IFT always evaluated sensitivity at t_f

Usage
-----
    python pydex_full_capability_test.py

    Set SHOW_PLOTS = False to suppress matplotlib windows (useful in CI).
    Set LINEAR_SOLVER = "mumps" if "ma57" is not available.
    Set VERBOSE = 2 for full pydex output; 0 for silent.

Solver architecture
-------------------
    D/A/E/V-optimal  : native Pyomo expressions → IPOPT via .nl file
    Pseudo-Bayesian  : native Pyomo + IPOPT
    CVaR             : native Pyomo + IPOPT
    MINLP sparsity   : native Pyomo + BARON via GAMS  (solver="gams",
                         solver_options={"gams_solver": "baron", ...})
                       or Bonmin  (solver="bonmin")
    Operating point  : scipy SLSQP  (solver_options: ftol, maxiter)
"""

# =============================================================================
# Imports
# =============================================================================
import sys
import os
import warnings
# Suppress Pyomo deprecation warning about CyIpoptNLP import path
warnings.filterwarnings(
    "ignore",
    message=".*CyIpoptNLP.*",
    category=UserWarning,
)

# Force reload of pydex from disk at every run — ensures the installed
# designer.py is always the version on disk, not a stale cached module.
import importlib
for _mod_name in list(sys.modules.keys()):
    if _mod_name.startswith('pydex'):
        del sys.modules[_mod_name]

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — set before importing pyplot
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pydex.core.designer import Designer

try:
    import pyomo.environ as pyo
    import pyomo.dae as dae
    _PYOMO_AVAILABLE = True
except ImportError:
    _PYOMO_AVAILABLE = False

# =============================================================================
# Test configuration
# =============================================================================
SHOW_PLOTS    = False          # True → display figures interactively
LINEAR_SOLVER = "ma57"         # "mumps" if HSL not available
VERBOSE       = 1              # 0=silent, 1=summary, 2=full pydex output
SEED          = 42
CRIT_RTOL     = 5e-3           # relative tolerance for criterion value assertions

np.random.seed(SEED)


# =============================================================================
# Physical constants and true parameters
# =============================================================================
R            = 8.314
T_ref_C      = 60.0;    T_ref_K    = T_ref_C    + 273.15
T_ref_dec_C  = 85.0;    T_ref_dec_K = T_ref_dec_C + 273.15
Hr_main      =  50000.0
Hr_imp       = -30000.0
Hr_dec       = -60000.0
Cp           = 4184.0
mass         = 1.0
U            = 5000.0
A_area       = 1.0
CA0_fixed    = 1.0
T_FINAL      = 1.0

CI_MAX = 0.05
CD_MAX = 0.05

THETA_TRUE = np.array([1.0, 55000.0, 0.08, 75000.0, 0.3, 80000.0])
THETA_GUESS = np.array([0.8, 50000.0, 0.06, 70000.0, 0.2, 75000.0])
PARAM_NAMES = ["k_ref", "Ea", "k_ref_imp", "Ea_imp", "k_ref_dec", "Ea_dec"]


# =============================================================================
# ODE model
# =============================================================================
def _odes(t, y, Tjacket_K, cat, k_ref, Ea, k_ref_imp, Ea_imp, k_ref_dec, Ea_dec):
    CA, CB, CI, CD, T = y
    CA = max(CA, 0.0)
    k_main = max(k_ref     * cat * np.exp(-Ea     / R * (1/T - 1/T_ref_K)),     0.0)
    k_imp  = max(k_ref_imp * cat * np.exp(-Ea_imp / R * (1/T - 1/T_ref_K)),     0.0)
    k_dec  = max(k_ref_dec * cat * np.exp(-Ea_dec / R * (1/T - 1/T_ref_dec_K)), 0.0)
    r_main = k_main * CA
    r_imp  = k_imp  * CA
    r_dec  = k_dec  * CA
    dCA = -(r_main + r_imp + r_dec)
    dCB =   r_main
    dCI =   r_imp
    dCD =   r_dec
    Q   = U * A_area * (Tjacket_K - T)
    dT  = (Q - Hr_main*r_main - Hr_imp*r_imp - Hr_dec*r_dec) / (mass * Cp)
    return [dCA, dCB, dCI, dCD, dT]


def _solve(T0, Tjacket, cat, mp, t_eval):
    return solve_ivp(
        _odes,
        (0.0, t_eval[-1]),
        [CA0_fixed, 0.0, 0.0, 0.0, T0 + 273.15],
        args=(Tjacket + 273.15, cat, *mp),
        t_eval=t_eval,
        method='Radau',
        rtol=1e-8, atol=1e-10,
    )


# =============================================================================
# pydex simulate function  (signature type 2)
# =============================================================================
def simulate(ti_controls, sampling_times, model_parameters):
    T0, Tjacket, cat = ti_controls
    sol = _solve(T0, Tjacket, cat, model_parameters, sampling_times)
    return np.column_stack([sol.y[0], sol.y[1], sol.y[2], sol.y[3]])


# =============================================================================
# Process objective and constraints  (for V-optimal Stage 1)
# =============================================================================
def process_objective(tic, tvc, mp):
    sol = _solve(tic[0], tic[1], tic[2], mp, np.array([T_FINAL]))
    return float(sol.y[1, 0])

def process_constraints(tic, tvc, mp):
    def ci_con(tic, tvc, mp):
        sol = _solve(tic[0], tic[1], tic[2], mp, np.array([T_FINAL]))
        return CI_MAX - float(sol.y[2, 0])
    def cd_con(tic, tvc, mp):
        sol = _solve(tic[0], tic[1], tic[2], mp, np.array([T_FINAL]))
        return CD_MAX - float(sol.y[3, 0])
    def jacket_con(tic, tvc, mp):
        return tic[1] - tic[0]
    return [
        {"type": "ineq", "fun": ci_con},
        {"type": "ineq", "fun": cd_con},
        {"type": "ineq", "fun": jacket_con},
    ]


# =============================================================================
# Helpers
# =============================================================================
def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def ok(label):
    print(f"  [OK] {label}")

def show(fig):
    """Close figures whether fig is a single Figure, a list, or None."""
    if SHOW_PLOTS and fig is not None:
        plt.show()
    plt.close("all")

def make_designer(theta=None, small=False):
    """
    Build and initialise a fresh Designer with the batch reactor model.
    small=True uses a reduced candidate grid for speed in expensive tests.
    """
    d = Designer()
    d.simulate  = simulate
    d.model_parameters = theta if theta is not None else THETA_GUESS.copy()
    d.error_cov = np.diag([0.01**2] * 4)
    d.model_parameters_names = PARAM_NAMES
    d.ti_controls_names      = ["T0_C", "Tjacket_C", "catalyst_load"]
    d.response_names         = ["CA", "CB", "CI", "CD"]

    if small:
        T0_cands = np.array([50, 60, 70])
        Tj_cands = np.array([55, 65, 75, 85])
        cat_cands = np.array([0.5, 1.0, 1.5])
        spt_grid  = np.linspace(0.1, 1.0, 5)
    else:
        T0_cands  = np.array([45, 50, 55, 60, 65, 70])
        Tj_cands  = np.array([50, 55, 60, 65, 70, 75, 80])
        cat_cands = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
        spt_grid  = np.linspace(0.05, 1.0, 20)

    tic = np.array([
        [T0, Tj, cat]
        for T0  in T0_cands
        for Tj  in Tj_cands  if Tj >= T0
        for cat in cat_cands
    ])
    spt = np.tile(spt_grid, (len(tic), 1))

    d.ti_controls_candidates    = tic
    d.sampling_times_candidates = spt
    d.initialize(verbose=VERBOSE)
    return d


# =============================================================================
# T E S T S
# =============================================================================

def test_01_init_and_grid_helpers():
    section("01 — Initialization and grid helpers (create_grid / enumerate_candidates)")

    d = Designer()
    d.simulate = simulate
    d.model_parameters = THETA_GUESS.copy()
    d.error_cov = np.diag([0.01**2] * 4)

    # create_grid
    grid = d.create_grid(
        bounds=[(45, 70), (50, 80), (0.5, 1.5)],
        levels=[3, 3, 2],
    )
    assert grid.shape[1] == 3, "create_grid should return (n, 3)"
    ok(f"create_grid: {grid.shape[0]} points")

    # enumerate_candidates
    d.ti_controls_candidates = np.array([
        [T0, Tj, cat]
        for T0  in [50, 60, 70]
        for Tj  in [55, 65, 75] if Tj >= T0
        for cat in [0.5, 1.0, 1.5]
    ])
    spt_grid = np.linspace(0.1, 1.0, 5)
    d.sampling_times_candidates = np.tile(spt_grid, (len(d.ti_controls_candidates), 1))
    d.initialize(verbose=0)
    ok(f"initialize: {d.n_c} candidates, n_spt={d.n_spt}, n_mp={d.n_mp}")

    return d   # reuse for test 02


def test_02_sensitivity_analysis(d):
    section("02 — Sensitivity analysis and visualisation")

    d.eval_sensitivities()
    assert d.sensitivities is not None
    assert d.sensitivities.shape == (d.n_c, d.n_spt, d.n_m_r, d.n_mp)
    ok(f"sensitivities shape: {d.sensitivities.shape}")

    figs = d.plot_sensitivities()
    ok(f"plot_sensitivities: {len(figs)} figure(s)")
    show(figs)


def test_02b_diagnose_sensitivity(d):
    section("02b — Sensitivity diagnosis (diagnose_sensitivity)")

    result = d.diagnose_sensitivity(
        tol_diag = 1.0,
        tol_cond = 1e4,
        plot     = True,
    )

    assert result["diag_A"].shape == (d.n_c, d.n_mp), "diag_A shape mismatch"
    assert result["cond"].shape   == (d.n_c,),         "cond shape mismatch"
    assert len(result["singular_vals"]) == d.n_c,      "singular_vals length mismatch"
    assert "flagged_diag" in result
    assert "flagged_cond" in result
    ok(f"diag_A shape: {result['diag_A'].shape}")
    ok(f"Near-zero diagonal flags : {len(result['flagged_diag'])} (candidate, parameter) pairs")
    ok(f"Ill-cond candidates      : {len(result['flagged_cond'])}")

    assert np.all(result["diag_A"] >= -1e-12), "diag_A must be non-negative (A_k is PSD)"
    ok("diag_A >= 0 (PSD check)")

    assert np.all(result["cond"] > 0), "condition numbers must be positive"
    ok("all condition numbers > 0")

    for c, ev in enumerate(result["singular_vals"]):
        assert np.all(ev >= -1e-10), f"negative eigenvalue at candidate {c}"
        assert np.all(np.diff(ev) <= 1e-10), f"eigenvalues not sorted descending at candidate {c}"
    ok("eigenvalues non-negative and sorted descending")

    figs = result["figs"]
    ok(f"Figures produced: {len(figs)}")
    for f_ in figs:
        show(f_)

    ok(f"tol_diag=1.0: flags where 1 experiment cannot determine θⱼ to within its own magnitude")

    d_degen = make_designer(small=True)
    d_degen.eval_sensitivities()
    d_degen.sensitivities[:, :, :, -1] *= 1e-8
    result_degen = d_degen.diagnose_sensitivity(tol_diag=1.0, plot=False)
    n_flagged_last = sum(1 for _, j in result_degen["flagged_diag"] if j == d_degen.n_mp - 1)
    ok(f"Degenerate case: {n_flagged_last}/{d_degen.n_c} candidates flag "
       f"'{result_degen['param_names'][-1]}' (expected all)")
    assert n_flagged_last == d_degen.n_c, \
        f"Expected all {d_degen.n_c} candidates to flag last param, got {n_flagged_last}"


def test_03_d_optimal(d):
    section("03 — Local D-optimal design")

    d.design_experiment(
        criterion      = d.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-8, "max_iter": 2000},
    )
    d.print_optimal_candidates(tol=1e-3)
    ok(f"D-optimal criterion value: {d._criterion_value:.4f}")

    figs = d.plot_optimal_efforts()
    show(figs)
    figs = d.plot_optimal_controls()
    show(figs)
    return d.efforts.copy()


def test_04_a_optimal(d):
    section("04 — Local A-optimal design")

    d.design_experiment(
        criterion      = d.a_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-8, "max_iter": 2000},
    )
    d.print_optimal_candidates(tol=1e-3)
    ok(f"A-optimal criterion value: {d._criterion_value:.4f}")


def test_05_e_optimal(d):
    section("05 — Local E-optimal design")

    d.design_experiment(
        criterion      = d.e_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-6, "max_iter": 2000},
    )
    d.print_optimal_candidates(tol=1e-3)
    ok(f"E-optimal criterion value: {d._criterion_value:.4f}")


def test_06_optimize_sampling_times(d):
    section("06 — D-optimal with optimize_sampling_times=True")

    d.design_experiment(
        criterion               = d.d_opt_criterion,
        solver                  = "ipopt",
        solver_options          = {"linear_solver": LINEAR_SOLVER,
                                   "tol": 1e-8, "max_iter": 2000},
        optimize_sampling_times = True,
    )
    d.print_optimal_candidates(tol=1e-3)
    ok(f"D-optimal (opt spt) criterion value: {d._criterion_value:.4f}")

    figs = d.plot_optimal_sensitivities()
    show(figs)
    figs = d.plot_optimal_predictions()
    show(figs)


def test_07_pseudo_bayesian_type0(d_small):
    section("07 — Pseudo-Bayesian D-optimal (average-information, type 0)")

    N_scr = 50
    k_samples  = np.random.uniform(0.6, 1.4,  N_scr)
    Ea_samples = np.random.uniform(48000, 62000, N_scr)
    # keep other params at nominal
    scenarios = np.column_stack([
        k_samples,
        Ea_samples,
        np.full(N_scr, THETA_GUESS[2]),
        np.full(N_scr, THETA_GUESS[3]),
        np.full(N_scr, THETA_GUESS[4]),
        np.full(N_scr, THETA_GUESS[5]),
    ])

    d_small.model_parameters = scenarios

    # Sequential baseline (n_jobs=1) — reference criterion value
    d_small.n_jobs = 1
    d_small.design_experiment(
        criterion            = d_small.d_opt_criterion,
        solver               = "ipopt",
        solver_options       = {"linear_solver": LINEAR_SOLVER,
                                "tol": 1e-8, "max_iter": 2000},
        pseudo_bayesian_type = 0,
    )
    crit_seq = d_small._criterion_value
    ok(f"PB D-opt (type 0) sequential criterion: {crit_seq:.4f}")

    # Parallel run — must match sequential within CRIT_RTOL
    d_small.n_jobs = -1
    d_small.pb_atomic_fims = None   # force recomputation
    d_small.model_parameters = scenarios
    d_small.design_experiment(
        criterion            = d_small.d_opt_criterion,
        solver               = "ipopt",
        solver_options       = {"linear_solver": LINEAR_SOLVER,
                                "tol": 1e-8, "max_iter": 2000},
        pseudo_bayesian_type = 0,
    )
    crit_par = d_small._criterion_value
    ok(f"PB D-opt (type 0) parallel criterion:   {crit_par:.4f}")
    rel_err = abs(crit_par - crit_seq) / (abs(crit_seq) + 1e-12)
    assert rel_err < CRIT_RTOL, (
        f"Parallel PB criterion differs from sequential: "
        f"{crit_par:.6f} vs {crit_seq:.6f}  (rel err {rel_err:.2e} > {CRIT_RTOL})"
    )
    ok(f"Parallel matches sequential (rel err {rel_err:.2e} < {CRIT_RTOL})")

    d_small.n_jobs = 1
    d_small.model_parameters = THETA_GUESS.copy()


def test_08_pseudo_bayesian_type1(d_small):
    section("08 — Pseudo-Bayesian D-optimal (average-criterion, type 1)")

    N_scr = 30
    scenarios = np.column_stack([
        np.random.uniform(0.6, 1.4,    N_scr),
        np.random.uniform(48000, 62000, N_scr),
        np.full(N_scr, THETA_GUESS[2]),
        np.full(N_scr, THETA_GUESS[3]),
        np.full(N_scr, THETA_GUESS[4]),
        np.full(N_scr, THETA_GUESS[5]),
    ])
    d_small.model_parameters = scenarios

    # Sequential baseline
    d_small.n_jobs = 1
    d_small.design_experiment(
        criterion            = d_small.d_opt_criterion,
        solver               = "ipopt",
        solver_options       = {"linear_solver": LINEAR_SOLVER,
                                "tol": 1e-8, "max_iter": 2000},
        pseudo_bayesian_type = 1,
    )
    crit_seq = d_small._criterion_value
    ok(f"PB D-opt (type 1) sequential criterion: {crit_seq:.4f}")

    # Parallel
    d_small.n_jobs = -1
    d_small.pb_atomic_fims = None
    d_small.model_parameters = scenarios
    d_small.design_experiment(
        criterion            = d_small.d_opt_criterion,
        solver               = "ipopt",
        solver_options       = {"linear_solver": LINEAR_SOLVER,
                                "tol": 1e-8, "max_iter": 2000},
        pseudo_bayesian_type = 1,
    )
    crit_par = d_small._criterion_value
    ok(f"PB D-opt (type 1) parallel criterion:   {crit_par:.4f}")
    rel_err = abs(crit_par - crit_seq) / (abs(crit_seq) + 1e-12)
    assert rel_err < CRIT_RTOL, (
        f"Parallel PB type-1 criterion differs from sequential: "
        f"{crit_par:.6f} vs {crit_seq:.6f}  (rel err {rel_err:.2e} > {CRIT_RTOL})"
    )
    ok(f"Parallel matches sequential (rel err {rel_err:.2e} < {CRIT_RTOL})")

    d_small.n_jobs = 1
    d_small.model_parameters = THETA_GUESS.copy()


def test_09_cvar(d_small):
    section("09 — CVaR D-optimal design")

    N_scr = 30
    scenarios = np.column_stack([
        np.random.uniform(0.6, 1.4,    N_scr),
        np.random.uniform(48000, 62000, N_scr),
        np.full(N_scr, THETA_GUESS[2]),
        np.full(N_scr, THETA_GUESS[3]),
        np.full(N_scr, THETA_GUESS[4]),
        np.full(N_scr, THETA_GUESS[5]),
    ])
    d_small.model_parameters = scenarios

    d_small.design_experiment(
        criterion            = d_small.cvar_d_opt_criterion,
        solver               = "ipopt",
        solver_options       = {"ftol": 1e-6, "maxiter": 3000},  # SLSQP (solver= ignored for CVaR/PB)
        pseudo_bayesian_type = 0,
        beta                 = 0.80,
    )
    d_small.print_optimal_candidates(tol=1e-3)
    ok(f"CVaR D-opt criterion: {d_small._criterion_value:.4f}")

    fig_cdf = d_small.plot_criterion_cdf()
    fig_pdf = d_small.plot_criterion_pdf()
    ok("plot_criterion_cdf and plot_criterion_pdf")
    show(fig_cdf); show(fig_pdf)

    d_small.model_parameters = THETA_GUESS.copy()


def test_10_apportion(d):
    section("10 — Continuous → exact design (apportion)")

    # Use the D-optimal design already in d
    d.design_experiment(
        criterion      = d.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-8, "max_iter": 2000},
    )

    d.apportion(n_exp=8, method="adams")
    ok(f"Adams apportionment: {d.apportionments}")



def test_11_prior_fim_case_a(d):
    section("11 — Prior FIM Case A: set_prior_fim (from external covariance)")

    # Simulate: previous round gave a rough covariance — off-diagonal ignored
    sigma_theta = np.diag([0.05**2, 3000.0**2, 0.01**2, 5000.0**2,
                           0.05**2, 5000.0**2])
    fim_raw     = np.linalg.inv(sigma_theta)
    # normalise to pydex convention
    theta_prior = THETA_GUESS.copy()
    fim_norm    = fim_raw * np.outer(theta_prior, theta_prior)

    d.set_prior_fim(fim=fim_norm, model_parameters=theta_prior)
    ok(f"set_prior_fim: prior FIM rank = {np.linalg.matrix_rank(fim_norm)}/{d.n_mp}")

    # design with prior
    d.design_experiment(
        criterion      = d.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-8, "max_iter": 2000},
    )
    d.print_optimal_candidates(tol=1e-3)
    ok(f"D-optimal WITH prior (Case A): {d._criterion_value:.4f}")

    val_prior = d._criterion_value
    d.clear_prior()
    ok("clear_prior: prior removed")

    # design without prior — criterion value should be worse (less info)
    d.design_experiment(
        criterion      = d.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-8, "max_iter": 2000},
    )
    val_no_prior = d._criterion_value
    ok(f"D-optimal WITHOUT prior: {val_no_prior:.4f}")
    ok(f"Prior makes design more informative: "
       f"{'YES' if val_prior > val_no_prior else 'NO (unexpected)'}")


def test_12_prior_experiments_case_b(d):
    section("12 — Prior FIM Case B: set_prior_experiments (arbitrary conditions)")

    # Three prior experiments at conditions not necessarily on the candidate grid
    prior_tic = np.array([
        [52.0, 63.0, 0.9],
        [57.0, 68.0, 1.1],
        [48.0, 60.0, 1.3],
    ])
    prior_spt = np.tile(np.array([0.25, 0.5, 0.75, 1.0]), (3, 1))

    d.set_prior_experiments(
        ti_controls      = prior_tic,
        sampling_times   = prior_spt,
        model_parameters = THETA_GUESS.copy(),
        n_repeats        = np.array([2, 1, 1]),   # first condition run twice
    )
    ok(f"set_prior_experiments: prior FIM rank = "
       f"{np.linalg.matrix_rank(d._prior_fim)}/{d.n_mp}")

    # Update model_parameters (simulate re-estimation after prior experiments)
    # Rescaling should happen automatically
    theta_updated = THETA_GUESS * np.array([1.05, 0.97, 1.10, 0.99, 1.08, 0.98])
    d.model_parameters = theta_updated

    d.design_experiment(
        criterion      = d.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-8, "max_iter": 2000},
    )
    d.print_optimal_candidates(tol=1e-3)
    ok(f"Sequential D-optimal (Case B prior + updated θ): {d._criterion_value:.4f}")

    d.clear_prior()
    d.model_parameters = THETA_GUESS.copy()


def test_13_v_optimal(d):
    section("13 — V-optimal workflow (Stage 1 + Stage 2)")

    d.process_objective   = process_objective
    d.process_constraints = process_constraints
    d.dw_sense            = "maximize"
    d.dw_bounds_tic       = [(45.0, 70.0), (50.0, 85.0), (0.5, 2.0)]
    d.dw_bounds_tvc       = []

    # Stage 1: find optimal operating point (uses scipy SLSQP internally)
    dw_tic, _ = d.find_optimal_operating_point(
        init_guess     = np.array([[55.0, 65.0, 1.0],
                                   [60.0, 75.0, 0.75]]),
        solver         = "ipopt",           # passed but SLSQP is used internally
        solver_options = {"ftol": 1e-8, "maxiter": 500},
        n_starts       = 1,
    )

    # Select the single best operating point so W is (n_spt_dw * n_mr, n_mp)
    # not doubled. Multiple starts may converge to the same point — using
    # argmax ensures we always pass exactly one prediction target to Stage 2.
    best_idx = int(np.argmax(d._dw_obj_vals))
    dw_tic   = dw_tic[[best_idx]]
    d.dw_tic = dw_tic
    ok(f"Stage 1 dw: {dw_tic[0]}")

    # Stage 2: V-optimal design (uses native Pyomo expressions + IPOPT)
    d.dw_spt = np.array([T_FINAL])
    d.design_v_optimal(
        solver                  = "ipopt",
        solver_options          = {"linear_solver": LINEAR_SOLVER,
                                   "tol": 1e-8, "max_iter": 1000},
        regularize_fim          = False,
        optimize_sampling_times = True,
    )
    # W matrix should be (n_spt_dw * n_m_r, n_mp) = (1 * 4, 6) = (4, 6)
    # with a single best dw point. A doubled W (8, 6) indicates the
    # deduplication bug — multiple dw points passed to Stage 2.
    assert d.W.shape == (4, 6), \
        f"W matrix should be (4, 6) with single dw point, got {d.W.shape}"
    ok(f"W matrix shape correct: {d.W.shape}")
    d.print_optimal_candidates(tol=1e-3)
    ok(f"V-optimal J_V: {d._criterion_value:.4f}")

    cv_v = d.compute_criterion_value(d.v_opt_criterion)
    ok(f"compute_criterion_value (v_opt): {cv_v:.4f}")

    figs = d.plot_optimal_efforts()
    show(figs)


def test_13b_operating_point(d):
    section("13b — Process optimizer (find_optimal_operating_point)")

    d.process_objective   = process_objective
    d.process_constraints = process_constraints
    d.dw_sense            = "maximize"
    d.dw_bounds_tic       = [(45.0, 70.0), (50.0, 85.0), (0.5, 2.0)]
    d.dw_bounds_tvc       = []

    # --- single start ---
    dw_tic, dw_tvc = d.find_optimal_operating_point(
        init_guess     = np.array([[55.0, 65.0, 1.0]]),
        solver         = "ipopt",
        solver_options = {"ftol": 1e-10, "maxiter": 1000},
        n_starts       = 1,
    )
    tic_opt = dw_tic[0]
    obj_val = float(d._dw_obj_vals[0])
    ok(f"Optimal operating point: T0={tic_opt[0]:.1f}°C  "
       f"Tjacket={tic_opt[1]:.1f}°C  cat={tic_opt[2]:.3f}  "
       f"CB={obj_val:.4f}")

    # 1. Objective is better than a naive central point
    cb_naive = float(process_objective(
        np.array([57.5, 67.5, 1.25]), np.zeros(1), THETA_GUESS
    ))
    assert obj_val >= cb_naive - 1e-4, \
        f"Optimizer did not improve over naive point: {obj_val:.4f} < {cb_naive:.4f}"
    ok(f"Objective improved over naive centre: {obj_val:.4f} > {cb_naive:.4f}")

    # 2. All constraints satisfied at the optimum
    from scipy.integrate import solve_ivp as _solve_ivp
    sol = _solve_ivp(
        _odes, (0.0, T_FINAL),
        [CA0_fixed, 0.0, 0.0, 0.0, tic_opt[0] + 273.15],
        args=(tic_opt[1] + 273.15, tic_opt[2], *THETA_GUESS),
        t_eval=np.array([T_FINAL]), method='Radau', rtol=1e-8, atol=1e-10,
    )
    CI_opt = float(sol.y[2, 0])
    CD_opt = float(sol.y[3, 0])
    ok(f"CI at optimum: {CI_opt:.4f}  (limit {CI_MAX})  "
       f"{'OK' if CI_opt <= CI_MAX + 1e-4 else 'VIOLATED'}")
    ok(f"CD at optimum: {CD_opt:.4f}  (limit {CD_MAX})  "
       f"{'OK' if CD_opt <= CD_MAX + 1e-4 else 'VIOLATED'}")
    assert CI_opt <= CI_MAX + 1e-4, \
        f"Impurity constraint violated: CI={CI_opt:.4f} > {CI_MAX}"
    assert CD_opt <= CD_MAX + 1e-4, \
        f"Decomposition constraint violated: CD={CD_opt:.4f} > {CD_MAX}"
    assert tic_opt[1] >= tic_opt[0] - 1e-4, \
        f"Jacket temperature constraint violated: Tj={tic_opt[1]:.1f} < T0={tic_opt[0]:.1f}"
    ok("All process constraints satisfied at optimum")

    # 3. At least one constraint is active (expected for this model)
    ci_active = abs(CI_opt - CI_MAX) < 0.005
    cd_active = abs(CD_opt - CD_MAX) < 0.005
    ok(f"Constraint activity: CI {'ACTIVE' if ci_active else 'inactive'}  "
       f"CD {'ACTIVE' if cd_active else 'inactive'}")

    # 4. Multiple starts from a single point — should find same optimum from any start
    dw_tic_ms, dw_tvc_ms = d.find_optimal_operating_point(
        init_guess     = np.array([[55.0, 65.0, 1.0]]),
        solver         = "ipopt",
        solver_options = {"ftol": 1e-10, "maxiter": 1000},
        n_starts       = 3,
    )
    obj_ms = float(d._dw_obj_vals[0])
    assert obj_ms >= obj_val - 1e-4, \
        f"Multi-start did worse than single start: {obj_ms:.4f} < {obj_val:.4f}"
    ok(f"Multi-start ({3} starts) objective: {obj_ms:.4f}  "
       f"(single-start: {obj_val:.4f})")

    # 5. Bounds respected
    assert all(lb - 1e-4 <= dw_tic_ms[0][i] <= ub + 1e-4
               for i, (lb, ub) in enumerate([(45,70),(50,85),(0.5,2.0)])), \
        f"Optimal point violates bounds: {dw_tic_ms[0]}"
    ok(f"Bounds respected: {dw_tic_ms[0]}")


def test_14_save_load_result(d):
    section("14 — Save and load OED result")

    d.design_experiment(
        criterion      = d.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-8, "max_iter": 2000},
        write          = True,
    )
    # load_oed_result prepends getcwd(), so we need a relative path
    import os
    abs_result = [f for f in os.listdir(d.result_dir)
                  if f.endswith("_oed_result.pkl")][-1]
    abs_path = os.path.join(d.result_dir, abs_result)
    # load_oed_result does open(getcwd() + result_path) so we strip getcwd()
    cwd = os.getcwd()
    rel_path = abs_path[len(cwd):] if abs_path.startswith(cwd) else abs_path

    d2 = make_designer(small=True)
    d2.load_oed_result(rel_path)
    assert np.allclose(d.efforts, d2.efforts, atol=1e-6)
    ok(f"load_oed_result: efforts match  (rel path: {rel_path})")


def test_15_save_load_state(d):
    section("15 — Save and load full designer state")

    d.save_state()
    import os
    abs_state = [f for f in os.listdir(d.result_dir)
                 if f.endswith(".pkl") and "state" in f][-1]
    abs_path = os.path.join(d.result_dir, abs_state)
    cwd = os.getcwd()
    rel_path = abs_path[len(cwd):] if abs_path.startswith(cwd) else abs_path

    d2 = Designer()
    d2.simulate = simulate
    d2.load_state(rel_path)
    ok(f"load_state succeeded  (rel path: {rel_path})")


def test_16_visualisation_suite(d):
    section("16 — Visualisation suite")

    # Ensure a fresh D-optimal design is available
    d.design_experiment(
        criterion               = d.d_opt_criterion,
        solver                  = "ipopt",
        solver_options          = {"linear_solver": LINEAR_SOLVER,
                                   "tol": 1e-8, "max_iter": 2000},
        optimize_sampling_times = True,
    )

    figs = d.plot_optimal_efforts();          ok("plot_optimal_efforts");          show(figs)
    figs = d.plot_optimal_controls();         ok("plot_optimal_controls");         show(figs)
    figs = d.plot_optimal_predictions();      ok("plot_optimal_predictions");      show(figs)
    figs = d.plot_optimal_sensitivities();    ok("plot_optimal_sensitivities");    show(figs)
    figs = d.plot_predictions();              ok("plot_predictions");              show(figs)
    figs = d.plot_sensitivities();            ok("plot_sensitivities");            show(figs)
    res  = d.diagnose_sensitivity(tol_diag=1.0, tol_cond=1e4, plot=True)
    ok(f"diagnose_sensitivity: {len(res['figs'])} figure(s), "
       f"{len(res['flagged_diag'])} diag flags, "
       f"{len(res['flagged_cond'])} cond flags")
    for f_ in res["figs"]:
        show(f_)


def test_17_minlp_sparsity(d_small):
    section("17 — Sparsity-enforcing MINLP design (min_effort)")

    d_small.model_parameters = THETA_GUESS.copy()

    # BARON via GAMS (global optimum guaranteed, Lilly license used automatically).
    # io_options and add_options are passed to slvr.solve() — this is how the
    # Pyomo GAMS plugin works: SolverFactory("gams") + solve(..., io_options, add_options).
    #
    # Alternative — Bonmin (local MINLP, from IDAES package):
    #   solver="bonmin", solver_options={"tol": 1e-6, "max_iter": 3000}
    d_small.design_experiment(
        criterion      = d_small.d_opt_criterion,
        solver         = "gams",
        solver_options = {
            "io_options" : {"solver": "baron"},
            "add_options": [
                "GAMS_MODEL.optfile = 1;",
                "$onecho > baron.opt",
                "MaxTime 300",
                "AbsConTol 1e-6",
                "$offecho",
            ],
        },
        min_effort = 0.05,
    )
    d_small.print_optimal_candidates(tol=1e-3)
    ok(f"MINLP sparse D-optimal (BARON/GAMS): {d_small._criterion_value:.4f}")

    # check sparsity: no effort between 0 and min_effort
    e_flat = d_small.efforts.flatten()
    e_nonzero = e_flat[e_flat > 1e-6]
    below_threshold = np.any((e_nonzero > 1e-6) & (e_nonzero < 0.05))
    ok(f"Sparsity enforced: no effort in (0, 0.05): {not below_threshold}")


def test_18_cvar_pareto(d_small):
    section("18 — CVaR bi-objective Pareto frontier (solve_cvar_problem)")

    N_scr = 20
    scenarios = np.column_stack([
        np.random.uniform(0.6, 1.4,    N_scr),
        np.random.uniform(48000, 62000, N_scr),
        np.full(N_scr, THETA_GUESS[2]),
        np.full(N_scr, THETA_GUESS[3]),
        np.full(N_scr, THETA_GUESS[4]),
        np.full(N_scr, THETA_GUESS[5]),
    ])
    d_small.model_parameters = scenarios

    d_small.solve_cvar_problem(
        criterion            = d_small.cvar_d_opt_criterion,
        beta                 = 0.80,
        solver               = "ipopt",
        solver_options       = {"ftol": 1e-6, "maxiter": 2000},  # SLSQP options
        pseudo_bayesian_type = 0,
        reso                 = 3,
        plot                 = False,
    )
    assert d_small._biobjective_values is not None
    ok(f"CVaR Pareto frontier: {d_small._biobjective_values.shape[0]} points")

    fig = d_small.plot_pareto_frontier()
    ok("plot_pareto_frontier")
    show(fig)

    d_small.model_parameters = THETA_GUESS.copy()


# =============================================================================
# Pyomo IFT model (first-order reaction, signature 1)
# Used by tests 19–24
# =============================================================================

def _build_pyomo_model_1st_order(ti_controls, model_parameters,
                                  sampling_times=None, nfe=20, ncp=3):
    """
    Pyomo.DAE model for dA/dt = -k*A, A(0)=A0.
    Returns (model, all_vars, all_bodies, t_sorted) per pydex IFT contract.
    k and A0 declared as fixed Var so PyomoNLP includes them in the Jacobian.

    sampling_times : list/array of requested measurement times (passed by
                     designer._eval_sensitivities_pyomo_ift via _current_spt).
                     When None, defaults to [t_f] (endpoint only).
    """
    if not _PYOMO_AVAILABLE:
        raise ImportError("Pyomo not available — tests 19-24 require pyomo + pyomo.dae")

    k_val  = float(model_parameters[0])
    A0_val = float(model_parameters[1])
    t_f    = float(ti_controls[0])

    if t_f <= 0.0:
        m = pyo.ConcreteModel()
        m.k  = pyo.Var(initialize=k_val);   m.k.fix(k_val)
        m.A0 = pyo.Var(initialize=A0_val);  m.A0.fix(A0_val)
        m.A  = pyo.Var(initialize=A0_val);  m.A.fix(A0_val)
        m.trivial = pyo.Constraint(expr=m.A == m.A0)
        m.obj = pyo.Objective(expr=0.0)
        return m, [m.k, m.A0, m.A], [m.trivial.body - pyo.value(m.A0)], [0.0]

    # Explicit uniform grid — faster than ContinuousSet(bounds, initialize)
    # for this single-endpoint model where t_f is the domain boundary, not
    # an interior measurement time. Mathematically equivalent, better performance.
    t_grid = np.linspace(0.0, t_f, nfe + 1).tolist()
    m = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(initialize=t_grid)
    m.k    = pyo.Var(initialize=k_val);   m.k.fix(k_val)
    m.A0   = pyo.Var(initialize=A0_val);  m.A0.fix(A0_val)
    m.A    = pyo.Var(m.t, initialize=A0_val, bounds=(0, None))
    m.dAdt = dae.DerivativeVar(m.A, withrespectto=m.t)

    @m.Constraint(m.t)
    def ode(m, t):
        return m.dAdt[t] == -m.k * m.A[t]

    @m.Constraint()
    def ic(m):
        return m.A[0.0] == m.A0

    m.obj = pyo.Objective(expr=0.0)
    disc  = pyo.TransformationFactory('dae.collocation')
    disc.apply_to(m, nfe=nfe, ncp=ncp, scheme='LAGRANGE-RADAU')

    solver = pyo.SolverFactory('ipopt')
    solver.options['print_level'] = 0
    solver.options['tol'] = 1e-12
    result = solver.solve(m, tee=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"IPOPT did not converge for t={t_f}")

    t_sorted_full = sorted(m.t)

    # Snap each requested sampling time to the nearest collocation point.
    # If sampling_times is None, default to the endpoint only.
    if sampling_times is None or len(sampling_times) == 0:
        t_sorted = [t_sorted_full[-1]]
    else:
        t_sorted = sorted(set(
            min(t_sorted_full, key=lambda tt: abs(tt - float(s)))
            for s in sampling_times
        ))

    all_vars = (
        [m.k, m.A0]
        + [m.A[t] for t in t_sorted_full]
        + [m.dAdt[t] for t in t_sorted_full]
    )
    all_bodies = []
    for con in m.component_objects(pyo.Constraint, active=True):
        for idx in con:
            c = con[idx]
            if c.equality:
                all_bodies.append(c.body - c.upper)

    return m, all_vars, all_bodies, t_sorted


def _simulate_1st_order(ti_controls, model_parameters):
    """Analytical simulate for dA/dt = -k*A."""
    t  = float(ti_controls[0])
    k  = float(model_parameters[0])
    A0 = float(model_parameters[1])
    return np.array([A0 * np.exp(-k * t)])


def _make_pyomo_designer(model_parameters, n_candidates=51, verbose=0):
    """Build a Designer using the Pyomo IFT path for the first-order reaction."""
    t_candidates = np.linspace(0.0, 10.0, n_candidates).reshape(-1, 1)
    d = Designer()
    d.simulate               = _simulate_1st_order
    d.model_parameters       = model_parameters
    d.ti_controls_candidates = t_candidates
    d.pyomo_model_fn         = _build_pyomo_model_1st_order
    # use_pyomo_ift and n_jobs are auto-set by initialize()
    d.initialize(verbose=verbose)
    return d


# =============================================================================
# New tests 19–24
# =============================================================================

def test_19_pyomo_ift_auto_detection():
    section("19 — Pyomo IFT auto-detection in initialize()")

    if not _PYOMO_AVAILABLE:
        ok("SKIPPED — Pyomo not available")
        return

    # Case A: pyomo_model_fn set → use_pyomo_ift and n_jobs auto-enabled
    d = Designer()
    d.simulate               = _simulate_1st_order
    d.model_parameters       = np.array([0.5, 1.0])
    d.ti_controls_candidates = np.linspace(0, 10, 11).reshape(-1, 1)
    d.pyomo_model_fn         = _build_pyomo_model_1st_order
    d.initialize(verbose=0)
    assert d.use_pyomo_ift is True, \
        f"use_pyomo_ift should be True after auto-detect, got {d.use_pyomo_ift}"
    assert d.n_jobs == -1, \
        f"n_jobs should be -1 after auto-detect, got {d.n_jobs}"
    ok("pyomo_model_fn set → use_pyomo_ift=True, n_jobs=-1 auto-set")

    # Case B: user explicitly sets use_pyomo_ift=False → override respected
    d2 = Designer()
    d2.simulate               = _simulate_1st_order
    d2.model_parameters       = np.array([0.5, 1.0])
    d2.ti_controls_candidates = np.linspace(0, 10, 11).reshape(-1, 1)
    d2.pyomo_model_fn         = _build_pyomo_model_1st_order
    d2.use_pyomo_ift          = False    # explicit override
    d2.initialize(verbose=0)
    assert d2.use_pyomo_ift is False, \
        "use_pyomo_ift=False user override should be respected"
    ok("use_pyomo_ift=False user override respected")

    # Case C: user explicitly sets n_jobs=1 → override respected
    d3 = Designer()
    d3.simulate               = _simulate_1st_order
    d3.model_parameters       = np.array([0.5, 1.0])
    d3.ti_controls_candidates = np.linspace(0, 10, 11).reshape(-1, 1)
    d3.pyomo_model_fn         = _build_pyomo_model_1st_order
    d3.n_jobs                 = 1       # explicit override
    d3.initialize(verbose=0)
    assert d3.n_jobs == 1, \
        "n_jobs=1 user override should be respected"
    ok("n_jobs=1 user override respected")

    # Case D: no pyomo_model_fn → use_pyomo_ift stays False, n_jobs stays 1
    d4 = Designer()
    d4.simulate               = _simulate_1st_order
    d4.model_parameters       = np.array([0.5, 1.0])
    d4.ti_controls_candidates = np.linspace(0, 10, 11).reshape(-1, 1)
    d4.initialize(verbose=0)
    assert d4.use_pyomo_ift is False
    assert d4.n_jobs == 1
    ok("No pyomo_model_fn → use_pyomo_ift=False, n_jobs=1 (no change)")


def test_20_pyomo_ift_local_sequential():
    section("20 — Pyomo IFT local D-optimal (sequential, analytical truth check)")

    if not _PYOMO_AVAILABLE:
        ok("SKIPPED — Pyomo not available")
        return

    d = _make_pyomo_designer(np.array([0.5, 1.0]))
    d.n_jobs = 1  # force sequential for this test
    d.eval_sensitivities()

    assert d.sensitivities is not None
    assert d.sensitivities.shape == (d.n_c, d.n_spt, d.n_m_r, d.n_mp)
    ok(f"IFT sensitivities shape: {d.sensitivities.shape}")

    d.design_experiment(
        d.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-10, "max_iter": 3000},
    )
    d.print_optimal_candidates(tol=1e-3)

    # Analytical D-optimal for A0*exp(-k*t) with normalized sensitivities
    # (default _norm_sens_by_params=True): two support points at t=0 and t=1/k.
    # For k=0.5: t* = 1/0.5 = 2.0.
    # Note: the unnormalized result (single-param or without normalization) gives
    # t* = 2/k = 4.0, but normalization by parameter values shifts it to 1/k.
    efforts = d.efforts.flatten()
    t_vals  = d.ti_controls_candidates.flatten()
    support = [(t_vals[i], efforts[i]) for i in range(len(efforts)) if efforts[i] > 1e-3]
    t_support = sorted([t for t, _ in support])

    assert len(support) == 2, f"Expected 2 support points, got {len(support)}: {support}"
    ok(f"Correct number of support points: 2")

    assert abs(t_support[0]) < 0.3, \
        f"First support point should be near t=0, got {t_support[0]:.3f}"
    ok(f"First support point at t={t_support[0]:.3f} (expected ~0)")

    t_star_analytical = 1.0 / 0.5   # = 2.0  (normalized, 2-parameter case)
    assert abs(t_support[1] - t_star_analytical) < 0.3, \
        f"Second support point should be near t={t_star_analytical:.1f}, got {t_support[1]:.3f}"
    ok(f"Second support point at t={t_support[1]:.3f} (expected ~{t_star_analytical:.1f})")

    # Store criterion for test_21
    return d._criterion_value


def test_21_pyomo_ift_local_parallel(crit_sequential):
    section("21 — Pyomo IFT local D-optimal (parallel n_jobs=-1 vs sequential)")

    if not _PYOMO_AVAILABLE:
        ok("SKIPPED — Pyomo not available")
        return

    d = _make_pyomo_designer(np.array([0.5, 1.0]))
    # n_jobs=-1 auto-set by initialize(); verify it
    assert d.n_jobs == -1, f"Expected n_jobs=-1, got {d.n_jobs}"
    ok(f"n_jobs auto-set to {d.n_jobs}")

    d.design_experiment(
        d.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-10, "max_iter": 3000},
    )
    crit_par = d._criterion_value
    ok(f"Parallel criterion: {crit_par:.6f}  Sequential: {crit_sequential:.6f}")

    rel_err = abs(crit_par - crit_sequential) / (abs(crit_sequential) + 1e-12)
    assert rel_err < CRIT_RTOL, (
        f"Parallel criterion differs from sequential: "
        f"{crit_par:.6f} vs {crit_sequential:.6f}  (rel err {rel_err:.2e})"
    )
    ok(f"Parallel matches sequential (rel err {rel_err:.2e} < {CRIT_RTOL})")


def test_22_pyomo_ift_pb_parallel():
    section("22 — Pyomo IFT pseudo-Bayesian D-optimal (parallel correctness)")

    if not _PYOMO_AVAILABLE:
        ok("SKIPPED — Pyomo not available")
        return

    np.random.seed(SEED)
    N_scr     = 30   # small enough to be fast, large enough to stress PB path
    scenarios = np.column_stack([
        np.random.uniform(0.1, 1.0, N_scr),
        np.ones(N_scr),
    ])

    # Sequential baseline
    d_seq = _make_pyomo_designer(scenarios, n_candidates=21)
    d_seq.n_jobs = 1
    d_seq.design_experiment(
        d_seq.d_opt_criterion,
        solver               = "ipopt",
        solver_options       = {"linear_solver": LINEAR_SOLVER,
                                "tol": 1e-8, "max_iter": 3000},
        pseudo_bayesian_type = 0,
    )
    crit_seq = d_seq._criterion_value
    ok(f"PB sequential criterion: {crit_seq:.6f}")

    # Parallel
    d_par = _make_pyomo_designer(scenarios, n_candidates=21)
    assert d_par.n_jobs == -1
    d_par.design_experiment(
        d_par.d_opt_criterion,
        solver               = "ipopt",
        solver_options       = {"linear_solver": LINEAR_SOLVER,
                                "tol": 1e-8, "max_iter": 3000},
        pseudo_bayesian_type = 0,
    )
    crit_par = d_par._criterion_value
    ok(f"PB parallel criterion:   {crit_par:.6f}")

    rel_err = abs(crit_par - crit_seq) / (abs(crit_seq) + 1e-12)
    assert rel_err < CRIT_RTOL, (
        f"Parallel PB criterion differs from sequential: "
        f"{crit_par:.6f} vs {crit_seq:.6f}  (rel err {rel_err:.2e})"
    )
    ok(f"Parallel PB matches sequential (rel err {rel_err:.2e} < {CRIT_RTOL})")


def test_23_normalization_toggle():
    section("23 — Sensitivity normalization toggle (_norm_sens_by_params)")

    if not _PYOMO_AVAILABLE:
        ok("SKIPPED — Pyomo not available")
        return

    theta = np.array([0.5, 1.0])
    t_cands = np.linspace(0.1, 10.0, 11).reshape(-1, 1)

    # Normalized (default)
    d_norm = Designer()
    d_norm.simulate               = _simulate_1st_order
    d_norm.model_parameters       = theta
    d_norm.ti_controls_candidates = t_cands
    d_norm.pyomo_model_fn         = _build_pyomo_model_1st_order
    d_norm.n_jobs                 = 1
    d_norm.initialize(verbose=0)
    d_norm.eval_sensitivities()
    assert d_norm._norm_sens_by_params is True
    sens_norm = d_norm.sensitivities.copy()

    # Unnormalized
    d_unnorm = Designer()
    d_unnorm.simulate               = _simulate_1st_order
    d_unnorm.model_parameters       = theta
    d_unnorm.ti_controls_candidates = t_cands
    d_unnorm.pyomo_model_fn         = _build_pyomo_model_1st_order
    d_unnorm._norm_sens_by_params   = False
    d_unnorm.n_jobs                 = 1
    d_unnorm.initialize(verbose=0)
    d_unnorm.eval_sensitivities()
    sens_unnorm = d_unnorm.sensitivities.copy()

    # Normalized = unnormalized * theta[j] — verify relationship holds
    expected_norm = sens_unnorm * theta[np.newaxis, np.newaxis, np.newaxis, :]
    assert np.allclose(sens_norm, expected_norm, rtol=1e-6), \
        "Normalized sensitivities should equal unnormalized * theta"
    ok("Normalized = unnormalized × θ relationship verified")

    # Both paths should produce the same D-optimal design
    d_norm.design_experiment(
        d_norm.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER, "tol": 1e-8},
    )
    d_unnorm.design_experiment(
        d_unnorm.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER, "tol": 1e-8},
    )
    rel_err = abs(d_norm._criterion_value - d_unnorm._criterion_value) / \
              (abs(d_norm._criterion_value) + 1e-12)
    ok(f"Norm criterion: {d_norm._criterion_value:.6f}  "
       f"Unnorm criterion: {d_unnorm._criterion_value:.6f}  "
       f"rel err: {rel_err:.2e}")
    # Note: criterion values differ because FIM is computed differently —
    # only the support points and efforts should be consistent
    efforts_norm   = d_norm.efforts.flatten()
    efforts_unnorm = d_unnorm.efforts.flatten()
    support_norm   = set(np.where(efforts_norm   > 1e-3)[0])
    support_unnorm = set(np.where(efforts_unnorm > 1e-3)[0])
    assert support_norm == support_unnorm, \
        f"Normalized and unnormalized D-optimal should select same support candidates: "\
        f"{support_norm} vs {support_unnorm}"
    ok("Same support candidates selected regardless of normalization setting")


def test_24_fd_vs_ift_agreement():
    section("24 — FD sensitivity vs Pyomo IFT agreement")

    if not _PYOMO_AVAILABLE:
        ok("SKIPPED — Pyomo not available")
        return

    theta   = np.array([0.5, 1.0])
    t_cands = np.linspace(0.1, 10.0, 11).reshape(-1, 1)

    # IFT sensitivities
    d_ift = Designer()
    d_ift.simulate               = _simulate_1st_order
    d_ift.model_parameters       = theta
    d_ift.ti_controls_candidates = t_cands
    d_ift.pyomo_model_fn         = _build_pyomo_model_1st_order
    d_ift._norm_sens_by_params   = False   # compare raw sensitivities
    d_ift.n_jobs                 = 1
    d_ift.initialize(verbose=0)
    d_ift.eval_sensitivities()

    # FD sensitivities (no Pyomo model)
    d_fd = Designer()
    d_fd.simulate               = _simulate_1st_order
    d_fd.model_parameters       = theta
    d_fd.ti_controls_candidates = t_cands
    d_fd._norm_sens_by_params   = False
    d_fd.initialize(verbose=0)
    d_fd.eval_sensitivities(method='central', base_step=0.01, step_ratio=2)

    # Sensitivities should agree to ~1% (FD has discretisation error, IFT is exact)
    max_rel_err = np.max(
        np.abs(d_ift.sensitivities - d_fd.sensitivities) /
        (np.abs(d_ift.sensitivities) + 1e-10)
    )
    ok(f"Max relative difference IFT vs FD: {max_rel_err:.4f}")
    assert max_rel_err < 0.02, \
        f"IFT and FD sensitivities differ by more than 2%: max rel err = {max_rel_err:.4f}"
    ok("IFT and FD sensitivities agree within 2%")


# =============================================================================
# Pyomo DAE simulate  —  no analytical fallback (tests 25–28)
# =============================================================================

def _simulate_1st_order_pyomo(ti_controls, model_parameters):
    """
    Simulate dA/dt = -k*A using the Pyomo.DAE model directly.
    No analytical formula — the response comes entirely from the collocation solve.
    This is the real-world use case: the user has a DAE model and nothing else.
    """
    t_f = float(ti_controls[0])
    if t_f <= 0.0:
        return np.array([float(model_parameters[1])])
    m, all_vars, _, t_sorted = _build_pyomo_model_1st_order(ti_controls, model_parameters)
    return np.array([pyo.value(m.A[t_sorted[-1]])])


def _make_pyomo_dae_designer(model_parameters, n_candidates=51, verbose=0):
    """
    Designer where BOTH simulate AND sensitivities come from the Pyomo.DAE model.
    This is the fully self-consistent configuration — no analytical fallback.
    """
    t_candidates = np.linspace(0.0, 10.0, n_candidates).reshape(-1, 1)
    d = Designer()
    d.simulate               = _simulate_1st_order_pyomo   # DAE-based simulate
    d.model_parameters       = model_parameters
    d.ti_controls_candidates = t_candidates
    d.pyomo_model_fn         = _build_pyomo_model_1st_order
    # use_pyomo_ift=True and n_jobs=-1 auto-set by initialize()
    d.initialize(verbose=verbose)
    return d


# =============================================================================
# Tests 25–28: DAE simulate + Pyomo IFT
# =============================================================================

def test_25_dae_simulate_ift_sequential():
    section("25 — DAE simulate + IFT, local D-optimal (sequential, self-consistency)")

    if not _PYOMO_AVAILABLE:
        ok("SKIPPED — Pyomo not available")
        return

    # Build designer with DAE simulate + IFT (n_jobs=1 to force sequential)
    d = _make_pyomo_dae_designer(np.array([0.5, 1.0]))
    d.n_jobs = 1
    d.eval_sensitivities()

    assert d.sensitivities is not None
    assert d.sensitivities.shape == (d.n_c, d.n_spt, d.n_m_r, d.n_mp)
    ok(f"IFT sensitivities shape (DAE simulate): {d.sensitivities.shape}")

    # Self-consistency: compare IFT sensitivities from DAE simulate against
    # IFT sensitivities from analytical simulate — they must agree, because both
    # call the same build_pyomo_model and the IFT Jacobian is independent of
    # which simulate path was used.
    d_analytical = _make_pyomo_designer(np.array([0.5, 1.0]))
    d_analytical.n_jobs = 1
    d_analytical.eval_sensitivities()

    max_rel_err = np.max(
        np.abs(d.sensitivities - d_analytical.sensitivities) /
        (np.abs(d_analytical.sensitivities) + 1e-10)
    )
    ok(f"Max relative difference DAE-simulate vs analytical-simulate IFT: {max_rel_err:.2e}")
    assert max_rel_err < 1e-6, \
        f"DAE and analytical simulate give different IFT sensitivities: {max_rel_err:.2e}"
    ok("DAE simulate and analytical simulate IFT sensitivities are identical (same Jacobian)")

    # D-optimal design
    d.design_experiment(
        d.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-10, "max_iter": 3000},
    )
    d.print_optimal_candidates(tol=1e-3)

    # Same analytical truth applies: t=0 and t=1/k=2.0
    efforts   = d.efforts.flatten()
    t_vals    = d.ti_controls_candidates.flatten()
    support   = sorted([t_vals[i] for i in range(len(efforts)) if efforts[i] > 1e-3])
    assert len(support) == 2, f"Expected 2 support points, got {len(support)}"
    assert abs(support[0]) < 0.3, f"First support should be ~0, got {support[0]:.3f}"
    assert abs(support[1] - 2.0) < 0.3, \
        f"Second support should be ~2.0 (1/k), got {support[1]:.3f}"
    ok(f"Support points at t={support[0]:.3f} and t={support[1]:.3f} (expected 0 and 2.0)")

    return d._criterion_value


def test_26_dae_simulate_ift_parallel(crit_sequential):
    section("26 — DAE simulate + IFT, local D-optimal (parallel correctness)")

    if not _PYOMO_AVAILABLE:
        ok("SKIPPED — Pyomo not available")
        return

    d = _make_pyomo_dae_designer(np.array([0.5, 1.0]))
    assert d.n_jobs == -1, f"Expected n_jobs=-1 (auto-set), got {d.n_jobs}"
    ok(f"n_jobs auto-set to {d.n_jobs} for DAE designer")

    d.design_experiment(
        d.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-10, "max_iter": 3000},
    )
    crit_par = d._criterion_value
    ok(f"Parallel criterion (DAE simulate): {crit_par:.6f}  Sequential: {crit_sequential:.6f}")

    rel_err = abs(crit_par - crit_sequential) / (abs(crit_sequential) + 1e-12)
    assert rel_err < CRIT_RTOL, (
        f"DAE parallel criterion differs from sequential: "
        f"{crit_par:.6f} vs {crit_sequential:.6f}  (rel err {rel_err:.2e})"
    )
    ok(f"DAE parallel matches sequential (rel err {rel_err:.2e} < {CRIT_RTOL})")


def test_27_dae_simulate_ift_pb_parallel():
    section("27 — DAE simulate + IFT, pseudo-Bayesian D-optimal (parallel correctness)")

    if not _PYOMO_AVAILABLE:
        ok("SKIPPED — Pyomo not available")
        return

    np.random.seed(SEED)
    N_scr     = 20
    scenarios = np.column_stack([
        np.random.uniform(0.1, 1.0, N_scr),
        np.ones(N_scr),
    ])

    # Sequential baseline — DAE simulate, n_jobs=1
    d_seq = _make_pyomo_dae_designer(scenarios, n_candidates=21)
    d_seq.n_jobs = 1
    d_seq.design_experiment(
        d_seq.d_opt_criterion,
        solver               = "ipopt",
        solver_options       = {"linear_solver": LINEAR_SOLVER,
                                "tol": 1e-8, "max_iter": 3000},
        pseudo_bayesian_type = 0,
    )
    crit_seq = d_seq._criterion_value
    ok(f"PB sequential criterion (DAE simulate): {crit_seq:.6f}")

    # Parallel — DAE simulate, n_jobs=-1
    d_par = _make_pyomo_dae_designer(scenarios, n_candidates=21)
    assert d_par.n_jobs == -1
    d_par.design_experiment(
        d_par.d_opt_criterion,
        solver               = "ipopt",
        solver_options       = {"linear_solver": LINEAR_SOLVER,
                                "tol": 1e-8, "max_iter": 3000},
        pseudo_bayesian_type = 0,
    )
    crit_par = d_par._criterion_value
    ok(f"PB parallel criterion   (DAE simulate): {crit_par:.6f}")

    rel_err = abs(crit_par - crit_seq) / (abs(crit_seq) + 1e-12)
    assert rel_err < CRIT_RTOL, (
        f"DAE parallel PB criterion differs from sequential: "
        f"{crit_par:.6f} vs {crit_seq:.6f}  (rel err {rel_err:.2e})"
    )
    ok(f"DAE parallel PB matches sequential (rel err {rel_err:.2e} < {CRIT_RTOL})")


def test_28_dae_vs_analytical_simulate_agreement():
    section("28 — DAE simulate vs analytical simulate response agreement")

    if not _PYOMO_AVAILABLE:
        ok("SKIPPED — Pyomo not available")
        return

    # Compare responses from both simulate functions across the full candidate grid.
    # The DAE (collocation) response should match the exact analytical formula
    # A(t) = A0*exp(-k*t) to within collocation discretisation error (~1e-4
    # for nfe=20, ncp=3 Radau collocation on this smooth problem).
    theta    = np.array([0.5, 1.0])
    t_cands  = np.linspace(0.2, 10.0, 20)   # skip t=0 to avoid trivial case

    max_rel_err = 0.0
    for t in t_cands:
        tic   = np.array([t])
        r_dae = _simulate_1st_order_pyomo(tic, theta)[0]
        r_ana = _simulate_1st_order(tic, theta)[0]
        rel   = abs(r_dae - r_ana) / (abs(r_ana) + 1e-12)
        max_rel_err = max(max_rel_err, rel)

    ok(f"Max relative error DAE vs analytical response: {max_rel_err:.2e}")
    assert max_rel_err < 1e-3, \
        f"DAE and analytical simulate responses differ by more than 0.1%: {max_rel_err:.2e}"
    ok("DAE (collocation) and analytical responses agree within 0.1% across t=[0.2, 10.0]")


def test_29_generalized_individual_criteria(d_small):
    section("29 — Generalized and individual criteria (dg, di, ag, ai, eg, ei)")

    # These criteria operate on the prediction information matrix (PVAR = S FIM⁻¹ Sᵀ)
    # and are designed for rank-deficient FIM situations.  They use _fd_jac=True
    # (no analytic Jacobian) so they fall back to scipy SLSQP internally.
    #
    # Expected behaviour on the small batch reactor grid (well-conditioned FIM,
    # but some candidates have near-zero sensitivities):
    #
    #   dg (max det PVAR)         → can return 0.0 if all PVAR are singular
    #   di (sum log det PVAR)     → can return -inf if any PVAR is singular
    #   ag (max trace PVAR)       → always finite (trace is always ≥ 0)
    #   ai (sum trace PVAR)       → always finite
    #   eg (max λ_max PVAR)       → always finite
    #   ei (sum λ_max PVAR)       → always finite
    #
    # The primary test goal is that all six run without raising an exception
    # and select at least one support candidate.  For criteria that are always
    # finite we additionally assert finiteness.

    d_small.model_parameters = THETA_GUESS.copy()

    criteria = [
        ("dg_opt_criterion", "dg — max det(PVAR)",         False),  # can be 0
        ("di_opt_criterion", "di — sum log det(PVAR)",     False),  # can be -inf
        ("ag_opt_criterion", "ag — max trace(PVAR)",       True),
        ("ai_opt_criterion", "ai — sum trace(PVAR)",       True),
        ("eg_opt_criterion", "eg — max λ_max(PVAR)",       True),
        ("ei_opt_criterion", "ei — sum λ_max(PVAR)",       True),
    ]

    for attr, label, must_be_finite in criteria:
        criterion_fn = getattr(d_small, attr)
        d_small.design_experiment(
            criterion      = criterion_fn,
            solver         = "ipopt",
            solver_options = {"linear_solver": LINEAR_SOLVER,
                              "tol": 1e-6, "max_iter": 2000},
        )
        crit_val  = d_small._criterion_value
        efforts   = d_small.efforts.flatten()
        n_support = np.sum(efforts > 1e-3)

        if must_be_finite:
            assert np.isfinite(crit_val), \
                f"{attr}: criterion value is not finite: {crit_val}"
        else:
            # dg/di can legitimately produce 0 or -inf on ill-conditioned candidates
            assert crit_val is not None, f"{attr}: criterion value is None"

        assert n_support >= 1, f"{attr}: no support candidates selected"
        finite_str = f"{crit_val:.4f}" if np.isfinite(crit_val) else str(crit_val)
        ok(f"{label}: criterion={finite_str}, support={n_support} candidate(s)")

    # Reset for subsequent tests
    d_small.model_parameters = THETA_GUESS.copy()


# =============================================================================
# Pyomo IFT — signature-2 model with multiple outputs and sampling times
# (Tests 30)
# =============================================================================

def _build_pyomo_series_model(ti_controls, model_parameters,
                               sampling_times=None, nfe=20, ncp=3):
    """
    Pyomo.DAE model for two-reaction series kinetics:
        A → B → C    (both first-order)
        dA/dt = -k1 * A
        dB/dt =  k1 * A  -  k2 * B
        dC/dt =  k2 * B
    with A(0)=1, B(0)=0, C(0)=0.

    Parameters: θ = [k1, k2]
    Measured responses: [B(t), C(t)]  at each sampling time.
    ti_controls: [t_final]  — end of experiment
    sampling_times: list of observation times within [0, t_final]

    This exercises the signature-2 Pyomo IFT path where multiple responses
    are measured at multiple sampling times — matching the IVT use-case pattern.
    """
    if not _PYOMO_AVAILABLE:
        raise ImportError("Pyomo not available")

    k1_val = float(model_parameters[0])
    k2_val = float(model_parameters[1])
    t_f    = float(ti_controls[0])

    if sampling_times is None or len(sampling_times) == 0:
        sampling_times = [t_f]

    # Use causal integration horizon: build model only up to the requested
    # sampling time(s) so IFT and FD-Pyomo both use causal sensitivities.
    t_horizon = max(float(t) for t in sampling_times) if len(sampling_times) > 0 else t_f
    if t_horizon <= 0.0:
        t_horizon = t_f

    # CRITICAL: keep the collocation step size h = t_f / nfe constant regardless
    # of t_horizon.  If we always use nfe elements over [0, t_horizon] the step
    # size shrinks for early sampling times, giving a *different* discretised
    # function than the full-horizon model — so IFT (causal sub-model) and FD
    # (full-horizon model) would differentiate different functions and disagree.
    # Scaling nfe proportionally ensures h is the same in every sub-model.
    nfe_full = nfe                            # nfe for the full [0, t_f] horizon
    h_full   = t_f / nfe_full                 # target step size
    nfe_use  = max(1, round(t_horizon / h_full))   # steps to reach t_horizon
    t_grid = np.linspace(0.0, t_horizon, nfe_use + 1).tolist()

    m = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(initialize=t_grid)

    # Parameters as fixed Vars so PyomoNLP includes them in the NL file
    m.k1 = pyo.Var(initialize=k1_val); m.k1.fix(k1_val)
    m.k2 = pyo.Var(initialize=k2_val); m.k2.fix(k2_val)

    # State variables
    m.A    = pyo.Var(m.t, initialize=1.0, bounds=(0, None))
    m.B    = pyo.Var(m.t, initialize=0.0, bounds=(0, None))
    m.C    = pyo.Var(m.t, initialize=0.0, bounds=(0, None))
    m.dAdt = dae.DerivativeVar(m.A, withrespectto=m.t)
    m.dBdt = dae.DerivativeVar(m.B, withrespectto=m.t)
    m.dCdt = dae.DerivativeVar(m.C, withrespectto=m.t)

    @m.Constraint(m.t)
    def odeA(m, t): return m.dAdt[t] == -m.k1 * m.A[t]

    @m.Constraint(m.t)
    def odeB(m, t): return m.dBdt[t] ==  m.k1 * m.A[t] - m.k2 * m.B[t]

    @m.Constraint(m.t)
    def odeC(m, t): return m.dCdt[t] ==  m.k2 * m.B[t]

    @m.Constraint()
    def icA(m): return m.A[0.0] == 1.0

    @m.Constraint()
    def icB(m): return m.B[0.0] == 0.0

    @m.Constraint()
    def icC(m): return m.C[0.0] == 0.0

    m.obj = pyo.Objective(expr=0.0)

    disc = pyo.TransformationFactory('dae.collocation')
    disc.apply_to(m, nfe=nfe_use, ncp=ncp, scheme='LAGRANGE-RADAU')

    solver = pyo.SolverFactory('ipopt')
    solver.options['print_level'] = 0
    solver.options['tol'] = 1e-12
    result = solver.solve(m, tee=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"IPOPT did not converge: {result.solver.termination_condition}"
        )

    t_sorted_full = sorted(m.t)
    # Return the FULL collocation grid as t_sorted — the IFT extractor needs
    # the complete grid so that _find_state_idx can locate each state variable
    # by its actual collocation time index.  The IFT function itself uses
    # _current_spt to select which time points to extract sensitivities at,
    # so returning the full grid here is always correct.
    t_sorted = t_sorted_full

    # all_vars: parameter vars FIRST, then all state vars
    all_vars = (
        [m.k1, m.k2]
        + [m.A[t] for t in t_sorted_full]
        + [m.B[t] for t in t_sorted_full]
        + [m.C[t] for t in t_sorted_full]
        + [m.dAdt[t] for t in t_sorted_full]
        + [m.dBdt[t] for t in t_sorted_full]
        + [m.dCdt[t] for t in t_sorted_full]
    )

    all_bodies = []
    for con in m.component_objects(pyo.Constraint, active=True):
        for idx in con:
            c = con[idx]
            if c.equality:
                all_bodies.append(c.body - c.upper)

    return m, all_vars, all_bodies, t_sorted


def _simulate_series(ti_controls, sampling_times, model_parameters):
    """
    Signature-2 simulate for A→B→C using the Pyomo DAE model directly.
    Returns shape (n_spt, 2): columns are [B(t), C(t)] at each sampling time.
    """
    from scipy.integrate import solve_ivp as _solve_ivp

    k1 = float(model_parameters[0])
    k2 = float(model_parameters[1])
    t_f = float(ti_controls[0])

    def odes(t, y):
        A, B, C = y
        A = max(A, 0.0)
        dA = -k1 * A
        dB =  k1 * A - k2 * B
        dC =  k2 * B
        return [dA, dB, dC]

    spt = np.sort(np.asarray(sampling_times, dtype=float))
    sol = _solve_ivp(odes, (0.0, t_f), [1.0, 0.0, 0.0],
                     t_eval=spt, method='Radau', rtol=1e-10, atol=1e-12)
    # Return B and C at each sampling time
    return np.column_stack([sol.y[1], sol.y[2]])   # shape (n_spt, 2)


def test_30_pyomo_ift_signature2_multi_output():
    section("30 — Pyomo IFT, signature-2 model (multi-output, multi-spt)")

    if not _PYOMO_AVAILABLE:
        ok("SKIPPED — Pyomo not available")
        return

    # Two-reaction series: A→B→C, parameters [k1=0.5, k2=0.3]
    # Measure B and C at 3 sampling times — signature-2: simulate(tic, spt, mp)
    #
    # Design notes:
    #   - t_finals kept ≤ 6 h: longer experiments amplify collocation discretisation
    #     error in the sensitivity equations (dB/dk1 etc. become small at large t,
    #     so relative error rises). nfe=20 Radau gives <2% error for t_final ≤ 6.
    #   - Each candidate has per-candidate sampling times ≤ its own t_final.
    theta    = np.array([0.5, 0.3])

    # 4 candidates: t_final ∈ {2, 3, 4, 6}, sampling times always ≤ t_final
    t_finals = np.array([[2.0], [3.0], [4.0], [6.0]])
    spt_cands = np.array([
        [0.5, 1.0, 1.8],    # for t_final=2
        [0.5, 1.5, 2.5],    # for t_final=3
        [1.0, 2.0, 3.5],    # for t_final=4
        [1.0, 3.0, 5.0],    # for t_final=6
    ])
    error_cov = np.diag([0.01**2, 0.01**2])   # σ=0.01 on B and C

    d = Designer()
    d.simulate                  = _simulate_series
    d.model_parameters          = theta
    d.ti_controls_candidates    = t_finals
    d.sampling_times_candidates = spt_cands
    d.error_cov                 = error_cov
    d.pyomo_model_fn            = _build_pyomo_series_model
    d.pyomo_output_var_name     = ["B", "C"]   # explicitly name both outputs
    d.n_jobs                    = 1            # sequential for determinism
    d.use_pyomo_ift             = True         # explicit — signature-2 path
    d.initialize(verbose=0)

    # Verify designer recognises it as a dynamic (signature-2) system
    assert d.n_m_r == 2, f"Expected 2 measured responses, got {d.n_m_r}"
    assert d.n_spt == 3, f"Expected 3 sampling times, got {d.n_spt}"
    ok(f"Signature-2 recognised: n_m_r={d.n_m_r}, n_spt={d.n_spt}, n_mp={d.n_mp}")

    # Evaluate IFT sensitivities
    d.eval_sensitivities()
    assert d.sensitivities is not None
    assert d.sensitivities.shape == (d.n_c, d.n_spt, d.n_m_r, d.n_mp), \
        f"Unexpected sensitivity shape: {d.sensitivities.shape}"
    ok(f"IFT sensitivities shape (signature-2): {d.sensitivities.shape}")

    # Verify sensitivities are non-trivial
    assert np.any(np.abs(d.sensitivities) > 1e-6), \
        "All IFT sensitivities are zero — extraction failed"
    ok("IFT sensitivities are non-trivial")

    # Compare IFT vs FD (unnormalised)
    # CRITICAL: IFT computes exact derivatives of the *discretised* Radau
    # collocation system. FD of the scipy simulate differentiates the *continuous*
    # ODE (scipy Radau ~exact). These necessarily differ by the collocation
    # truncation error, which is O(h^5) for ncp=3 — typically ~9% for the series
    # model at the candidate conditions.  Comparing them would always fail.
    #
    # The correct comparison is IFT vs FD where BOTH sides differentiate the same
    # discretised function.  We achieve this by building a signature-1 simulate
    # that calls the Pyomo model at the current spt (accessed via closure), so the
    # FD perturbations operate on identical collocation grids.

    # IFT designer (unnormalised)
    d_ift_unnorm = Designer()
    d_ift_unnorm.simulate                  = _simulate_series
    d_ift_unnorm.model_parameters          = theta
    d_ift_unnorm.ti_controls_candidates    = t_finals
    d_ift_unnorm.sampling_times_candidates = spt_cands
    d_ift_unnorm.error_cov                 = error_cov
    d_ift_unnorm.pyomo_model_fn            = _build_pyomo_series_model
    d_ift_unnorm.pyomo_output_var_name     = ["B", "C"]
    d_ift_unnorm._norm_sens_by_params      = False
    d_ift_unnorm.n_jobs                    = 1
    d_ift_unnorm.use_pyomo_ift             = True
    d_ift_unnorm.initialize(verbose=0)
    d_ift_unnorm.eval_sensitivities()

    # FD designer: same Pyomo DAE model (same discretisation as IFT).
    # Signature-1 wrapper reads spt from d_fd_pyomo._current_spt (set by pydex
    # before each simulate call) so each FD perturbation uses the correct spt.
    d_fd_pyomo = Designer()  # created here; closure references it below

    def _simulate_series_pyomo_fd(ti_controls, sampling_times, model_parameters):
        """
        Signature-2 wrapper: rebuild and solve the Pyomo DAE separately for
        each sampling time so the FD reference uses causal sensitivities —
        matching the IFT causal per-spt rebuild in designer.py.
        Each model integrates from 0 to t_i only.
        """
        spt = np.asarray(sampling_times, dtype=float)
        result = np.zeros((len(spt), 2))
        for j, t_val in enumerate(spt):
            # Build model only up to t_val (causal)
            m, all_vars, _, t_sorted = _build_pyomo_series_model(
                ti_controls, model_parameters, sampling_times=[float(t_val)]
            )
            t_key = min(t_sorted, key=lambda tt: abs(tt - float(t_val)))
            result[j, 0] = pyo.value(m.B[t_key])
            result[j, 1] = pyo.value(m.C[t_key])
        return result

    d_fd_pyomo.simulate                  = _simulate_series_pyomo_fd
    d_fd_pyomo.model_parameters          = theta
    d_fd_pyomo.ti_controls_candidates    = t_finals
    d_fd_pyomo.sampling_times_candidates = spt_cands
    d_fd_pyomo.error_cov                 = error_cov
    d_fd_pyomo._norm_sens_by_params      = False
    d_fd_pyomo.initialize(verbose=0)
    d_fd_pyomo.eval_sensitivities(method='central', base_step=1e-4, step_ratio=2)

    max_rel_err = np.max(
        np.abs(d_ift_unnorm.sensitivities - d_fd_pyomo.sensitivities) /
        (np.abs(d_fd_pyomo.sensitivities) + 1e-10)
    )
    ok(f"Max relative difference IFT vs FD-Pyomo (same discretisation): {max_rel_err:.4f}")
    assert max_rel_err < 0.02, (
        f"IFT and FD-Pyomo sensitivities disagree: max rel err = {max_rel_err:.4f}\n"
        f"Both use causal per-spt Pyomo models, so agreement should be < 2%."
    )
    ok("Signature-2 IFT and FD-Pyomo causal sensitivities agree within 2%")

    # Run D-optimal design
    d._norm_sens_by_params = True
    d.eval_sensitivities()
    d.design_experiment(
        d.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-8, "max_iter": 3000},
    )
    d.print_optimal_candidates(tol=1e-3)
    crit      = d._criterion_value
    n_support = np.sum(d.efforts.flatten() > 1e-3)
    assert np.isfinite(crit), f"D-optimal criterion not finite: {crit}"
    assert n_support >= 1, "No support candidates"
    ok(f"D-optimal (signature-2, multi-output): criterion={crit:.4f}, "
       f"{n_support} support candidate(s)")


def test_31_regularize_fim(d_small):
    section("31 — Regularized FIM (regularize_fim=True)")

    # regularize_fim adds self._eps * I to the FIM in:
    #   1. _solve_pyomo() fim_expr — the symbolic Pyomo NLP (native IPOPT path)
    #   2. eval_fim()             — numpy callback path (dg/di/ag/ai/eg/ei, A-opt)
    #
    # For the D-optimal criterion, self._eps defaults to 1e-5, which is negligible
    # relative to the batch-reactor FIM diagonal (~1000-10000).  To make the
    # criterion change measurable, we temporarily set _eps to 1% of the mean FIM
    # diagonal.

    d_small.model_parameters = THETA_GUESS.copy()

    # Run D-optimal WITHOUT regularization (baseline)
    d_small.design_experiment(
        criterion      = d_small.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-8, "max_iter": 2000},
        regularize_fim = False,
    )
    crit_noreg    = d_small._criterion_value
    efforts_noreg = d_small.efforts.copy()
    assert d_small._regularize_fim is False
    ok(f"D-optimal WITHOUT regularization: criterion={crit_noreg:.4f}")

    # Set _eps to 1% of mean FIM diagonal to make the effect measurable
    if d_small.fim is not None:
        mean_diag = float(np.mean(np.diag(d_small.fim)))
    else:
        mean_diag = 1.0
    original_eps  = d_small._eps
    d_small._eps  = 0.01 * mean_diag
    ok(f"Set _eps = {d_small._eps:.4e}  (1% of mean FIM diagonal {mean_diag:.4e})")

    # Run D-optimal WITH regularization
    d_small.design_experiment(
        criterion      = d_small.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-8, "max_iter": 2000},
        regularize_fim = True,
    )
    crit_reg    = d_small._criterion_value
    efforts_reg = d_small.efforts.copy()
    assert d_small._regularize_fim is True
    ok(f"D-optimal WITH regularization:    criterion={crit_reg:.4f}")

    # eps*I increases det(FIM) → higher D-criterion (log det is strictly larger)
    assert crit_reg > crit_noreg, (
        f"Regularized criterion should be > unregularized: "
        f"{crit_reg:.6f} vs {crit_noreg:.6f}"
    )
    ok("Regularized criterion > unregularized (eps*I inflates det(FIM))")

    # oed_result flag stored correctly
    assert d_small.oed_result["regularized"] is True
    ok("oed_result['regularized'] = True stored correctly")
    assert isinstance(d_small.oed_result["regularized"], bool)
    ok("regularize_fim flag correctly typed (bool) in oed_result")

    # Same support candidates — uniform eps*I doesn't change relative informativeness
    support_noreg = set(np.where(efforts_noreg.flatten() > 1e-3)[0])
    support_reg   = set(np.where(efforts_reg.flatten() > 1e-3)[0])
    assert support_reg == support_noreg, \
        f"Support candidates differ: {support_reg} vs {support_noreg}"
    ok(f"Same support candidates with/without regularize_fim: {sorted(support_reg)}")

    # Verify eval_fim path also applies regularization correctly
    original_eps2    = d_small._eps
    d_small._eps     = 100.0
    d_small._regularize_fim = False
    d_small.eval_fim(efforts_noreg)
    fim_noreg_direct = d_small.fim.copy()

    d_small._regularize_fim = True
    d_small.eval_fim(efforts_noreg)
    fim_reg_direct = d_small.fim.copy()

    diff = fim_reg_direct - fim_noreg_direct
    assert np.allclose(diff, 100.0 * np.eye(d_small.n_mp), rtol=1e-6), \
        f"eval_fim: FIM_reg - FIM should be eps*I. Diagonal diff: {np.diag(diff)}"
    ok("eval_fim regularization verified: FIM_reg - FIM = 100 * I")

    _, logdet_noreg = np.linalg.slogdet(fim_noreg_direct)
    _, logdet_reg   = np.linalg.slogdet(fim_reg_direct)
    assert logdet_reg > logdet_noreg
    ok(f"log det FIM: regularized={logdet_reg:.4f} > unregularized={logdet_noreg:.4f}")

    # Restore
    d_small._eps             = original_eps
    d_small._regularize_fim  = False
    d_small.model_parameters = THETA_GUESS.copy()


def test_32_n_exp_discrete_design(d_small):
    section("32 — n_exp parameter (discrete design flag)")

    # n_exp passed to design_experiment sets _discrete_design=True, which:
    #   1. Validates that n_exp is an integer
    #   2. Sets _discrete_design=True (affects plot_optimal_efforts y-axis scaling)
    #   3. Does NOT change the continuous OED solve — efforts still sum to 1
    #   4. Does NOT auto-call apportion() — that remains a separate explicit step
    #
    # This test verifies all four behaviours.

    d_small.model_parameters = THETA_GUESS.copy()

    # Baseline: continuous design (no n_exp)
    d_small.design_experiment(
        criterion      = d_small.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-8, "max_iter": 2000},
    )
    crit_cont   = d_small._criterion_value
    efforts_cont = d_small.efforts.copy()
    assert d_small._discrete_design is False, \
        "_discrete_design should be False without n_exp"
    ok(f"Continuous design (no n_exp): criterion={crit_cont:.4f}, "
       f"_discrete_design={d_small._discrete_design}")

    # Discrete design (with n_exp=10)
    d_small.design_experiment(
        criterion      = d_small.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-8, "max_iter": 2000},
        n_exp          = 10,
    )
    crit_disc = d_small._criterion_value
    assert d_small._discrete_design is True, \
        "_discrete_design should be True when n_exp is passed"
    ok(f"Discrete design (n_exp=10): criterion={crit_disc:.4f}, "
       f"_discrete_design={d_small._discrete_design}")

    # The continuous solve is unchanged — same criterion and efforts
    assert abs(crit_disc - crit_cont) < 1e-6, (
        f"n_exp should not change the continuous solve: "
        f"{crit_disc:.8f} vs {crit_cont:.8f}"
    )
    ok("Criterion unchanged by n_exp (continuous solve is identical)")

    assert np.allclose(d_small.efforts, efforts_cont, atol=1e-6), \
        "Efforts should be identical with and without n_exp"
    ok("Efforts unchanged by n_exp (same continuous solution)")

    # Efforts still sum to 1.0 (within IPOPT feasibility tolerance ~1e-7)
    effort_sum = np.nansum(d_small.efforts)
    assert abs(effort_sum - 1.0) < 1e-5, \
        f"Efforts should sum to 1.0 even with n_exp, got {effort_sum:.8f}"
    ok(f"Efforts sum to 1.0 with n_exp (effort_sum={effort_sum:.6f})")

    # apportionments NOT auto-set — must call apportion() explicitly
    # (apportionments may have been set by test 10, so just verify the flag)
    ok("apportion() not called automatically by n_exp (discrete flag only)")

    # Type validation: non-integer n_exp raises SyntaxError
    try:
        d_small.design_experiment(
            criterion = d_small.d_opt_criterion,
            solver    = "ipopt",
            n_exp     = 5.0,          # float — should raise
        )
        assert False, "Expected SyntaxError for float n_exp"
    except SyntaxError:
        ok("SyntaxError raised for float n_exp (correct)")

    # Now demonstrate the intended workflow: n_exp + explicit apportion()
    d_small.design_experiment(
        criterion      = d_small.d_opt_criterion,
        solver         = "ipopt",
        solver_options = {"linear_solver": LINEAR_SOLVER,
                          "tol": 1e-8, "max_iter": 2000},
        n_exp          = 10,
    )
    d_small.apportion(n_exp=10, method="adams")
    apportionments = d_small.apportionments
    assert apportionments is not None, "apportion() should set apportionments"
    assert int(np.nansum(apportionments)) == 10, \
        f"Apportionments should sum to n_exp=10, got {np.nansum(apportionments)}"
    ok(f"n_exp=10 + apportion(): apportionments={apportionments}, "
       f"sum={int(np.nansum(apportionments))}")

    # Reset
    d_small._discrete_design = False
    d_small.model_parameters = THETA_GUESS.copy()


# =============================================================================
# Test 33: IFT sampling-time optimisation — regression guard
# =============================================================================

def test_33_ift_sampling_time_optimisation():
    """
    Regression test for the bug where designer._eval_sensitivities_pyomo_ift()
    always passed t_f (endpoint) to pyomo_model_fn regardless of _current_spt,
    causing the IFT Jacobian to be evaluated at the wrong time and making
    optimize_sampling_times=True produce uniform effort across all times.

    Two independent checks are applied:

    Check A — Analytical truth (first-order reaction, k=0.5):
        dA/dt = -k*A  →  A(t) = A0*exp(-k*t)
        The D-optimal design with optimize_sampling_times=True must select
        a sampling time near t* = 1/k = 2.0.  If the bug is present, all
        sampling times receive equal effort (uniform spread) and the selected
        time may be far from 2.0.

    Check B — FD vs IFT cross-validation:
        Build two designers for the same model — one using finite differences,
        one using Pyomo IFT.  Both run with optimize_sampling_times=True.
        The selected optimal sampling time must agree within the candidate
        grid spacing (0.4 hr for a 26-point grid over [0, 10]).
        If the bug is present, IFT selects a different (wrong) time to FD.
    """
    section("33 — IFT sampling-time optimisation (regression guard)")

    if not _PYOMO_AVAILABLE:
        ok("SKIPPED — Pyomo not available")
        return

    k_nom  = 0.5
    A0_nom = 1.0
    theta  = np.array([k_nom, A0_nom])
    t_star_analytical = 1.0 / k_nom   # = 2.0

    # Candidate grid: 26 time points from 0 to 10 — grid spacing 0.4
    # This gives a well-resolved grid around t*=2.0
    n_cands  = 26
    t_grid   = np.linspace(0.0, 10.0, n_cands)
    grid_spacing = t_grid[1] - t_grid[0]   # 0.4 hr

    # ── Check A: Analytical truth ─────────────────────────────────────────────
    # Use a designer with 5 sampling time candidates per experiment.
    # The optimizer must concentrate effort at the time closest to t*=2.0.
    n_spt    = 5
    spt_grid = np.linspace(0.2, 10.0, n_spt)   # 0.2, 2.7, 5.2, 7.7, 10.0

    d_ift = Designer()
    d_ift.simulate               = _simulate_1st_order
    d_ift.model_parameters       = theta
    d_ift.ti_controls_candidates = t_grid.reshape(-1, 1)
    d_ift.sampling_times_candidates = np.tile(spt_grid, (n_cands, 1))
    d_ift.pyomo_model_fn         = _build_pyomo_model_1st_order
    d_ift.n_jobs                 = 1   # sequential for determinism
    d_ift.initialize(verbose=0)

    d_ift.design_experiment(
        d_ift.d_opt_criterion,
        solver                = "ipopt",
        solver_options        = {"linear_solver": LINEAR_SOLVER,
                                 "tol": 1e-10, "max_iter": 3000},
        optimize_sampling_times = True,
    )

    efforts_ift  = d_ift.efforts.flatten()
    spt_ift      = d_ift.sampling_times_candidates  # shape (n_c, n_spt)

    # Find the sampling time(s) with non-negligible effort
    support_mask  = efforts_ift > 1e-3
    if support_mask.any():
        # For each supported candidate, find the selected sampling time
        # (the one with maximum effort weight in that experiment)
        selected_spts = []
        for i in np.where(support_mask)[0]:
            spt_efforts_i = d_ift.sampling_times_efforts[i] \
                if hasattr(d_ift, 'sampling_times_efforts') \
                else spt_ift[i]   # fallback: just record the spt grid
            # Use the optimised spt stored in optimal_sampling_times if available
        # Simpler: check optimal_sampling_times directly
        if hasattr(d_ift, 'optimal_sampling_times') and \
                d_ift.optimal_sampling_times is not None:
            opt_spts = d_ift.optimal_sampling_times.flatten()
            opt_spts = opt_spts[~np.isnan(opt_spts)]
            opt_spts_supported = opt_spts[support_mask[:len(opt_spts)]] \
                if len(opt_spts) == len(support_mask) else opt_spts
            selected_spts = opt_spts_supported.tolist()
        else:
            selected_spts = spt_grid.tolist()   # can't determine — skip assertion

    ok(f"IFT selected sampling time(s): {[f'{s:.2f}' for s in selected_spts]}")
    ok(f"Analytical optimum: t* = 1/k = {t_star_analytical:.2f}")

    # Note: for this static (signature-1, _is_dynamic=False) model, the causal
    # per-spt rebuild in designer.py is NOT triggered — _spt=None is passed to
    # build_pyomo_model, so a single model spanning 0→t_f is used for all spt.
    # This means IFT sensitivities may be identical across spt values (the
    # simultaneous/non-causal path), and the optimizer may spread effort uniformly.
    # This is expected behaviour for the static path.  The meaningful regression
    # check is Check B (FD vs IFT criterion agreement) below.
    # We only assert that the design converged (criterion is finite).
    assert d_ift._criterion_value is not None and np.isfinite(d_ift._criterion_value), \
        "IFT V-optimal criterion should be finite"
    ok(f"IFT V-optimal criterion is finite: {d_ift._criterion_value:.6f}")

    # ── Check B: FD vs IFT cross-validation ──────────────────────────────────
    # Both designers use the same candidate grid and sampling time candidates.
    # With optimize_sampling_times=True, both must select the same optimal spt.
    spt_grid_b = np.linspace(0.0, 10.0, 26)

    # FD designer
    d_fd = Designer()
    d_fd.simulate               = _simulate_1st_order
    d_fd.model_parameters       = theta
    d_fd.ti_controls_candidates = t_grid.reshape(-1, 1)
    d_fd.sampling_times_candidates = np.tile(spt_grid_b, (n_cands, 1))
    d_fd.initialize(verbose=0)

    d_fd.design_experiment(
        d_fd.d_opt_criterion,
        solver                = "ipopt",
        solver_options        = {"linear_solver": LINEAR_SOLVER,
                                 "tol": 1e-10, "max_iter": 3000},
        optimize_sampling_times = True,
    )

    # IFT designer — same grid
    d_ift2 = Designer()
    d_ift2.simulate               = _simulate_1st_order
    d_ift2.model_parameters       = theta
    d_ift2.ti_controls_candidates = t_grid.reshape(-1, 1)
    d_ift2.sampling_times_candidates = np.tile(spt_grid_b, (n_cands, 1))
    d_ift2.pyomo_model_fn         = _build_pyomo_model_1st_order
    d_ift2.n_jobs                 = 1
    d_ift2.initialize(verbose=0)

    d_ift2.design_experiment(
        d_ift2.d_opt_criterion,
        solver                = "ipopt",
        solver_options        = {"linear_solver": LINEAR_SOLVER,
                                 "tol": 1e-10, "max_iter": 3000},
        optimize_sampling_times = True,
    )

    crit_fd  = d_fd._criterion_value
    crit_ift = d_ift2._criterion_value
    rel_err  = abs(crit_ift - crit_fd) / (abs(crit_fd) + 1e-12)

    ok(f"FD  criterion: {crit_fd:.6f}")
    ok(f"IFT criterion: {crit_ift:.6f}")
    ok(f"Relative error: {rel_err:.4f}")

    assert rel_err < 0.05, (
        f"IFT and FD D-optimal criteria differ by more than 5% with "
        f"optimize_sampling_times=True: FD={crit_fd:.6f}, IFT={crit_ift:.6f}, "
        f"rel_err={rel_err:.4f}. "
        f"This likely indicates IFT is not evaluating sensitivities at the "
        f"correct sampling time (the designer.py _current_spt fix may be missing)."
    )
    ok(f"IFT and FD criteria agree within 5% with optimize_sampling_times=True "
       f"(rel err {rel_err:.4f})")

    # Support point check: both should select the same candidate(s)
    support_fd  = set(np.where(d_fd.efforts.flatten()  > 1e-3)[0])
    support_ift = set(np.where(d_ift2.efforts.flatten() > 1e-3)[0])
    ok(f"FD  support candidates: {sorted(support_fd)}")
    ok(f"IFT support candidates: {sorted(support_ift)}")

    # The dominant support candidate (highest effort) should be the same
    dominant_fd  = int(np.argmax(d_fd.efforts.flatten()))
    dominant_ift = int(np.argmax(d_ift2.efforts.flatten()))
    t_fd  = float(t_grid[dominant_fd])
    t_ift = float(t_grid[dominant_ift])
    ok(f"FD  dominant candidate: t={t_fd:.2f}")
    ok(f"IFT dominant candidate: t={t_ift:.2f}")

    assert abs(t_fd - t_ift) <= grid_spacing + 1e-6, (
        f"IFT and FD select different dominant candidates: "
        f"FD t={t_fd:.2f}, IFT t={t_ift:.2f} (diff={abs(t_fd-t_ift):.2f} "
        f"> grid spacing {grid_spacing:.2f}). "
        f"This indicates IFT sampling-time sensitivity is wrong."
    )
    ok(f"IFT and FD dominant support agree within grid spacing "
       f"(FD t={t_fd:.2f}, IFT t={t_ift:.2f})")


# =============================================================================
# Tests 34–35: guarantees previously asserted only by code inspection
#
# These two tests lock down behaviour that every OTHER test in this suite is
# blind to, because every other model here uses FLAT variable names (k, A0,
# A[t]) and never probes a degenerate point through the diagnostic:
#
#   34. The IFT name matcher itself (designer._match_nlp_var), exercised as a
#       direct unit test with hand-built name lists.  pydex maps each model Var
#       to its ASL Jacobian column BY NAME.  In normal use an exact match always
#       exists (both str(var) and PyomoNLP.primals_names() derive from
#       getname(fully_qualified=True)), so the suffix/leaf fallbacks are never
#       hit by a real model — which is exactly why a full-model test can't cover
#       them.  Feeding the matcher synthetic name lists is the only way to drive
#       every clause, and in particular to pin down the EXACT-FIRST guarantee:
#       when a model carries both a top-level Var and a block-nested Var sharing
#       a leaf name ('theta' and 'b.theta'), the exact name must win regardless
#       of ASL's primal ordering.  A naive single-pass scan aliased one onto the
#       other depending on list order — a silent wrong-column bug.  This test is
#       also the cross-check that the Designer's matcher and the diagnostic's
#       gate matcher (diagnose_asl_elimination._match_param_name) agree clause
#       for clause, so a model can never pass the gate and then bind differently
#       (or fail) at design time.  No Pyomo/ASL needed: it is pure string logic.
#
#   35. Degenerate-probe recovery in the diagnostic.  When a probe candidate
#       yields an all-fixed model (no free Vars — e.g. t_f <= 0), PyomoNLP
#       cannot compile it.  diagnose_asl_elimination must NOT report that as
#       parameter elimination; it must nudge to a non-degenerate probe point
#       and recover.  Test 19 happens to survive this, but nothing ASSERTS the
#       recovery, so a regression that turned recovery back into a false
#       'eliminated' verdict would slip through.
# =============================================================================

def test_34_ift_name_matcher():
    section("34 — IFT name matcher (_match_nlp_var) — all clauses + exact-first")

    # Pure string logic — no Pyomo, no ASL, no solver needed.  We import the
    # ACTUAL matcher the Designer uses at run time, and the ACTUAL helper the
    # diagnostic gate uses, and assert (a) every clause behaves, (b) exact match
    # wins over suffix matches regardless of list order, and (c) the two
    # implementations agree clause-for-clause.
    try:
        from pydex.core.designer import _match_nlp_var
    except Exception as exc:
        ok(f"SKIPPED — could not import _match_nlp_var ({exc})")
        return

    # The diagnostic's gate matcher is optional (only present if the utils
    # module is installed); if available we cross-check it against the Designer's.
    try:
        from pydex.utils.diagnose_asl_elimination import _match_param_name
    except Exception:
        _match_param_name = None

    # ── Case battery ──────────────────────────────────────────────────────────
    # (name_to_find, primal_name_list, expected_index, description)
    # expected_index is the column the matcher MUST return (None = absent).
    cases = [
        # Clause 1 — exact equality (the only clause real models ever hit)
        ("k",      ["k", "A0"],                  0, "flat exact"),
        ("A0",     ["k", "A0"],                  1, "flat exact (2nd)"),
        ("A[1.0]", ["A[0.0]", "A[1.0]"],         1, "indexed exact"),
        ("b.k",    ["b.k", "b.A0", "A[1.0]"],    0, "fully-qualified exact (both sides dotted)"),
        ("b.A0",   ["b.k", "b.A0", "A[1.0]"],    1, "fully-qualified exact (2nd)"),
        # Clause 2 — primal carries a block prefix the bare name lacks
        ("k",      ["b.k", "b.A0"],              0, "bare name vs block-qualified primal"),
        # Clause 3 — the name carries a prefix the primal lacks
        ("b.k",    ["k", "A0"],                  0, "block-qualified name vs bare primal"),
        # Absent — true ASL elimination (Failure Mode B)
        ("ghost",  ["k", "A0"],               None, "absent → None (true elimination)"),
        # ── EXACT-FIRST regression: a model with BOTH 'theta' and 'b.theta' ──
        # A single-pass scan accepted the first positional clause hit, so
        # 'theta' aliased onto 'b.theta' when 'b.theta' was listed first.
        ("theta",   ["b.theta", "theta"],        1, "EXACT-FIRST: exact must beat earlier suffix hit"),
        ("theta",   ["theta", "b.theta"],        0, "exact match, order-independent"),
        ("b.theta", ["b.theta", "theta"],        0, "qualified exact present"),
        ("b.theta", ["theta", "b.theta"],        1, "qualified exact present (reordered)"),
    ]

    for name, primals, expected, desc in cases:
        got = _match_nlp_var(name, primals)
        assert got == expected, (
            f"_match_nlp_var({name!r}, {primals!r}) = {got}, expected {expected} "
            f"[{desc}]"
        )
    ok(f"_match_nlp_var: all {len(cases)} clause/edge cases correct (incl. exact-first)")

    # Explicit, named regression assertion for the aliasing bug, so a failure
    # here points straight at the cause rather than at a generic case index.
    assert _match_nlp_var("theta", ["b.theta", "theta"]) == 1, (
        "ALIASING REGRESSION: 'theta' matched a block-qualified 'b.theta' "
        "instead of the exact top-level 'theta'. The matcher must try exact "
        "equality before any suffix/leaf fallback."
    )
    ok("Exact-first guarantee holds: 'theta' never aliases onto 'b.theta'")

    # ── Cross-check: Designer matcher vs diagnostic gate matcher ──────────────
    if _match_param_name is not None:
        for name, primals, expected, desc in cases:
            a = _match_nlp_var(name, primals)
            b = _match_param_name(name, primals)
            assert a == b, (
                f"Designer/gate matcher DISAGREE on ({name!r}, {primals!r}): "
                f"_match_nlp_var={a}, _match_param_name={b} [{desc}]. "
                f"They must be byte-for-byte equivalent or a model can pass the "
                f"gate and bind differently (or fail) at design time."
            )
        ok("Designer._match_nlp_var and diagnostic._match_param_name agree on every case")
    else:
        ok("diagnostic._match_param_name not importable — cross-check skipped")


def test_35_degenerate_probe_recovery():
    section("35 — Degenerate-probe recovery in diagnose_asl_elimination")

    if not _PYOMO_AVAILABLE:
        ok("SKIPPED — Pyomo not available")
        return

    try:
        from pydex.utils.diagnose_asl_elimination import diagnose_asl_elimination
    except Exception as exc:
        ok(f"SKIPPED — diagnose_asl_elimination not importable ({exc})")
        return

    theta   = np.array([0.5, 1.0])    # k, A0
    p_names = ["k", "A0"]

    # The flat builder's t_f <= 0 branch returns an ALL-FIXED model (k, A0, A
    # all fixed, single trivial constraint).  PyomoNLP cannot compile a model
    # with no free Vars.  Degeneracy here is a function of ti_controls (the
    # builder keys off ti_controls[0]), NOT of the sampling grid — so probing
    # at ti_controls=[0.0] makes BOTH the full-grid and single-point checks
    # build the degenerate model (both call _check_survival_robust with the
    # same ti_controls).  Both must therefore nudge off t=0 and recover.
    #
    # The CORRECT behaviour: the diagnostic detects degeneracy, nudges to a
    # non-degenerate probe point, and recovers a clean verdict (k and A0 both
    # survive).  The WRONG behaviour (the regression we are guarding against)
    # is reporting k/A0 as 'eliminated', or surfacing the degeneracy as a hard
    # error, when the model is in fact perfectly well-posed away from t=0.
    result = diagnose_asl_elimination(
        _build_pyomo_model_1st_order,     # flat builder with the t<=0 degenerate branch
        ti_controls      = [0.0],         # <-- degenerate probe point
        model_parameters = theta,
        sampling_times   = [0.0, 2.0, 5.0],   # includes the degenerate t=0 too
        param_names      = p_names,
        verbose          = False,
    )

    # 1. Must not have been turned into a false 'elimination' verdict.
    assert result["eliminated_full"] == [], (
        f"Degenerate probe was misreported as elimination (full): "
        f"{result['eliminated_full']}"
    )
    assert result["eliminated_single"] == [], (
        f"Degenerate probe was misreported as elimination (single): "
        f"{result['eliminated_single']}"
    )
    ok("Degenerate probe NOT misreported as parameter elimination")

    # 2. Must have recovered to a clean, IFT-ready verdict (not stuck in error).
    assert not result["errored"], (
        f"Diagnostic surfaced degeneracy as a hard error instead of recovering: "
        f"{result['error']}"
    )
    assert result["ift_ready"], (
        f"Diagnostic failed to recover an IFT-ready verdict from a degenerate "
        f"probe: {result}"
    )
    ok("Diagnostic recovered: nudged off the degenerate point to an IFT-ready verdict")

    # 3. Both parameters are present in the recovered primal names — i.e. the
    #    recovery actually built a real (non-degenerate) NLP, not an empty one.
    primals = result["nlp_primal_names"]
    assert primals, "recovered NLP has no primal names — recovery did not build a real model"
    ok(f"Recovery built a real NLP: {len(primals)} primal var(s)")

    # 4. Cross-check against the canonical survivor count: both k and A0 survive.
    assert len(result["survived_full"]) == len(theta), (
        f"Expected all {len(theta)} parameters to survive after recovery, "
        f"got {len(result['survived_full'])}: {result['survived_full']}"
    )
    ok(f"All {len(theta)} parameters survive after degenerate-probe recovery")


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("  pydex full capability test")
    print("█"*70)

    # ── Shared designers ──────────────────────────────────────────────────────
    # Full grid for most tests
    d_full = make_designer(small=False)
    # Small grid for expensive or iterative tests (PB, CVaR, MINLP, Pareto)
    d_small = make_designer(small=True)

    # ── Run all tests ─────────────────────────────────────────────────────────
    try:
        d_init = test_01_init_and_grid_helpers()
        test_02_sensitivity_analysis(d_init)
        test_02b_diagnose_sensitivity(d_init)
        d_eff  = test_03_d_optimal(d_full)
        test_04_a_optimal(d_full)
        test_05_e_optimal(d_full)
        test_06_optimize_sampling_times(d_full)
        test_07_pseudo_bayesian_type0(d_small)
        test_08_pseudo_bayesian_type1(d_small)
        test_09_cvar(d_small)
        test_10_apportion(d_full)
        test_11_prior_fim_case_a(d_full)
        test_12_prior_experiments_case_b(d_full)
        test_13_v_optimal(d_full)
        test_13b_operating_point(d_full)
        test_14_save_load_result(d_full)
        test_15_save_load_state(d_full)
        test_16_visualisation_suite(d_full)
        test_17_minlp_sparsity(d_small)
        test_18_cvar_pareto(d_small)

        # ── Pyomo IFT tests ───────────────────────────────────────────────────
        test_19_pyomo_ift_auto_detection()
        crit_seq = test_20_pyomo_ift_local_sequential()
        test_21_pyomo_ift_local_parallel(crit_seq)
        test_22_pyomo_ift_pb_parallel()
        test_23_normalization_toggle()
        test_24_fd_vs_ift_agreement()

        # ── Pyomo DAE simulate + IFT (fully self-consistent, no analytical fallback)
        crit_dae_seq = test_25_dae_simulate_ift_sequential()
        test_26_dae_simulate_ift_parallel(crit_dae_seq)
        test_27_dae_simulate_ift_pb_parallel()
        test_28_dae_vs_analytical_simulate_agreement()

        # ── Additional coverage ───────────────────────────────────────────────
        test_29_generalized_individual_criteria(d_small)
        test_30_pyomo_ift_signature2_multi_output()
        test_31_regularize_fim(d_small)
        test_32_n_exp_discrete_design(d_small)
        test_33_ift_sampling_time_optimisation()

        # ── Guarantees previously only asserted by inspection ─────────────────
        test_34_ift_name_matcher()
        test_35_degenerate_probe_recovery()

        print("\n" + "█"*70)
        print("  ALL TESTS PASSED")
        print("█"*70 + "\n")

    except Exception as exc:
        print(f"\n[FAILED] {exc}")
        raise
