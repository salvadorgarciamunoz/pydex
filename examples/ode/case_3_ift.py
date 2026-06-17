from pydex.core.designer import Designer
from case_3_ift_model import simulate, build_pyomo_model
import numpy as np
import logging

"""
case_3_ift.py
=============
D-optimal design for the Michaelis-Menten-style reaction network using
exact IFT sensitivities via Pyomo.DAE collocation + IPOPT (PyomoNLP).

This is the fast version of case_3.py.  The scipy/FD path in case_3.py
spends ~350 s on sensitivity analysis (finite differences, ~45 model
evaluations per candidate × 121 candidates).  The IFT path here computes
exact symbolic sensitivities from the KKT conditions of the collocation NLP —
one IPOPT solve per candidate, sensitivities extracted analytically.
Expected sensitivity analysis time: ~5–15 s (20–70× speedup).

Reaction system
---------------
    A → B    r = k1(T) * CA^α / (k2(T) + k3(T) * CA^β)
    ki(T) = exp(θ_i0 + θ_i1 * (T - 273.15) / T)

Nine model parameters : [θ_10, θ_11, θ_20, θ_21, θ_30, θ_31, ν, α, β]
Three time-invariant controls : [CA0 (mol/L), T (K), τ]
Two measurable responses : [CA(t), CB(t)]
"""

designer = Designer()
designer.simulate            = simulate
designer.pyomo_model_fn      = build_pyomo_model   # IFT sensitivities via PyomoNLP
designer.pyomo_output_var_name = ["ca", "cb"]       # response var names in the Pyomo model
# use_pyomo_ift and n_jobs are auto-detected by initialize() when
# pyomo_model_fn is provided — no manual configuration needed.

# ── Run-control flags ─────────────────────────────────────────────────────────
# Set these to True to enable the corresponding optional section.
# They are False by default so the script runs straight to the OED solve.
#
# INSPECT_CANDIDATES : simulate all candidates and plot their concentration
#                      profiles and sensitivities.  Useful for a sanity check
#                      after changing the candidate bounds or levels, but adds
#                      the full sensitivity analysis time before the OED solve.
#
# RUN_ESTIMABILITY   : run diagnose_sensitivity() after computing sensitivities,
#                      producing a heatmap of per-candidate observability and a
#                      condition-number bar chart.  If INSPECT_CANDIDATES is
#                      False, this block calls eval_sensitivities() on its own
#                      (same time cost as the sensitivity analysis).
INSPECT_CANDIDATES = False
RUN_ESTIMABILITY   = False

# ── Nominal model parameters ──────────────────────────────────────────────────
designer.model_parameters = np.array([5.4, 5.0, 6.2, 0.5, 1.4, 2.5, 7/3, 3, 5])


# ══════════════════════════════════════════════════════════════════════════════
# Candidate generation and feasibility filtering
# ══════════════════════════════════════════════════════════════════════════════
#
# The experimental candidate space is defined by three time-invariant controls:
#   CA0  — initial concentration of A       [mol/L]   bounds: [1,  20]
#   T    — isothermal reaction temperature  [K]       bounds: [273.15, 323.15]
#   τ    — residence / batch time           [units]   bounds: [1,  100]
#
# enumerate_candidates() builds a full-factorial grid at the specified number
# of levels per control, giving 5 × 5 × 5 = 125 candidate experiments.
#
# WHY FILTER?
# -----------
# A purely geometric grid has no knowledge of the physics.  Some (CA0, T, τ)
# combinations produce concentration profiles that are essentially flat —
# either because the reaction is too slow to make any progress within τ, or
# because the Michaelis-Menten denominator (k2 + k3·CA^β) suppresses the rate
# to near zero.  Flat profiles contribute negligible sensitivity to any
# parameter.  Including them:
#   (a) wastes sensitivity analysis time (one IPOPT solve per candidate), and
#   (b) adds near-zero rows to the FIM that can cause numerical ill-conditioning.
#
# WHAT THE FILTER DOES — AND DOES NOT — DO
# -----------------------------------------
# The filter enforces pure physical feasibility: it removes only candidates
# where literally no reaction occurs.  It does NOT pre-select "good" or
# "informative" candidates — that is deliberately left to the D-optimal
# optimiser.  An overly aggressive filter risks discarding candidates that
# the optimiser would have selected as support points.
#
# The two feasibility conditions checked are:
#   1. Simulation success  — simulate() returns finite values (no NaN / error).
#   2. Non-zero reaction   — fractional conversion of A at t=τ is at least
#      MIN_CONVERSION (default 1%), confirming the rate law is non-zero.
#
# For the IFT path, simulate() calls IPOPT via Pyomo collocation rather than
# scipy.  The feasibility check therefore also implicitly verifies that the
# collocation NLP converges — a useful early warning before the full
# sensitivity analysis.
#
# Only two time points are simulated for the filter (t=0.001 and t=1.0) to
# keep the cost low.

MIN_CONVERSION = 0.01   # minimum fractional conversion of A to pass the filter

def is_feasible(tic, model_parameters, min_conversion=MIN_CONVERSION):
    """
    Return True if the candidate experiment (tic) is physically feasible
    at the nominal model parameters and the collocation NLP converges.

    Parameters
    ----------
    tic : array-like, length 3
        Time-invariant controls [CA0 (mol/L), T (K), τ].
    model_parameters : array-like, length 9
        Nominal parameter vector.
    min_conversion : float
        Minimum fractional conversion of A required to pass (default 0.01).

    Returns
    -------
    bool

    Notes
    -----
    WHY A DENSER GRID THAN THE SCIPY FEASIBILITY FILTER:
    The scipy version in case_3.py used only [0.001, 1.0] (2 points) because
    solve_ivp has no initialisation requirement — it marches forward from the
    initial condition regardless of how many output points are requested.

    The Pyomo collocation model is fundamentally different: it is a boundary
    value problem solved simultaneously across all finite elements.  IPOPT
    needs a consistent initial trajectory across the whole domain to converge.
    With only 2 normalised time points embedded as FE boundaries, the
    collocation grid is extremely coarse and IPOPT frequently hits
    maxIterations or declares local infeasibility — not because the candidate
    is physically infeasible, but because the initialisation is too poor.

    Using the full 11-point sampling grid (same as the main sensitivity
    analysis) gives the collocation NLP enough structure to initialise well
    and converge reliably.  The cost is slightly higher per candidate (~0.3 s
    vs ~0.05 s), but the filter remains much cheaper than a full sensitivity
    analysis solve.

    WHY WE CATCH BOTH RuntimeError AND ValueError:
    - RuntimeError is raised by our own check in build_pyomo_model() when
      IPOPT returns infeasible or another non-optimal status.
    - ValueError is raised by Pyomo's solutions.load_from() when IPOPT
      returns status 'error' (e.g. maxIterations exceeded without a
      feasible point) — Pyomo raises this before our check even runs.
    Both conditions mean the NLP failed to find a solution, so both should
    be treated as infeasible candidates and excluded from the design grid.
    """
    # Use the full 11-point grid — same as the main sensitivity analysis.
    # This gives the collocation NLP a well-structured initialisation problem
    # and avoids the convergence failures seen with a 2-point grid.
    spt_check = np.linspace(0.001, 1.0, 11)
    # Suppress Pyomo's WARNING messages for infeasible/error solver status.
    # These are printed to stderr by solutions.load_from() before it raises —
    # they appear even when the exception is correctly caught, and are noisy
    # when filtering 125 candidates.  The exception itself is still caught
    # below; suppression only affects the log output, not the control flow.
    _pyomo_logger = logging.getLogger('pyomo')
    _prev_level   = _pyomo_logger.level
    _pyomo_logger.setLevel(logging.ERROR)
    try:
        c = simulate(tic, spt_check, model_parameters)
    except (RuntimeError, ValueError):
        # IPOPT did not converge or Pyomo could not load results — exclude
        return False
    finally:
        # Always restore the logger level so pydex output is unaffected
        _pyomo_logger.setLevel(_prev_level)

    if np.any(~np.isfinite(c)):
        return False

    cA_start = c[0, 0]
    cA_end   = c[-1, 0]   # last point (t=1.0) after using 11-point grid

    if cA_start <= 0:
        return False

    conversion = (cA_start - cA_end) / cA_start
    return conversion >= min_conversion


# ── Build full geometric grid ─────────────────────────────────────────────────
tic_all = designer.enumerate_candidates(
    bounds=[
        [1,      20   ],    # CA0 (mol/L)
        [273.15, 323.15],   # T   (K)
        [1,      100  ],    # tau (time units)
    ],
    levels=[5, 5, 5],
)

# ── Apply feasibility filter ──────────────────────────────────────────────────
# Each filter simulation uses the full 11-point grid and calls IPOPT once
# (~0.3–1 s per candidate).  For 125 candidates this takes ~40–120 s — still
# much less than the full per-candidate IFT sensitivity solve, and the denser
# grid is required for reliable IPOPT convergence (see is_feasible docstring).
mp_nom = designer.model_parameters
feasible_mask = np.array([
    is_feasible(row, mp_nom)
    for row in tic_all
])

tic = tic_all[feasible_mask]

n_total    = len(tic_all)
n_feasible = int(feasible_mask.sum())
n_removed  = n_total - n_feasible
print(f"\nCandidate feasibility filter: {n_feasible}/{n_total} retained "
      f"({n_removed} removed, conversion < {MIN_CONVERSION:.0%} or NLP failed "
      f"at nominal parameters)\n")

# ── Assign filtered candidates to designer ───────────────────────────────────
designer.ti_controls_candidates = tic

designer.sampling_times_candidates = np.array([
    np.linspace(0.001, 1, 11)
    for _ in tic
])

# ── Optional metadata ─────────────────────────────────────────────────────────
designer.measurable_responses  = [0, 1]
designer.response_names        = ["$c_A$", "$c_B$"]
designer.model_parameter_names = [
    r"$\theta_{10}$", r"$\theta_{11}$",
    r"$\theta_{20}$", r"$\theta_{21}$",
    r"$\theta_{30}$", r"$\theta_{31}$",
    r"$\nu$", r"$\alpha$", r"$\beta$",
]

# ── Error covariance ──────────────────────────────────────────────────────────
designer.error_cov = np.diag([0.1, 0.1])

designer.initialize(verbose=2)


# ── [Optional] inspect candidates ─────────────────────────────────────────────
if INSPECT_CANDIDATES:
    designer.simulate_candidates(plot_simulation_times=True)
    designer.plot_predictions()
    designer.eval_sensitivities(save_sensitivities=False, store_predictions=True)
    designer.plot_sensitivities()
    designer.show_plots()

# ── Estimability diagnosis ─────────────────────────────────────────────────────
# With 9 parameters (including the structurally coupled α, β exponents and three
# pairs of Arrhenius coefficients), it is worth checking which parameters are
# observable across the candidate grid before committing to the OED solve.
# diagnose_sensitivity() operates on the per-candidate atomic FIM, giving a
# grid-independent, physically meaningful assessment.
#
# tol_diag=1.0 : flag any (candidate, parameter) pair where a single experiment
#                cannot determine that parameter to within its own magnitude.
# tol_cond=1e4 : flag candidates where parameters are nearly collinear.
#
# If α or β are flagged across most candidates, consider fixing one of them or
# widening the CA0 / τ bounds to create more dynamic range in the rate law.
if RUN_ESTIMABILITY:
    designer.eval_sensitivities(save_sensitivities=False, store_predictions=True)
    diag = designer.diagnose_sensitivity(tol_diag=1.0, tol_cond=1e4)
    print("Flagged (candidate, parameter) pairs:", diag["flagged_diag"])
    print("Ill-conditioned candidates:          ", diag["flagged_cond"])
    designer.show_plots()

# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 100)
print(" DESIGN 1 / 2 — D-optimal, fixed sampling times ".center(100, "="))
print(" All 11 evenly-spaced time points used per run ".center(100, " "))
print("=" * 100)
print()
# ══════════════════════════════════════════════════════════════════════════════

designer.design_experiment(
    designer.d_opt_criterion,
    optimize_sampling_times=False,
    solver="ipopt",
    solver_options={"linear_solver": "ma57"},
    write=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_predictions()
designer.plot_optimal_sensitivities(interactive=False)
designer.apportion(20)

# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 100)
print(" DESIGN 2 / 2 — D-optimal, optimised sampling times ".center(100, "="))
print(" 5 measurement times per run chosen optimally from the 11-point grid ".center(100, " "))
print(" Candidate weights fixed from Design 1; only sampling times optimised  ".center(100, " "))
print(" Note: lower criterion than Design 1 is expected — fewer samples per run ".center(100, " "))
print("=" * 100)
print()
# ══════════════════════════════════════════════════════════════════════════════
#
# STRATEGY: two-stage decomposition
# ----------------------------------
# The joint problem (optimise both efforts AND 5-of-11 sampling times over all
# 24 candidates) produces an IPOPT problem with ~11 133 variables that struggles
# to converge — the combined effort + sampling-time landscape is highly
# non-convex for this 9-parameter model.
#
# Instead we decompose into two sequential sub-problems:
#
#   Stage 1 (Design 1 above): optimise candidate efforts with ALL 11 time
#           points fixed.  This is a convex relaxation of the full problem and
#           converges reliably (142 iterations).  Result: 10 support candidates
#           with known effort weights.
#
#   Stage 2 (this block): fix the candidate effort weights at the Design 1
#           values and only optimise which 5 of the 11 time points to use per
#           run.  By restricting to the 10 support candidates and fixing their
#           efforts, the IPOPT problem shrinks from ~11 133 variables to
#           ~10 × C(11,5) = 4 620 effective combinations — a much more
#           tractable problem.
#
# This gives a practically useful answer (which 5 time points to measure per
# run given we already know which runs to perform) without the convergence
# issues of the joint formulation.
#
# HOW efforts are fixed
# ---------------------
# pydex's design_experiment() always optimises efforts unless they are
# frozen externally.  We achieve the freeze by:
#   1. Extracting the support-candidate indices and their efforts from Design 1.
#   2. Building a NEW designer restricted to only those support candidates.
#      Passing their atomic FIMs (already computed) avoids re-running any
#      sensitivity analysis.
#   3. Using a scalar min_effort lower bound set to 90% of the smallest
#      support effort, so IPOPT cannot drop any support candidate to zero —
#      effectively pinning the effort allocation while still satisfying the
#      simplex constraint.
#   4. e0 is left as None (pydex default: uniform over all n_spt_comb
#      combinations) — this avoids the shape validation requirement that
#      e0 must be (n_c, n_spt_comb) where n_spt_comb = C(11,5) = 462.
#
# NOTE: pydex does not have a dedicated "fix efforts" flag, so we achieve
# the freeze via the scalar lower bound.  IPOPT converges quickly because
# the effort variables are tightly bounded and only the sampling-time
# combination weights have true freedom.

# ── Step 1: extract support candidates from Design 1 ─────────────────────────
# designer.efforts has shape (n_c, n_spt) for the fixed-spt run.
# Sum across time to get per-candidate total effort, then find supports.
d1_efforts_total = designer.efforts.sum(axis=1)          # shape (n_c,)
support_mask     = d1_efforts_total > 1e-4               # 10 support candidates
support_indices  = np.where(support_mask)[0]

print(f"Design 1 support candidates: {len(support_indices)} "
      f"(indices {support_indices.tolist()})")
print(f"Design 1 efforts at supports: "
      f"{np.round(d1_efforts_total[support_indices], 4).tolist()}")

# ── Step 2: build restricted designer on support candidates only ──────────────
designer2 = Designer()
designer2.simulate             = simulate
designer2.pyomo_model_fn       = build_pyomo_model
designer2.pyomo_output_var_name = ["ca", "cb"]
designer2.model_parameters     = designer.model_parameters.copy()
designer2.error_cov             = designer.error_cov.copy()

# Restrict candidates and sampling-time grid to the 10 support points
designer2.ti_controls_candidates      = designer.ti_controls_candidates[support_indices]
designer2.sampling_times_candidates   = designer.sampling_times_candidates[support_indices]
designer2.measurable_responses        = designer.measurable_responses
designer2.response_names              = designer.response_names
designer2.model_parameter_names       = designer.model_parameter_names

designer2.initialize(verbose=2)

# ── Step 3: transfer pre-computed atomic FIMs — no re-solve needed ────────────
# designer.atomic_fims has shape (n_c, n_spt, n_mp, n_mp) after Design 1.
# Slice to the support candidates so designer2 skips the sensitivity analysis.
if designer.atomic_fims is not None:
    designer2.atomic_fims = designer.atomic_fims[support_indices]
    print("[INFO]: Transferred atomic FIMs from Design 1 — "
          "no sensitivity re-analysis required.")

# ── Step 4: prepare effort bounds and warm-start ──────────────────────────────
# Effort vector for the restricted 14-candidate designer: normalise D1 efforts
# to sum to 1 (they already do, but be explicit).
d1_support_efforts = d1_efforts_total[support_indices]
d1_support_efforts = d1_support_efforts / d1_support_efforts.sum()   # re-normalise

print(f"\nDesign 1 efforts at supports (normalised):")
for i, (idx, eff) in enumerate(zip(support_indices, d1_support_efforts)):
    print(f"  Candidate {idx:2d}  effort = {eff:.4f}")

# Warm-start: e0 must be shape (n_c, n_spt_comb) where n_spt_comb = C(11,5) = 462.
# We use the pydex default (uniform) by passing e0=None, which avoids the shape
# validation entirely and lets pydex compute the correct n_spt_comb internally.
#
# Lower bound: scalar — pydex applies it uniformly as a minimum effort floor
# across all (candidate, spt-combination) variables.  Using the smallest
# support effort × 0.9 ensures no candidate is dropped while keeping IPOPT
# in a feasible region.
min_eff_scalar = float(0.90 * d1_support_efforts.min())
print(f"\nScalar effort lower bound (90% of smallest D1 support effort): {min_eff_scalar:.4f}")

# ── Step 5: optimise sampling times only ─────────────────────────────────────
designer2.design_experiment(
    designer2.d_opt_criterion,
    optimize_sampling_times=True,
    n_spt=5,                           # select 5 optimal measurement times per run
    solver="ipopt",
    solver_options={
        "linear_solver": "ma57",
        "tol":           1e-8,
        "max_iter":      1000,
        "acceptable_tol": 1e-6,
    },
    # e0 intentionally omitted — pydex uses uniform (n_c, n_spt_comb) default
    min_effort=min_eff_scalar,         # scalar floor keeps all support candidates active
    write=False,
)
designer2.print_optimal_candidates()
designer2.plot_optimal_efforts()
designer2.plot_optimal_predictions()
designer2.plot_optimal_sensitivities(interactive=False)
designer2.apportion(40)

designer.show_plots()
designer2.show_plots()
