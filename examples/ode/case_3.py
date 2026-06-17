from pydex.core.designer import Designer
from case_3_model import simulate
import numpy as np

"""
case_3.py
=========
D-optimal design for the Michaelis-Menten-style reaction network:
    A → B   with rate  r = k1 * cA^α / (k2 + k3 * cA^β)

Nine model parameters : [θ_10, θ_11, θ_20, θ_21, θ_30, θ_31, ν, α, β]
Three time-invariant controls : [cA0, T (K), τ]
Two measurable responses : [cA, cB]
"""

designer = Designer()
designer.simulate = simulate

# ── Nominal model parameters ──────────────────────────────────────────────────
designer.model_parameters = np.array([5.4, 5.0, 6.2, 0.5, 1.4, 2.5, 7/3, 3, 5])


# ══════════════════════════════════════════════════════════════════════════════
# Candidate generation and feasibility filtering
# ══════════════════════════════════════════════════════════════════════════════
#
# The experimental candidate space is defined by three time-invariant controls:
#   cA0  — initial concentration of A       [mol/L]   bounds: [1,  20]
#   T    — isothermal reaction temperature  [K]       bounds: [273.15, 323.15]
#   τ    — residence / batch time           [units]   bounds: [1,  100]
#
# enumerate_candidates() builds a full-factorial grid at the specified number
# of levels per control, giving 5 × 5 × 5 = 125 candidate experiments.
#
# WHY FILTER?
# -----------
# A purely geometric grid has no knowledge of the physics.  Some (cA0, T, τ)
# combinations produce concentration profiles that are essentially flat — either
# because the reaction is too slow to make any progress within τ, or because the
# Michaelis-Menten denominator (k2 + k3·cA^β) suppresses the rate to near zero
# at the given conditions.  Flat profiles contribute negligible sensitivity to
# any parameter, so including them:
#   (a) wastes sensitivity analysis time (~3 s per candidate × 125 = ~6 min), and
#   (b) adds near-zero rows to the FIM that can cause numerical ill-conditioning.
#
# WHAT THE FILTER DOES — AND DOES NOT — DO
# -----------------------------------------
# The filter enforces pure physical feasibility: it removes only candidates
# where the ODE integration fails outright or where literally no reaction occurs.
# It does NOT attempt to pre-select "good" or "informative" candidates — that
# is deliberately left to the D-optimal optimiser.  An overly aggressive filter
# risks discarding candidates that the optimiser would have selected as support
# points, which would corrupt the OED result.
#
# The two feasibility conditions checked are:
#   1. Integration success  — simulate() returns finite values (no NaN).
#      The scipy Radau solver already handles stiffness robustly, so NaN only
#      appears for truly pathological parameter/control combinations.
#   2. Non-zero reaction    — fractional conversion of A at t=1 is at least
#      MIN_CONVERSION (default 1%).  This guards against the silent degeneracy
#      where the rate law evaluates to zero (e.g. cA0 so small that the
#      Michaelis-Menten denominator completely dominates the numerator), which
#      produce no solver failure but contribute nothing to parameter estimation.
#
# Only two time points are simulated for the filter (t=0.001 and t=1.0) — just
# enough to measure whether any conversion occurred.  Running the full 11-point
# profile for every candidate would take as long as the sensitivity analysis
# itself and defeat the purpose of the filter.
#
# THRESHOLD CHOICE
# ----------------
# MIN_CONVERSION = 0.01 (1%) is intentionally permissive.  A candidate with
# only 1% conversion is almost certainly not a support point, but it is
# physically real and the optimiser should be allowed to confirm that by
# assigning it zero effort rather than having it silently removed.  Raising
# this threshold (e.g. to 0.05 or 0.10) would be pre-empting the optimiser.

MIN_CONVERSION = 0.01   # minimum fractional conversion of A to pass the filter

def is_feasible(tic, model_parameters, min_conversion=MIN_CONVERSION):
    """
    Return True if the candidate experiment (tic) is physically feasible
    at the nominal model parameters.

    Parameters
    ----------
    tic : array-like, length 3
        Time-invariant controls [cA0 (mol/L), T (K), τ].
    model_parameters : array-like, length 9
        Nominal parameter vector [θ_10, θ_11, θ_20, θ_21, θ_30, θ_31, ν, α, β].
    min_conversion : float
        Minimum fractional conversion of A required to pass (default 0.01).

    Returns
    -------
    bool
        True  → candidate is feasible and should be included in the OED grid.
        False → candidate is degenerate and should be excluded.

    Notes
    -----
    Only the initial and final time points are simulated.  The full sampling
    grid is not needed because we are only checking whether any reaction
    occurred, not characterising the profile shape.

    The function intentionally uses the same simulate() that pydex will call
    during sensitivity analysis, so feasibility is assessed on exactly the
    same model as the OED.  There is no separate "feasibility model" to
    maintain or keep in sync.
    """
    # Use only two time points — start and end — for a lightweight check.
    # t=0.001 rather than t=0 avoids a division-by-zero edge case in some
    # integrators when cA=cA0 and the rate expression is evaluated at t=0.
    spt_check = np.array([0.001, 1.0])

    c = simulate(tic, spt_check, model_parameters)

    # Condition 1: integration must have succeeded.
    # simulate() returns an array of NaN if scipy's solve_ivp fails (e.g. due
    # to an unbounded blow-up at extreme parameter perturbations).  At nominal
    # parameters this should never trigger, but it is cheap to check.
    if np.any(~np.isfinite(c)):
        return False

    # Condition 2: at least min_conversion of A must have been consumed.
    # c[0, 0] is cA at the first time point (≈ cA0 after the tiny initial step).
    # c[1, 0] is cA at t=1 (end of the normalised batch).
    cA_start = c[0, 0]
    cA_end   = c[1, 0]

    # Guard against a degenerate initial condition (cA0 ≤ 0) which would cause
    # a zero-division.  This should not occur given the bounds [1, 20] mol/L,
    # but defensive programming is warranted here since the filter runs before
    # pydex's own input validation.
    if cA_start <= 0:
        return False

    conversion = (cA_start - cA_end) / cA_start
    return conversion >= min_conversion


# ── Build full geometric grid ─────────────────────────────────────────────────
tic_all = designer.enumerate_candidates(
    bounds=[
        [1,      20   ],    # cA0 (mol/L)
        [273.15, 323.15],   # T   (K)
        [1,      100  ],    # tau (time units)
    ],
    levels=[5, 5, 5],
)

# ── Apply feasibility filter ──────────────────────────────────────────────────
# Evaluate each candidate against the two feasibility conditions above.
# This loop runs 125 quick two-point simulations; at ~0.03 s each it takes
# only a few seconds — much less than the full sensitivity analysis.
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
      f"({n_removed} removed, conversion < {MIN_CONVERSION:.0%} at nominal parameters)\n")

# ── Assign filtered candidates to designer ───────────────────────────────────
designer.ti_controls_candidates = tic

# Each candidate gets the same 11-point sampling grid over the normalised
# batch time [0.001, 1].  The lower bound avoids t=0 so that the integrator
# never has to evaluate the rate at the exact initial condition (minor
# numerical convenience; has no effect on the OED result).
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
if False:
    designer.simulate_candidates(plot_simulation_times=True)
    designer.plot_predictions()
    designer.eval_sensitivities(save_sensitivities=False, store_predictions=True)
    designer.plot_sensitivities()
    designer.show_plots()

# ── Estimability diagnosis ─────────────────────────────────────────────────────
# With 9 parameters (including the structurally coupled α, β exponents and three
# pairs of Arrhenius coefficients), it is worth checking which parameters are
# observable across the candidate grid before committing to the OED solve.
# diagnose_sensitivity() replaces the old estimability_study_fim() — it operates
# on the per-candidate atomic FIM rather than the aggregate, giving a
# grid-independent, physically meaningful assessment.
#
# tol_diag=1.0 : flag any (candidate, parameter) pair where a single experiment
#                cannot determine that parameter to within its own magnitude.
# tol_cond=1e4 : flag candidates where parameters are nearly collinear.
#
# If α or β are flagged across most candidates, consider fixing one of them or
# widening the cA0 / τ bounds to create more dynamic range in the rate law.
if False:
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
print(" Note: lower criterion than Design 1 is expected — fewer samples per run ".center(100, " "))
print("=" * 100)
print()
# ══════════════════════════════════════════════════════════════════════════════

# Reset atomic_fims: the fixed-spt run cached a (n_c × 11) atomic FIM array;
# the n_spt=5 run needs a different layout so pydex must recompute from scratch.
designer.atomic_fims = None
designer.design_experiment(
    designer.d_opt_criterion,
    optimize_sampling_times=True,
    n_spt=5,                        # select 5 optimal measurement times per run
    solver="ipopt",
    solver_options={"linear_solver": "ma57"},
    write=False,
)
designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_predictions()
designer.plot_optimal_sensitivities(interactive=False)
designer.apportion(40)

designer.show_plots()
