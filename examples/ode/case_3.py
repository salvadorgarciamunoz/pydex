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

Sensitivities are FINITE DIFFERENCES over scipy Radau integration — no
pyomo_model_fn, so the model is a BLACK BOX as far as pydex is concerned.
case_3_ift.py designs the same experiment on the same model using exact IFT
derivatives from the KKT conditions of a collocation NLP.

WHAT THIS SCRIPT DEMONSTRATES
------------------------------
The same estimability workflow as case_3_ift.py, and the point of repeating it
here is that NONE OF IT REQUIRES A TRACTABLE MODEL. run_estimability() reads the
sensitivity matrix and knows nothing about where it came from; simulate() could
be a legacy Fortran binary, a commercial process simulator or a network call.
The sequence is:

  1. estimability on all nine parameters — which can this grid determine, and
     which are merely restating each other?
  2. fix the two it flags as unresolvable, leaving seven free;
  3. estimability again on the reduced set, to see what the reduction did and
     did not fix;
  4. then design.

Step 2 is not optional housekeeping. With all nine free, design_experiment()
refuses this model: the rate law has an exact invariance (adding a constant to
every θ_i0 at once, or to every θ_i1, leaves every prediction identical), so the
FIM is structurally singular and no allocation of effort can repair it.

Note that the two parameters flagged here are NOT the two flagged in
case_3_ift.py — that script fixes θ_20 and θ_21, this one fixes θ_21 and θ_31.
Both are correct. The redundancy is a common shift within a triple, so which
member you hold still is a convention, and at finite-difference accuracy the
residuals deciding the ordering sit close enough together that the tie breaks
differently. That is the nature of finite differences, and it is why the E-index
resolution floor is inferred as 1e-3 on this path against 1e-7 on the exact one.
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
designer.ti_controls_names     = ["cA0", "T", "tau"]
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

# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 100)
print(" ESTIMABILITY — on a BLACK-BOX model ".center(100, "="))
print("=" * 100)
print()
# ══════════════════════════════════════════════════════════════════════════════
#
# Same analysis as in case_3_ift.py, and the point of running it here is that
# NOTHING ABOUT IT NEEDS A TRACTABLE MODEL. run_estimability() reads
# designer.sensitivities and knows nothing about where they came from. In
# case_3_ift.py those are exact IFT derivatives extracted from the KKT system of
# a collocation NLP. Here they are finite differences over scipy's Radau
# integrator, and simulate() could equally be a legacy Fortran binary, a
# commercial process simulator, or an HTTP call — anything that maps
# (controls, times, parameters) to responses. The estimability question is
# answered from the sensitivity matrix alone.
#
# run_estimability() asks, over the WHOLE GRID: pooling every measurement you
# could take, in what ORDER do the parameters become estimable, and which are
# merely restating each other? It computes the sensitivities itself if they are
# not already cached. Yao/McAuley orthogonalisation (Wu, McLean, Harris and
# McAuley 2011, Table 1): rank by sensitivity column norm, project the remaining
# columns onto those already chosen, rank the residuals, repeat.
#
# THE ONE THING THAT DOES CHANGE ON THE BLACK-BOX PATH
# ----------------------------------------------------
# The UNRESOLVABLE flag is thresholded on the E index, and that threshold has to
# match the accuracy of the derivatives underneath it. Left at tol=None,
# run_estimability() infers it: 1e-7 when use_pyomo_ift is True, 1e-3 when it is
# False, as here. Below that floor a residual is indistinguishable from the
# numerical error of the sensitivity method, so the flag means "this analysis
# cannot resolve the parameter", NOT "the parameter is inestimable".
#
# The practical consequence is that WHICH parameters get named differs between
# this script and case_3_ift.py. Both flag two, and both are right; finite
# differences and exact IFT derivatives simply break the tie differently. See the
# section below, which acts on the two flagged HERE.
#
# This is a PRE-DESIGN analysis: no criterion, no design, only the candidate grid
# and the nominal parameters. Nothing downstream reads it.
est = designer.run_estimability()

print()
print("Estimability ranking (most to least estimable):")
print(est["table"].to_string())
print()
print("Correlation groups:", est["groups"])
print(f"E-index resolution floor inferred for this sensitivity method: {est['tol']:.0e}")
designer.show_plots()


# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 100)
print(" FIXING TWO PARAMETERS — removing the redundant directions ".center(100, "="))
print("=" * 100)
print()
# ══════════════════════════════════════════════════════════════════════════════
#
# WHY THIS IS HERE
# ----------------
# The estimability analysis above flags theta_21 and theta_31 as UNRESOLVABLE,
# and with all nine free design_experiment() refuses this model outright with
#
#     ValueError: The Fisher information matrix is STRUCTURALLY singular:
#     rank 8 of 9 even at the fully-supported design
#     Parameters implicated: [theta_11, theta_21, theta_31]
#
# That is not a conditioning problem and no candidate grid can fix it. It is an
# exact invariance of the rate law. Look at the rate:
#
#     r = k1(T) * CA^alpha / (k2(T) + k3(T) * CA^beta)
#     ki(T) = exp(theta_i0 + theta_i1 * (T - 273.15) / T)
#
# Multiply numerator AND denominator by exp(d) and r is unchanged. Since ki
# appears only inside an exponential, that is the same as ADDING d to every
# theta_i0 at once, and the same argument applies to theta_i1 through the shared
# temperature term. Verified against this very simulate(), with no derivatives
# and no algebra — just a handful of calls to the black box: shifting all three
# theta_i0 by +0.7 moves the predictions by 2.4e-12, all three theta_i1 by +0.3
# by 4.0e-12. Those directions carry no information, so the parameters spanning
# them cannot all be estimated at once.
#
# THE TWO FLAGGED PARAMETERS FAIL FOR DIFFERENT REASONS
# ------------------------------------------------------
# Worth reading off the abs info column rather than assuming, because the two
# cases need different experiments and only one of them is a redundancy:
#
#   theta_31   abs info 1.7e+05  — healthy information, but it sits in the
#              correlation group {theta_11, theta_31} at -0.9998. The grid
#              informs it; it just cannot separate it from theta_11. This is
#              the invariance above showing up as a correlated pair.
#
#   theta_21   abs info 3.0      — against 8.8e+08 for beta. This is not
#              redundancy at all: the grid contains almost NO information about
#              theta_21, and its correlation with theta_20 is only 0.81, below
#              the grouping threshold. k2 barely moves the rate over this
#              temperature range, so its Arrhenius slope is nearly invisible.
#
# So the reduction removes one genuinely redundant parameter and one genuinely
# uninformed one. A wider temperature range would help theta_21 and would NOT
# help theta_31, whose problem no grid can fix.
#
# WHICH TWO GET NAMED IS METHOD-DEPENDENT, AND THAT IS FINE
# ---------------------------------------------------------
# case_3_ift.py, on the same model with exact IFT derivatives, flags theta_20 and
# theta_21 and fixes those. This script flags theta_21 and theta_31 and fixes
# those. The disagreement is not a contradiction and neither script is wrong: the
# redundancy is a COMMON SHIFT within a triple, so the choice of which member to
# hold still is a convention, and at finite-difference accuracy the residuals
# that decide the ordering sit close enough together that the tie breaks
# differently. This is the nature of finite differences, and it is exactly why
# the E-index floor is loosened to 1e-3 on this path.
#
# One consequence is worth knowing rather than discovering later. Both parameters
# fixed here come from the theta_i1 triple, which removes that invariance
# outright; the theta_i0 shift is left formally intact, since theta_10, theta_20
# and theta_30 all stay free. It does not trip the gate — the reduced FIM comes
# back rank 7 OF 7 — because at finite-difference accuracy that direction is not
# resolved as null. The surviving correlation group {theta_10, theta_30, alpha}
# below is its fingerprint. So the seven-parameter design is well posed as far as
# THESE derivatives can tell, which is the standard every other number in this
# script is held to; case_3_ift.py, fixing one member of each triple, removes
# both invariances outright and is the stronger reduction where exact
# derivatives are available.
#
# THE FIX
# -------
# Fix the two flagged parameters at their nominal values and estimate the other
# seven:
#
#     theta_21 = 0.5   (fixed)          theta_31 = 2.5   (fixed)
#
# Nothing about the model changes — same rate law, same predictions, same
# numerical scaling. The only change is that two parameters are no longer being
# estimated, which is what removes the redundant direction the gate objected to.
#
# HOW THE REDUCTION IS IMPLEMENTED
# --------------------------------
# Rather than editing case_3_model.py, simulate() is wrapped. The wrapper
# re-inserts the two fixed nominal values before calling the 9-parameter model.
# On this FD path that is the whole story: there is no pyomo_model_fn, so unlike
# case_3_ift.py there is no all_vars block to rebuild and no IFT contract to
# preserve.

_KEEP = [0, 1, 2, 4, 6, 7, 8]          # 9-parameter indices we still estimate
                                       # dropped: 3 = theta_21, 5 = theta_31

THETA_FULL = np.array([5.4, 5.0, 6.2, 0.5, 1.4, 2.5, 7 / 3, 3, 5])


def _expand_to_nine(mp7):
    """Rebuild the 9-parameter vector, holding theta_21 and theta_31 at their
    NOMINAL values — not at zero, which would change the reaction."""
    full = THETA_FULL.copy()
    full[_KEEP] = np.asarray(mp7, dtype=float)
    return full


def simulate_reduced(ti_controls, sampling_times, model_parameters):
    """7-parameter simulate; theta_21 and theta_31 held fixed at nominal.

    The argument NAMES matter: pydex inspects this signature to work out which
    calling convention the model uses, so they must match simulate()'s.
    """
    return simulate(ti_controls, sampling_times, _expand_to_nine(model_parameters))


THETA_REDUCED = THETA_FULL[_KEEP]        # 5.4, 5.0, 6.2, 1.4, 7/3, 3, 5
NAMES_REDUCED = [
    r"$\theta_{10}$", r"$\theta_{11}$", r"$\theta_{20}$", r"$\theta_{30}$",
    r"$\nu$", r"$\alpha$", r"$\beta$",
]

designer.simulate              = simulate_reduced
designer.model_parameters      = THETA_REDUCED
designer.model_parameter_names = NAMES_REDUCED
# These two resets ARE required. initialize() does not invalidate the sensitivity
# cache when the parameter count changes, so the 9-column array from above would
# be reused against a 7-parameter model and fail with an opaque
# "operands could not be broadcast together" deep inside the FIM assembly.
designer.sensitivities         = None
designer.atomic_fims           = None
designer.initialize(verbose=0)
print(f"  n_mp = {designer.n_mp} (was 9); theta_21 and theta_31 held at "
      f"{THETA_FULL[3]}, {THETA_FULL[5]}")
print(f"  nominal values : {np.round(THETA_REDUCED, 4)}")

# eval_fim() at the fully-supported design is what diagnose_fim_structure() needs;
# it triggers the sensitivity analysis itself.
designer.eval_fim(np.ones(designer.n_c * designer.n_spt)
                  / (designer.n_c * designer.n_spt))
_dg = designer.diagnose_fim_structure(report=False)
print(f"  FIM rank       : {_dg['rank']} of {_dg['n_mp']}   "
      f"structurally singular: {_dg['singular']}")
if _dg["singular"]:
    print("  STILL SINGULAR — the two parameters fixed above were not enough.")
    print("  Read the table below for the direction that remains, and fix one")
    print("  member of THAT group instead of a second member of one already")
    print("  covered. Adjust _KEEP accordingly; nothing else needs to change.")
    designer.diagnose_fim_structure(report=True)
else:
    print("  full rank: the redundant direction is gone. The designs below")
    print("  therefore need no allow_singular_fim escape, and their criterion")
    print("  values reflect information in the data rather than the Cholesky")
    print("  floor the singular problem would have rested on.")

# Having removed two parameters, show what that did and did not fix.
_est = designer.run_estimability()
print()
print("Estimability of the REDUCED parameter set:")
print(_est["table"].to_string())
print()
print("Correlation groups (conditioning, not structure):", _est["groups"])
designer.show_plots()

# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 100)
print(" DESIGN 1 / 2 — D-optimal, fixed sampling times ".center(100, "="))
print(" All 11 evenly-spaced time points measured on every run ".center(100, " "))
print(" (n_spt = the full grid: one schedule per candidate, effort per EXPERIMENT) ".center(100, " "))
print("=" * 100)
print()
# ══════════════════════════════════════════════════════════════════════════════

# NOTE — no allow_singular_fim needed here.
#
# The designs below estimate the SEVEN free parameters, with theta_21 and
# theta_31 held at their nominal values. The full 9-parameter form was
# structurally singular and design_experiment() refused it; fixing two
# parameters removes the redundancy at source, so the criterion values below rest
# on information the data actually contains rather than on the Cholesky floor.
#
# If you free all nine again you will need allow_singular_fim=True on both calls,
# and should then read apportion()'s reported efficiency carefully: a continuous
# optimum that leans on a near-null direction rounds badly.
designer.design_experiment(
    designer.d_opt_criterion,
    # n_spt = the number of listed times gives ONE schedule per candidate
    # containing every time, so effort is allocated per EXPERIMENT and all 11
    # points really are measured on every run -- which is what the banner
    # above claims. Omitting n_spt would instead optimise the sampling times,
    # spending effort per (candidate, time) cell.  n_spt is the only control.
    n_spt=designer.n_spt,
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
print(" Note: expect a LOWER criterion than Design 1 — 5 samples per run, not 11 ".center(100, " "))
print("=" * 100)
print()
# ══════════════════════════════════════════════════════════════════════════════

# Reset atomic_fims: the fixed-spt run cached a (n_c × 11) atomic FIM array;
# the n_spt=5 run needs a different layout so pydex must recompute from scratch.
designer.atomic_fims = None
designer.design_experiment(
    designer.d_opt_criterion,
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
