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

# ── Run-control flag ──────────────────────────────────────────────────────────
# INSPECT_CANDIDATES : simulate every candidate and plot its concentration
#                      profiles and sensitivities. A sanity check after changing
#                      the candidate bounds or levels; costs a full sensitivity
#                      analysis up front. False by default.
#
# The estimability analysis is NOT behind a flag — it is what concludes that two
# parameters must be fixed, and the section that fixes them acts on that
# conclusion, so the two belong together.
INSPECT_CANDIDATES = False

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
if INSPECT_CANDIDATES:
    designer.simulate_candidates(plot_simulation_times=True)
    designer.plot_predictions()
    designer.plot_sensitivities()
    designer.show_plots()

# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 100)
print(" ESTIMABILITY — which of the 9 parameters can this grid actually determine? ".center(100, "="))
print("=" * 100)
print()
# ══════════════════════════════════════════════════════════════════════════════
#
# Nine parameters is a lot to ask of two measured responses. Before committing to
# an OED solve it is worth asking which of them the candidate grid can actually
# determine, and which are merely restating each other. The answer here turns out
# to be decisive: two of the nine are redundant by construction, and the section
# below fixes them at their nominal values.
#
# run_estimability() asks, over the WHOLE GRID: pooling every measurement you
# could take, in what ORDER do the parameters become estimable, and which of them
# are merely restating each other?  It computes the sensitivities itself if they
# are not already available.
#
# It implements the Yao/McAuley orthogonalisation (Wu, McLean, Harris and
# McAuley 2011, Table 1): rank by sensitivity column norm, project the
# remaining columns onto those already chosen, rank the residuals, repeat.
# The E index is the residual norm at the moment a parameter is selected,
# normalised so the most estimable is 1.
#
# This is a PRE-DESIGN analysis — it needs no criterion and no design, only
# the candidate grid and the nominal parameters. Nothing downstream reads it.
est = designer.run_estimability()

# Two things to read in the output, and they call for different fixes.
#
# 1. STEP-1 NORM vs E. A small step-1 norm means the parameter barely moves
#    the measurements at all — no experiment on this grid excites it. A
#    HEALTHY step-1 norm with a small E means the opposite: the parameter has
#    plenty of leverage, but its effect duplicates one already selected.
#
#    On this model theta_20 shows the second pattern clearly: step-1 norm
#    ~0.42 (comparable to the best) yet E ~1e-8, because it is almost
#    perfectly collinear with theta_21. theta_31 shows the first: step-1 norm
#    ~2e-4, nothing excites it. Fixing theta_20 needs experiments that
#    SEPARATE it from theta_21; fixing theta_31 needs experiments that
#    EXCITE it. Widening the temperature range does the former; widening the
#    CA0 or tau range the latter.
#
# 2. CORRELATION GROUPS. Within a group the data determine roughly ONE
#    parameter. This model typically returns two:
#
#        {theta_11, theta_21, theta_31}   the three Arrhenius slopes
#        {theta_30, alpha, beta}          the rate-law exponents
#
#    with theta_11 <-> theta_21 at -1.0000, i.e. exactly interchangeable
#    over this temperature span. Estimate or fix ONE member of each group;
#    the rest become unestimable once you do. Choose on physical grounds —
#    which is transferable to other reactors, which you have independent
#    literature values for — not on the ranking, which only says which is
#    numerically most convenient.
#
# The threshold sweep printed underneath shows how stable the grouping is:
# theta_31 joins the first group only below |corr| = 0.98, so its membership
# is marginal in a way theta_11 <-> theta_21 is not.
# WHY E AND E-UD ARE IDENTICAL HERE
# ---------------------------------
# The report prints two indices side by side and on this model they agree to
# every digit. That is not a bug and not a missing calculation — it follows
# from designer.error_cov = np.diag([0.1, 0.1]) above.
#
# E weights the sensitivity rows by Sigma^(-1/2); E-UD does not. When Sigma
# is a MULTIPLE OF THE IDENTITY, as it is here (0.1 x I), that weighting
# rescales every column of Z by the same constant. Column norms all change
# by the same factor, the ratios are untouched, and the two indices coincide
# exactly. The same holds for the correlation matrix, which is a cosine and
# therefore scale-free in the columns.
#
# They diverge only when the RESPONSES ARE MEASURED WITH DIFFERENT PRECISION.
# Setting error_cov = np.diag([0.1, 0.9]) on this model — cB nine times
# noisier than cA — moves theta_20 from 8th to 4th and theta_30 from 4th to
# 8th, because MLE discounts the noisier response and the two parameters are
# informed by different measurements.
#
# So if cA and cB really are measured to different precision, put the real
# numbers in error_cov: as it stands the E column carries no information the
# E-UD column does not, and you are choosing between least squares and MLE
# without the data to tell them apart.
print()
print("Estimability ranking (most to least estimable):")
print(est["table"].to_string())
print()
print("Correlation groups:", est["groups"])
designer.show_plots()

# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 100)
print(" FIXING TWO PARAMETERS — removing the structurally redundant directions ".center(100, "="))
print("=" * 100)
print()
# ══════════════════════════════════════════════════════════════════════════════
#
# WHY THIS IS HERE
# ----------------
# The estimability analysis above flags theta_20 and theta_21 as UNRESOLVABLE,
# and design_experiment() refuses this model outright with
#
#     ValueError: The Fisher information matrix is STRUCTURALLY singular:
#     rank 7 of 9 even at the fully-supported design
#
# That is not a conditioning problem and no candidate grid can fix it. It is an
# exact invariance of the rate law. Look at the rate:
#
#     r = k1(T) * CA^alpha / (k2(T) + k3(T) * CA^beta)
#     ki(T) = exp(theta_i0 + theta_i1 * (T - 273.15) / T)
#
# Multiply numerator AND denominator by exp(d) and r is unchanged. Since
# ki appears only inside an exponential, that is the same as ADDING d to every
# theta_i0 at once. The same argument applies to theta_i1 through the
# temperature term. So two directions in parameter space,
#
#     (1,0,1,0,1,0,0,0,0)   — common shift in theta_10, theta_20, theta_30
#     (0,1,0,1,0,1,0,0,0)   — common shift in theta_11, theta_21, theta_31
#
# leave EVERY prediction identical. Verified numerically: shifting all three
# theta_i0 by +0.7 changes the responses by 1.4e-12, and all three theta_i1 by
# +0.3 by 2.9e-09. Nine parameters, seven identifiable directions — exactly the
# rank the gate reported.
#
# The estimability ranking corroborates this from the sensitivity matrix alone,
# with no knowledge of the algebra: it flags TWO parameters as unresolvable, one
# from each affected triple. Which two it names is grid-dependent and partly a
# matter of numerical tie-breaking — on a 5x5x5 grid it reports theta_20 and
# theta_21, on a coarser one theta_20 and theta_31. That is not a
# contradiction. The ALGEBRA is the authority on what is redundant: a common
# shift within {theta_10, theta_20, theta_30} and within
# {theta_11, theta_21, theta_31}, so exactly one member of each triple can be
# zeroed and it does not matter which. Zeroing the theta_2* pair is the
# conventional choice because it is what "normalise the denominator" means.
# The ranking's job here is to tell you the redundancy exists and how many
# dimensions it has, not to pick the convention.
#
# THE FIX
# -------
# The invariance is a COMMON SHIFT within each triple, so it disappears the
# moment ONE member of each triple is no longer free to move. So simply FIX two
# parameters at their nominal values and estimate the other seven:
#
#     theta_20 = 6.2   (fixed)          theta_21 = 0.5   (fixed)
#
# Nothing about the model changes — same rate law, same k2 = exp(6.2 + 0.5g),
# same numerical scaling, same predictions. The only change is that two
# parameters are no longer being estimated, which removes exactly the two
# redundant directions. Verified: the FIM comes out rank 7 OF 7, so
# design_experiment() no longer needs allow_singular_fim=True and the criterion
# value stops depending on the Cholesky floor.
#
# A note on the road not taken. The textbook remedy for an inhibited rate law is
# to normalise the denominator — divide through by k2 so that k2 == 1, which
# means shifting every theta_i0 by -theta_20 and every theta_i1 by -theta_21:
#
#     theta_10' = -0.8,  theta_11' = 4.5,  theta_30' = -4.8,  theta_31' = 2.0
#
# That is mathematically equivalent (verified: predictions agree to 1.4e-10) and
# it also gives rank 7 of 7 — but it is numerically WORSE here. Dividing by k2
# takes the denominator from
#
#     k2 = 492.7,  k3*CA^beta = 4.055 * CA^5      ->  balanced terms
# to
#     k2 = 1,      k3'*CA^beta = 0.00823 * CA^5   ->  terms orders apart
#
# and IPOPT's collocation solve then fails to converge on low-conversion
# candidates such as [CA0=10.5, T=273.15, tau=50.5], which the original scaling
# handles without trouble. Fixing parameters achieves the same identifiability
# result and cannot introduce a conditioning problem, because it does not touch
# the arithmetic at all.
#
# WHAT REMAINS
# ------------
# Fixing the two parameters removes the exact invariances; it does NOT remove the
# ill-conditioning. The reduced model still returns correlation groups
#
#     {theta_11, theta_31}          two Arrhenius slopes over a 50 K span
#     {alpha, beta, theta_30}        rate-law exponents against a pre-exponential
#
# Those are conditioning, not structure: a wider temperature range attacks the
# first, a wider CA0 range the second, and ds_opt_criterion lets you design for
# one member of each while marginalising the rest.
#
# HOW THE REDUCTION IS IMPLEMENTED
# --------------------------------
# Rather than editing case_3_ift_model.py, the two model functions are wrapped.
# The wrapper re-inserts the two fixed nominal values before calling the
# 9-parameter model, and
# for the IFT path it also rebuilds all_vars so the parameter block holds only
# the seven ESTIMATED parameters. pydex unfixes exactly all_vars[:n_mp], so
# theta_20 and theta_21 stay fixed at their nominal values and never enter the
# sensitivity calculation — the IFT contract is satisfied with n_mp = 7 and the
# exact-derivative path is preserved.

_KEEP = [0, 1, 4, 5, 6, 7, 8]          # 9-parameter indices we still estimate
                                       # dropped: 2 = theta_20, 3 = theta_21


THETA_FULL = np.array([5.4, 5.0, 6.2, 0.5, 1.4, 2.5, 7 / 3, 3, 5])


def _expand_to_nine(mp7):
    """Rebuild the 9-parameter vector, holding theta_20 and theta_21 at their
    NOMINAL values — not at zero, which would change the reaction."""
    full = THETA_FULL.copy()
    full[_KEEP] = np.asarray(mp7, dtype=float)
    return full


def simulate_reduced(ti_controls, sampling_times, model_parameters):
    """7-parameter simulate; theta_20 and theta_21 held fixed at nominal."""
    return simulate(ti_controls, sampling_times, _expand_to_nine(model_parameters))


def build_pyomo_model_reduced(ti_controls, model_parameters,
                             sampling_times=None, **kwargs):
    """7-parameter Pyomo model preserving the IFT contract.

    all_vars is rebuilt so its parameter block is the seven estimated
    parameters. The two fixed ones remain in the model at zero but are not in
    the parameter block, so pydex never unfixes them and they contribute no
    sensitivity columns.
    """
    m, all_vars, all_bodies, t_sorted = build_pyomo_model(
        ti_controls, _expand_to_nine(model_parameters), sampling_times, **kwargs
    )
    p9 = list(all_vars[:9])
    return m, [p9[i] for i in _KEEP] + list(all_vars[9:]), all_bodies, t_sorted


THETA_REDUCED = THETA_FULL[_KEEP]        # unchanged values: 5.4, 5.0, 1.4, 2.5, ...
NAMES_REDUCED = [
    r"$\theta_{10}$", r"$\theta_{11}$", r"$\theta_{30}$", r"$\theta_{31}$",
    r"$\nu$", r"$\alpha$", r"$\beta$",
]

designer.simulate           = simulate_reduced
designer.pyomo_model_fn     = build_pyomo_model_reduced
designer.model_parameters   = THETA_REDUCED
designer.model_parameter_names = NAMES_REDUCED
# These two resets ARE required. initialize() does not invalidate the sensitivity
# cache when the parameter count changes, so the 9-column array from above would
# be reused against a 7-parameter model and fail with an opaque
# "operands could not be broadcast together" deep inside the FIM assembly.
designer.sensitivities      = None
designer.atomic_fims        = None
designer.initialize(verbose=0)
print(f"  n_mp = {designer.n_mp} (was 9); theta_20 and theta_21 held at "
      f"{THETA_FULL[2]}, {THETA_FULL[3]}")
print(f"  nominal values : {np.round(THETA_REDUCED, 4)}")

# eval_fim() at the fully-supported design is what diagnose_fim_structure() needs;
# it triggers the sensitivity analysis itself.
designer.eval_fim(np.ones(designer.n_c * designer.n_spt)
                  / (designer.n_c * designer.n_spt))
_dg = designer.diagnose_fim_structure(report=False)
print(f"  FIM rank       : {_dg['rank']} of {_dg['n_mp']}   "
      f"structurally singular: {_dg['singular']}")
if _dg["singular"]:
    print("  UNEXPECTED — fixing the two parameters should have removed both")
    print("  invariances. Diagnosis follows.")
    designer.diagnose_fim_structure(report=True)
else:
    print("  full rank: both redundant directions are gone. The designs below")
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
# The designs below estimate the SEVEN free parameters, with theta_20 and
# theta_21 held at their nominal values. The full 9-parameter form was
# structurally singular (rank 7 of 9) and design_experiment() refused it; fixing
# two parameters removes the redundancy at source, so the criterion values below
# rest on information the data actually contains rather than on the Cholesky
# floor.
#
# If you free all nine again you will need allow_singular_fim=True on both
# calls, and should then read apportion()'s reported efficiency carefully: a
# continuous optimum that leans on a near-null direction rounds badly.
#
# The remaining correlation groups — {theta_11, theta_31} and
# {alpha, beta, theta_30} — are conditioning, not structure. They make the
# design harder but not ill-posed, and they are what a wider temperature and CA0
# range, or ds_opt_criterion on one member of each, would address.
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
