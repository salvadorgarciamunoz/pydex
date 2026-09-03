"""
van_laar_criteria.py
====================
D-, A- and E-optimal designs for the same van Laar problem, then the same
problem designed under UNCERTAINTY in the parameters (pseudo-Bayesian).

Companion to van_laar_design.py, which designs once, D-optimally, at a single
nominal parameter vector. This script answers the two questions that raises:

    1. What changes if I pick a different optimality criterion?
    2. van_laar_design.py shows the design MOVING when the nominal parameters
       move. If I do not know the parameters well, what should I design?

HOW THIS SCRIPT IS WRITTEN
--------------------------
Every design is built, solved and read out IN FULL, in its own numbered
section, with nothing factored into helper functions. That makes the file
long on purpose: you can read any one section top to bottom, or copy it
straight into your own work, without chasing a helper. The sections differ
only in the criterion and in the parameter argument, and seeing them written
out side by side is the lesson.

    Section 0   the problem: grid, noise model, nominal parameters
    Section 1   D-optimal      build -> solve -> retrieve, spelled out
    Section 2   A-optimal      the same, one line different
    Section 3   E-optimal      the same, one line different
    Section 4   the three designs side by side
    Section 5   rounding a continuous design to a real budget
    Section 6   pseudo-Bayesian D-optimal, type 0 (average the INFORMATION)
    Section 7   pseudo-Bayesian D-optimal, type 1 (average the CRITERION)
    Section 8   every design scored by every criterion
    Section 9   the figure: parameter confidence regions

TWO TRAPS THIS SCRIPT DEMONSTRATES RATHER THAN DESCRIBES
--------------------------------------------------------
1. CRITERION VALUES ARE NOT COMPARABLE BETWEEN CRITERIA.
   `designer._criterion_value` is the NEGATED Pyomo objective, for every
   criterion. For D-optimality that gives a natural +log det(FIM); for a
   minimised-positive criterion like A it gives a NEGATIVE number for a
   quantity that cannot be negative. Measured here: D reports 16.78, A
   reports -5.1e-04, E reports 2651.05. Three different quantities on three
   different scales. Compare DESIGNS (Section 4), or score every design
   through one common criterion (Section 8).

2. `pseudo_bayesian_type` IS A `design_experiment()` KEYWORD.
   Setting `designer._pseudo_bayesian_type` on the instance is silently
   overwritten by the keyword default, and the run then reports the wrong
   type with no error at all. The default when the keyword is omitted is
   type 0, average-information -- despite the docstring calling the argument
   "Required" for a scenario array, which it is not. That mistake produced two identical
   "different" designs while this script was being written. Sections 6 and 7
   pass it as a keyword.

Sampling times do not appear anywhere here: the model has no time axis, so
`n_spt` does not apply. See `examples/ode/` for that.

MEASURED RESULTS (pydex 0.7.5, 21 candidates, theta = [1.65, 0.95])
-------------------------------------------------------------------
Local designs, effort per condition and that effort rounded to 8 runs:

                        candidate 3         candidate 9
                        x1=0.05, T=80       x1=0.35, T=80    criterion value
    D-optimal           47.46%  4 runs      52.54%  4 runs        16.782682
    A-optimal           43.53%  4 runs      56.46%  4 runs        -0.000515
    E-optimal           42.10%  3 runs      57.90%  5 runs      2651.046115

    rounding cost: D 99.88%, A 98.50%, E 98.75% as informative as continuous

All three chose the SAME two experiments here and differed only in how they
split effort between them. That is an observation about this grid, not a
rule: the support of a D-optimal design on p parameters is at least p and, by
Caratheodory, at most p(p+1)/2 -- so 2 or 3 points were both available and
nothing forced the three criteria to agree. The split is nonetheless enough
to change the ROUNDED design: E gives 3 and 5 where D and A give 4 and 4.

Pseudo-Bayesian designs, 40 scenarios drawn about the nominals:

    D-optimal, pb type 0   46.64%   53.36%    16.541379
    D-optimal, pb type 1   46.98%   53.02%    16.397751

Type 0 at or above type 1 is the expected direction, since
log det(mean FIM) >= mean(log det FIM).

Section 8's cross-scoring. All three criteria are MINIMISED, so lower is
better DOWN a column; '<-' marks the winner:

    design                 d_opt_criterion   a_opt_criterion   e_opt_criterion
    D-optimal               -1.678268e+01 <-   5.175329e-04     -2.608845e+03
    A-optimal               -1.677700e+01      5.146692e-04 <-  -2.647850e+03
    E-optimal               -1.677221e+01      5.150384e-04     -2.651046e+03 <-
    D-optimal, pb type 0    -1.678243e+01      5.164482e-04     -2.620637e+03
    D-optimal, pb type 1    -1.678260e+01      5.168728e-04     -2.615905e+03

Each local design is best on EXACTLY the criterion it was optimised for and
worse on the other two, by 0.03% to 1.59%. On this problem the criterion is a
refinement, not a different experiment. Do not generalise that from a
2-parameter model.

Note D's d_opt score of -16.78268 is the NEGATION of the 16.782682 reported
as its criterion value. That is trap 1, made visible.

Run directly:   python van_laar_criteria.py
"""

import numpy as np
from matplotlib import pyplot as plt

from pydex.core.designer import Designer

from van_laar_model import simulate


# =============================================================================
# SECTION 0 — The problem, defined once
#
# What every design below shares: the candidate grid, the measurement noise
# and the labels. Written as module constants rather than a factory function
# so that each section can be read on its own.
# =============================================================================

# Candidate experiments: 7 liquid compositions x 3 temperatures = 21 points.
GRID_BOUNDS = [
    [0.05, 0.95],      # x1, liquid mole fraction of component 1
    [40.0, 80.0],      # T, degC
]
GRID_LEVELS = [7, 3]

# van Laar activity-coefficient parameters. The Antoine constants are known
# rather than fitted, so these two are the whole estimation problem.
THETA_NOM = np.array([1.65, 0.95])

# Total pressure (kPa) and vapour composition live on very different scales.
# The identity default would treat a 0.3 kPa pressure error and a 0.004
# mole-fraction error as equally serious; error_cov is how you say they are
# not. A NON-UNIFORM error_cov can change which design is optimal, not merely
# rescale the criterion value.
ERROR_COV = np.diag([0.3 ** 2, 0.004 ** 2])

BUDGET = 8         # experiments we can actually afford, used in Section 5

print()
print("=" * 78)
print("  van Laar: D- vs A- vs E-optimal, locally and under uncertainty")
print("=" * 78)


# =============================================================================
# SECTION 1 — D-optimal design at the nominal parameters
#
# Spelled out in full. Sections 2 and 3 repeat this verbatim except for one
# line, so that the difference between the three criteria is visible as one
# line of code.
#
# D-optimality maximises log det(FIM), which minimises the VOLUME of the
# parameter confidence region. It is the usual default.
# =============================================================================
print("\n" + "-" * 78)
print("  SECTION 1 — D-optimal design")
print("-" * 78)

designer_d = Designer()

# --- BUILD -------------------------------------------------------------------

# The model. pydex inspects this function's SIGNATURE to decide the calling
# convention, so the argument names matter: (ti_controls, model_parameters)
# marks a static model. Renaming them raises a SyntaxError from pydex.
designer_d.simulate = simulate

# Where in parameter space we are designing. A local design is optimal AT
# this point and nowhere in particular else, which is what Sections 6 and 7
# are about.
designer_d.model_parameters = THETA_NOM

# The candidate pool: every experiment we are willing to consider.
designer_d.ti_controls_candidates = designer_d.enumerate_candidates(
    bounds=GRID_BOUNDS,
    levels=GRID_LEVELS,
)

designer_d.error_cov = ERROR_COV

# Labels. Set these. Anything left unset gets a generated default such as
# "Time-invariant Control 0", so an unlabelled design still prints something
# plausible -- which is exactly why it goes unnoticed.
designer_d.model_parameter_names = ["A12", "A21"]
designer_d.ti_controls_names = ["x1", "T"]
designer_d.response_names = ["P", "y1"]

designer_d.initialize(verbose=0)

# --- SOLVE -------------------------------------------------------------------

designer_d.design_experiment(
    designer_d.d_opt_criterion,      # the criterion, passed by reference
    solver="ipopt",
    write=False,                     # do not write a log file to disk
)

# --- RETRIEVE ----------------------------------------------------------------
# Capture the results before reporting them: print_optimal_candidates_table()
# trims sub-tolerance efforts and renormalises designer.efforts in place.

d_criterion_value = float(designer_d._criterion_value)
d_efforts = np.array(designer_d.efforts, copy=True)
d_fim = np.array(designer_d.fim, copy=True)      # FIM of the optimal design

print(f"\n  criterion value (log det FIM) : {d_criterion_value:.6f}")
print(f"  efforts array shape           : {d_efforts.shape}")
print(f"  FIM shape                     : {d_fim.shape}")

# The human-readable report. print_optimal_candidates_table() is the "what do
# I actually run" view: one row per suggested experiment.
print()
designer_d.print_optimal_candidates_table()


# =============================================================================
# SECTION 2 — A-optimal design at the nominal parameters
#
# Identical to Section 1 except for the criterion on the design_experiment
# call. A-optimality minimises trace(FIM^-1), the SUM of the parameter
# variances. It cares about total uncertainty rather than the volume of the
# region, so it is less willing to accept one badly-determined direction in
# exchange for a tight region overall.
# =============================================================================
print("\n" + "-" * 78)
print("  SECTION 2 — A-optimal design")
print("-" * 78)

# --- BUILD -------------------------------------------------------------------
designer_a = Designer()
designer_a.simulate = simulate
designer_a.model_parameters = THETA_NOM
designer_a.ti_controls_candidates = designer_a.enumerate_candidates(
    bounds=GRID_BOUNDS,
    levels=GRID_LEVELS,
)
designer_a.error_cov = ERROR_COV
designer_a.model_parameter_names = ["A12", "A21"]
designer_a.ti_controls_names = ["x1", "T"]
designer_a.response_names = ["P", "y1"]
designer_a.initialize(verbose=0)

# --- SOLVE -------------------------------------------------------------------
designer_a.design_experiment(
    designer_a.a_opt_criterion,      # <-- THE ONLY DIFFERENCE from Section 1
    solver="ipopt",
    write=False,
)

# --- RETRIEVE ----------------------------------------------------------------
a_criterion_value = float(designer_a._criterion_value)
a_efforts = np.array(designer_a.efforts, copy=True)
a_fim = np.array(designer_a.fim, copy=True)

print(f"\n  criterion value : {a_criterion_value:.6e}")
print("  ^ NEGATIVE, for a sum of variances that cannot be negative.")
print("    _criterion_value negates the Pyomo objective for every criterion,")
print("    which reads naturally for D and not at all for A. Trap 1.")
print()
designer_a.print_optimal_candidates_table()


# =============================================================================
# SECTION 3 — E-optimal design at the nominal parameters
#
# Identical again, criterion aside. E-optimality maximises the SMALLEST
# eigenvalue of the FIM, i.e. minimises the LONGEST axis of the confidence
# region. It is the pessimist's criterion: it improves the worst-determined
# direction and does not care what that costs elsewhere.
# =============================================================================
print("\n" + "-" * 78)
print("  SECTION 3 — E-optimal design")
print("-" * 78)

# --- BUILD -------------------------------------------------------------------
designer_e = Designer()
designer_e.simulate = simulate
designer_e.model_parameters = THETA_NOM
designer_e.ti_controls_candidates = designer_e.enumerate_candidates(
    bounds=GRID_BOUNDS,
    levels=GRID_LEVELS,
)
designer_e.error_cov = ERROR_COV
designer_e.model_parameter_names = ["A12", "A21"]
designer_e.ti_controls_names = ["x1", "T"]
designer_e.response_names = ["P", "y1"]
designer_e.initialize(verbose=0)

# --- SOLVE -------------------------------------------------------------------
designer_e.design_experiment(
    designer_e.e_opt_criterion,      # <-- THE ONLY DIFFERENCE from Section 1
    solver="ipopt",
    write=False,
)

# --- RETRIEVE ----------------------------------------------------------------
e_criterion_value = float(designer_e._criterion_value)
e_efforts = np.array(designer_e.efforts, copy=True)
e_fim = np.array(designer_e.fim, copy=True)

print(f"\n  criterion value (smallest eigenvalue of FIM) : "
      f"{e_criterion_value:.6f}")
print()
designer_e.print_optimal_candidates_table()


# =============================================================================
# SECTION 4 — The three designs side by side
#
# Read out of the arrays captured above, with the CONDITIONS printed in the
# header: "candidate 8" on its own is unreadable, and a bare list of numbers
# cannot tell you which condition each belongs to.
#
# The supported candidates are listed in ASCENDING CANDIDATE INDEX, because
# that is the order apportion() returns its counts in (Section 5). Sorting by
# effort instead would silently mis-align the two when printed together.
# =============================================================================
print("\n" + "-" * 78)
print("  SECTION 4 — the three local designs side by side")
print("-" * 78)

tic = np.asarray(designer_d.ti_controls_candidates, dtype=float)

supported = sorted({
    int(i)
    for efforts in (d_efforts, a_efforts, e_efforts)
    for i in np.flatnonzero(np.asarray(efforts).flatten() > 1e-3)
})

header = "  " + " " * 14
subhead = "  " + " " * 14
for i in supported:
    # +1: print_optimal_candidates_table() reports the 1-INDEXED pool
    # position, so match it rather than printing raw numpy indices.
    header += f"candidate {i + 1}".ljust(22)
    subhead += f"x1={tic[i][0]:.2f}, T={tic[i][1]:.0f}".ljust(22)
print()
print(header + "criterion value")
print(subhead)
print("  " + "-" * (14 + 22 * len(supported) + 15))

for label, efforts, value in (("D-optimal", d_efforts, d_criterion_value),
                              ("A-optimal", a_efforts, a_criterion_value),
                              ("E-optimal", e_efforts, e_criterion_value)):
    flat = np.asarray(efforts).flatten()
    row = f"  {label:<14s}"
    for i in supported:
        row += f"{flat[i]:.2%}".ljust(22)
    print(row + f"{value:>15.6f}")

print("\n  The three criteria chose the SAME two experiments and differ only")
print("  in how they split effort between them. The last column is NOT")
print("  comparable down the rows -- three quantities, three scales.")


# =============================================================================
# SECTION 5 — Rounding a continuous design to a real budget
#
# design_experiment() allocates a UNIT budget: fractional efforts summing to
# 1, independent of how many runs you can afford. Your budget is not an input
# to the optimisation and often is not known when you solve.
#
# apportion(n) rounds an already-solved design to n actual experiments and
# reports what the rounding cost. This is where the effort split in Section 4
# stops being a decimal and becomes a different experimental protocol.
#
# apportion() reads the design currently held on the designer, so the three
# designers from Sections 1-3 are reused directly here -- there is no need to
# rebuild or re-solve anything. The efforts captured in those sections are
# assigned back first, so what gets rounded is the solver's own output.
# =============================================================================
print("\n" + "-" * 78)
print(f"  SECTION 5 — rounding to {BUDGET} experiments with apportion()")
print("-" * 78)

for label, designer, captured_efforts in (
        ("D-optimal", designer_d, d_efforts),
        ("A-optimal", designer_a, a_efforts),
        ("E-optimal", designer_e, e_efforts)):

    print(f"\n  {label}")

    designer.efforts = np.array(captured_efforts, copy=True)

    # compute_actual_efficiency is TRI-STATE. The default None means "compute
    # it only if it will be reported", and at verbose=0 the whole report block
    # is skipped -- so the efficiency silently comes back None. Force it.
    apportionment = designer.apportion(BUDGET, compute_actual_efficiency=True)

    # One entry per supported candidate, in ascending candidate index. Total
    # each with nansum: the return is a RAGGED object array when the support
    # is ragged, and astype(int) raises on that.
    runs = [int(np.nansum(a)) for a in apportionment]
    efficiency = float(np.squeeze(designer.rounding_efficiency))

    for i, r in zip(supported, runs):
        print(f"    candidate {i + 1} (x1={tic[i][0]:.2f}, T={tic[i][1]:.0f}): "
              f"{r} run{'s' if r != 1 else ''}")
    print(f"    the rounded design is {efficiency:.2%} as informative as the "
          f"continuous one")

print(f"\n  E-optimal splits {BUDGET} runs differently from D and A: the effort")
print("  difference in Section 4 becoming a different protocol.")


# =============================================================================
# SECTION 6 — Pseudo-Bayesian D-optimal, TYPE 0
#
# Sections 1-3 designed at ONE parameter vector. van_laar_design.py shows the
# support moving to different candidates when the nominals move, so a local
# design is only as good as the guess it was built on.
#
# A pseudo-Bayesian design takes a SET of parameter scenarios and designs for
# all of them at once. Two things change from Section 1, and nothing else:
#
#     * model_parameters becomes 2-D, (n_scenarios, n_parameters)
#     * design_experiment() takes pseudo_bayesian_type
#
# TYPE 0 averages the INFORMATION: it builds the mean FIM over the scenarios
# and applies the criterion to that. It solves natively in Pyomo.
# =============================================================================
print("\n" + "-" * 78)
print("  SECTION 6 — pseudo-Bayesian D-optimal, type 0 (average information)")
print("-" * 78)

N_SCENARIOS = 40
SCENARIO_SEED = 7
THETA_SPREAD = np.array([0.25, 0.20])    # not calibrated to any real prior

rng = np.random.default_rng(SCENARIO_SEED)
scenarios = np.column_stack([
    rng.normal(THETA_NOM[0], THETA_SPREAD[0], N_SCENARIOS),
    rng.normal(THETA_NOM[1], THETA_SPREAD[1], N_SCENARIOS),
])

print(f"\n  {N_SCENARIOS} scenarios, array shape {scenarios.shape}, "
      f"seed {SCENARIO_SEED}")
print(f"  A12 spans {scenarios[:, 0].min():.3f} to {scenarios[:, 0].max():.3f}")
print(f"  A21 spans {scenarios[:, 1].min():.3f} to {scenarios[:, 1].max():.3f}")

# --- BUILD -------------------------------------------------------------------
designer_pb0 = Designer()
designer_pb0.simulate = simulate
# A 2-D array here is what TRIGGERS the pseudo-Bayesian calculation: pydex
# reads model_parameters.ndim and sets _pseudo_bayesian from it, so the shape
# of this one array is the whole switch. 1-D (n_mp,) = local design;
# 2-D (n_scr, n_mp) = pseudo-Bayesian, with n_scr scenarios of n_mp parameters.
designer_pb0.model_parameters = scenarios       # <-- 2-D: triggers pseudo-Bayesian
designer_pb0.ti_controls_candidates = designer_pb0.enumerate_candidates(
    bounds=GRID_BOUNDS,
    levels=GRID_LEVELS,
)
designer_pb0.error_cov = ERROR_COV
designer_pb0.model_parameter_names = ["A12", "A21"]
designer_pb0.ti_controls_names = ["x1", "T"]
designer_pb0.response_names = ["P", "y1"]
designer_pb0.initialize(verbose=0)

# --- SOLVE -------------------------------------------------------------------
designer_pb0.design_experiment(
    designer_pb0.d_opt_criterion,
    solver="ipopt",
    write=False,
    # A KEYWORD, not an attribute -- see trap 2. Note 0 is also what you get
    # by omitting this argument entirely, so type 0 is the silent default.
    pseudo_bayesian_type=0,
)

# --- RETRIEVE ----------------------------------------------------------------
pb0_criterion_value = float(designer_pb0._criterion_value)
pb0_efforts = np.array(designer_pb0.efforts, copy=True)

print(f"\n  criterion value : {pb0_criterion_value:.6f}")
print()
designer_pb0.print_optimal_candidates_table()


# =============================================================================
# SECTION 7 — Pseudo-Bayesian D-optimal, TYPE 1
#
# TYPE 1 averages the CRITERION: it evaluates log det(FIM) for each scenario
# and averages those numbers. Type 0 averages the matrices; type 1 averages
# the scores. Jensen's inequality means type 0 comes out at or above type 1.
#
# Type 1 also falls back to scipy SLSQP rather than solving natively, because
# the native path requires `(not is_pb or _pb_type0)`. On a larger problem
# that is the slower route.
# =============================================================================
print("\n" + "-" * 78)
print("  SECTION 7 — pseudo-Bayesian D-optimal, type 1 (average criterion)")
print("-" * 78)

# --- BUILD -------------------------------------------------------------------
designer_pb1 = Designer()
designer_pb1.simulate = simulate
designer_pb1.model_parameters = scenarios
designer_pb1.ti_controls_candidates = designer_pb1.enumerate_candidates(
    bounds=GRID_BOUNDS,
    levels=GRID_LEVELS,
)
designer_pb1.error_cov = ERROR_COV
designer_pb1.model_parameter_names = ["A12", "A21"]
designer_pb1.ti_controls_names = ["x1", "T"]
designer_pb1.response_names = ["P", "y1"]
designer_pb1.initialize(verbose=0)

# --- SOLVE -------------------------------------------------------------------
designer_pb1.design_experiment(
    designer_pb1.d_opt_criterion,
    solver="ipopt",
    write=False,
    pseudo_bayesian_type=1,          # <-- the only difference from Section 6
)

# --- RETRIEVE ----------------------------------------------------------------
pb1_criterion_value = float(designer_pb1._criterion_value)
pb1_efforts = np.array(designer_pb1.efforts, copy=True)

print(f"\n  criterion value, type 1 : {pb1_criterion_value:.6f}")
print(f"  criterion value, type 0 : {pb0_criterion_value:.6f}")
print("  type 0 >= type 1 is expected: log det(mean FIM) >= mean(log det FIM)")
print()
designer_pb1.print_optimal_candidates_table()


# =============================================================================
# SECTION 8 — Every design scored by every criterion
#
# Section 4's last column could not be compared across rows, because each
# design was reported in its own criterion's units. The fix is to score every
# design through the SAME criterion, and pydex will do that for you:
#
#     eval_fim(efforts)                 assemble the FIM for this design
#     compute_criterion_value(crit)     evaluate a criterion on it
#
# There is nothing to reimplement here: no determinant, no trace, no
# eigenvalues. All three criteria are MINIMISED, so lower is better DOWN a
# column. Values are still not comparable ACROSS columns.
#
# Note this reuses designer_d rather than building a scorer -- see below.
# =============================================================================
print("\n" + "-" * 78)
print("  SECTION 8 — every design scored by every criterion")
print("-" * 78)

all_designs = [
    ("D-optimal", d_efforts),
    ("A-optimal", a_efforts),
    ("E-optimal", e_efforts),
    ("D-optimal, pb type 0", pb0_efforts),
    ("D-optimal, pb type 1", pb1_efforts),
]

# One scorer suffices, and we already have it: designer_d was built at the
# NOMINAL parameters, so its cached atomic FIMs are exactly what is needed to
# score any design on the same footing. Put a design's efforts in, ask
# eval_fim() to assemble the FIM, then ask each criterion what it makes of it.
# No rebuild, no re-solve.
#
# The pseudo-Bayesian designs are scored here at the nominal parameters too,
# which is one lens on them rather than the quantity they were optimised for.
scorer = designer_d

scores = {}
for label, efforts in all_designs:
    # eval_fim() takes the effort-weighted sum of the candidates' atomic FIMs
    # and stores the result on scorer.fim -- which is what the criteria read.
    scorer.efforts = np.asarray(efforts, dtype=float)
    scorer.eval_fim(np.asarray(efforts, dtype=float))

    scores[label] = (
        float(scorer.compute_criterion_value(scorer.d_opt_criterion)),
        float(scorer.compute_criterion_value(scorer.a_opt_criterion)),
        float(scorer.compute_criterion_value(scorer.e_opt_criterion)),
        np.array(scorer.fim, copy=True),
    )

print("\n  All three criteria are MINIMISED: LOWER IS BETTER down a column.")
print("  '<-' marks the winner. Columns are not comparable with each other.")
print()
print(f"  {'design':<22s}{'d_opt_criterion':>18s}{'a_opt_criterion':>18s}"
      f"{'e_opt_criterion':>18s}")

best = [min(scores[label][j] for label in scores) for j in range(3)]
for label in scores:
    row = f"  {label:<22s}"
    for j in range(3):
        value = scores[label][j]
        row += f"{value:>16.6e}" + ("  <-" if value == best[j] else "    ")
    print(row)

winners = [min(scores, key=lambda k: scores[k][j]) for j in range(3)]
print(f"\n  best by d_opt -> {winners[0]}")
print(f"  best by a_opt -> {winners[1]}")
print(f"  best by e_opt -> {winners[2]}")
print("  Each local design wins EXACTLY the criterion it was optimised for.")


# =============================================================================
# SECTION 9 — The figure: parameter confidence regions
#
# With exactly two parameters, FIM^-1 is a 2x2 covariance, so the approximate
# confidence region is an ellipse that can simply be drawn. That makes the
# three criteria concrete, because each minimises a different feature of the
# SAME ellipse:
#
#     D-optimal    det(FIM^-1)          the AREA
#     A-optimal    trace(FIM^-1)        the SUM of squared semi-axes
#     E-optimal    lambda_max(FIM^-1)   the LONGEST semi-axis
#
# pydex has ten plot_* methods and none draws a confidence region, so this
# section is ordinary matplotlib applied to the FIMs captured in Section 8.
# The figure is DISPLAYED, not saved: generated PNGs are not committed in this
# repository.
#
# The ellipse axes are RELATIVE parameter deviations, because pydex normalises
# sensitivities by parameter magnitude by default (_norm_sens_by_params). Any
# closed-form comparison has to apply the same scaling before it means
# anything.
# =============================================================================
print("\n" + "-" * 78)
print("  SECTION 9 — confidence regions")
print("-" * 78)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 5.2))

N_SIGMA = 2.0
for (label, _), style in zip(all_designs, ["-", "-", "-", "--", ":"]):
    fim = scores[label][3]

    # The ellipse {x : x^T FIM x <= N_SIGMA^2}, traced from the eigen-
    # decomposition: semi-axis lengths are N_SIGMA / sqrt(eigenvalue), along
    # the corresponding eigenvectors.
    eigenvalues, eigenvectors = np.linalg.eigh(fim)
    semi_axes = N_SIGMA / np.sqrt(np.clip(eigenvalues, 1e-30, None))
    angles = np.linspace(0.0, 2.0 * np.pi, 400)
    unit_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    ellipse = unit_circle * semi_axes[None, :] @ eigenvectors.T

    ax.plot(ellipse[:, 0], ellipse[:, 1], style, lw=1.8, label=label)

ax.axhline(0.0, color="0.8", lw=0.8, zorder=0)
ax.axvline(0.0, color="0.8", lw=0.8, zorder=0)
ax.set_xlabel("relative deviation in A12")
ax.set_ylabel("relative deviation in A21")
ax.set_title(f"Approximate {N_SIGMA:.0f}-sigma confidence regions\n"
             "(they very nearly coincide -- see the right panel)")
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="upper right")

# The ellipses nearly coincide, so the left panel alone would suggest the
# criterion does not matter. This panel plots how much WORSE each design is
# than the best one on each criterion, as a percentage: zero is best, so the
# winner has no bar and taller unambiguously means worse.
labels = list(scores)
titles = ["d_opt_criterion\n(ellipse area)",
          "a_opt_criterion\n(sum of squared axes)",
          "e_opt_criterion\n(longest axis)"]
bar_width = 0.8 / len(labels)

for k, label in enumerate(labels):
    percent_worse = []
    for j in range(3):
        value, reference = scores[label][j], best[j]
        # All three criteria are minimised, but d_opt and e_opt report
        # negative values, so normalise by MAGNITUDE to keep "worse" positive.
        percent_worse.append(100.0 * (value - reference) / abs(reference))
    bars = bx.bar(np.arange(3) + k * bar_width, percent_worse, bar_width,
                  label=label)
    for rect, value in zip(bars, percent_worse):
        if value < 1e-9:
            bx.annotate("best", (rect.get_x() + rect.get_width() / 2, 0),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7, rotation=90)

bx.set_xticks(np.arange(3) + 0.4 - bar_width / 2)
bx.set_xticklabels(titles, fontsize=8)
bx.axhline(0.0, color="0.3", lw=1.0)
bx.set_ylabel("% WORSE than the best design (0% = best)")
bx.set_title("Each criterion is best on its own measure\n"
             "and worse on the others -- by under 2% here")
bx.legend(fontsize=7)

fig.tight_layout()
print("\n  close the figure window to finish")
Designer.show_plots()
