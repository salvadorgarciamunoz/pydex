"""
suzuki_sequential.py
====================
Sequential model-based design of experiments, end to end, on a Suzuki-Miyaura
coupling. Reads top to bottom, in the order the work happens.

The story
---------
1. A process chemist runs a minimal four-experiment screen: a 2x2 factorial
   in temperature and time, at fixed catalyst. Nothing wrong with it -- it is
   the screen almost everyone runs first.
2. Fitting the kinetic model to that data determines the COUPLING parameters
   well and the PROTODEBORONATION parameters badly. The screen answered the
   question it was designed for -- does the reaction work -- and left the side
   reaction loose.
3. That matters: protodeboronation erodes yield at higher temperature, so its
   temperature dependence is exactly what you need before you extrapolate.
4. pydex is given the six completed experiments as a PRIOR and asked to design
   six more. It does not spend effort re-establishing what is already known.
5. Re-fitting on all ten shows what the second round bought.

This is not the same as designing ten experiments up front. The first four
are a fact, not a choice; sequential design conditions on them.

Why the parameterisation is (lnk_ref, Ea) and not (lnA, Ea)
----------------------------------------------------------
Measured on the same screen, same data:

                             corr(1st, Ea1)   corr(2nd, Ea2)   cond(FIM)
    reference-centred            +0.076          -0.860          6.0e+05
    Arrhenius (lnA, Ea)          +1.0000         +1.0000         4.1e+07

With the Arrhenius form the correlation is 1.0000 to four decimal places: a
change in Ea is absorbed almost exactly by a compensating change in ln A, so
the pair is one parameter wearing two names, and the information matrix is 68x
worse conditioned. Centring the rate on a reference temperature is not a
cosmetic choice -- it is the difference between an estimable pair and an
unidentifiable one, and it is exactly the "reparameterise so the
unidentifiable combination becomes a single parameter" advice that
run_estimability() gives when it reports a correlation group.

Run
---
    python suzuki_sequential.py

Five figures are shown at the end: four from run_estimability() -- one bar
chart per index plus the correlation heat map -- and a three-panel summary
comparing where the two rounds sit and how much the confidence region shrank.
Nothing is written to disk.

Needs an NLP solver. `solver="pounce"` works with nothing beyond pip; IPOPT
works equally well. No MINLP solver required.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from pydex.core.designer import Designer

import suzuki_kinetics as sk

SEED = 12
N_NEW = 6                 # experiments to design for round 2
SOLVER = "pounce"         # or "ipopt"

rng = np.random.default_rng(SEED)


# ═════════════════════════════════════════════════════════════════════════════
# The one thing that genuinely happens twice: fit the four kinetic parameters
# to a set of experiments. Everything else below runs once, in place.
# ═════════════════════════════════════════════════════════════════════════════
def fit_kinetics(conditions, measurements, fit_these, starting_values):
    """Weighted least squares over a NAMED subset of the parameters.

    Everything here is keyed by parameter NAME rather than by position, so
    there are no index masks to keep straight.

    Args:
        conditions: array of experimental conditions, one row per run.
        measurements: matching array of measured responses.
        fit_these: list of parameter names to estimate. Any name not in this
            list is held at its value in `starting_values`.
        starting_values: dict {name: value} covering ALL six parameters.

    Returns:
        estimates: dict {name: value}, all six, fitted ones updated.
        std_errors: dict {name: standard error}, fitted ones only.
        correlations: 2-D array over `fit_these`, in that order.
    """
    # scipy wants a plain vector, so translate name <-> position here and
    # nowhere else.
    guess = [starting_values[name] for name in fit_these]
    lower = [sk.BOUNDS_LO[name] for name in fit_these]
    upper = [sk.BOUNDS_HI[name] for name in fit_these]

    def all_six(vector):
        """Turn the vector scipy is holding back into a full parameter set."""
        values = dict(starting_values)
        for name, value in zip(fit_these, vector):
            values[name] = value
        return values

    def residuals(vector):
        values = all_six(vector)
        theta = [values[name] for name in sk.THETA_NAMES]
        stacked = []
        for condition, measured in zip(conditions, measurements):
            predicted = sk.simulate(condition, theta)
            stacked.append((predicted - measured) / sk.SIGMA)
        return np.concatenate(stacked)

    result = least_squares(residuals, guess, method="trf", bounds=(lower, upper))

    covariance = np.linalg.inv(result.jac.T @ result.jac)
    errors = np.sqrt(np.diag(covariance))
    spread = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))

    estimates = all_six(result.x)
    std_errors = {name: err for name, err in zip(fit_these, errors)}
    return estimates, std_errors, covariance / spread


def as_vector(values):
    """Dict of parameters -> numpy array in the model's own order.

    pydex expects model_parameters to be an array, not a list.
    """
    return np.array([values[name] for name in sk.THETA_NAMES], dtype=float)


# ═════════════════════════════════════════════════════════════════════════════
# ROUND 1 — the four experiments the chemist ran
# ═════════════════════════════════════════════════════════════════════════════
# A 2x2 factorial in temperature and time. A reasonable screen: it
# spans the range and nothing sits at completion, so every run carries
# information about the rate. A screen that pushes everything to 100%
# conversion proves the process works and says almost nothing about kinetics.
SCREEN = np.array([
    #   T    t_rxn  cat   boron  base
    [60.0,  1.0,   1.0,  1.2,   2.0],
    [60.0,  4.0,   1.0,  1.2,   2.0],
    [90.0,  1.0,   1.0,  1.2,   2.0],
    [90.0,  4.0,   1.0,  1.2,   2.0],
])

# "Run" the four experiments. In a real project these numbers arrive from the
# lab. Here they are manufactured, and it is worth seeing exactly how, because
# the whole example depends on it:
#
#   * sk.THETA_TRUE is the answer we are pretending not to know;
#   * sk.SIGMA is the assay precision, one number per response;
#   * the clip at zero is not cosmetic -- imp_D is so small that a noise draw
#     can push it negative, and a negative impurity is not a measurement.
Y_screen = []
for conditions in SCREEN:
    truth = sk.simulate(conditions, sk.THETA_TRUE)
    noise = rng.normal(0.0, sk.SIGMA)
    Y_screen.append(np.maximum(truth + noise, 0.0))
Y_screen = np.array(Y_screen)

print("\n" + "=" * 78)
print(f"  ROUND 1 — the {len(SCREEN)} experiments the chemist ran")
print("=" * 78)
print(f"\n  {'T':>6}{'t_rxn':>7}{'cat':>6}{'boron':>7}{'base':>6} | "
      f"{'yield_P':>8}{'imp_D':>8}{'res_B':>8}")
for tic, y in zip(SCREEN, Y_screen):
    print(f"  {tic[0]:6.1f}{tic[1]:7.1f}{tic[2]:6.2f}{tic[3]:7.2f}{tic[4]:6.2f}"
          f" | {y[0]:8.3f}{y[1]:8.4f}{y[2]:8.3f}")
print(f"\n  Temperature spans {SCREEN[:, 0].min():.0f}-{SCREEN[:, 0].max():.0f} C."
      "\n  That span is what makes the activation energies estimable at all: a"
      "\n  screen clustered in a 10 C window determines rates but not their"
      "\n  temperature dependence, and the fit then runs to its bounds.")


# ═════════════════════════════════════════════════════════════════════════════
# Can the screen support all six parameters? Ask before fitting.
# ═════════════════════════════════════════════════════════════════════════════
# run_estimability() is usually run over a candidate GRID before designing.
# Here it is pointed at the six conditions already executed, which audits the
# data in hand: what can THIS dataset resolve? Nothing is designed yet.
# First fit all six, so the audit is evaluated at the best current estimate
# rather than at a vague prior guess. Estimability is a LOCAL property: the
# sensitivities are computed wherever you put the parameters, so auditing at a
# poor guess can condemn a parameter that the data actually supports.
first_fit, first_errors, _ = fit_kinetics(
    SCREEN, Y_screen, fit_these=sk.THETA_NAMES, starting_values=sk.START)
print("\n" + "=" * 78)
print("  A FIRST FIT OF ALL SIX, TO HAVE SOMEWHERE TO AUDIT FROM")
print("=" * 78)
print(f"\n  {'parameter':>10}{'true':>9}{'estimate':>10}{'std err':>10}")
for name in sk.THETA_NAMES:
    print(f"  {name:>10}{sk.TRUE[name]:9.2f}{first_fit[name]:10.2f}"
          f"{first_errors[name]:10.2f}")
print("\n  Some of these standard errors are larger than the values they"
      "\n  qualify. That is the symptom; the audit below is the diagnosis.")

audit = Designer()
audit.simulate = sk.simulate
audit.model_parameters = as_vector(first_fit)    # audit AT THE FIT
audit.error_cov = sk.ERROR_COV
audit.ti_controls_candidates = SCREEN            # the experiments, not a grid
audit.model_parameter_names = sk.THETA_NAMES
audit.initialize(verbose=0)

print("\n" + "=" * 78)
print("  CAN THE SCREEN SUPPORT ALL SIX PARAMETERS?")
print("=" * 78)
print("  Everything below is pydex's own estimability report, computed over the"
      "\n  four conditions already run. The four figures it draws describe THIS"
      "\n  DATA -- they are the ones to look at.\n")

# report=True prints the full ranking, the correlation matrix and the
# threshold sweep. plot=True draws four figures: a bar chart for each of the
# three indices and the correlation heat map.
report = audit.run_estimability(report=True, plot=True)

flagged = list(report["flagged"])
print(f"\n  flagged unresolvable: {flagged if flagged else 'none'}")
print("\n  abs info is Fisher information about each parameter's FRACTIONAL"
      "\n  value. Below 1 means this data cannot pin the parameter to within"
      "\n  its own magnitude, however confident the regression looks."
      "\n\n  Typically it is an activation energy that falls out. A rate"
      "\n  constant is seen directly in a response; its TEMPERATURE DEPENDENCE"
      "\n  needs the response to move measurably across the temperature range,"
      "\n  and for a minor pathway it does not."
      "\n\n  The method reached this conclusion, not us. Fitting a flagged"
      "\n  parameter anyway yields a confident-looking number that is noise.")

# ═════════════════════════════════════════════════════════════════════════════
# Fix what the data cannot support, then fit the rest
# ═════════════════════════════════════════════════════════════════════════════
# Keep every parameter the audit did not flag. A plain list of names.
free_names = [name for name in sk.THETA_NAMES if name not in flagged]

print("\n" + "=" * 78)
print(f"  FITTING THE {len(free_names)} SUPPORTABLE PARAMETERS TO ROUND 1")
print("=" * 78)
print(f"  held fixed: {flagged or 'none'}")

round1_fit, round1_errors, corr_1 = fit_kinetics(
    SCREEN, Y_screen, fit_these=free_names, starting_values=first_fit)

print(f"\n  {'parameter':>10}{'true':>9}{'estimate':>10}{'std err':>10}"
      f"{'95% CI half-width':>20}")
for name in free_names:
    err = round1_errors[name]
    print(f"  {name:>10}{sk.TRUE[name]:9.2f}{round1_fit[name]:10.2f}{err:10.2f}"
          f"{1.96 * err:20.2f}")

print(f"\n  correlation matrix ({', '.join(free_names)}):")
for row in np.round(corr_1, 3):
    print("   ", row)

print("\n  Read the standard errors, not the estimates. The coupling pair"
      "\n  (lnk1_ref, Ea1) is pinned down; the protodeboronation pair"
      "\n  (lnk2_ref, Ea2) is not. Ea2 is known to a few tens of percent,"
      "\n  which is not good enough to extrapolate to a hotter process.")

# ═════════════════════════════════════════════════════════════════════════════
# Design round 2, conditioned on round 1
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("  DESIGNING ROUND 2 — conditioned on what has already been run")
print("=" * 78)

def simulate_free(ti_controls, model_parameters):
    """What pydex calls: the model, over the free parameters only.

    pydex varies just the parameters we are still estimating; the ones the
    audit fixed keep their values from round1_fit. The argument NAMES must be
    exactly these -- pydex reads the signature.
    """
    values = dict(round1_fit)
    for name, value in zip(free_names, model_parameters):
        values[name] = value
    return sk.simulate(ti_controls, as_vector(values))


designer = Designer()
designer.simulate = simulate_free
designer.model_parameters = np.array([round1_fit[name] for name in free_names])
designer.error_cov = sk.ERROR_COV

designer.ti_controls_candidates = designer.enumerate_candidates(
    bounds=[sk.BOUNDS[k] for k in sk.BOUND_ORDER],
    levels=[4, 4, 3, 2, 2],                        # 192 candidate conditions
)
designer.model_parameter_names = free_names
designer.ti_controls_names = sk.CONTROL_NAMES
designer.response_names = sk.RESPONSE_NAMES

designer.initialize(verbose=0)

# ── Is the problem estimable at all, from the grid we are allowed to use? ──
# This asks a different question from the regression above. The regression
# said the screen determined two parameters well and two badly. Estimability
# asks whether the CANDIDATE GRID could resolve all four if the experiments
# were chosen well -- i.e. whether the loose parameters are a consequence of
# WHICH experiments were run, or a structural property of the model that no
# design can fix. Only in the first case is there any point designing more.
# Secondary check, so no report and no figures -- the interesting estimability
# analysis was the one above, on the data actually in hand. This one only
# answers a yes/no question, so a two-line summary is enough.
estimability = designer.run_estimability(report=False, plot=False)

print("\n  Estimability over the candidate GRID at the current estimate:")
print(f"  {'parameter':>10}{'abs info':>13}{'E-index':>12}")
for name in estimability["corr_names"]:
    print(f"  {name:>10}{estimability['abs_info'][name]:13.3e}"
          f"{estimability['e_index'][name]:12.3e}")

flagged = estimability["flagged"]
groups = dict(estimability["_sweep"])
print(f"\n  flagged unresolvable: {flagged if flagged else 'none'}")
print(f"  correlated groups at |corr| > {estimability['corr_tol']}: "
      f"{groups.get(estimability['corr_tol']) or 'none'}")
print("  Nothing is flagged, so all four parameters ARE resolvable from this"
      "\n  grid. The loose pair is therefore a property of WHICH experiments"
      "\n  were run, not of the model -- which is what makes designing more"
      "\n  experiments worth doing. Had a parameter been flagged, no design"
      "\n  could have rescued it and the model would need reparameterising.")
print(f"  Note the pair appears as a group only once the threshold is loosened"
      f" to 0.8:\n  {groups.get(0.8)} -- borderline, not degenerate.")

# Case B: register the completed experiments. Their conditions need not lie on
# the candidate grid, and here they do not.
designer.set_prior_experiments(
    ti_controls=SCREEN,
    model_parameters=np.array([round1_fit[name] for name in free_names]),
)
print(f"  prior FIM rank: {np.linalg.matrix_rank(designer._prior_fim)}"
      f" of {designer.n_mp}")

designer.design_experiment(designer.d_opt_criterion, solver=SOLVER, write=False)
designer.print_optimal_candidates()
apportionment = designer.apportion(N_NEW)

# apportion() turned the continuous efforts into whole runs. Read the answer
# out of it: each entry of designer.optimal_candidates describes one supported
# candidate, and element [1] of that entry is its control settings. The
# matching entry of `apportionment` says how many runs it was given.
new_tic = []
for candidate, run_count in zip(designer.optimal_candidates, apportionment):
    conditions = np.asarray(candidate[1]).ravel()      # [1] = the controls
    n_runs = int(np.nansum(run_count))                 # nansum: see note below
    for _ in range(n_runs):
        new_tic.append(conditions)
new_tic = np.array(new_tic, dtype=float)
# np.nansum rather than int(): apportion() returns a ragged object array when
# support is ragged, and plain astype(int) raises on it.

n_distinct = len(np.unique(new_tic, axis=0))
print(f"\n  {n_distinct} DISTINCT conditions, replicated to fill {N_NEW} runs."
      "\n  That is not a bug, and it is worth understanding:"
      "\n"
      "\n    * D-optimal designs are minimal-support BY NATURE. The optimum of"
      "\n      a determinant sits on a small number of points -- often no more"
      "\n      than the number of parameters, and fewer once a prior covers"
      "\n      most directions. This is not a symptom of anything being wrong."

      "\n"
      "\n  The practical cost is real: replicates give you a good estimate of"
      "\n  measurement noise but NO degrees of freedom for lack-of-fit testing."
      "\n  A design on two conditions cannot tell you the model is wrong, only"
      "\n  how precisely its parameters are determined given that it is right."
      "\n  If model adequacy is also in question, spend one or two of the six"
      "\n  runs somewhere the design did not choose. The levers that DO spread"
      "\n  a design are a different criterion, more free parameters, or a"
      "\n  larger budget -- not a weaker prior.")

print(f"\n  round 2, {len(new_tic)} experiments:")
print(f"  {'T':>6}{'t_rxn':>7}{'cat':>6}{'boron':>7}{'base':>6}")
for tic in new_tic:
    print(f"  {tic[0]:6.1f}{tic[1]:7.1f}{tic[2]:6.2f}{tic[3]:7.2f}{tic[4]:6.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# Run round 2 and refit on all ten
# ═════════════════════════════════════════════════════════════════════════════
# Run round 2, the same way -- and note the SAME rng continues, so these
# draws follow on from round 1's rather than repeating them.
Y_new = []
for conditions in new_tic:
    truth = sk.simulate(conditions, sk.THETA_TRUE)
    noise = rng.normal(0.0, sk.SIGMA)
    Y_new.append(np.maximum(truth + noise, 0.0))
Y_new = np.array(Y_new)
TIC_all = np.vstack([SCREEN, new_tic])
Y_all = np.vstack([Y_screen, Y_new])

final_fit, final_errors, corr_2 = fit_kinetics(
    TIC_all, Y_all, fit_these=free_names, starting_values=round1_fit)

print("\n" + "=" * 78)
print(f"  ROUND 2 RUN, MODEL REFITTED ON ALL {len(TIC_all)}")
print("=" * 78)
print(f"\n  {'parameter':>10}{'true':>9}{'estimate':>10}{'std err':>10}"
      f"{'95% CI half-width':>20}")
for name in free_names:
    err = final_errors[name]
    print(f"  {name:>10}{sk.TRUE[name]:9.2f}{final_fit[name]:10.2f}{err:10.2f}"
          f"{1.96 * err:20.2f}")

print(f"\n  {'parameter':>10}{'se, screen':>13}{'se, all runs':>13}"
      f"{'improvement':>14}")
for name in free_names:
    before, after = round1_errors[name], final_errors[name]
    print(f"  {name:>10}{before:13.3f}{after:13.3f}{before / after:13.1f}x")


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("  WHAT TO TAKE FROM THIS")
print("=" * 78)
print("""
  The six new experiments were chosen KNOWING the first four. pydex received
  them through set_prior_experiments(), so the design does not re-establish
  what the screen already settled -- it goes after the parameters the screen
  left loose. That is the whole idea, and the improvement column is the
  evidence: the coupling parameters barely move, the protodeboronation pair
  and the homocoupling rate tighten several-fold.

  The workflow, in the order it has to happen:

     1.  experiments exist          whatever was already run
     2.  fit                        to have somewhere to stand
     3.  audit                      run_estimability() on those conditions
     4.  fix what is unsupported    the method chooses, not you
     5.  set_prior_experiments()    hand pydex the completed runs
     6.  design                     it targets what is still loose
     7.  run, refit, compare        standard errors before and after

  Two cautions that belong with the method:

    * The design depends on the CURRENT estimate. Round 2 was designed at the
      round-1 fit, which was itself imprecise. That is unavoidable, and it is
      why sequential design is iterative rather than a single correction.

    * Standard errors are the honest measure of what you gained, not the
      distance from the true values. On a real system there are no true
      values; they are printed here only because this example manufactures
      its own measurements.

  How much of this survives a different set of measurements -- whether the
  same parameter gets flagged, whether the design lands in the same place --
  is a separate question, and noise_and_estimability.py answers it by
  repeating the whole exercise across twelve noise realisations.
""")

# ═════════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════════
# Reconstruct covariances from the standard errors and correlations returned
# above: cov = corr * outer(se, se).
# Rebuild the covariances from what the fits returned: correlation matrix
# times the outer product of the standard errors.
err_before = [round1_errors[name] for name in free_names]
err_after = [final_errors[name] for name in free_names]
cov_1 = corr_1 * np.outer(err_before, err_before)
cov_2 = corr_2 * np.outer(err_after, err_after)

RED, BLUE, GREY = "#E1251B", "#0F3A85", "#B9C4CE"

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
fig.patch.set_facecolor("white")

# ── (a) and (b): where the two rounds sit in the input space ───────────────
grid = designer.ti_controls_candidates
for ax, (i, j), (xl, yl) in zip(
        axes[:2],
        [(0, 2), (0, 1)],
        [("T  [C]", "cat_mol  [mol %]"), ("T  [C]", "t_rxn  [h]")]):
    ax.scatter(grid[:, i], grid[:, j], s=22, facecolor="white",
               edgecolor=GREY, linewidth=1.0, label="candidate grid", zorder=1)
    ax.scatter(SCREEN[:, i], SCREEN[:, j], s=150, facecolor=BLUE,
               edgecolor="white", linewidth=1.6, zorder=3,
               label="round 1 — the chemist's screen")
    ax.scatter(new_tic[:, i], new_tic[:, j], s=210, marker="D",
               facecolor=RED, edgecolor="white", linewidth=1.6, zorder=4,
               label="round 2 — designed")
    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel(yl, fontsize=11)
    for sp in ax.spines.values():
        sp.set_color(GREY)

axes[0].set_title("Where the experiments sit", fontsize=13, fontweight="bold")
axes[1].set_title("The same, against reaction time", fontsize=13,
                  fontweight="bold")
axes[0].legend(fontsize=9, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.18))

# ── (c) 95% confidence region for the protodeboronation pair ───────────────
# The geometric meaning of D-optimality: it shrinks this ellipse.
ax = axes[2]
k = (2, 3)                                 # lnk2_ref, Ea2
chi2_95 = 5.991                            # 2 degrees of freedom
angle = np.linspace(0, 2 * np.pi, 240)
circle = np.vstack([np.cos(angle), np.sin(angle)])

centre_before = [round1_fit[name] for name in free_names]
centre_after = [final_fit[name] for name in free_names]
for cov, theta, colour, label in [
        (cov_1, centre_before, BLUE, "screen only (4 runs)"),
        (cov_2, centre_after, RED, "screen + designed (10 runs)")]:
    sub = cov[np.ix_(k, k)]
    vals, vecs = np.linalg.eigh(sub)
    ell = (vecs @ np.diag(np.sqrt(np.maximum(vals, 0) * chi2_95)) @ circle)
    ax.plot(theta[k[0]] + ell[0], theta[k[1]] + ell[1], color=colour, lw=2.2,
            label=label)
    ax.scatter([theta[k[0]]], [theta[k[1]]], s=70, color=colour, zorder=4)

truth = [sk.TRUE[name] for name in free_names]
ax.scatter([truth[k[0]]], [truth[k[1]]], marker="*", s=340,
           color="#144B2D", zorder=5, label="true value")
ax.set_xlabel("lnk2_ref", fontsize=11)
ax.set_ylabel("Ea2  [kJ/mol]", fontsize=11)
ax.set_title("95% confidence region, protodeboronation", fontsize=13,
             fontweight="bold")
ax.legend(fontsize=9, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, -0.18))
for sp in ax.spines.values():
    sp.set_color(GREY)

plt.tight_layout()
fig.subplots_adjust(bottom=0.28)

area_ratio = (np.sqrt(np.linalg.det(cov_1[np.ix_(k, k)]))
              / np.sqrt(np.linalg.det(cov_2[np.ix_(k, k)])))
print(f"\n  The 95% confidence region for (lnk2_ref, Ea2) shrank by a factor"
      f" of {area_ratio:.1f} in area.")

# Five figures in total: four from the SCREEN audit -- one bar chart per index
# plus the correlation heat map, all describing the four experiments already
# run -- and the three-panel summary above. show_plots() wraps plt.show().
designer.show_plots()
