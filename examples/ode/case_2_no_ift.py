from pydex.core.designer import Designer
from case_2_no_ift_model import simulate
import numpy as np

"""
case_2_no_ift.py
================
D-optimal design for the A→B reaction with Arrhenius kinetics.
Finite-difference sensitivities via Pyomo collocation wrapper.

pyomo_model_fn is intentionally NOT assigned — pydex computes
sensitivities by finite differences on top of simulate().

The optimal design should match case_2.py — same collocation model,
different sensitivity method.

IMPORTANT — the sampling-time grid is a trap on collocation models
-------------------------------------------------------------------
`np.linspace(0.001, 200, 11)` below looks harmless. It is the trigger for a
failure that shipped undetected in this example for a long time.

The model normalises time by tau = max(sampling_times) = 200, so the first
sampling time maps to 0.001/200 = 5e-6 -- five microseconds away from the
collocation node at 0.0, but not equal to it. Embedding both produces a finite
element of width ~1e-16 next to elements of width 5e-2, and the collocation
solve then converges to a non-physical branch WHILE IPOPT REPORTS SUCCESS:
CA rises to 31 mol/L from CA0 = 5, and the exact invariant CA + CB/nu = CA0 is
violated by ~70 mol/L. Refining nfe does not help.

The model file now guards against this (see build_collocation_grid in
case_2_no_ift_model.py, and the PITFALL section of its module docstring for the full
diagnosis). You will see a RuntimeWarning saying the first sampling time was
read from the nearest node instead of being embedded exactly -- that warning is
the guard working, not a problem.

If you write your own collocation model, three cheap habits catch this class of
bug: assert a conservation law inside simulate(); print the min/max
finite-element width after building the grid; and cross-check one candidate
against an independent integrator. Details in the model file.

Two ways to avoid triggering it at all:
  * start the sampling grid at a time that is a comfortable fraction of the
    horizon -- np.linspace(1.0, 200, 11) is exact to 1e-13 here, versus a
    600% error for 0.001; or
  * guard the grid construction, as the model file now does, which is the
    better option because it protects every future sampling grid.


WHAT THIS SCRIPT DOES, AND IN WHAT ORDER
----------------------------------------
Three D-optimal design rounds on the same model and the same 25-candidate grid,
differing only in how much freedom the optimiser has over WHEN to sample. Each
round prints a report and produces three figures (one per plot call), so the run
ends with nine figures numbered in the order below.

  ROUND 1 — FIXED sampling grid       design_experiment(..., n_spt=<all times>)
      Every selected experiment is measured at ALL ELEVEN sampling times. The
      optimiser only chooses WHICH conditions (CA0, T) to run and how much of
      the budget each gets.

      pydex offers exactly three treatments of sampling times, selected by
      n_spt:
        * omit n_spt          -- sampling times are OPTIMIZED: effort is spent
                                 per (condition, time) cell and the optimiser
                                 picks which listed times to measure.
        * n_spt=k             -- exactly k samples per run; the optimiser
                                 chooses WHICH k.
        * n_spt=<all listed>  -- the grid is FIXED: one schedule per candidate
                                 holding every listed time, so effort is spent
                                 per EXPERIMENT and all of them are measured.
      (This example claimed the fixed-grid behaviour while requesting the
      optimized one until pydex 0.6.0.)
      -> Figures 1-3: optimal efforts, predicted responses, sensitivities
      -> apportion(2)

  ROUND 2 — sampling times optimised  design_experiment(...)   [the default]
      The optimiser now also chooses WHEN to sample, one time point per
      experiment.
      -> Figures 4-6
      -> apportion(12)

  ROUND 3 — two samples per run       design_experiment(..., n_spt=2)
      Each experiment collects exactly TWO samples; the optimiser picks the
      best PAIR of times for each condition.
      -> Figures 7-9
      -> apportion(12)

Unlike case_2.py this script does not clear designer_1.atomic_fims before
round 3. It does not need to -- round 3 returns 10.671126 with or without the
reset -- and neither does case_2.py, whose comment to the contrary is stale.


READING THE PREDICTED-RESPONSE FIGURES
--------------------------------------
plot_optimal_predictions() draws cA(t) and cB(t) as dashed lines for each
selected candidate, with MARKERS at the times the design says to sample. Marker
size is proportional to the effort allocated there.

In rounds 2 and 3 the legend gains entries reading "Sampling schedule 1",
"Sampling schedule 2", each with its own marker shape (circle, square, hexagon,
plus). A SAMPLING SCHEDULE is one particular set of sampling times for a given
experimental condition; two schedules on the same candidate mean run that same
(CA0, T) condition more than once, sampling at different times each time. Read
the times off the marker positions; exact values are in the printed report.


HOW THIS COMPARES WITH THE OTHER TWO VARIANTS
---------------------------------------------
Round 1 D-optimal criterion, all three sensitivity paths:

    case_2.py                        (collocation + IFT)   10.657395
    case_2_no_ift.py                 (collocation + FD)    10.724136   <- here
    case_2_no_ift_no_collocation.py  (scipy + FD)          10.724134

The two FD paths agree to seven significant figures despite using different
forward solvers. IFT sits ~0.6% away, and that gap is discretisation rather
than error: pydex's causal IFT rebuild solves the model once per sampling time
(tau = t) while the FD paths use one solve with tau = max(spt). IFT is the more
accurate of the two against a closed-form reference.

This variant is the slowest of the three -- it solves a ~285-variable NLP for
every finite-difference perturbation -- and it will emit "Converged to a locally
infeasible point" warnings when a perturbed parameter pushes CA toward zero.
Those are handled by the NaN fallback in the model file and do not invalidate
the result.

"""

designer_1 = Designer()
designer_1.simulate = simulate
# pyomo_model_fn intentionally NOT assigned — finite differences only

print("Sensitivity path: finite differences (Pyomo collocation wrapper)")

# ── Nominal model parameter values ───────────────────────────────────────────
pre_exp_constant = 0.1
activ_energy     = 5000
theta_0 = np.log(pre_exp_constant) - activ_energy / (8.314159 * 273.15)
theta_1 = activ_energy / (8.314159 * 273.15)
theta_nom = np.array([theta_0, theta_1, 1.0, 0.5])
designer_1.model_parameters = theta_nom

# ── Experimental candidates ───────────────────────────────────────────────────
tic = designer_1.enumerate_candidates(
    bounds=[
        [1, 5],
        [273.15, 323.15],
    ],
    levels=[5, 5],
)
designer_1.ti_controls_candidates = tic

# NOTE the first sampling time. 0.001 with a 200-minute horizon normalises to
# 5e-6, which sits a hair off the collocation node at 0 and used to produce a
# machine-epsilon finite element that silently corrupted the solve. The model
# file guards against it now and warns when it engages; see the PITFALL section
# in case_2_no_ift_model.py for the full story.
designer_1.sampling_times_candidates = np.array([
    np.linspace(0.001, 200, 11)
    for _ in tic
])

# ── Optional metadata ─────────────────────────────────────────────────────────
designer_1.measurable_responses = [0, 1]
designer_1.candidate_names = np.array([f"Candidate {i+1}" for i, _ in enumerate(tic)])
designer_1.response_names  = ["$c_A$", "$c_B$"]
designer_1.model_parameter_names = [
    r"$\theta_0$", r"$\theta_1$", r"$\alpha$", r"$\nu$",
]

# ── Error covariance ──────────────────────────────────────────────────────────
designer_1.error_cov = np.diag([0.1, 0.1])

designer_1.initialize(verbose=2)

# ── D-optimal design (fixed sampling times) ───────────────────────────────────
designer_1.design_experiment(
    designer_1.d_opt_criterion,
    n_spt=designer_1.n_spt,
    solver="ipopt",
    solver_options={"linear_solver": "ma57"},
    write=False,
)
designer_1.print_optimal_candidates()
designer_1.plot_optimal_efforts()
designer_1.plot_optimal_predictions()
designer_1.plot_optimal_sensitivities(interactive=False)
designer_1.apportion(2)

# ── D-optimal design (sampling times optimised) ───────────────────────────────
designer_1.design_experiment(
    designer_1.d_opt_criterion,
    solver="ipopt",
    solver_options={"linear_solver": "ma57"},
    write=False,
)
designer_1.print_optimal_candidates()
designer_1.plot_optimal_efforts()
designer_1.plot_optimal_predictions()
designer_1.plot_optimal_sensitivities(interactive=False)
designer_1.apportion(12)

# ── D-optimal design (exactly 2 sampling times) ───────────────────────────────
designer_1.design_experiment(
    designer_1.d_opt_criterion,
    n_spt=2,
    solver="ipopt",
    solver_options={"linear_solver": "ma57"},
    write=False,
)
designer_1.print_optimal_candidates()
designer_1.plot_optimal_efforts()
designer_1.plot_optimal_predictions()
designer_1.plot_optimal_sensitivities(interactive=False)
designer_1.apportion(12)

designer_1.show_plots()
