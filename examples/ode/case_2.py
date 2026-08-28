from pydex.core.designer import Designer
from case_2_model import simulate, build_pyomo_model
import numpy as np

"""
case_2.py
=========
D-optimal design for the A→B reaction with Arrhenius kinetics.
IFT sensitivities via Pyomo collocation + IPOPT (PyomoNLP).

Model:  dCA/dt = -k * CA^α
        dCB/dt =  ν * k * CA^α
        k = exp(θ₀ + θ₁ * (T - 273.15) / T)

Four model parameters : [θ₀, θ₁, α, ν]
Two time-invariant controls : [CA0, T]
Two measurable responses : [CA, CB]

WHAT THIS SCRIPT DOES, AND IN WHAT ORDER
----------------------------------------
Three D-optimal design rounds on the same model and the same 25-candidate grid.
They differ only in how much freedom the optimiser has over WHEN to sample.
Each round prints a report and then produces three figures, so the run ends
with nine figures numbered in the order below.

  ROUND 1 — FIXED sampling grid             design_experiment(..., n_spt=<all times>)
      Every selected experiment is measured at ALL ELEVEN sampling times.
      The optimiser only chooses WHICH conditions (CA0, T) to run and how
      much of the budget each one gets.

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

      n_spt is the ONLY control over sampling times. There is no flag that
      switches optimisation on or off.
      -> Figures 1-3: optimal efforts, predicted responses, sensitivities
      -> apportion(2): the continuous design rounded to 2 physical runs

  ROUND 2 — sampling times optimised        design_experiment(...)   [the default]
      Now the optimiser also chooses WHEN to sample, one time point per
      experiment. The same experimental condition may appear more than once
      with different sampling times (see "sampling schedules" below).
      -> Figures 4-6
      -> apportion(12)

  ROUND 3 — exactly two samples per run     design_experiment(..., n_spt=2)
      As round 2, but each experiment collects exactly TWO samples. The
      optimiser picks the best PAIR of times for each condition.
      -> Figures 7-9
      -> apportion(12)

  Note the  designer_1.atomic_fims = None  line before round 3. It clears the
  cached atomic FIMs so they are rebuilt for the new 2-sample layout. It is not
  required: round 3 returns 13.429393 bit-identically with or without it, on
  both the IFT and the finite-difference paths. Kept as a cheap precaution when
  changing n_spt between rounds -- the cache is keyed on the sampling-time
  layout, so a stale entry would be a wrong answer rather than an error.


READING THE PREDICTED-RESPONSE FIGURES
--------------------------------------
plot_optimal_predictions() draws, for each selected candidate, the model
trajectories cA(t) and cB(t) as dashed lines, with MARKERS at the times where
the design says to take a sample. Marker size is proportional to the effort
allocated there, so a big marker means "most of your budget goes here".

In round 1 there is one marker per sampling time on the fixed grid. In rounds 2
and 3 the legend gains entries reading "Sampling schedule 1", "Sampling
schedule 2", and so on, each drawn with its own marker shape (circle, square,
hexagon, plus). Read the times off the marker positions on the time axis; the
exact values are also listed in the printed report above each figure.

A SAMPLING SCHEDULE is one particular set of sampling times for a given
experimental condition. Two schedules on the same candidate mean: run this
same (CA0, T) condition more than once, and sample at different times each
time. For example round 3 typically returns

    Candidate 21  (CA0 = 5, T = 273.15)
      schedule 1 ~ [ 60, 80]:   ~23% of experiments
      schedule 2 ~ [180, 200]:  ~25% of experiments

i.e. run that condition twice -- one batch sampled early, one late. Sampling
early and late on a decay curve is the usual pattern: the two regions carry
different information, so splitting the runs beats putting every sample in one
window. (Which schedule informs which parameter is not something this script
measures -- if you need that, compare the sensitivity magnitudes at those times
via plot_optimal_sensitivities.)

Earlier pydex versions labelled these "Variant 1", "Variant 2", which said
nothing about what a variant was.


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
case_2_model.py, and the PITFALL section of its module docstring for the full
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

"""

designer_1 = Designer()
designer_1.simulate       = simulate
designer_1.pyomo_model_fn = build_pyomo_model  # IFT sensitivities via Pyomo

print("IFT path: Collocation + IPOPT (PyomoNLP)")

# ── Nominal model parameter values ───────────────────────────────────────────
pre_exp_constant = 0.1
activ_energy     = 5000
theta_0 = np.log(pre_exp_constant) - activ_energy / (8.314159 * 273.15)
theta_1 = activ_energy / (8.314159 * 273.15)
theta_nom = np.array([theta_0, theta_1, 1.0, 0.5])   # [θ₀, θ₁, α, ν]
designer_1.model_parameters = theta_nom

# ── Experimental candidates ───────────────────────────────────────────────────
tic = designer_1.enumerate_candidates(
    bounds=[
        [1, 5],             # initial CA concentration (mol/L)
        [273.15, 323.15],   # reaction temperature (K)
    ],
    levels=[5, 5],
)
designer_1.ti_controls_candidates = tic

# Control labels, so the design table reads "CA0"/"T" rather than
# "Time-invariant Control 0"/"1".
designer_1.ti_controls_names = ["CA0", "T"]

# NOTE the first sampling time. 0.001 with a 200-minute horizon normalises to
# 5e-6, which sits a hair off the collocation node at 0. Embedding both would
# create a machine-epsilon finite element and silently corrupt the solve, so the
# model file snaps such times to the nearest node and warns when it does; see
# the PITFALL section in case_2_model.py for the full story.
designer_1.sampling_times_candidates = np.array([
    np.linspace(0.001, 200, 11)   # avoid t=0 with normalised time convention
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

# ── D-optimal design (FIXED sampling grid: every listed time measured) ────────
# n_spt equal to the NUMBER of listed times gives exactly one sampling
# schedule per candidate -- C(n, n) == 1 -- containing every time, so effort
# is allocated per EXPERIMENT rather than per (condition, time) cell.
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
# Clear the cached atomic FIMs so they are rebuilt for the new n_spt=2 layout.
# Not required -- round 3 returns 13.429393 with or without this line, on both
# sensitivity paths -- but a cheap precaution when changing n_spt between
# rounds, since a stale cache would be a wrong answer rather than an error.
designer_1.atomic_fims = None
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
