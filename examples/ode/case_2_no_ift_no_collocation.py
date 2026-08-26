"""
case_2_no_ift_no_collocation.py
================================
D-optimal Model-Based Design of Experiments for the A→B reaction using the
scipy.integrate ODE path with finite-difference (FD) sensitivities.

This is the third of three variants that demonstrate the full spectrum of
sensitivity methods available in pydex for this model:

  case_2.py                        — Pyomo.DAE collocation + IFT
                                     (exact symbolic sensitivities; fastest)

  case_2_no_ift.py                 — Pyomo.DAE collocation + FD
                                     (same NLP model; slower, needs robustness
                                     precautions against IPOPT infeasibility)

  case_2_no_ift_no_collocation.py  — scipy.integrate + FD  ← THIS FILE
                                     (no Pyomo; direct IVP integration;
                                     simplest and most robust FD path)

Why scipy instead of Pyomo collocation for the FD path
-------------------------------------------------------
Pyomo collocation + FD pays the cost of solving a large NLP (IPOPT, ~285
variables) for *every* FD perturbation step while still getting only approximate
derivatives.  When numdifftools perturbs a parameter to push CA near zero,
IPOPT's interior-point barrier may declare infeasibility — requiring parameter
clamping, relaxed bounds, and NaN fallbacks as defensive machinery.

The deeper reason to keep this variant around, though, is as an INDEPENDENT
CROSS-CHECK.  A collocation time-grid defect once made case_2_no_ift.py report
a D-optimal criterion of 45.31 against the correct 10.72 while IPOPT reported
"Optimal Solution Found" — and it was this scipy path disagreeing that localised
it.  See the PITFALL section in case_2_model.py.

scipy.integrate.solve_ivp avoids all of this:
  • No hard bounds — the integrator adapts step size; CA near zero is fine.
  • No symbolic expression tree — CA**α is a plain float at each timestep.
  • Much faster per call — an adaptive RK5 solve takes milliseconds vs. the
    ~0.1s IPOPT NLP solve.  Over 25 candidates × 4 parameters × 15 Richardson
    steps = 1500 evaluations, the total time drops from ~30s to ~2–4s.
  • No Pyomo dependency — this script imports only numpy, scipy, and pydex.

The only defensive line needed is `max(CA, 0.0)` in the ODE RHS to prevent
complex-valued powers near depletion.

Experimental design setup
--------------------------
The design problem is identical to case_2_no_ift.py and case_2.py:

  Reaction     : A → B  (irreversible power-law Arrhenius)
  Responses    : CA(t), CB(t)  [mol/L]
  Parameters   : θ = [θ₀, θ₁, α, ν]  (4 parameters)
  Controls     : CA0 (mol/L) ∈ {1, 2, 3, 4, 5}
                 T   (K)     ∈ {273.15, 285.65, 298.15, 310.65, 323.15}
  Candidates   : 5 × 5 = 25 (CA0, T) combinations
  Sampling     : 11 time points from 0 to 200 min (same for all candidates)

Three design rounds are performed:
  Round 1 — D-optimal, FIXED sampling grid (all 11 measured)
  Round 2 — D-optimal, n_spt=1
  Round 3 — D-optimal, n_spt=2

Expected results
----------------
Measured D-optimal criterion, round 1 (FIXED sampling grid, all 11 times):

    SUPERSEDED IN 0.6.0 -- re-measure before quoting. The values below were
    recorded when round 1 requested OPTIMIZED sampling times rather than a
    fixed grid: effort stayed free per (condition, time) cell, so those
    numbers describe a ONE-SAMPLE-per-run design, not the eleven-sample design
    the heading claims. Round 1 now passes n_spt = the full grid.

    For reference, case_2.py under the corrected round 1 reports 19.489976,
    and its round 2 (free per-cell effort) reports 10.657395 -- which is
    exactly the old round-1 figure below, confirming the two rounds were the
    same problem.

    old, mislabelled values:
    case_2.py                        (collocation + IFT)   10.657395
    case_2_no_ift.py                 (collocation + FD)    10.724136
    case_2_no_ift_no_collocation.py  (scipy + FD)          10.724134   <- here

The two FD paths agree to seven significant figures despite using completely
different forward solvers, which is the strong statement: Richardson-
extrapolated FD is reproducible across integrators.

IFT sits ~0.6% away, and that gap is real rather than error.  pydex's causal
IFT rebuild solves the model once per sampling time, so tau = t and each
measurement lands at normalised 1.0; the FD paths use a single solve with
tau = max(spt).  Different collocation grids, different truncation error.  IFT
is the more accurate of the two -- against a closed-form model it agrees to
1e-07 where FD sits near 1e-03.

All three select the same support (candidates 21 and 25) and, in round 3, the
same sampling-time variants to two decimal places.

Sensitivity path
----------------
  Method : Finite differences (Richardson extrapolation via numdifftools)
  Model  : scipy Radau integrator (stiff-safe, order 5)
  No Pyomo, no IPOPT, no collocation grid.

WHAT THIS SCRIPT DOES, AND IN WHAT ORDER
----------------------------------------
Three D-optimal design rounds on the same model and the same 25-candidate grid,
differing only in how much freedom the optimiser has over WHEN to sample.

  ROUND 1 — FIXED sampling grid    design_experiment(..., n_spt=<all times>)
      Every selected experiment is measured at ALL ELEVEN sampling times. The
      optimiser only chooses WHICH conditions (CA0, T) to run and their shares.

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
      -> Figure 1: optimal efforts
      -> apportion(2)

  ROUND 2 — one sample per EXPERIMENT   design_experiment(..., n_spt=1)
      n_spt = 1 constrains each individual experiment to a single sample. It
      does NOT constrain a candidate to a single sampling time: the same
      (CA0, T) condition may be run several times over, each run sampled at a
      different instant. Those alternatives appear as separate sampling
      schedules under one candidate. A typical result here is

          [Candidate 21]  (CA0 = 5, T = 273.15)
            Schedule 1 ~ [ 60.00]:   9.72% of experiments
            Schedule 2 ~ [ 80.00]:  14.65% of experiments
            Schedule 3 ~ [200.00]:  25.50% of experiments
          [Candidate 25]  (CA0 = 5, T = 323.15)
            Schedule 1 ~ [ 60.00]:  26.40% of experiments
            Schedule 2 ~ [160.00]:  23.73% of experiments

      i.e. five single-sample experiments in total, three of them at the same
      conditions but at different times. apportion(12) then rounds those shares
      to whole runs while tracking the proportions -- 9.72/14.65/25.50 becomes
      1/6, 2/6, 3/6 for candidate 21 -- so uneven efforts are preserved rather
      than flattened.

      (case_2.py and case_2_no_ift.py leave n_spt unset in their round 2, which
      behaves identically: both also return three schedules for candidate 21 and
      two for candidate 25.)
      -> Figure 2: optimal efforts
      -> apportion(12)

  ROUND 3 — two samples per run    design_experiment(..., n_spt=2)
      Each experiment collects exactly TWO samples; the optimiser picks the best
      PAIR of times per condition.
      -> Figure 3: optimal efforts
      -> apportion(12)

  THEN, once, after all three rounds:
      -> Figure 4: plot_optimal_predictions()
      -> Figure 5: plot_optimal_sensitivities()

  MIND THE FIGURE LAYOUT. Unlike case_2.py and case_2_no_ift.py, which call all
  three plot functions inside every round and so produce nine figures (three per
  round), this script only plots efforts per round and calls predictions and
  sensitivities once at the very end. Those last two figures therefore show ONLY
  ROUND 3's design -- the two-sample-per-run one. If you want to compare
  predicted responses across rounds, move those two calls inside each round as
  the other two scripts do.


READING THE PREDICTED-RESPONSE FIGURE
-------------------------------------
plot_optimal_predictions() draws CA(t) and CB(t) as dashed lines for each
selected candidate, with MARKERS at the times the design says to sample. Marker
size is proportional to the effort allocated there.

Because the figure reflects round 3 (n_spt = 2), its legend carries entries
reading "Sampling schedule 1", "Sampling schedule 2", each with its own marker
shape. A SAMPLING SCHEDULE is one particular set of sampling times for a given
experimental condition; two schedules on the same candidate mean run that same
(CA0, T) condition more than once, sampling at different times each time --
typically one batch early and one late, because the two regions of a decay curve
carry different information. Read the times off the marker positions; exact
values are in the printed report.


MEASURED RESULTS, ALL THREE VARIANTS
------------------------------------
Round 1 D-optimal criterion:

    case_2.py                        (collocation + IFT)   10.657395
    case_2_no_ift.py                 (collocation + FD)    10.724136
    case_2_no_ift_no_collocation.py  (scipy + FD)          10.724134   <- here

The two FD paths agree to seven significant figures despite completely
different forward solvers, which is the strong statement here: Richardson-
extrapolated FD is reproducible across integrators. IFT sits ~0.6% away and
that gap is discretisation, not error -- see the note in case_2_no_ift.py.

All three select the same support (candidates 21 and 25) and, in round 3, the
same sampling schedules to two decimal places. This variant runs the fastest
and is the only one that emits no infeasible-solve warnings at all.

"""

import os
import sys
import numpy as np
from matplotlib import pyplot as plt

# ── pydex import ──────────────────────────────────────────────────────────────
from pydex.core.designer import Designer

# ── model import ─────────────────────────────────────────────────────────────
# Add the examples/ode directory to sys.path so the model file can be found
# regardless of the working directory.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from case_2_no_ift_no_collocation_model import simulate

print("Sensitivity path: scipy Radau integration + finite differences")

# =============================================================================
# Nominal parameters
# =============================================================================
# k(T) = exp(θ₀ + θ₁*(T - 273.15)/T)
# At T=273.15 K: k_ref = 0.1 L/(mol·min)
# Ea = 5000 J/mol, R = 8.314159 J/(mol·K), T_ref = 273.15 K
pre_exp_constant = 0.1
activ_energy     = 5000.0
R                = 8.314159
T_ref            = 273.15

theta_0   = np.log(pre_exp_constant) - activ_energy / (R * T_ref)
theta_1   = activ_energy / (R * T_ref)
theta_nom = np.array([theta_0, theta_1, 1.0, 0.5])

# =============================================================================
# Candidate grid — 5 CA0 levels × 5 temperature levels = 25 candidates
# =============================================================================
CA0_candidates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])    # mol/L
T_candidates   = np.array([273.15, 285.65, 298.15, 310.65, 323.15])  # K

tic_candidates = np.array([
    [CA0, T]
    for CA0 in CA0_candidates
    for T   in T_candidates
])  # shape (25, 2)

# 11 equally-spaced sampling times from ~0 to 200 min
# t=0 is excluded (no information at t=0); small positive t_start instead
spt_grid = np.linspace(0.001, 200, 11)   # [0.001, 20, 40, ..., 200] min

# All candidates share the same sampling time grid
spt_candidates = np.tile(spt_grid, (len(tic_candidates), 1))  # shape (25, 11)

# =============================================================================
# Initialise pydex designer
# =============================================================================
designer_1 = Designer()
designer_1.simulate = simulate
# designer_1.pyomo_model_fn is NOT set → FD sensitivity path is used

designer_1.model_parameters          = theta_nom
designer_1.ti_controls_candidates    = tic_candidates
designer_1.sampling_times_candidates = spt_candidates
designer_1.error_cov = np.diag([0.1, 0.1])   # measurement noise: σ²=0.1 for CA, CB

designer_1.model_parameters_names = ["θ₀", "θ₁", "α", "ν"]
designer_1.ti_controls_names      = ["CA0 (mol/L)", "T (K)"]
designer_1.response_names         = ["CA", "CB"]

designer_1.initialize(verbose=1)

# =============================================================================
# Round 1 — D-optimal, FIXED sampling grid (all 11 measured per run)
# =============================================================================
designer_1.design_experiment(
    designer_1.d_opt_criterion,
    n_spt                   = designer_1.n_spt,
    solver                  = "ipopt",
    solver_options          = {"linear_solver": "ma57"},
    write                   = False,
)
designer_1.print_optimal_candidates()
designer_1.plot_optimal_efforts()
designer_1.apportion(2)

# =============================================================================
# Round 2 — D-optimal, sampling times optimised (the default: no n_spt)
# =============================================================================
designer_1.design_experiment(
    designer_1.d_opt_criterion,
    n_spt                   = 1,
    solver                  = "ipopt",
    solver_options          = {"linear_solver": "ma57"},
    write                   = False,
)
designer_1.print_optimal_candidates()
designer_1.plot_optimal_efforts()
designer_1.apportion(12)

# =============================================================================
# Round 3 — D-optimal, 2 sampling times per experiment
# =============================================================================
designer_1.design_experiment(
    designer_1.d_opt_criterion,
    n_spt                   = 2,
    solver                  = "ipopt",
    solver_options          = {"linear_solver": "ma57"},
    write                   = False,
)
designer_1.print_optimal_candidates()
designer_1.plot_optimal_efforts()
designer_1.apportion(12)

# =============================================================================
# Visualisation
# =============================================================================
designer_1.plot_optimal_predictions()
designer_1.plot_optimal_sensitivities(interactive=False)

designer_1.show_plots()
