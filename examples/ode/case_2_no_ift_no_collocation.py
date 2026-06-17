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
Pyomo collocation + FD is the worst of all worlds: you pay the cost of solving
a large NLP (IPOPT, ~285 variables) for *every* FD perturbation step, while
still getting only approximate derivatives.  When numdifftools perturbs a
parameter to push CA near zero, IPOPT's interior-point barrier may declare
infeasibility — requiring parameter clamping, relaxed bounds, and NaN fallbacks
as defensive machinery.

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
  Round 1 — D-optimal, fixed sampling times (all 11)
  Round 2 — D-optimal, optimize_sampling_times=True, n_spt=1
  Round 3 — D-optimal, optimize_sampling_times=True, n_spt=2

Expected results
----------------
The scipy FD sensitivities should be close to (but not identical with) the
Pyomo collocation FD sensitivities from case_2_no_ift.py, since both use
Richardson-extrapolated FD — the only difference is the integrator.  Both
will differ from the IFT sensitivities in case_2.py because FD is approximate
and the two integrators discretise the ODE differently.

Sensitivity path
----------------
  Method : Finite differences (Richardson extrapolation via numdifftools)
  Model  : scipy Radau integrator (stiff-safe, order 5)
  No Pyomo, no IPOPT, no collocation grid.
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
# Round 1 — D-optimal, fixed sampling times (all 11 per candidate)
# =============================================================================
designer_1.design_experiment(
    designer_1.d_opt_criterion,
    optimize_sampling_times = False,
    solver                  = "ipopt",
    solver_options          = {"linear_solver": "ma57"},
    write                   = False,
)
designer_1.print_optimal_candidates()
designer_1.plot_optimal_efforts()
designer_1.apportion(2)

# =============================================================================
# Round 2 — D-optimal, 1 sampling time per experiment (optimize_sampling_times)
# =============================================================================
designer_1.design_experiment(
    designer_1.d_opt_criterion,
    optimize_sampling_times = True,
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
    optimize_sampling_times = True,
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
