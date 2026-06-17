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
    optimize_sampling_times=False,
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
    optimize_sampling_times=True,
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
    optimize_sampling_times=True,
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
