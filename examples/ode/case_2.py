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
# Reset atomic_fims so pydex recomputes them for the new n_spt=2 grid.
# Without this, pydex reuses the shape (n_c*11, 4, 4) array from the
# previous round and tries to index it with the new (n_c*2) layout,
# causing an IndexError.
designer_1.atomic_fims = None
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
