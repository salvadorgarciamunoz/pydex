from pydex.core.designer import Designer
from case_1_no_ift_no_collocation_model import simulate, build_pyomo_model
import numpy as np

"""
case_1_no_ift_no_collocation.py
================================
D-optimal design for the first-order reaction using finite-difference
sensitivities.  No IFT, no orthogonal collocation.

The Pyomo Simulator (scipy/vode) integrates the DAE forward in time.
simulate() is a thin wrapper around build_pyomo_model(), so pydex's
finite differences perturb k and re-integrate each time.

The commented-out block below demonstrates what happens if
build_pyomo_model is also assigned to designer.pyomo_model_fn:
the safety check in designer.py detects active DerivativeVar components
(the model was never discretised) and raises a RuntimeError before
any IFT computation is attempted.

The optimal design should match case_1_no_ift.py — same FD path,
same model, different solver backend (Simulator vs collocation+IPOPT).
"""

designer_1 = Designer()
designer_1.simulate = simulate
# pyomo_model_fn is intentionally NOT assigned — finite differences only.
#
# Uncommenting the line below would trigger the safety switch in designer.py:
# designer_1.pyomo_model_fn = build_pyomo_model  # ← triggers RuntimeError

print("Sensitivity path: finite differences (Pyomo Simulator wrapper)")

theta_nom = np.array([0.25])  # value of k
designer_1.model_parameters = theta_nom

tic = designer_1.enumerate_candidates(
    bounds=[
        [0.1, 5],
    ],
    levels=[
        5,
    ],
)
designer_1.ti_controls_candidates = tic
designer_1.sampling_times_candidates = np.array([
    np.linspace(0, 50, 101)
    for _ in tic
])
designer_1._num_steps = 15
designer_1.initialize(verbose=2)

"""
===============================================================
[Optional]: check responses and sensitivities of all candidates
===============================================================
"""
if False:
    designer_1.simulate_candidates(plot_simulation_times=True)
    designer_1.plot_predictions()
    sens = designer_1.eval_sensitivities(save_sensitivities=False, store_predictions=True)
    designer_1.plot_sensitivities()

""" solve OED problem """
designer_1.design_experiment(
    designer_1.d_opt_criterion,
    solver="ipopt",
    solver_options={"linear_solver": "ma57"},
    optimize_sampling_times=True,
)

designer_1.print_optimal_candidates()
for n_exp in [2, 3, 4, 5, 6]:
    designer_1.apportion(n_exp)
designer_1.plot_optimal_efforts()
designer_1.plot_optimal_predictions()
designer_1.plot_optimal_sensitivities()
designer_1.show_plots()
